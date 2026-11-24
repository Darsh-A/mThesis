"""v19 mega-test: Z*-sampled potluck across every Pop II yield table.

The v19 prior samples the pre-enrichment metallicity ``Z*`` inside the Pop II
table's nonzero range and derives ``M_dil = M_Zpre / Z*``.  This removes the
low-Z clamp that affected the earlier M_dil-prior runs.

For each table the runner writes a self-contained subdirectory with:

- the generated dataset (50k accepted mocks),
- explanation plots (the old v7 set),
- per-element contours (the old v14 set),
- random-forest performance and feature importance,
- Li+ calibration/support/candidate scoring,
- a Z*/M_dil coverage panel,
- the Salvadori Eq. 13 bookkeeping check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..calibration import run_phase_b
from ..config import ModelConfig
from ..dataset import generate_dataset
from ..io import load_dataset, save_dataset
from ..observation import load_li_records
from ..validation import eq13_check
from ..yields.popii import z_config_for_table
from .classifier_plots import make_plots
from .element_contours import run_element_contours
from .explanation_plots import run_explanation_plots

DEFAULT_TABLES = ("Limongi18", "Nomoto13", "WW95")

# Nonzero metallicity span of each table.  Sampling Z* inside this span is the
# whole point: the Pop II yields are never asked for a metallicity the table
# does not resolve.
TABLE_Z_RANGES: dict[str, tuple[float, float]] = {
    "Limongi18": (3.236e-5, 1.345e-2),
    "Nomoto13": (1.0e-3, 5.0e-2),
    "Nomoto13_CK20": (1.0e-3, 5.0e-2),
    "WW95": (1.345e-6, 1.345e-2),
}

# The full table range is reported, but the sampled range stops at 1e-2.  Above
# that the final [Fe/H] is outside the Li+ box, and a 1e3 Msun burst cannot fit
# inside the implied 10% cloud budget.  Sampling the unphysical tail only
# produces rejections and a Pop II contribution that is zero by construction.
TABLE_Z_SAMPLED: dict[str, tuple[float, float]] = {
    table: (low, min(high, 1.0e-2))
    for table, (low, high) in TABLE_Z_RANGES.items()
}

# CK20 needs the low-Z sampler nodes from v14_1; the other tables use their
# default sampler grids.
CK20_Z_NODES = (
    1.0e-12,
    1.0e-9,
    1.0e-6,
    1.0e-4,
    1.0e-3,
    4.0e-3,
    8.0e-3,
    2.0e-2,
    5.0e-2,
)


def table_grid(table: str) -> tuple[Any, int]:
    """Return the Pop II sampler grid for one table."""
    if table == "Nomoto13_CK20":
        return CK20_Z_NODES, len(CK20_Z_NODES)
    return z_config_for_table(table)


def table_config(
    base: ModelConfig,
    table: str,
    n_samples: int,
    seed: int,
    outdir: Path,
) -> ModelConfig:
    """Return the v19 configuration for one Pop II table."""
    z_lo, z_hi = TABLE_Z_SAMPLED[table]
    nodes, size = table_grid(table)
    return replace(
        base,
        prior=replace(
            base.prior,
            z_star_range=(z_lo, z_hi),
            z_cr=z_lo,
            z_max=z_hi,
        ),
        yields=replace(
            base.yields,
            popii_table=table,
            popii_z_nodes=nodes,
            popii_z_grid_size=size,
        ),
        runtime=replace(
            base.runtime,
            n_samples=n_samples,
            chunk_size=max(1000, min(n_samples, 5000)),
            seed=seed,
            max_draw_factor=60,
            output_dir=str(outdir),
        ),
    )


def _impute(x: np.ndarray) -> np.ndarray:
    """Median-impute NaNs column by column."""
    x = np.asarray(x, dtype=np.float32).copy()
    for column in range(x.shape[1]):
        values = x[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if np.any(finite) else 0.0
        values[~finite] = fill
    return x


def _importance(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    outdir: Path,
) -> dict[str, Any]:
    """Fit random forests and report feature importance for both blocks."""
    outdir.mkdir(parents=True, exist_ok=True)
    weight = arrays["importance_weight"].astype(float)
    y = arrays["y_f_05"].astype(int)
    target = arrays["f_zp"].astype(float)

    blocks = {
        "noiseless": (
            _impute(arrays["features"]),
            list(meta["feature_names"]),
        ),
        "observed": (
            _impute(arrays["obs_features"]),
            list(meta.get("observation_feature_names", meta["feature_names"])),
        ),
    }
    result: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for column, (block, (x, names)) in enumerate(blocks.items()):
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )
        classifier.fit(x, y, sample_weight=weight)
        regressor = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )
        regressor.fit(x, target, sample_weight=weight)

        for row, (model, label) in enumerate(
            (
                (classifier, r"classifier $y_{f\geq0.5}$"),
                (regressor, r"regressor $f_Z^P$"),
            )
        ):
            values = np.asarray(model.feature_importances_, dtype=float)
            order = np.argsort(values)[::-1][:20]
            axis = axes[row, column]
            axis.barh(
                np.arange(order.size)[::-1],
                values[order],
                color="tab:blue",
                alpha=0.8,
            )
            axis.set_yticks(np.arange(order.size)[::-1])
            axis.set_yticklabels([names[i] for i in order], fontsize=7)
            axis.set_xlabel("random-forest importance")
            axis.set_title(f"{block}: {label}")
            axis.grid(axis="x", alpha=0.25)
        result[block] = {
            "features": names,
            "classifier_importance": dict(
                zip(names, np.asarray(classifier.feature_importances_).tolist())
            ),
            "regressor_importance": dict(
                zip(names, np.asarray(regressor.feature_importances_).tolist())
            ),
        }
    fig.tight_layout()
    fig.savefig(outdir / "rf_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (outdir / "rf_importance.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def _zstar_budget(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    z_range: tuple[float, float],
    outdir: Path,
) -> dict[str, Any]:
    """Summarise the Z*/M_dil coverage of one table."""
    outdir.mkdir(parents=True, exist_ok=True)
    z = arrays["z_star"].astype(float)
    mdil = arrays["mdil"].astype(float)
    good = np.isfinite(z) & (z > 0.0) & np.isfinite(mdil) & (mdil > 0.0)
    log_z = np.log10(z[good])
    log_mdil = np.log10(mdil[good])
    z_lo, z_hi = z_range
    summary = {
        "table": meta["config"]["yields"]["popii_table"],
        "z_star_range": [z_lo, z_hi],
        "z_star_sampled": list(TABLE_Z_SAMPLED[meta["config"]["yields"]["popii_table"]]),
        "z_star": {
            "p1": float(np.percentile(z[good], 1)),
            "p16": float(np.percentile(z[good], 16)),
            "p50": float(np.percentile(z[good], 50)),
            "p84": float(np.percentile(z[good], 84)),
            "p99": float(np.percentile(z[good], 99)),
            "min": float(z[good].min()),
            "max": float(z[good].max()),
            "fraction_below_range": float(np.mean(z[good] < z_lo)),
            "fraction_above_range": float(np.mean(z[good] > z_hi)),
        },
        "mdil": {
            "p16": float(np.percentile(mdil[good], 16)),
            "p50": float(np.percentile(mdil[good], 50)),
            "p84": float(np.percentile(mdil[good], 84)),
            "min": float(mdil[good].min()),
            "max": float(mdil[good].max()),
        },
        "feh": {
            "p16": float(np.nanpercentile(arrays["feh"], 16)),
            "p50": float(np.nanpercentile(arrays["feh"], 50)),
            "p84": float(np.nanpercentile(arrays["feh"], 84)),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(log_z, bins=70, color="tab:blue", alpha=0.85)
    axes[0].axvline(np.log10(z_lo), color="tab:orange", lw=1.5, ls="--", label="table Z* range")
    axes[0].axvline(np.log10(z_hi), color="tab:orange", lw=1.5, ls="--")
    axes[0].set_xlabel(r"$\log_{10} Z_\star$")
    axes[0].set_ylabel("mocks")
    axes[0].set_title("Pre-enrichment metallicity")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].hist(log_mdil, bins=70, color="tab:green", alpha=0.85)
    axes[1].axvline(np.log10(2.0e5), color="0.5", lw=1.2, ls=":", label="old prior 2e5–1e9")
    axes[1].axvline(np.log10(1.0e9), color="0.5", lw=1.2, ls=":")
    axes[1].set_xlabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
    axes[1].set_ylabel("mocks")
    axes[1].set_title("Derived dilution mass")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "zstar_mdil_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (outdir / "zstar_budget.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_table(
    table: str,
    base: ModelConfig,
    outdir: Path,
    n_samples: int,
    seed: int,
    force: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Generate one table's dataset and run every subtest."""
    table_dir = outdir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    dataset = table_dir / "dataset.npz"
    config = table_config(base, table, n_samples, seed, table_dir)

    if force or not dataset.exists():
        if verbose:
            print(f"[{table}] generating {n_samples} mocks", flush=True)
        arrays, meta = generate_dataset(config, verbose=verbose)
        save_dataset(dataset, arrays, meta)
    else:
        if verbose:
            print(f"[{table}] reusing {dataset}", flush=True)
        arrays, meta = load_dataset(dataset)

    if verbose:
        print(f"[{table}] explanation plots", flush=True)
    explanation = run_explanation_plots(
        dataset, outdir=table_dir / "explanation_plots", seed=seed
    )
    if verbose:
        print(f"[{table}] element contours", flush=True)
    contours = run_element_contours(
        dataset, outdir=table_dir / "element_contours"
    )
    if verbose:
        print(f"[{table}] RF diagnostics", flush=True)
    rf_paths = make_plots(
        dataset, table_dir / "rf_performance", model="rf"
    )
    importance = _importance(arrays, meta, table_dir / "rf_importance")
    if verbose:
        print(f"[{table}] Li+ validation", flush=True)
    try:
        li = run_phase_b(
            dataset,
            table_dir / "li_validation",
            model="rf",
            n_boot=100,
        )
    except Exception as error:  # keep the rest of the table's results
        li = {"error": str(error)}
    if verbose:
        print(f"[{table}] Z*/M_dil budget and Eq. 13", flush=True)
    budget = _zstar_budget(
        arrays,
        meta,
        TABLE_Z_SAMPLED[table],
        table_dir / "zstar_budget",
    )
    eq13 = eq13_check(
        config,
        pisn_mass=200.0,
        f_pisn=0.5,
        f_ratio=1.0e-3,
        tpop2_myr=10.0,
    )

    return {
        "table": table,
        "dataset": str(dataset),
        "z_star_range": list(TABLE_Z_RANGES[table]),
        "z_star_sampled": list(TABLE_Z_SAMPLED[table]),
        "n_accepted": int(meta["n_accepted"]),
        "n_drawn": int(meta["n_drawn"]),
        "acceptance": float(meta["acceptance"]),
        "explanation_plots": explanation,
        "element_contours": {
            "n_elements": len(contours["elements"]),
            "n_ratios": len(contours["ratios"]),
        },
        "rf_performance": json.loads(
            (table_dir / "rf_performance" / "metrics.json").read_text()
        ),
        "rf_importance": importance,
        "li_validation": li,
        "zstar_budget": budget,
        "eq13": eq13,
    }


def make_summary(outdir: str | Path) -> dict[str, Any]:
    """Write the cross-table comparison figure and a compact summary."""
    outdir = Path(outdir)
    manifest = json.loads((outdir / "manifest.json").read_text())
    records = load_li_records()
    li_median = float(np.nanmedian(records.feh))
    rows: list[dict[str, Any]] = []
    for table, result in manifest["results"].items():
        budget = result["zstar_budget"]
        rows.append(
            {
                "table": table,
                "acceptance": result["acceptance"],
                "z16": budget["z_star"]["p16"],
                "z50": budget["z_star"]["p50"],
                "z84": budget["z_star"]["p84"],
                "mdil16": budget["mdil"]["p16"],
                "mdil50": budget["mdil"]["p50"],
                "mdil84": budget["mdil"]["p84"],
                "feh50": budget["feh"]["p50"],
                "auc": result["rf_performance"]["auc"],
                "brier": result["rf_performance"]["brier"],
                "positive_rate": result["rf_performance"]["positive_rate"],
                "precision_fpr1": result["rf_performance"][
                    "precision_at_fpr_0.001_pi_2.5e-3"
                ],
                "out_of_model": result["li_validation"].get("n_out_of_model"),
                "eq13_max": result["eq13"]["max_abs_difference"],
            }
        )
    summary = {
        "li_feh_median": li_median,
        "rows": rows,
    }
    summary_dir = outdir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    tables = [row["table"] for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axis = axes[0, 0]
    axis.errorbar(
        positions,
        [row["z50"] for row in rows],
        yerr=[
            [row["z50"] - row["z16"] for row in rows],
            [row["z84"] - row["z50"] for row in rows],
        ],
        fmt="o",
        capsize=4,
        color="tab:blue",
    )
    for position, row in zip(positions, rows):
        low, high = TABLE_Z_RANGES[row["table"]]
        axis.plot([position, position], [low, high], color="0.75", lw=8, alpha=0.5, zorder=0)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(tables, rotation=20, ha="right")
    axis.set_ylabel(r"$Z_\star$")
    axis.set_title("Pre-enrichment metallicity (band: table range)")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[0, 1]
    axis.errorbar(
        positions,
        [row["mdil50"] for row in rows],
        yerr=[
            [row["mdil50"] - row["mdil16"] for row in rows],
            [row["mdil84"] - row["mdil50"] for row in rows],
        ],
        fmt="o",
        capsize=4,
        color="tab:green",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(tables, rotation=20, ha="right")
    axis.set_ylabel(r"$M_{\rm dil}$ [$M_\odot$]")
    axis.set_title("Derived dilution mass")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 0]
    bins = np.linspace(-4.2, -1.4, 55)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(rows)))
    for color, row in zip(colors, rows):
        data = np.load(outdir / row["table"] / "dataset.npz")
        feh = data["feh"].astype(float)
        axis.hist(
            feh[np.isfinite(feh)],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.6,
            color=color,
            label=row["table"],
        )
    axis.axvline(li_median, color="black", ls="--", lw=1.2, label="Li+ median")
    axis.set_xlabel("[Fe/H]")
    axis.set_ylabel("density")
    axis.set_title("Final metallicity")
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    auc = [row["auc"] for row in rows]
    brier = [row["brier"] for row in rows]
    pos = [row["positive_rate"] for row in rows]
    sub = axis.twinx()
    axis.bar(positions - 0.18, auc, width=0.34, color="tab:blue", alpha=0.8, label="AUC")
    axis.bar(positions + 0.18, brier, width=0.34, color="tab:orange", alpha=0.8, label="Brier")
    sub.plot(positions, pos, "o-", color="black", label="positive rate")
    axis.set_xticks(positions)
    axis.set_xticklabels(tables, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("AUC / Brier")
    sub.set_ylabel("positive fraction")
    axis.set_title("RF performance and class balance")
    axis.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([], [], color="tab:blue", lw=6, label="AUC"),
        plt.Line2D([], [], color="tab:orange", lw=6, label="Brier"),
        plt.Line2D([], [], color="black", marker="o", label="positive rate"),
    ]
    axis.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("v19: Z*-sampled potluck across Pop II tables", fontsize=13)
    fig.tight_layout()
    fig.savefig(summary_dir / "table_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def make_transfer(outdir: str | Path) -> dict[str, Any]:
    """Train on one yield table and test on every other table."""
    outdir = Path(outdir)
    manifest = json.loads((outdir / "manifest.json").read_text())
    tables = list(manifest["results"])
    arrays = {
        table: load_dataset(outdir / table / "dataset.npz")[0]
        for table in tables
    }

    def block(table: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = arrays[table]
        if name == "observed":
            x = data["obs_features"].astype(float)
        else:
            x = data["features"].astype(float)
        return x, data["y_f_05"].astype(int), data["importance_weight"].astype(float)

    def impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train = train.copy()
        test = test.copy()
        for column in range(train.shape[1]):
            values = train[:, column]
            finite = np.isfinite(values)
            fill = float(np.median(values[finite])) if np.any(finite) else 0.0
            train[~finite, column] = fill
            test[~np.isfinite(test[:, column]), column] = fill
        return train, test

    result: dict[str, Any] = {"tables": tables, "blocks": {}}
    for block_name in ("noiseless", "observed"):
        auc = np.full((len(tables), len(tables)), np.nan)
        brier = np.full((len(tables), len(tables)), np.nan)
        for i, train_table in enumerate(tables):
            x_train, y_train, w_train = block(train_table, block_name)
            for j, test_table in enumerate(tables):
                x_test, y_test, w_test = block(test_table, block_name)
                train_x, test_x = impute(x_train, x_test)
                model = RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=0,
                )
                model.fit(train_x, y_train, sample_weight=w_train)
                probability = model.predict_proba(test_x)[:, 1]
                auc[i, j] = roc_auc_score(y_test, probability, sample_weight=w_test)
                brier[i, j] = brier_score_loss(y_test, probability, sample_weight=w_test)
        result["blocks"][block_name] = {
            "auc": dict(
                zip(
                    tables,
                    [
                        dict(zip(tables, row.tolist()))
                        for row in auc
                    ],
                )
            ),
            "brier": dict(
                zip(
                    tables,
                    [
                        dict(zip(tables, row.tolist()))
                        for row in brier
                    ],
                )
            ),
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for axis, matrix, title, cmap, vmin, vmax in (
            (axes[0], auc, "AUC", "viridis", 0.5, 1.0),
            (axes[1], brier, "Brier", "magma", 0.0, 0.3),
        ):
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks(np.arange(len(tables)))
            axis.set_xticklabels(tables, rotation=30, ha="right", fontsize=8)
            axis.set_yticks(np.arange(len(tables)))
            axis.set_yticklabels(tables, fontsize=8)
            axis.set_xlabel("test table")
            axis.set_ylabel("train table")
            axis.set_title(f"{block_name}: {title}")
            for i in range(len(tables)):
                for j in range(len(tables)):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] < 0.7 else "black",
                        fontsize=7,
                    )
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            outdir / "summary" / f"rf_transfer_{block_name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    with (outdir / "summary" / "rf_transfer.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_v19(
    outdir: str | Path = "src/potluck/tests/v19",
    n_samples: int = 50000,
    seed: int = 42,
    tables: tuple[str, ...] = DEFAULT_TABLES,
    force: bool = False,
    verbose: bool = True,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Run the full v19 mega-test."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = ModelConfig() if config is None else config
    manifest: dict[str, Any] = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "tables": list(tables),
        "z_star_sampling": "log-uniform Z* inside the sampled part of each table's nonzero range; M_dil = M_Zpre/Z*",
        "mstar_ii_range": "default 1e3-1e7 Msun",
        "results": {},
    }
    for table in tables:
        if table not in TABLE_Z_RANGES:
            raise ValueError(f"unknown table {table!r}")
        manifest["results"][table] = run_table(
            table,
            base,
            outdir,
            n_samples,
            seed,
            force,
            verbose,
        )
    with (outdir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    make_summary(outdir)
    make_transfer(outdir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="src/potluck/tests/v19")
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tables", default=",".join(DEFAULT_TABLES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if args.summary_only:
        print(json.dumps(make_summary(args.outdir), indent=2))
        make_transfer(args.outdir)
        return 0
    tables = tuple(t.strip() for t in args.tables.split(",") if t.strip())
    manifest = run_v19(
        outdir=args.outdir,
        n_samples=args.n,
        seed=args.seed,
        tables=tables,
        force=args.force,
        verbose=not args.quiet,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
