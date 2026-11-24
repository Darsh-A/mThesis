"""Highest-f_ZP Li+ stars: observed abundance pattern under each Pop II policy.

The APOGEE counterpart (``apogee_fzp``) plots the observed pattern of every star
above a predicted f_ZP threshold. This does the same for the Li+ catalogue, using
the same trained models and the same observation-layer input that
``li_histograms`` summarises.

For each policy the regressor is trained exactly as in ``calibration.py``, every
Li+ star is scored, and the top candidates are drawn one per panel with their
catalogue error bars. Elements that the catalogue does not measure for that star
are shaded, so the reader can see how much of the pattern is actually present.

Outputs (``--outdir``):
    li_top_fzp_<policy>.png     observed pattern, top candidates
    li_top_fzp_<policy>.csv     the ranked candidate list behind the figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..calibration import fit_phase_b_models, real_matrix
from ..constants import LI_FEATURE_ELEMENTS, REFERENCE
from ..diagnostics.ceiling import observation_mask
from ..io import load_dataset
from ..observation import GIANT_LOGG, load_li_records

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)
PLOT_ELEMENTS = tuple(e for e in LI_FEATURE_ELEMENTS if e != REFERENCE)


def score_policy(
    base: Path, policy: str, seed: int = 0
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Train on one policy and score every Li+ star."""
    arrays, _ = load_dataset(base / "application" / policy / "dataset.npz")
    models = fit_phase_b_models(arrays, seed=seed, model="gb")

    records = load_li_records()
    x_real = real_matrix(records, keep=observation_mask(arrays))
    expected = np.clip(models.regressor.predict(x_real), 0.0, 1.0)
    score = models.classifier.predict_proba(x_real)[:, 1]
    posterior = models.calibrator.predict(score)

    table = pd.DataFrame(
        {
            "star": records.labels,
            "feh": records.feh,
            "logg": records.logg,
            "is_giant": records.logg < GIANT_LOGG,
            "expected_fz": expected,
            "p_calibrated": posterior,
            "n_measured": np.isfinite(
                records.xfe[:, [LI_FEATURE_ELEMENTS.index(e) for e in PLOT_ELEMENTS]]
            ).sum(axis=1),
        }
    ).sort_values("expected_fz", ascending=False, ignore_index=True)

    xfe = records.xfe[:, [LI_FEATURE_ELEMENTS.index(e) for e in PLOT_ELEMENTS]]
    sigma = records.sigma[:, [LI_FEATURE_ELEMENTS.index(e) for e in PLOT_ELEMENTS]]
    flags = records.flags[:, [LI_FEATURE_ELEMENTS.index(e) for e in PLOT_ELEMENTS]]
    return table, {
        "xfe": xfe,
        "sigma": sigma,
        "flags": flags,
        "catalogue_index": {label: i for i, label in enumerate(records.labels)},
    }


def _panel_values(
    table: pd.DataFrame,
    data: dict[str, np.ndarray],
    top: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every value and error bar drawn in this figure's panels."""
    values: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for index in range(min(top, len(table))):
        row_index = data["catalogue_index"][table.iloc[index]["star"]]
        observed = data["xfe"][row_index]
        sigma = data["sigma"][row_index]
        finite = np.isfinite(observed)
        values.append(observed[finite])
        errors.append(np.where(np.isfinite(sigma[finite]), sigma[finite], 0.0))
    if not values:
        return np.empty(0), np.empty(0)
    return np.concatenate(values), np.concatenate(errors)


def shared_ylim(
    scored: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]],
    top: int,
    pad: float = 0.12,
) -> tuple[float, float]:
    """One y range covering every panel of every policy, so figures compare."""
    values: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for table, data in scored.values():
        v, e = _panel_values(table, data, top)
        values.append(v)
        errors.append(e)
    values = np.concatenate(values)
    errors = np.concatenate(errors)
    low = float(np.min(values - errors))
    high = float(np.max(values + errors))
    span = high - low
    low -= pad * span
    high += pad * span
    return (low, high)


def _panel(
    axis: plt.Axes,
    index: int,
    table: pd.DataFrame,
    data: dict[str, np.ndarray],
    ylim: tuple[float, float],
) -> None:
    row = table.iloc[index]
    row_index = data["catalogue_index"][row["star"]]
    observed = data["xfe"][row_index]
    error = np.where(np.isfinite(data["sigma"][row_index]), data["sigma"][row_index], 0.0)
    flags = data["flags"][row_index]

    x = np.arange(len(PLOT_ELEMENTS))
    axis.axhline(0.0, color="0.7", lw=0.8, zorder=0)

    detected = flags == 1
    limits = flags == 2
    axis.errorbar(
        x[detected],
        observed[detected],
        yerr=error[detected],
        fmt="o",
        ms=5,
        color="black",
        ecolor="0.35",
        capsize=2,
        zorder=3,
        label="Li+",
    )
    if np.any(limits):
        axis.errorbar(
            x[limits],
            observed[limits],
            fmt="v",
            ms=5,
            color="tab:orange",
            zorder=3,
            label="upper limit",
        )
    for position in np.flatnonzero(~detected & ~limits):
        axis.axvspan(position - 0.4, position + 0.4, color="tab:red", alpha=0.10, zorder=0)

    axis.set_xticks(x, labels=list(PLOT_ELEMENTS), rotation=45, ha="right", fontsize=8)
    axis.set_ylim(*ylim)
    axis.set_ylabel("[X/Fe]")
    axis.grid(axis="y", alpha=0.25)
    kind = "giant" if row["is_giant"] else "dwarf"
    axis.set_title(
        f"{row['star']}   [Fe/H]={row['feh']:.2f}   {kind}   "
        f"fZP={row['expected_fz']:.3f}   ({int(row['n_measured'])}/"
        f"{len(PLOT_ELEMENTS)} measured)",
        fontsize=8,
        loc="left",
    )


def plot_policy(
    table: pd.DataFrame,
    data: dict[str, np.ndarray],
    policy: str,
    path: Path,
    ylim: tuple[float, float],
    top: int = 6,
) -> None:
    ncols = 2
    nrows = int(np.ceil(top / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.1 * nrows), squeeze=False, sharex=True, sharey=True
    )
    for axis, index in zip(axes.ravel(), range(min(top, len(table)))):
        _panel(axis, index, table, data, ylim)
    for axis in axes.ravel()[min(top, len(table)):]:
        axis.axis("off")

    axes.ravel()[0].legend(fontsize=6, frameon=False)

    fig.suptitle(
        f"Li+ observed abundance pattern, {policy}: top {top} by predicted "
        r"$f_Z^P$   (red bands = not measured; shared y range "
        f"{ylim[0]:.2f} to {ylim[1]:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_star(
    star: str,
    scored: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]],
    path: Path,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """One star, the same observed pattern, one panel per yield policy.

    The observed points are identical in every panel; only the predicted
    ``f_Z^P`` changes, so the figure isolates the yield systematic.
    """
    policies = list(scored)
    fig, axes = plt.subplots(
        len(policies), 1, figsize=(11.0, 2.6 * len(policies)),
        sharex=True, sharey=True, squeeze=False,
    )
    x = np.arange(len(PLOT_ELEMENTS))
    rows = []
    for axis, policy in zip(axes.ravel(), policies):
        table, data = scored[policy]
        match = table.index[table["star"] == star]
        if len(match) == 0:
            raise KeyError(f"{star} not present in policy {policy}")
        index = int(match[0])
        row = table.iloc[index]
        row_index = data["catalogue_index"][star]
        observed = data["xfe"][row_index]
        error = np.where(np.isfinite(data["sigma"][row_index]), data["sigma"][row_index], 0.0)
        flags = data["flags"][row_index]

        detected = flags == 1
        limits = flags == 2
        axis.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        axis.errorbar(
            x[detected], observed[detected], yerr=error[detected],
            fmt="o", ms=5, color="black", ecolor="0.35", capsize=2,
            zorder=3, label="Li+",
        )
        if np.any(limits):
            axis.errorbar(
                x[limits], observed[limits], fmt="v", ms=5,
                color="tab:orange", zorder=3, label="upper limit",
            )
        for position in np.flatnonzero(~detected & ~limits):
            axis.axvspan(position - 0.4, position + 0.4, color="tab:red", alpha=0.10, zorder=0)

        axis.set_ylabel("[X/Fe]")
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(
            f"{policy}   fZP = {row['expected_fz']:.3f}   "
            f"(rank {index + 1} of {len(table)})",
            fontsize=9, loc="left",
        )
        rows.append(
            {"star": star, "policy": policy, "expected_fz": float(row["expected_fz"]),
             "rank": int(index) + 1, "n_stars": int(len(table))}
        )

    axes.ravel()[0].legend(fontsize=6, frameon=False)
    axes.ravel()[-1].set_xticks(x, labels=list(PLOT_ELEMENTS), rotation=45, ha="right", fontsize=8)
    if ylim is not None:
        axes.ravel()[0].set_ylim(*ylim)
    first = scored[policies[0]][0]
    info = first[first["star"] == star].iloc[0]
    kind = "giant" if info["is_giant"] else "dwarf"
    fig.suptitle(
        f"{star}   [Fe/H] = {info['feh']:.2f}   {kind}   "
        f"{int(info['n_measured'])}/{len(PLOT_ELEMENTS)} measured   "
        "same data, different yield table",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _tag(star: str) -> str:
    """Filename-safe form of a star name."""
    return star.replace("+", "p").replace("-", "m")


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/li_top_fzp",
    policies: tuple[str, ...] = POLICIES,
    top: int = 6,
    seed: int = 0,
    stars: tuple[str, ...] = (),
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scored: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    for policy in policies:
        scored[policy] = score_policy(base, policy, seed=seed)

    ylim = shared_ylim(scored, top)
    print(f"shared y range: {ylim[0]:.2f} to {ylim[1]:.2f}")

    summary: dict[str, Any] = {"shared_ylim": list(ylim)}
    for policy in policies:
        table, data = scored[policy]
        table.to_csv(outdir / f"li_top_fzp_{policy}.csv", index=False)
        plot_policy(
            table, data, policy, outdir / f"li_top_fzp_{policy}.png", ylim, top=top
        )
        summary[policy] = {
            "n_stars": int(len(table)),
            "max_expected_fz": float(table["expected_fz"].max()),
            "median_expected_fz": float(table["expected_fz"].median()),
            "top": table.head(top)[
                ["star", "feh", "is_giant", "expected_fz", "n_measured"]
            ].to_dict(orient="records"),
        }
        print(f"{policy:20s} max fZP = {summary[policy]['max_expected_fz']:.4f}")

    with (outdir / "li_top_fzp_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    if stars:
        comparison = pd.concat(
            [
                plot_single_star(
                    star,
                    scored,
                    outdir / f"li_star_{_tag(star)}_fzp.png",
                    ylim=ylim,
                )
                for star in stars
            ],
            ignore_index=True,
        )
        comparison.to_csv(outdir / "li_star_comparison.csv", index=False)
        summary["stars"] = comparison.to_dict(orient="records")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default="src/potluck/tests/v20/processed/li_top_fzp")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--star",
        action="append",
        default=[],
        help="star name to plot across all policies (repeatable)",
    )
    args = parser.parse_args(argv)
    run(args.base, args.outdir, top=args.top, seed=args.seed, stars=tuple(args.star))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
