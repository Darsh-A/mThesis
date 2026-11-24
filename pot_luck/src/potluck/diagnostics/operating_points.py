"""Operating-point report: TPR at fixed FPR, precision at survey prevalence,
expected true/false positives, reliability and Brier.

Headline feature set is the general 8-element set (Zn, Si, Al, S, Ca, Ti, Cr,
Ni) with Gaussian element-level noise. The Li+ 11-element observed block is
reported as a comparison, not the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from ..io import load_dataset
from .element_selection import POLICIES, _ratios
from .v20_mega import _balance

GENERAL8 = ("Zn", "Si", "Al", "S", "Ca", "Ti", "Cr", "Ni")
PREVALENCES = (1.0e-3, 2.5e-3, 1.0e-2)
FPR_TARGETS = (1.0e-2, 1.0e-3)


def _oof(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, n_trees: int
) -> np.ndarray:
    probability = np.zeros(len(y))
    for train, test in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ).split(x, y):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def _tpr_at_fpr(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, target: float
) -> float:
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    return float(np.interp(target, fpr, tpr))


def _precision(tpr: float, fpr: float, pi: float) -> float:
    return tpr * pi / (tpr * pi + fpr * (1.0 - pi))


def _bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    n_boot: int = 200,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(0)
    rows: dict[str, list[float]] = {
        "tpr_1e-2": [],
        "tpr_1e-3": [],
    }
    for prevalence in PREVALENCES:
        rows[f"precision_{prevalence:.1e}"] = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        tpr1 = _tpr_at_fpr(y[index], score[index], weight[index], 1.0e-2)
        tpr2 = _tpr_at_fpr(y[index], score[index], weight[index], 1.0e-3)
        rows["tpr_1e-2"].append(tpr1)
        rows["tpr_1e-3"].append(tpr2)
        for prevalence in PREVALENCES:
            rows[f"precision_{prevalence:.1e}"].append(
                _precision(tpr2, 1.0e-3, prevalence)
            )
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in rows.items()
        if values
    }


def _fpr_groups(
    y: np.ndarray,
    score: np.ndarray,
    f_zp: np.ndarray,
    weight: np.ndarray,
    target: float,
) -> dict[str, float]:
    fpr, _, thresholds = roc_curve(y, score, sample_weight=weight)
    index = min(np.searchsorted(fpr, target), len(thresholds) - 1)
    threshold = thresholds[index]
    predicted = score >= threshold
    negative = y == 0
    clear = negative & (f_zp < 0.25)
    near = negative & (f_zp >= 0.25) & (f_zp < 0.5)
    return {
        "threshold": float(threshold),
        "clear_n": int(clear.sum()),
        "near_n": int(near.sum()),
        "fpr_clear": float(predicted[clear].mean()) if clear.any() else float("nan"),
        "fpr_near": float(predicted[near].mean()) if near.any() else float("nan"),
        "fpr_all": float(predicted[negative].mean()),
    }


def _reliability(
    y: np.ndarray, score: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    edges = np.quantile(score, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    points = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (score >= lower) & (score <= upper)
        if not mask.any():
            continue
        points.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": int(mask.sum()),
                "mean_score": float(score[mask].mean()),
                "observed": float(y[mask].mean()),
            }
        )
    return points


def _top_precision(
    y: np.ndarray, score: np.ndarray, fraction: float
) -> float:
    k = max(1, int(fraction * len(score)))
    return float(y[np.argsort(-score)[:k]].mean())


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    n_trees: int = 100,
    pi: float = 2.5e-3,
    n_boot: int = 200,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "operating_points"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    reliability = {}
    for policy in POLICIES:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, meta = load_dataset(path)
        y = data["y_f_05"].astype(int)
        weight = _balance(y)
        f_zp = data["f_zp"].astype(float)
        observed_names = tuple(meta["observation_feature_names"])
        observed_columns = list(range(len(observed_names)))
        feature_sets = {
            "general8": _ratios(data, GENERAL8, sigma, seed=0),
            "observed11": data["obs_features"][:, observed_columns],
        }
        for name, x in feature_sets.items():
            score = _oof(x, y, weight, n_trees)
            metrics: dict[str, Any] = {
                "label": "f05",
                "feature_set": name,
                "policy": policy,
                "n": int(len(y)),
                "positive_rate": float(y.mean()),
                "brier": float(np.mean((score - y) ** 2)),
                "top1_precision": _top_precision(y, score, 0.01),
                "top5_precision": _top_precision(y, score, 0.05),
                "top10_precision": _top_precision(y, score, 0.10),
            }
            for target in FPR_TARGETS:
                label = f"{target:.0e}"
                tpr = _tpr_at_fpr(y, score, weight, target)
                metrics[f"tpr_fpr_{label}"] = tpr
                for prevalence in PREVALENCES:
                    p = _precision(tpr, target, prevalence)
                    metrics[f"precision_{prevalence:.1e}_at_fpr_{label}"] = p
                metrics[f"tp_per_1e4_at_fpr_{label}"] = (
                    pi * 1.0e4 * tpr
                )
                metrics[f"fp_per_1e4_at_fpr_{label}"] = (
                    (1.0 - pi) * 1.0e4 * target
                )
            metrics["fpr_groups_1e-3"] = _fpr_groups(
                y, score, f_zp, weight, 1.0e-3
            )
            metrics["fpr_groups_1e-2"] = _fpr_groups(
                y, score, f_zp, weight, 1.0e-2
            )
            metrics["bootstrap"] = _bootstrap(
                y, score, weight, n_boot=n_boot
            )
            rows.append(metrics)
            reliability.setdefault(f"{name}", {})[policy] = _reliability(
                y, score
            )
            print(
                f"{policy:22s} {name:11s} "
                f"TPR(1e-2)={metrics['tpr_fpr_1e-02']:.3f} "
                f"TPR(1e-3)={metrics['tpr_fpr_1e-03']:.3f} "
                f"prec(1e-3)={metrics['precision_1.0e-03_at_fpr_1e-03']:.3f} "
                f"Brier={metrics['brier']:.4f}",
                flush=True,
            )

    with (outdir / "operating_points.json").open("w") as handle:
        json.dump({"rows": rows, "reliability": reliability}, handle, indent=2)
    keys = [key for key in rows[0] if key not in ("bootstrap", "fpr_groups_1e-3", "fpr_groups_1e-2")]
    with (outdir / "operating_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})

    # TPR bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, target in zip(axes, FPR_TARGETS):
        label = f"{target:.0e}"
        policies = sorted({row["policy"] for row in rows})
        width = 0.38
        for offset, name in ((-width / 2, "general8"), (width / 2, "observed11")):
            values = [
                next(
                    row[f"tpr_fpr_{label}"]
                    for row in rows
                    if row["policy"] == policy and row["feature_set"] == name
                )
                for policy in policies
            ]
            axis.bar(
                np.arange(len(policies)) + offset,
                values,
                width,
                label=name,
            )
        axis.set_xticks(np.arange(len(policies)))
        axis.set_xticklabels(policies, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel(f"TPR at FPR = {target:g}")
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "tpr_at_fpr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Precision vs prevalence
    fig, axis = plt.subplots(figsize=(8, 5))
    for name in ("general8", "observed11"):
        for policy in sorted({row["policy"] for row in rows}):
            values = []
            for prevalence in PREVALENCES:
                row = next(
                    row
                    for row in rows
                    if row["policy"] == policy and row["feature_set"] == name
                )
                values.append(
                    row[f"precision_{prevalence:.1e}_at_fpr_1e-03"]
                )
            axis.plot(
                np.array(PREVALENCES) * 100,
                values,
                marker="o",
                alpha=0.5 if name == "observed11" else 1.0,
                label=f"{name} {policy}",
            )
    axis.set_xscale("log")
    axis.set_xlabel("prevalence (%)")
    axis.set_ylabel("precision at FPR = 1e-3")
    axis.set_ylim(0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "precision_vs_prevalence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Reliability
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, name in zip(axes, ("general8", "observed11")):
        for policy, points in reliability.get(name, {}).items():
            axis.plot(
                [point["mean_score"] for point in points],
                [point["observed"] for point in points],
                marker="o",
                label=policy,
            )
        axis.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0)
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed positive fraction")
        axis.set_title(name)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # FPR groups
    fig, axis = plt.subplots(figsize=(9, 4.5))
    policies = sorted({row["policy"] for row in rows})
    width = 0.38
    for offset, name in ((-width / 2, "general8"), (width / 2, "observed11")):
        clear = []
        near = []
        for policy in policies:
            row = next(
                row
                for row in rows
                if row["policy"] == policy and row["feature_set"] == name
            )
            clear.append(row["fpr_groups_1e-3"]["fpr_clear"])
            near.append(row["fpr_groups_1e-3"]["fpr_near"])
        axis.bar(
            np.arange(len(policies)) + offset,
            clear,
            width,
            label=f"{name} clear",
        )
        axis.bar(
            np.arange(len(policies)) + offset,
            near,
            width,
            bottom=clear,
            label=f"{name} near",
        )
    axis.set_xticks(np.arange(len(policies)))
    axis.set_xticklabels(policies, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("FPR at overall FPR = 1e-3")
    axis.set_title("FPR split: clear (f_Z < 0.25) vs near-threshold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fpr_groups.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"rows": rows, "reliability": reliability}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--pi", type=float, default=2.5e-3)
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args(argv)
    run(
        args.base,
        args.outdir,
        sigma=args.sigma,
        n_trees=args.n_trees,
        pi=args.pi,
        n_boot=args.n_boot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
