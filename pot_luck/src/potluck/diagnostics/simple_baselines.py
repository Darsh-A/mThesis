"""Does the random forest beat a simple cut at the same FPR?

Compare the general 8-element RF against:
- a single-feature threshold ([Zn/Fe], [Zn/Mg], [Na/Mg], [Si/Fe]),
- a 2-feature logistic rule ([Na/Mg] + [Zn/Mg]),
- an RF on the same two features,
- a logistic rule on the full general 8.

All scores are OOF; thresholds are chosen at the same target FPR on the
negative class. Gaussian element-level noise sigma=0.2, f05 label.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..constants import ELEMENT_INDEX, REFERENCE
from ..io import load_dataset
from .v20_mega import _balance

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
)
GENERAL8 = ("Zn", "Si", "Al", "S", "Ca", "Ti", "Cr", "Ni")
SIMPLE = {
    "ZnFe": ("Zn", "Fe"),
    "SiFe": ("Si", "Fe"),
    "ZnMg": ("Zn", "Mg"),
    "NaMg": ("Na", "Mg"),
}
FPR_TARGETS = (1.0e-2, 1.0e-3)
PREVALENCE = 2.5e-3


def _ratios(
    data: dict[str, np.ndarray], pairs: tuple[tuple[str, str], ...], sigma: float
) -> np.ndarray:
    loga = data["log_abundance"].astype(float)
    noisy = loga + np.random.default_rng(0).normal(0.0, sigma, loga.shape)
    return np.column_stack(
        [
            noisy[:, ELEMENT_INDEX[numerator]]
            - noisy[:, ELEMENT_INDEX[denominator]]
            for numerator, denominator in pairs
        ]
    )


def _oof(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    model: str,
    n_trees: int = 100,
) -> np.ndarray:
    probability = np.zeros(len(y))
    for train, test in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ).split(x, y):
        if model == "rf":
            estimator = RandomForestClassifier(
                n_estimators=n_trees,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=0,
            )
        else:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000, class_weight="balanced"
                ),
            )
        estimator.fit(x[train], y[train])
        probability[test] = estimator.predict_proba(x[test])[:, 1]
    return probability


def _tpr_at_fpr(
    y: np.ndarray, score: np.ndarray, target: float
) -> tuple[float, float]:
    negative = score[y == 0]
    threshold = np.quantile(negative, 1.0 - target)
    tpr = float((score[y == 1] >= threshold).mean())
    fpr = float((negative >= threshold).mean())
    return tpr, fpr


def _precision(tpr: float, fpr: float, pi: float) -> float:
    return tpr * pi / (tpr * pi + fpr * (1.0 - pi))


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    n_trees: int = 100,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "simple_baselines"
    outdir.mkdir(parents=True, exist_ok=True)

    general_pairs = tuple((element, "Fe") for element in GENERAL8)
    simple_pairs = tuple(SIMPLE.values())
    all_pairs = general_pairs + simple_pairs
    rows = []
    for policy in POLICIES:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, _ = load_dataset(path)
        y = data["y_f_05"].astype(int)
        weight = _balance(y)
        ratios = _ratios(data, all_pairs, sigma)
        general = ratios[:, : len(general_pairs)]
        columns = {
            name: ratios[:, len(general_pairs) + j]
            for j, name in enumerate(SIMPLE)
        }
        two = np.column_stack(
            [columns["ZnMg"], columns["NaMg"]]
        )

        models = {
            "rf_general8": _oof(general, y, weight, "rf", n_trees),
            "logit_general8": _oof(general, y, weight, "logit"),
            "logit_NaMg_ZnMg": _oof(two, y, weight, "logit"),
            "rf_NaMg_ZnMg": _oof(two, y, weight, "rf", n_trees),
        }
        for name in ("ZnFe", "SiFe", "ZnMg", "NaMg"):
            score = columns[name]
            if roc_auc_score(y, score) < 0.5:
                score = -score
            models[f"single_{name}"] = score

        for model, score in models.items():
            row: dict[str, Any] = {
                "policy": policy,
                "model": model,
                "auc": float(roc_auc_score(y, score)),
            }
            for target in FPR_TARGETS:
                tpr, fpr = _tpr_at_fpr(y, score, target)
                label = f"{target:.0e}"
                row[f"tpr_{label}"] = tpr
                row[f"fpr_{label}"] = fpr
                row[f"precision_{label}"] = _precision(
                    tpr, fpr, PREVALENCE
                )
            rows.append(row)
            print(
                f"{policy:22s} {model:20s} "
                f"TPR(1e-2)={row['tpr_1e-02']:.3f} "
                f"TPR(1e-3)={row['tpr_1e-03']:.3f} "
                f"prec(2.5e-3)={row['precision_1e-03']:.3f}",
                flush=True,
            )

    with (outdir / "simple_baselines.json").open("w") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    keys = list(rows[0])
    with (outdir / "simple_baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    models = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, target in zip(axes, FPR_TARGETS):
        label = f"{target:.0e}"
        means = []
        lows = []
        highs = []
        for model in models:
            values = [
                row[f"tpr_{label}"]
                for row in rows
                if row["model"] == model
            ]
            means.append(float(np.mean(values)))
            lows.append(float(np.min(values)))
            highs.append(float(np.max(values)))
        x = np.arange(len(models))
        axis.bar(x, means, color="tab:blue")
        axis.errorbar(
            x,
            means,
            yerr=[np.array(means) - lows, np.array(highs) - means],
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel(f"TPR at FPR = {target:g}")
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"FPR = {target:g}")
    fig.tight_layout()
    fig.savefig(outdir / "baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--n-trees", type=int, default=100)
    args = parser.parse_args(argv)
    run(args.base, args.outdir, args.sigma, args.n_trees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
