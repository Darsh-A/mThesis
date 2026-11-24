"""v21: does the RF transfer across yield tables better than a logistic model?

Train on one yield table, test on every other table. General 8 features,
Gaussian element-level noise sigma=0.2, f05 label. Thresholds are chosen on the
training negatives at the target FPR; the test FPR is reported because it can
shift under the table change.
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

from ..io import load_dataset
from .element_influence import POLICIES
from .element_selection import _ratios
from .v20_mega import _balance

GENERAL8 = ("Zn", "Si", "Al", "S", "Ca", "Ti", "Cr", "Ni")
FPR_TARGETS = (1.0e-2, 1.0e-3)
PREVALENCE = 2.5e-3


def _model(kind: str, n_trees: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def _threshold(score: np.ndarray, y: np.ndarray, target: float) -> float:
    return float(np.quantile(score[y == 0], 1.0 - target))


def _tpr(score: np.ndarray, y: np.ndarray, threshold: float) -> float:
    return float((score[y == 1] >= threshold).mean())


def _fpr(score: np.ndarray, y: np.ndarray, threshold: float) -> float:
    return float((score[y == 0] >= threshold).mean())


def _precision(tpr: float, fpr: float, pi: float) -> float:
    return tpr * pi / (tpr * pi + fpr * (1.0 - pi))


def _within(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    kind: str,
    n_trees: int,
) -> np.ndarray:
    score = np.zeros(len(y))
    for train, test in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ).split(x, y):
        model = _model(kind, n_trees)
        model.fit(x[train], y[train])
        score[test] = model.predict_proba(x[test])[:, 1]
    return score


def _metrics(
    y: np.ndarray, score: np.ndarray, train_score: np.ndarray, train_y: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {"auc": float(roc_auc_score(y, score))}
    for target in FPR_TARGETS:
        label = f"{target:.0e}"
        threshold = _threshold(train_score, train_y, target)
        tpr = _tpr(score, y, threshold)
        fpr = _fpr(score, y, threshold)
        result[f"tpr_train_thr_{label}"] = tpr
        result[f"test_fpr_{label}"] = fpr
        result[f"precision_train_thr_{label}"] = _precision(
            tpr, fpr, PREVALENCE
        )
        # oracle threshold on the test negatives: same test FPR for both models
        test_threshold = _threshold(score, y, target)
        result[f"tpr_test_thr_{label}"] = _tpr(score, y, test_threshold)
    return result


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    n_trees: int = 100,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else Path("src/potluck/tests/v21")
    outdir.mkdir(parents=True, exist_ok=True)
    data = {}
    for policy in POLICIES:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        arrays, _ = load_dataset(path)
        data[policy] = {
            "x": _ratios(arrays, GENERAL8, sigma, seed=0),
            "y": arrays["y_f_05"].astype(int),
        }
    policies = list(data)
    rows = []
    within_rows = []

    for train_policy in policies:
        x_train = data[train_policy]["x"]
        y_train = data[train_policy]["y"]
        weight = _balance(y_train)
        fitted = {}
        for kind in ("rf", "logit"):
            model = _model(kind, n_trees)
            model.fit(x_train, y_train)
            fitted[kind] = model
            oof_score = _within(x_train, y_train, weight, kind, n_trees)
            within = _metrics(y_train, oof_score, oof_score, y_train)
            within_rows.append(
                {"train": train_policy, "test": train_policy, "model": kind, **within}
            )
        for test_policy in policies:
            x_test = data[test_policy]["x"]
            y_test = data[test_policy]["y"]
            for kind in ("rf", "logit"):
                score = fitted[kind].predict_proba(x_test)[:, 1]
                train_score = fitted[kind].predict_proba(x_train)[:, 1]
                metrics = _metrics(y_test, score, train_score, y_train)
                rows.append(
                    {
                        "train": train_policy,
                        "test": test_policy,
                        "model": kind,
                        **metrics,
                    }
                )

    with (outdir / "cross_table.json").open("w") as handle:
        json.dump({"rows": rows, "within": within_rows}, handle, indent=2)
    keys = list(rows[0])
    with (outdir / "cross_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def matrix(kind: str, key: str) -> np.ndarray:
        out = np.full((len(policies), len(policies)), np.nan)
        for i, train_policy in enumerate(policies):
            for j, test_policy in enumerate(policies):
                value = next(
                    row[key]
                    for row in rows
                    if row["train"] == train_policy
                    and row["test"] == test_policy
                    and row["model"] == kind
                )
                out[i, j] = value
        return out

    def heatmap(
        values: np.ndarray, path: Path, title: str, center: bool = False
    ) -> None:
        if center:
            limit = max(float(np.nanmax(np.abs(values))), 0.02)
            vmin, vmax = -limit, limit
            cmap = "RdBu_r"
        else:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            cmap = "viridis"
        fig, axis = plt.subplots(figsize=(8.5, 7))
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(np.arange(len(policies)))
        axis.set_xticklabels(policies, rotation=35, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(policies)))
        axis.set_yticklabels(policies, fontsize=8)
        for i in range(len(policies)):
            for j in range(len(policies)):
                if np.isfinite(values[i, j]):
                    axis.text(
                        j,
                        i,
                        f"{values[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if not center else "black",
                    )
        axis.set_xlabel("test policy")
        axis.set_ylabel("train policy")
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    auc_rf = matrix("rf", "auc")
    auc_logit = matrix("logit", "auc")
    tpr_rf = matrix("rf", "tpr_test_thr_1e-03")
    tpr_logit = matrix("logit", "tpr_test_thr_1e-03")
    heatmap(auc_rf, outdir / "auc_rf.png", "Cross-table AUC: RF")
    heatmap(auc_logit, outdir / "auc_logit.png", "Cross-table AUC: logistic")
    heatmap(auc_rf - auc_logit, outdir / "delta_auc.png", "AUC: RF minus logistic", center=True)
    heatmap(tpr_rf, outdir / "tpr_rf.png", "Cross-table TPR at FPR 0.1%: RF")
    heatmap(tpr_logit, outdir / "tpr_logit.png", "Cross-table TPR at FPR 0.1%: logistic")
    heatmap(
        tpr_rf - tpr_logit,
        outdir / "delta_tpr.png",
        "TPR at FPR 0.1%: RF minus logistic",
        center=True,
    )

    mask = ~np.eye(len(policies), dtype=bool)
    summary = {
        "rf_auc_mean": float(np.nanmean(auc_rf[mask])),
        "logit_auc_mean": float(np.nanmean(auc_logit[mask])),
        "rf_tpr_mean": float(np.nanmean(tpr_rf[mask])),
        "logit_tpr_mean": float(np.nanmean(tpr_logit[mask])),
        "rf_auc_better": int(np.sum((auc_rf - auc_logit)[mask] > 0)),
        "rf_tpr_better": int(np.sum((tpr_rf - tpr_logit)[mask] > 0)),
        "n_pairs": int(mask.sum()),
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return {"rows": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default="src/potluck/tests/v21")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--n-trees", type=int, default=100)
    args = parser.parse_args(argv)
    result = run(args.base, args.outdir, args.sigma, args.n_trees)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
