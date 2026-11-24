"""Head-to-head comparison of tree-ensemble backends on a potluck dataset.

Every backend shares the same folds, weights, imputation, and metrics, so the
only thing that changes between rows of the result is the learner:

- random forest (``rf``)
- histogram gradient boosting (``gb``) -- the v15 model
- XGBoost (``xgb``)

For each feature block the script reports out-of-fold classification
(weighted AUC, Brier) and regression (f_Z, weighted R2 and MAE) scores, the
out-of-fold permutation importance of every column, and how well the backends
agree on the ranking (Spearman rho).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score

from ..io import load_dataset
from .feature_importance import BLOCK_LABELS, BLOCKS, _block, _folds, _impute, _weighted_auc

SEED = 0
REPEATS = 3
MODELS = ("rf", "gb", "xgb")

Factory = Callable[[int], Any]


def _rf_classifier(n_rows: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300 if n_rows >= 5000 else 60,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )


def _rf_regressor(n_rows: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300 if n_rows >= 5000 else 60,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )


def _gb_classifier(n_rows: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150, max_depth=3, learning_rate=0.05, random_state=SEED
    )


def _gb_regressor(n_rows: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=150, max_depth=3, learning_rate=0.05, random_state=SEED
    )


def _xgb_classifier(n_rows: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=SEED,
    )


def _xgb_regressor(n_rows: int) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )


BACKENDS: dict[str, dict[str, Factory]] = {
    "rf": {"classifier": _rf_classifier, "regressor": _rf_regressor},
    "gb": {"classifier": _gb_classifier, "regressor": _gb_regressor},
    "xgb": {"classifier": _xgb_classifier, "regressor": _xgb_regressor},
}


def _oof(
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    splits,
    backend: dict[str, Factory],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Out-of-fold predictions plus per-fold metrics for uncertainty."""
    probability = np.zeros(len(y), dtype=np.float64)
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    per_fold: dict[str, list[float]] = {"auc": [], "brier": [], "r2": [], "mae": []}
    for train, test in splits:
        classifier = backend["classifier"](len(x))
        classifier.fit(x[train], y[train], sample_weight=weight[train])
        fold_probability = classifier.predict_proba(x[test])[:, 1]
        probability[test] = fold_probability
        per_fold["auc"].append(_weighted_auc(y[test], fold_probability, weight[test]))
        per_fold["brier"].append(
            float(brier_score_loss(y[test], fold_probability, sample_weight=weight[test]))
        )

        regressor = backend["regressor"](len(x))
        regressor.fit(x[train], target[train], sample_weight=weight[train])
        fold_prediction = regressor.predict(x[test])
        prediction[test] = fold_prediction
        fold_regression = _regression_scores(target[test], fold_prediction, weight[test])
        per_fold["r2"].append(fold_regression["r2"])
        per_fold["mae"].append(fold_regression["mae"])
    return probability, prediction, per_fold


def _fold_se(values: list[float]) -> float:
    """Standard error of the per-fold mean."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size < 2:
        return 0.0
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def _permutation(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    splits,
    backend: dict[str, Factory],
    repeats: int,
) -> np.ndarray:
    """Out-of-fold permutation importance (AUC drop) per column, shape (n, repeats)."""
    n_columns = x.shape[1]
    drop = np.zeros((n_columns, repeats))
    rng = np.random.default_rng(SEED)
    for train, test in splits:
        model = backend["classifier"](len(x))
        model.fit(x[train], y[train], sample_weight=weight[train])
        base = _weighted_auc(y[test], model.predict_proba(x[test])[:, 1], weight[test])
        for column in range(n_columns):
            for repeat in range(repeats):
                shuffled = x[test].copy()
                rng.shuffle(shuffled[:, column])
                score = model.predict_proba(shuffled)[:, 1]
                drop[column, repeat] += (
                    base - _weighted_auc(y[test], score, weight[test])
                ) / len(splits)
    return drop


def _regression_scores(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    finite = np.isfinite(target) & np.isfinite(prediction)
    if finite.sum() < 2:
        return {"r2": float("nan"), "mae": float("nan")}
    return {
        "r2": float(r2_score(target[finite], prediction[finite], sample_weight=weight[finite])),
        "mae": float(mean_absolute_error(target[finite], prediction[finite], sample_weight=weight[finite])),
    }


def _plot_metrics(summary: dict[str, Any], path: Path) -> None:
    models = list(summary["models"])
    blocks = list(summary["blocks"])
    metrics = (
        ("auc", "auc_se", "weighted out-of-fold AUC", "high"),
        ("brier", "brier_se", "weighted Brier", "low"),
        ("r2", "r2_se", "f_Z weighted R2", "high"),
        ("mae", "mae_se", "f_Z weighted MAE", "low"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    width = 0.8 / len(models)
    for axis, (metric, se_metric, label, better) in zip(axes.ravel(), metrics):
        for i, model in enumerate(models):
            values = [summary["blocks"][b]["models"][model][metric] for b in blocks]
            errors = [summary["blocks"][b]["models"][model][se_metric] for b in blocks]
            axis.bar(
                np.arange(len(blocks)) + i * width,
                values,
                width,
                yerr=errors,
                capsize=2,
                label=model,
            )
        axis.set_xticks(np.arange(len(blocks)) + width * (len(models) - 1) / 2)
        axis.set_xticklabels([BLOCK_LABELS[b] for b in blocks], fontsize=8)
        axis.set_ylabel(label)
        axis.set_title(f"{label} ({better} is better, error = fold SE)", fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{summary['dataset']} -- backend comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(
    names: list[str], importances: dict[str, np.ndarray], block: str, path: Path
) -> None:
    models = list(importances)
    reference = importances[models[0]]
    order = np.argsort(reference)
    position = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(9.0, 0.32 * len(order) + 2.0))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        axis.barh(position + i * width, importances[model][order], width, label=model)
    axis.set_yticks(position + width * (len(models) - 1) / 2)
    axis.set_yticklabels([names[j] for j in order], fontsize=8)
    axis.set_xlabel("out-of-fold permutation importance (AUC drop)")
    axis.set_title(f"{BLOCK_LABELS[block]} -- permutation importance by backend", fontsize=10)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    dataset: str | Path | tuple[dict[str, np.ndarray], dict[str, Any]],
    outdir: str | Path = "src/potluck/tests/v15/model_comparison",
    models: tuple[str, ...] = MODELS,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """Compare classifier and regressor backends on every feature block."""
    arrays, meta = (
        load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    )
    label = str(dataset) if isinstance(dataset, (str, Path)) else "<in-memory>"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)
    target = arrays["f_zp"].astype(float)

    summary: dict[str, Any] = {
        "dataset": label,
        "n_mocks": int(len(y)),
        "models": list(models),
        "repeats": repeats,
        "blocks": {},
    }

    for block in BLOCKS:
        x, names = _block(arrays, meta, block)
        splits = _folds(x, y)
        entry: dict[str, Any] = {"n_features": len(names), "features": list(names), "models": {}}
        importances: dict[str, np.ndarray] = {}

        for model in models:
            backend = BACKENDS[model]
            probability, prediction, per_fold = _oof(x, y, target, weight, splits, backend)
            drop = _permutation(x, y, weight, splits, backend, repeats)
            importances[model] = drop.mean(axis=1)
            entry["models"][model] = {
                "auc": _weighted_auc(y, probability, weight),
                "auc_se": _fold_se(per_fold["auc"]),
                "brier": float(brier_score_loss(y, probability, sample_weight=weight)),
                "brier_se": _fold_se(per_fold["brier"]),
                **_regression_scores(target, prediction, weight),
                "r2_se": _fold_se(per_fold["r2"]),
                "mae_se": _fold_se(per_fold["mae"]),
                "per_fold": per_fold,
                "permutation_mean": dict(zip(names, importances[model].tolist())),
                "permutation_std": dict(zip(names, drop.std(axis=1).tolist())),
            }

        agreement: dict[str, float] = {}
        for i, first in enumerate(models):
            for second in models[i + 1:]:
                rho = float(spearmanr(importances[first], importances[second]).statistic)
                agreement[f"{first}_vs_{second}"] = rho
        entry["rank_agreement"] = agreement
        entry["ranking"] = {
            model: [names[i] for i in np.argsort(-importances[model])] for model in models
        }
        summary["blocks"][block] = entry
        _plot_importance(names, importances, block, outdir / f"importance_{block}.png")

    _plot_metrics(summary, outdir / "metrics.png")
    with (outdir / "comparison.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--outdir", default="src/potluck/tests/v15/model_comparison")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    unknown = set(models) - set(BACKENDS)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; choose from {sorted(BACKENDS)}")

    summary = run_comparison(
        args.dataset, outdir=args.outdir, models=models, repeats=args.repeats
    )
    for block, entry in summary["blocks"].items():
        print(f"\n=== {BLOCK_LABELS[block]} ({entry['n_features']} features) ===")
        for model in models:
            payload = entry["models"][model]
            print(
                f"  {model:>3}: AUC={payload['auc']:.4f}+-{payload['auc_se']:.4f}  "
                f"Brier={payload['brier']:.4f}+-{payload['brier_se']:.4f}  "
                f"fZ R2={payload['r2']:.4f}+-{payload['r2_se']:.4f}  "
                f"MAE={payload['mae']:.4f}+-{payload['mae_se']:.4f}"
            )
        for pair, rho in entry["rank_agreement"].items():
            print(f"       rank agreement {pair}: rho={rho:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
