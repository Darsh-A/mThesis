"""Feature importance and performance-based element selection.

Two matrices are ranked: the noiseless mock ratios (what the physics encodes)
and the noisy observed ratios (what a star actually delivers). Importance comes
from random-forest impurity and from out-of-fold permutation; the subset size
comes from a cross-validated selection curve.
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..io import load_dataset

FOLDS = 5
SEED = 0
REPEATS = 5
BLOCKS = ("noiseless", "observed")
BLOCK_LABELS = {
    "noiseless": "Noiseless mock ratios",
    "observed": "Observed ratios (with noise)",
}


def _impute(matrix: np.ndarray) -> np.ndarray:
    """Fill missing values with the column median."""
    filled = np.array(matrix, dtype=np.float32, copy=True)
    for column in range(filled.shape[1]):
        values = filled[:, column]
        missing = ~np.isfinite(values)
        if missing.all():
            filled[:, column] = 0.0
        elif missing.any():
            filled[missing, column] = np.median(values[~missing])
    return filled


def _block(arrays: dict[str, np.ndarray], meta: dict[str, Any], block: str):
    """Return the imputed matrix and column names for one feature block."""
    if block == "noiseless":
        return _impute(arrays["features"]), tuple(meta["feature_names"])
    if block == "observed":
        return _impute(arrays["obs_features"]), tuple(meta["observation_feature_names"])
    raise ValueError(f"unknown block {block!r}")


def _folds(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(y)
    splits = max(2, min(FOLDS, int(counts[counts > 0].min())))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
    return list(splitter.split(x, y))


def _forest(n_rows: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300 if n_rows >= 5000 else 60,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )


def _boost() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=SEED,
    )


def _classifier(model: str, n_rows: int):
    """Return an unfitted classifier for the requested backend."""
    if model == "rf":
        return _forest(n_rows)
    if model == "gb":
        return _boost()
    raise ValueError(f"unknown model {model!r}; choose 'rf' or 'gb'")


def _weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    return float(roc_auc_score(y, score, sample_weight=weight))


def _importance(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, splits
) -> dict[str, Any]:
    """Impurity and out-of-fold permutation importance of every column."""
    n_columns = x.shape[1]
    repeats = REPEATS if len(x) >= 5000 else 3
    impurity = np.zeros(n_columns)
    drop = np.zeros((n_columns, repeats))
    rng = np.random.default_rng(SEED)

    for train, test in splits:
        model = _forest(len(x))
        model.fit(x[train], y[train], sample_weight=weight[train])
        impurity += model.feature_importances_ / len(splits)

        reference = model.predict_proba(x[test])[:, 1]
        base = _weighted_auc(y[test], reference, weight[test])
        for column in range(n_columns):
            for repeat in range(repeats):
                shuffled = x[test].copy()
                rng.shuffle(shuffled[:, column])
                score = model.predict_proba(shuffled)[:, 1]
                drop[column, repeat] += (
                    base - _weighted_auc(y[test], score, weight[test])
                ) / len(splits)

    return {
        "impurity": impurity / impurity.sum() if impurity.sum() > 0 else impurity,
        "permutation_mean": drop.mean(axis=1),
        "permutation_std": drop.std(axis=1),
    }


def _score_subset(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    splits,
    columns: np.ndarray,
    model: str,
) -> tuple[list[float], float]:
    """Per-fold AUC and out-of-fold Brier for one column subset."""
    score = np.zeros(len(y))
    fold_auc = []
    for train, test in splits:
        classifier = _classifier(model, len(x))
        classifier.fit(x[train][:, columns], y[train], sample_weight=weight[train])
        score[test] = classifier.predict_proba(x[test][:, columns])[:, 1]
        fold_auc.append(_weighted_auc(y[test], score[test], weight[test]))
    return fold_auc, float(brier_score_loss(y, score, sample_weight=weight))


def _selection_curve(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    splits,
    names: tuple[str, ...],
    ranking: np.ndarray,
    model: str,
) -> dict[str, Any]:
    """AUC and Brier for the top-k ranked features, k = 1..n."""
    rows = []
    for k in range(1, len(names) + 1):
        columns = np.sort(ranking[:k])
        fold_auc, brier = _score_subset(x, y, weight, splits, columns, model)
        rows.append(
            {
                "k": k,
                "features": [names[i] for i in ranking[:k]],
                "auc": float(np.mean(fold_auc)),
                "auc_se": float(np.std(fold_auc) / np.sqrt(len(fold_auc))),
                "brier": brier,
            }
        )
    return {"rows": rows}


def _choose(curve: dict[str, Any]) -> dict[str, Any]:
    """Smallest subset within one standard error of the best AUC."""
    rows = curve["rows"]
    best = max(rows, key=lambda row: row["auc"])
    threshold = best["auc"] - best["auc_se"]
    chosen = next(row for row in rows if row["auc"] >= threshold)
    return {
        "k": chosen["k"],
        "features": chosen["features"],
        "auc": chosen["auc"],
        "auc_se": chosen["auc_se"],
        "brier": chosen["brier"],
        "best_k": best["k"],
        "best_auc": best["auc"],
        "tolerance": best["auc_se"],
    }


def _plot_importance(
    names: tuple[str, ...],
    stats: dict[str, Any],
    block: str,
    path: Path,
) -> None:
    order = np.argsort(stats["permutation_mean"])
    position = np.arange(len(order))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 0.30 * len(order) + 2.2), sharey=True
    )
    left.barh(position, stats["impurity"][order], color="tab:gray", alpha=0.8)
    left.set_yticks(position)
    left.set_yticklabels([names[i] for i in order], fontsize=8)
    left.set_xlabel("impurity importance (normalised)")
    left.set_title("Random-forest impurity (MDI)", fontsize=10)
    left.grid(axis="x", alpha=0.25)

    right.errorbar(
        stats["permutation_mean"][order],
        position,
        xerr=stats["permutation_std"][order],
        fmt="o",
        ms=4,
        color="tab:blue",
        ecolor="0.6",
        elinewidth=0.8,
        capsize=2,
    )
    right.axvline(0.0, color="0.3", lw=0.8)
    right.set_xlabel("permutation importance (AUC drop)")
    right.set_title("Out-of-fold permutation", fontsize=10)
    right.grid(axis="x", alpha=0.25)

    fig.suptitle(f"{BLOCK_LABELS[block]} — feature importance", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(curves: dict[str, Any], chosen: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(11.0, 4.2))
    for axis, block in zip(np.atleast_1d(axes), BLOCKS):
        rows = curves[block]["rows"]
        ks = [row["k"] for row in rows]
        aucs = [row["auc"] for row in rows]
        axis.plot(ks, aucs, "o-", color="tab:blue", ms=4)
        pick = chosen[block]
        axis.axhline(pick["best_auc"], color="0.6", ls=":", lw=1.0)
        axis.axhline(pick["best_auc"] - pick["tolerance"], color="0.6", ls="--", lw=0.9)
        axis.plot([pick["k"]], [pick["auc"]], "o", ms=9, mfc="none", mec="tab:red", mew=1.6)
        axis.set_title(
            f"{BLOCK_LABELS[block]}\nchosen k = {pick['k']} "
            f"(AUC {pick['auc']:.3f})",
            fontsize=10,
        )
        axis.set_xlabel("features kept (ranked by permutation)")
        axis.set_ylabel("out-of-fold AUC")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_feature_importance(
    dataset: str | Path,
    outdir: str | Path = "src/potluck/tests/v15",
    model: str = "gb",
) -> dict[str, Any]:
    """Rank every feature and select the subset that keeps performance.

    ``model`` chooses the backend for the selection curve (``"gb"`` for the
    production gradient-boosting classifier, ``"rf"`` for the random forest).
    The impurity/permutation ranking always uses the random forest.
    """
    arrays, meta = (
        load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)

    summary: dict[str, Any] = {"dataset": str(dataset), "n_mocks": int(len(y)),
                               "model": model, "blocks": {}}
    curves: dict[str, Any] = {}
    chosen: dict[str, Any] = {}

    for block in BLOCKS:
        x, names = _block(arrays, meta, block)
        splits = _folds(x, y)
        stats = _importance(x, y, weight, splits)
        ranking = np.argsort(-stats["permutation_mean"])
        curve = _selection_curve(x, y, weight, splits, names, ranking, model)
        curves[block] = curve
        chosen[block] = _choose(curve)

        summary["blocks"][block] = {
            "n_features": len(names),
            "features": list(names),
            "impurity": dict(zip(names, stats["impurity"].tolist())),
            "permutation_mean": dict(
                zip(names, stats["permutation_mean"].tolist())
            ),
            "permutation_std": dict(zip(names, stats["permutation_std"].tolist())),
            "ranking": [names[i] for i in ranking],
            "curve": curve["rows"],
            "chosen": chosen[block],
        }
        _plot_importance(names, stats, block, outdir / f"importance_{block}.png")

    _plot_curve(curves, chosen, outdir / "selection_curve.png")

    with (outdir / "feature_importance.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (outdir / "chosen_features.json").open("w") as handle:
        json.dump(chosen, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the importance ranking from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--outdir", default="src/potluck/tests/v15")
    parser.add_argument("--model", choices=("rf", "gb"), default="gb")
    args = parser.parse_args(argv)
    summary = run_feature_importance(
        args.dataset, outdir=args.outdir, model=args.model
    )
    for block, payload in summary["blocks"].items():
        pick = payload["chosen"]
        print(
            f"{block}: {payload['n_features']} features -> "
            f"k={pick['k']} AUC={pick['auc']:.4f} Brier={pick['brier']:.4f}"
        )
        print("  " + ", ".join(pick["features"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
