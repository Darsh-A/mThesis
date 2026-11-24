"""How much does each element affect the ranking and the top-1% candidate list?

All 26 feature ratios are used, with generic element-level Gaussian noise. For
each element we compare the full set against the set with that element removed.
Metrics: paired Delta AUC with a bootstrap interval, top-1% turnover, and
precision at fixed false-positive rates.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from ..constants import (
    ELEMENT_INDEX,
    ELEMENTS,
    REFERENCE,
    SOLAR_MASS_FRACTION,
)
from ..ejecta import PISN
from ..io import load_dataset
from .v20_mega import _balance

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)
CORE7 = ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni")
PREVALENCE = 2.5e-3


def _detection_label(
    data: dict[str, np.ndarray],
    meta: dict[str, Any],
    k: float = 3.0,
    min_detected: int = 4,
) -> np.ndarray:
    """Reconstruct the counterfactual label y = 1[D > k] from stored arrays.

    This follows the observed branch of detect_labels: remove the PISN row,
    compare the remaining pattern to the noisy observed ratios, and normalise
    by the observation errors. Ratios flagged invalid or non-finite are skipped.
    """
    n = len(data["M_tot"])
    y = np.zeros(n, dtype=bool)
    has_pisn = data["n_pisn"] > 0
    observation_names = tuple(meta["observation_feature_names"])
    solar = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENTS])
    reference_index = ELEMENT_INDEX[REFERENCE]
    for index in np.flatnonzero(has_pisn):
        used = [
            column
            for column in range(len(observation_names))
            if data["obs_flags"][index, column] == 1
            and np.isfinite(data["obs_features"][index, column])
        ]
        if len(used) < min_detected:
            continue
        used_indices = [
            ELEMENT_INDEX[observation_names[column].split("/")[0]]
            for column in used
        ] + [reference_index]
        m_minus = data["M_tot"][index] - data["matrix"][index, PISN, :]
        if np.any(m_minus[used_indices] <= 0.0):
            y[index] = True
            continue
        log_minus = np.log10(m_minus[used_indices] / solar[used_indices])
        x_minus = log_minus[:-1] - log_minus[-1]
        ratio_values = data["obs_features"][index, used]
        sigma_values = data["obs_sigma"][index, used]
        delta = ratio_values - x_minus
        y[index] = np.sqrt(
            max(float(np.sum(delta * delta / sigma_values**2)), 0.0)
        ) > k
    return y


def _ratios(
    data: dict[str, np.ndarray],
    elements: tuple[str, ...],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    loga = data["log_abundance"].astype(float)
    noisy = loga + rng.normal(0.0, sigma, loga.shape)
    ref = ELEMENT_INDEX[REFERENCE]
    return np.column_stack(
        [noisy[:, ELEMENT_INDEX[element]] - noisy[:, ref] for element in elements]
    )


def _oof(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, n_trees: int
) -> np.ndarray:
    probability = np.zeros(len(y))
    splits = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=0
    ).split(x, y)
    for train, test in splits:
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


def _precision(tpr: float, fpr: float, pi: float = PREVALENCE) -> float:
    return tpr * pi / (tpr * pi + fpr * (1.0 - pi))


def _precision_at_fpr(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, target: float
) -> float:
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    index = min(np.searchsorted(fpr, target), len(fpr) - 1)
    return float(_precision(float(tpr[index]), float(fpr[index])))


def _top_overlap(
    score_a: np.ndarray, score_b: np.ndarray, fraction: float = 0.01
) -> float:
    k = max(1, int(fraction * len(score_a)))
    top_a = set(np.argsort(-score_a)[:k])
    top_b = set(np.argsort(-score_b)[:k])
    return len(top_a & top_b) / k


def _top_precision(
    score: np.ndarray, y: np.ndarray, fraction: float = 0.01
) -> float:
    k = max(1, int(fraction * len(score)))
    return float(y[np.argsort(-score)[:k]].mean())


def _paired(
    score_full: np.ndarray,
    score_without: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    n_boot: int = 500,
) -> dict[str, float]:
    auc_full = float(roc_auc_score(y, score_full, sample_weight=weight))
    auc_without = float(roc_auc_score(y, score_without, sample_weight=weight))
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[index], score_full[index], sample_weight=weight[index])
            - roc_auc_score(
                y[index], score_without[index], sample_weight=weight[index]
            )
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "auc_full": auc_full,
        "auc_without": auc_without,
        "delta": auc_full - auc_without,
        "ci_low": float(low),
        "ci_high": float(high),
        "top1_overlap": _top_overlap(score_full, score_without),
        "precision_1pct_full": _precision_at_fpr(y, score_full, weight, 0.01),
        "precision_1pct_without": _precision_at_fpr(
            y, score_without, weight, 0.01
        ),
        "precision_01pct_full": _precision_at_fpr(y, score_full, weight, 0.001),
        "precision_01pct_without": _precision_at_fpr(
            y, score_without, weight, 0.001
        ),
    }


def _plot_bars(
    rows: dict[str, dict[str, float]],
    key_full: str,
    key_without: str,
    ylabel: str,
    title: str,
    path: Path,
    highlight: tuple[str, ...] = ("Zn", "Cu", "Co"),
) -> None:
    names = list(rows)
    values = [rows[name][key_full] - rows[name][key_without] for name in names]
    order = np.argsort(values)
    colors = [
        "tab:red" if names[i] == "Zn" else
        "tab:orange" if names[i] == "Cu" else
        "tab:purple" if names[i] == "Co" else "tab:blue"
        for i in order
    ]
    fig, axis = plt.subplots(figsize=(9.5, 0.35 * len(names) + 2.0))
    axis.barh(np.arange(len(order)), [values[i] for i in order], color=colors)
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([names[i] for i in order], fontsize=8)
    axis.set_xlabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_turnover(rows: dict[str, dict[str, float]], path: Path) -> None:
    names = list(rows)
    turnover = [1.0 - rows[name]["top1_overlap"] for name in names]
    order = np.argsort(turnover)
    colors = [
        "tab:red" if names[i] == "Zn" else
        "tab:orange" if names[i] == "Cu" else
        "tab:purple" if names[i] == "Co" else "tab:blue"
        for i in order
    ]
    fig, axis = plt.subplots(figsize=(9.5, 0.35 * len(names) + 2.0))
    axis.barh(np.arange(len(order)), [turnover[i] for i in order], color=colors)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([names[i] for i in order], fontsize=8)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("top-1% turnover (1 - overlap)")
    axis.set_title("How much removing one element changes the top-1% list")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_auc(rows: dict[str, dict[str, float]], path: Path) -> None:
    names = list(rows)
    values = [rows[name]["delta"] for name in names]
    order = np.argsort(values)
    colors = [
        "tab:red" if names[i] == "Zn" else
        "tab:orange" if names[i] == "Cu" else
        "tab:purple" if names[i] == "Co" else "tab:blue"
        for i in order
    ]
    fig, axis = plt.subplots(figsize=(9.5, 0.35 * len(names) + 2.0))
    axis.barh(np.arange(len(order)), [values[i] for i in order], color=colors)
    low = [values[i] - rows[names[i]]["ci_low"] for i in order]
    high = [rows[names[i]]["ci_high"] - values[i] for i in order]
    axis.errorbar(
        [values[i] for i in order],
        np.arange(len(order)),
        xerr=[low, high],
        fmt="none",
        color="black",
        capsize=2,
    )
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([names[i] for i in order], fontsize=8)
    axis.set_xlabel("Delta AUC: full - without element")
    axis.set_title("Marginal AUC of each element in the full 26-ratio set")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_zn_scores(
    full_scores: np.ndarray,
    no_zn_scores: np.ndarray,
    core_scores: np.ndarray,
    core_zn_scores: np.ndarray,
    y: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    panels = (
        (axes[0], no_zn_scores, full_scores, "all 26", "all 26 minus Zn"),
        (axes[1], core_scores, core_zn_scores, "core 7 + Zn", "core 7"),
    )
    for axis, x, yv, high_label, low_label in panels:
        negative = y == 0
        positive = y == 1
        axis.scatter(
            x[negative], yv[negative], s=3, color="0.65", alpha=0.18,
            rasterized=True,
        )
        axis.scatter(
            x[positive], yv[positive], s=5, color="tab:red", alpha=0.35,
            rasterized=True,
        )
        axis.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0)
        threshold_x = np.quantile(x, 0.99)
        threshold_y = np.quantile(yv, 0.99)
        axis.axvline(threshold_x, color="0.3", ls=":", lw=1.0)
        axis.axhline(threshold_y, color="0.3", ls=":", lw=1.0)
        overlap = _top_overlap(yv, x)
        precision_high = _top_precision(yv, y)
        precision_low = _top_precision(x, y)
        axis.text(
            0.03,
            0.96,
            f"top-1% overlap {overlap:.2f}\n"
            f"top-1% precision {precision_high:.2f} vs {precision_low:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_xlabel(f"score, {low_label}")
        axis.set_ylabel(f"score, {high_label}")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{high_label} vs {low_label}")
        axis.grid(alpha=0.25)
    fig.suptitle("Zn and the top 1% of the candidate list", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_at_k(
    scores: dict[str, np.ndarray], y: np.ndarray, path: Path
) -> None:
    fractions = np.array(
        [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, score in scores.items():
        precision = [
            _top_precision(score, y, fraction) for fraction in fractions
        ]
        axis.plot(100 * fractions, precision, marker="o", label=label)
    axis.axvline(1.0, color="0.4", ls=":", lw=1.0)
    axis.set_xscale("log")
    axis.set_xlabel("top K% of candidates")
    axis.set_ylabel("precision (fraction of true PISN)")
    axis.set_title("Zn helps below the top 1%, not at it")
    axis.set_ylim(0.4, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    max_samples: int = 5000,
    n_trees: int = 50,
    policies: tuple[str, ...] = POLICIES,
    label: str = "f05",
    k: float = 3.0,
) -> dict[str, Any]:
    base = Path(base)
    if outdir is None:
        outdir = base / f"element_influence_{label}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores_full = []
    scores_without: dict[str, list[np.ndarray]] = {}
    labels = []
    weights = []
    first_policy_scores = None

    for policy in policies:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, meta = load_dataset(path)
        names = tuple(meta["feature_names"])
        elements = tuple(name.split("/")[0] for name in names)
        for element in elements:
            scores_without.setdefault(element, [])
        n = len(data["y_f_05"])
        if n > max_samples:
            index = np.sort(
                np.random.default_rng(0).choice(n, max_samples, replace=False)
            )
            data = {key: value[index] for key, value in data.items()}
        if label == "D":
            y = _detection_label(data, meta, k=k).astype(int)
        else:
            y = data["y_f_05"].astype(int)
        weight = _balance(y)
        ratios = _ratios(data, elements, sigma, np.random.default_rng(0))
        full_columns = list(range(len(elements)))
        full_score = _oof(ratios[:, full_columns], y, weight, n_trees)
        scores_full.append(full_score)
        labels.append(y)
        weights.append(weight)
        for element in elements:
            columns = [
                j for j, name in enumerate(elements) if name != element
            ]
            score = _oof(ratios[:, columns], y, weight, n_trees)
            scores_without[element].append(score)
        if policy == "Limongi18_clamp":
            no_zn_columns = [
                j for j, name in enumerate(elements) if name != "Zn"
            ]
            no_zn_score = scores_without["Zn"][-1]
            core_columns = [elements.index(element) for element in CORE7]
            core_zn_columns = core_columns + [elements.index("Zn")]
            core_score = _oof(ratios[:, core_columns], y, weight, n_trees)
            core_zn_score = _oof(
                ratios[:, core_zn_columns], y, weight, n_trees
            )
            first_policy_scores = (
                full_score,
                no_zn_score,
                core_score,
                core_zn_score,
                y,
            )

    if not scores_full:
        raise FileNotFoundError("no v20 datasets")
    y_all = np.concatenate(labels)
    w_all = np.concatenate(weights)
    full_all = np.concatenate(scores_full)
    rows: dict[str, Any] = {}
    for element, chunks in scores_without.items():
        without_all = np.concatenate(chunks)
        rows[element] = _paired(full_all, without_all, y_all, w_all)

    result = {
        "sigma": sigma,
        "label": label,
        "k": k,
        "n_elements": len(rows),
        "elements": rows,
    }
    with (outdir / "element_influence.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    _plot_delta_auc(rows, outdir / "leave_one_out_auc.png")
    _plot_turnover(rows, outdir / "leave_one_out_top1.png")
    _plot_bars(
        rows,
        "precision_1pct_full",
        "precision_1pct_without",
        "Delta precision at FPR = 1%",
        "Precision change when one element is removed",
        outdir / "leave_one_out_precision_1pct.png",
    )
    _plot_bars(
        rows,
        "precision_01pct_full",
        "precision_01pct_without",
        "Delta precision at FPR = 0.1%",
        "Precision change when one element is removed (strict cut)",
        outdir / "leave_one_out_precision_01pct.png",
    )
    if first_policy_scores is not None:
        full_score, no_zn_score, core_score, core_zn_score, y_first = (
            first_policy_scores
        )
        _plot_zn_scores(
            full_score,
            no_zn_score,
            core_score,
            core_zn_score,
            y_first,
            outdir / "zn_top1_scores.png",
        )
        _plot_precision_at_k(
            {
                "all 26": full_score,
                "all 26 minus Zn": no_zn_score,
                "core 7": core_score,
                "core 7 + Zn": core_zn_score,
            },
            y_first,
            outdir / "zn_precision_at_k.png",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--label", default="f05", choices=("f05", "D"))
    parser.add_argument("--k", type=float, default=3.0)
    args = parser.parse_args(argv)
    result = run(
        args.base,
        args.outdir,
        sigma=args.sigma,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
        label=args.label,
        k=args.k,
    )
    for element, row in sorted(
        result["elements"].items(), key=lambda kv: -kv[1]["delta"]
    ):
        print(
            f"{element:2s} dAUC={row['delta']:+.4f} "
            f"CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}] "
            f"turnover={1 - row['top1_overlap']:.2f} "
            f"dPrec1%={row['precision_1pct_full'] - row['precision_1pct_without']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
