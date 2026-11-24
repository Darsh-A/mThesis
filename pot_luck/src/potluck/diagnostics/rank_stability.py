"""Is the element ranking stable, or is it retraining noise?

``element_influence`` reports a paired bootstrap interval on a *fixed* pair of
out-of-fold score arrays. That interval answers "how much would this delta move
if I drew a different sample and kept these predictions?" It does not answer the
question a ranking needs: "how much would it move if I retrained?"

This module varies the noise realisation and the CV split together, retrains
everything, and reports the spread of each leave-one-out delta across seeds.
The across-seed spread is the noise floor a ranking must clear.

Outputs (``rank_stability/`` by default):
    rank_stability.json
    delta_auc_by_seed.png
    rank_correlation.png

Usage:
    python -m src.potluck.diagnostics.rank_stability --seeds 5
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..constants import ELEMENT_INDEX, REFERENCE
from ..io import load_dataset
from .element_influence import _detection_label
from .v20_mega import _balance

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
)


def _ratios(
    data: dict[str, np.ndarray],
    elements: tuple[str, ...],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Element-level Gaussian noise, then iron-normalised ratios."""
    loga = data["log_abundance"].astype(float)
    noisy = loga + rng.normal(0.0, sigma, loga.shape)
    reference = ELEMENT_INDEX[REFERENCE]
    return np.column_stack(
        [
            noisy[:, ELEMENT_INDEX[element]] - noisy[:, reference]
            for element in elements
        ]
    )


def _oof(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    n_trees: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold probabilities, with the split varied by ``seed``."""
    probability = np.zeros(len(y))
    splits = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed
    ).split(x, y)
    for train, test in splits:
        model = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def _auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    return float(roc_auc_score(y, score, sample_weight=weight))


def _bootstrap_half_width(
    y: np.ndarray,
    score_full: np.ndarray,
    score_without: np.ndarray,
    weight: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> float:
    """Half-width of the paired bootstrap CI on a fixed score pair.

    This reproduces the interval in ``element_influence._paired``: the same
    predictions are reused, only the rows are resampled.
    """
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size < 2:
            continue
        deltas.append(
            _auc(y[index], score_full[index], weight[index])
            - _auc(y[index], score_without[index], weight[index])
        )
    if not deltas:
        return float("nan")
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(0.5 * (high - low))


def _subsample(
    data: dict[str, np.ndarray], n: int, seed: int
) -> dict[str, np.ndarray]:
    if n >= len(data["y_f_05"]):
        return data
    index = np.sort(
        np.random.default_rng(seed).choice(len(data["y_f_05"]), n, replace=False)
    )
    return {key: value[index] for key, value in data.items()}


def _rank_correlation(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    """Spearman correlation of two element rankings."""
    order_a = [name for name, _ in sorted(a, key=lambda kv: -kv[1])]
    order_b = [name for name, _ in sorted(b, key=lambda kv: -kv[1])]
    position_a = {name: i for i, name in enumerate(order_a)}
    position_b = {name: i for i, name in enumerate(order_b)}
    names = list(position_a)
    x = np.array([position_a[name] for name in names], dtype=float)
    y = np.array([position_b[name] for name in names], dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    policies: tuple[str, ...] = POLICIES,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    sigma: float = 0.2,
    max_samples: int = 2500,
    n_trees: int = 40,
    label: str = "f05",
    k: float = 3.0,
    top_k: int = 3,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "rank_stability"
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, dict[str, float]] = {}
    auc_full: dict[int, float] = {}
    elements: tuple[str, ...] = ()
    policies_used: list[str] = []

    for policy in policies:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, meta = load_dataset(path)
        data = _subsample(data, max_samples, seed=0)
        elements = tuple(name.split("/")[0] for name in meta["feature_names"])
        policies_used.append(policy)

        y = (
            _detection_label(data, meta, k=k).astype(int)
            if label == "D"
            else data["y_f_05"].astype(int)
        )
        weight = _balance(y)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            ratios = _ratios(data, elements, sigma, rng)
            full = _oof(ratios, y, weight, n_trees, seed)
            auc_full[seed] = _auc(y, full, weight)
            deltas = {}
            for element in elements:
                columns = [
                    column
                    for column, name in enumerate(elements)
                    if name != element
                ]
                without = _oof(ratios[:, columns], y, weight, n_trees, seed)
                deltas[element] = auc_full[seed] - _auc(y, without, weight)
            per_seed[seed] = deltas

    if not per_seed:
        raise FileNotFoundError("no v20 datasets found")

    table: dict[str, Any] = {}
    for element in elements:
        values = np.array([per_seed[seed][element] for seed in seeds])
        ranks = [
            sorted(elements, key=lambda name: -per_seed[seed][name]).index(element)
            for seed in seeds
        ]
        top_hits = sum(
            element
            in sorted(elements, key=lambda name: -per_seed[seed][name])[:top_k]
            for seed in seeds
        )
        table[element] = {
            "delta_mean": float(values.mean()),
            "delta_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "delta_min": float(values.min()),
            "delta_max": float(values.max()),
            "snr": (
                float(values.mean() / values.std(ddof=1))
                if values.size > 1 and values.std(ddof=1) > 0
                else float("inf")
            ),
            "rank_median": float(np.median(ranks)),
            "rank_range": int(max(ranks) - min(ranks)),
            "top_k_frequency": top_hits / len(seeds),
            "sign_consistent": bool(
                len({bool(value > 0) for value in values}) == 1
            ),
        }

    # Reproduce the fixed-prediction bootstrap for comparison. Rebuild one seed.
    first_policy = policies_used[0]
    data, meta = load_dataset(base / first_policy / "dataset.npz")
    data = _subsample(data, max_samples, seed=0)
    y = (
        _detection_label(data, meta, k=k).astype(int)
        if label == "D"
        else data["y_f_05"].astype(int)
    )
    weight = _balance(y)
    seed = seeds[0]
    ratios = _ratios(data, elements, sigma, np.random.default_rng(seed))
    full = _oof(ratios, y, weight, n_trees, seed)
    for element in elements:
        columns = [
            column for column, name in enumerate(elements) if name != element
        ]
        without = _oof(ratios[:, columns], y, weight, n_trees, seed)
        table[element]["bootstrap_half_width"] = _bootstrap_half_width(
            y, full, without, weight
        )

    rankings = [
        [(name, per_seed[seed][name]) for name in elements] for seed in seeds
    ]
    correlations = [
        _rank_correlation(rankings[i], rankings[j])
        for i in range(len(rankings))
        for j in range(i + 1, len(rankings))
    ]

    aucs = np.array([auc_full[seed] for seed in seeds])
    # Seed-to-seed spread pooled over elements: the size of an effect that is
    # indistinguishable from retraining noise. An element must clear both this
    # floor and a signal-to-noise floor, since a stable but negligible effect
    # (tiny mean, tiny sd) otherwise passes on SNR alone.
    noise_floor = float(np.median([row["delta_sd"] for row in table.values()]))
    resolvable = [
        name
        for name, row in table.items()
        if row["snr"] > 2.0
        and row["sign_consistent"]
        and row["delta_mean"] > noise_floor
    ]

    result = {
        "label": label,
        "sigma": sigma,
        "policies": policies_used,
        "seeds": list(seeds),
        "n_samples": int(max_samples),
        "n_trees": n_trees,
        "auc_full": {
            "mean": float(aucs.mean()),
            "sd": float(aucs.std(ddof=1)) if aucs.size > 1 else 0.0,
            "values": [float(value) for value in aucs],
        },
        "rank_correlation": {
            "mean": float(np.mean(correlations)) if correlations else float("nan"),
            "min": float(np.min(correlations)) if correlations else float("nan"),
            "max": float(np.max(correlations)) if correlations else float("nan"),
        },
        "resolvable_elements": resolvable,
        "noise_floor": noise_floor,
        "elements": table,
    }
    with (outdir / "rank_stability.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    _plot_deltas(table, aucs, outdir / "delta_auc_by_seed.png", label)
    _plot_correlations(rankings, correlations, outdir / "rank_correlation.png", label)
    return result


def _plot_deltas(
    table: dict[str, dict[str, float]],
    aucs: np.ndarray,
    path: Path,
    label: str,
) -> None:
    names = sorted(table, key=lambda name: -table[name]["delta_mean"])
    mean = np.array([table[name]["delta_mean"] for name in names])
    sd = np.array([table[name]["delta_sd"] for name in names])
    order = np.arange(len(names))[::-1]

    fig, axis = plt.subplots(figsize=(8.5, 0.32 * len(names) + 2.2))
    axis.barh(order, mean, xerr=sd, color="tab:blue", alpha=0.85, capsize=2)
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_yticks(order)
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xlabel(r"$\Delta$AUC: full $-$ without element")
    axis.set_title(
        f"Element influence with retraining noise ({label}), "
        f"{len(aucs)} seeds\n"
        f"AUC across seeds = {aucs.mean():.4f} $\\pm$ {aucs.std(ddof=1):.4f}"
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_correlations(
    rankings: list[list[tuple[str, float]]],
    correlations: list[float],
    path: Path,
    label: str,
) -> None:
    n = len(rankings)
    matrix = np.eye(n)
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = correlations[index]
            index += 1

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    image = axes[0].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("seed")
    axes[0].set_title("Rank correlation between seeds")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    for rank, (name, _) in enumerate(rankings[0]):
        positions = [
            [item[0] for item in ranking].index(name) for ranking in rankings
        ]
        axes[1].plot(positions, [rank] * n, "o", ms=3, alpha=0.5, color="tab:blue")
    axes[1].plot([0, len(rankings[0]) - 1], [0, len(rankings[0]) - 1],
                 color="0.6", ls="--", lw=1.0)
    axes[1].set_xlabel("rank under one seed")
    axes[1].set_ylabel("rank under the first seed")
    axes[1].set_title(f"Rank movement across {n} seeds ({label})")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=2500)
    parser.add_argument("--n-trees", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--label", default="f05", choices=("f05", "D"))
    parser.add_argument("--k", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = run(
        base=args.base,
        outdir=args.outdir,
        policies=tuple(name for name in args.policies.split(",") if name),
        seeds=tuple(range(args.seeds)),
        sigma=args.sigma,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
        label=args.label,
        k=args.k,
        top_k=args.top_k,
    )
    print(
        f"AUC = {result['auc_full']['mean']:.4f} "
        f"+/- {result['auc_full']['sd']:.4f} across seeds"
    )
    print(
        f"rank correlation: mean {result['rank_correlation']['mean']:.2f} "
        f"[{result['rank_correlation']['min']:.2f}, "
        f"{result['rank_correlation']['max']:.2f}]"
    )
    print(f"resolvable: {', '.join(result['resolvable_elements']) or 'none'}")
    print(f"noise floor (median delta sd): {result['noise_floor']:.4f}")
    print(f"\n{'el':3s} {'dAUC':>9s} {'sd':>8s} {'SNR':>6s} {'95% CI':>8s} "
          f"{'rank':>5s} {'top-k':>6s}")
    for name, row in sorted(
        result["elements"].items(), key=lambda kv: -kv[1]["delta_mean"]
    ):
        print(
            f"{name:3s} {row['delta_mean']:+9.4f} {row['delta_sd']:8.4f} "
            f"{row['snr']:6.1f} {row['bootstrap_half_width']:8.4f} "
            f"{row['rank_median']:5.0f} {row['top_k_frequency']:6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
