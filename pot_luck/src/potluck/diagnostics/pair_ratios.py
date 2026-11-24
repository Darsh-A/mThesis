"""Do non-Fe pair ratios such as Mg/Si beat the single [X/Fe] features?

The production feature set is [X/Fe] for all 26 elements. This script tests
explicit [X/Y] ratios: standalone, added to the core 7, and added to the full
[X/Fe] set. Gaussian element-level noise sigma=0.2.
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
from .v20_mega import _balance

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)
KEY_ELEMENTS = (
    "C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn", "O", "Al",
)
CORE7 = ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni")
PAIRS = tuple(
    (numerator, denominator)
    for i, numerator in enumerate(KEY_ELEMENTS)
    for denominator in KEY_ELEMENTS[i + 1:]
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


def _auc(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, n_trees: int
) -> float:
    return float(
        roc_auc_score(
            y, _oof(x, y, weight, n_trees), sample_weight=weight
        )
    )


def _plot_bars(
    labels: list[str], values: list[float], path: Path, title: str, xlabel: str,
    highlight: tuple[str, ...] = (),
) -> None:
    order = np.argsort(values)
    colors = [
        "tab:red" if labels[i] in highlight else "tab:blue" for i in order
    ]
    fig, axis = plt.subplots(figsize=(9.5, 0.28 * len(labels) + 2.0))
    axis.barh(np.arange(len(order)), [values[i] for i in order], color=colors)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([labels[i] for i in order], fontsize=7)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    max_samples: int = 5000,
    n_trees: int = 40,
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "pair_ratios"
    outdir.mkdir(parents=True, exist_ok=True)
    key_indices = [ELEMENT_INDEX[element] for element in KEY_ELEMENTS]
    fe_index = ELEMENT_INDEX[REFERENCE]
    single_results: dict[str, list[float]] = {
        element: [] for element in KEY_ELEMENTS
    }
    pair_results: dict[str, list[float]] = {
        f"{a}/{b}": [] for a, b in PAIRS
    }
    core_base: list[float] = []
    core_pair: dict[str, list[float]] = {
        f"{a}/{b}": [] for a, b in PAIRS
    }
    full_base: list[float] = []
    full_pair: dict[str, list[float]] = {
        f"{a}/{b}": [] for a, b in PAIRS
    }

    for policy in policies:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, meta = load_dataset(path)
        loga = data["log_abundance"].astype(float)
        finite = np.isfinite(loga[:, key_indices]).all(axis=1)
        index = np.flatnonzero(finite)
        if len(index) > max_samples:
            index = np.sort(
                np.random.default_rng(0).choice(
                    index, max_samples, replace=False
                )
            )
        y = data["y_f_05"][index].astype(int)
        weight = _balance(y)
        noisy = loga[index] + np.random.default_rng(0).normal(
            0.0, sigma, (len(index), loga.shape[1])
        )
        singles = np.column_stack(
            [
                noisy[:, ELEMENT_INDEX[element]] - noisy[:, fe_index]
                for element in KEY_ELEMENTS
            ]
        )
        pairs = np.column_stack(
            [
                noisy[:, ELEMENT_INDEX[a]] - noisy[:, ELEMENT_INDEX[b]]
                for a, b in PAIRS
            ]
        )
        full_names = tuple(meta["feature_names"])
        full_elements = tuple(name.split("/")[0] for name in full_names)
        full = np.column_stack(
            [
                noisy[:, ELEMENT_INDEX[element]] - noisy[:, fe_index]
                for element in full_elements
            ]
        )
        full = np.where(
            np.isfinite(full), full, np.nanmedian(full, axis=0)
        )
        core_columns = [full_elements.index(element) for element in CORE7]
        core = full[:, core_columns]

        for j, element in enumerate(KEY_ELEMENTS):
            single_results[element].append(
                _auc(singles[:, [j]], y, weight, n_trees)
            )
        for j, (a, b) in enumerate(PAIRS):
            name = f"{a}/{b}"
            pair_results[name].append(_auc(pairs[:, [j]], y, weight, n_trees))
            core_pair[name].append(
                _auc(
                    np.column_stack([core, pairs[:, j]]), y, weight, n_trees
                )
            )
            full_pair[name].append(
                _auc(
                    np.column_stack([full, pairs[:, j]]), y, weight, n_trees
                )
            )
        core_base.append(_auc(core, y, weight, n_trees))
        full_base.append(_auc(full, y, weight, n_trees))

    singles = {
        element: float(np.mean(values))
        for element, values in single_results.items()
    }
    pairs = {
        name: float(np.mean(values))
        for name, values in pair_results.items()
    }
    core_delta = {
        name: float(np.mean(core_pair[name]) - np.mean(core_base))
        for name in core_pair
    }
    full_delta = {
        name: float(np.mean(full_pair[name]) - np.mean(full_base))
        for name in full_pair
    }
    result = {
        "sigma": sigma,
        "n_pairs": len(PAIRS),
        "singles": singles,
        "pairs": pairs,
        "core7_base": float(np.mean(core_base)),
        "core7_delta": core_delta,
        "full_base": float(np.mean(full_base)),
        "full_delta": full_delta,
    }
    with (outdir / "pair_ratios.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    _plot_bars(
        list(pairs),
        list(pairs.values()),
        outdir / "pair_standalone_auc.png",
        "Standalone AUC of non-Fe pair ratios",
        "AUC",
        highlight=("Mg/Si", "Zn/Mg", "Co/Ni", "Mn/Cr", "Na/Mg"),
    )
    _plot_bars(
        list(core_delta),
        list(core_delta.values()),
        outdir / "pair_added_to_core7.png",
        "Delta AUC when a pair ratio is added to the core 7",
        "Delta AUC (core 7 + pair minus core 7)",
        highlight=("Mg/Si", "Zn/Mg", "Co/Ni", "Mn/Cr", "Na/Mg"),
    )
    _plot_bars(
        list(full_delta),
        list(full_delta.values()),
        outdir / "pair_added_to_full.png",
        "Delta AUC when a pair ratio is added to the full [X/Fe] set",
        "Delta AUC (full + pair minus full)",
        highlight=("Mg/Si", "Zn/Mg", "Co/Ni", "Mn/Cr", "Na/Mg"),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-trees", type=int, default=40)
    args = parser.parse_args(argv)
    result = run(
        args.base,
        args.outdir,
        sigma=args.sigma,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
    )
    print("best standalone pairs")
    for name, value in sorted(
        result["pairs"].items(), key=lambda kv: -kv[1]
    )[:12]:
        print(f"  {name:8s} {value:.3f}")
    print("best single [X/Fe]")
    for name, value in sorted(
        result["singles"].items(), key=lambda kv: -kv[1]
    )[:6]:
        print(f"  {name:8s} {value:.3f}")
    print(f"core7 base {result['core7_base']:.3f}")
    for name, value in sorted(
        result["core7_delta"].items(), key=lambda kv: -kv[1]
    )[:10]:
        print(f"  core7 + {name:8s} {value:+.4f}")
    print(f"full base {result['full_base']:.3f}")
    for name, value in sorted(
        result["full_delta"].items(), key=lambda kv: -kv[1]
    )[:6]:
        print(f"  full + {name:8s} {value:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
