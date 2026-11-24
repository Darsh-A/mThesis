"""General element subset selection with Gaussian noise, no catalogue bias.

Greedy forward selection over a practical set of measurable elements. The
candidate set is physics-motivated: alpha elements, odd-even elements,
iron-peak elements, and the light elements with usable lines. Noble gases,
halogens and Ga/Ge are excluded because they are not practical survey targets.

Runs for both labels: y_f_05 and y = 1[D > 3].
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
PRACTICAL = (
    "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "P", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn",
)
CORE7 = ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni")


def _ratios(
    data: dict[str, np.ndarray],
    elements: tuple[str, ...],
    sigma: float,
    seed: int,
) -> np.ndarray:
    loga = data["log_abundance"].astype(float)
    noisy = loga + np.random.default_rng(seed).normal(
        0.0, sigma, loga.shape
    )
    ref = ELEMENT_INDEX[REFERENCE]
    return np.column_stack(
        [noisy[:, ELEMENT_INDEX[element]] - noisy[:, ref] for element in elements]
    )


def _oof(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, n_trees: int
) -> np.ndarray:
    probability = np.zeros(len(y))
    for train, test in StratifiedKFold(
        n_splits=3, shuffle=True, random_state=0
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


def _greedy(
    ratios: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    elements: tuple[str, ...],
    max_k: int,
    n_trees: int,
) -> list[dict[str, Any]]:
    cache: dict[frozenset[str], float] = {}

    def score(subset: tuple[str, ...]) -> float:
        key = frozenset(subset)
        if key not in cache:
            values = []
            for policy, x_all in ratios.items():
                y = labels[policy]
                weight = _balance(y)
                x = x_all[:, [elements.index(e) for e in subset]]
                values.append(
                    roc_auc_score(
                        y,
                        _oof(x, y, weight, n_trees),
                        sample_weight=weight,
                    )
                )
            cache[key] = float(np.mean(values))
        return cache[key]

    selected: list[str] = []
    history = []
    for step in range(max_k):
        best = max(
            (element for element in elements if element not in selected),
            key=lambda element: score(tuple(selected + [element])),
        )
        selected.append(best)
        history.append(
            {
                "k": step + 1,
                "added": best,
                "subset": list(selected),
                "auc": score(tuple(selected)),
            }
        )
    return history


def _plot(
    history: list[dict[str, Any]],
    title: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5))
    k = [row["k"] for row in history]
    auc = [row["auc"] for row in history]
    axis.plot(k, auc, marker="o", color="tab:blue")
    for row in history:
        axis.annotate(
            row["added"],
            (row["k"], row["auc"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    axis.set_xlabel("number of elements")
    axis.set_ylabel("mean AUC")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigma: float = 0.2,
    max_samples: int = 5000,
    n_trees: int = 40,
    max_k: int = 10,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "element_selection"
    outdir.mkdir(parents=True, exist_ok=True)
    ratios = {}
    labels_f05 = {}
    labels_d = {}
    for policy in POLICIES:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data, meta = load_dataset(path)
        n = len(data["y_f_05"])
        if n > max_samples:
            index = np.sort(
                np.random.default_rng(0).choice(n, max_samples, replace=False)
            )
            data = {key: value[index] for key, value in data.items()}
        ratios[policy] = _ratios(data, PRACTICAL, sigma, seed=0)
        labels_f05[policy] = data["y_f_05"].astype(int)
        labels_d[policy] = _detection_label(data, meta, k=3.0).astype(int)
    if not ratios:
        raise FileNotFoundError("no v20 datasets")

    f05 = _greedy(ratios, labels_f05, PRACTICAL, max_k, n_trees)
    dlabel = _greedy(ratios, labels_d, PRACTICAL, max_k, n_trees)
    result = {
        "sigma": sigma,
        "n_trees": n_trees,
        "max_k": max_k,
        "candidates": list(PRACTICAL),
        "f05": f05,
        "D": dlabel,
    }
    with (outdir / "element_selection.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    _plot(
        f05,
        "y = 1[f_ZP >= 0.5]",
        outdir / "selection_curves_f05.png",
    )
    _plot(
        dlabel,
        "y = 1[D > 3]",
        outdir / "selection_curves_D.png",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-trees", type=int, default=40)
    parser.add_argument("--max-k", type=int, default=10)
    args = parser.parse_args(argv)
    result = run(
        args.base,
        args.outdir,
        sigma=args.sigma,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
        max_k=args.max_k,
    )
    for label in ("f05", "D"):
        print(label)
        for row in result[label]:
            print(
                f"  k={row['k']:2d} add={row['added']:2s} "
                f"auc={row['auc']:.4f} subset={'+'.join(row['subset'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
