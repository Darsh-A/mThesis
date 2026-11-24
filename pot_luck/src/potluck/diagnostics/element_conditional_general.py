"""Conditional add-one-element map with single-element bases, no Li+ bias.

Rows are single-element bases; columns are elements added to that base. Value is
the change in mean AUC. Gaussian element-level noise sigma=0.2, three yield
tables, both labels. N is included in the candidate set.
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
from sklearn.metrics import roc_auc_score

from ..io import load_dataset
from .element_influence import _detection_label
from .element_selection import POLICIES, PRACTICAL, _oof, _ratios
from .v20_mega import _balance


def _conditional(
    ratios: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    elements: tuple[str, ...],
    n_trees: int,
) -> tuple[dict[str, float], np.ndarray]:
    cache: dict[frozenset[str], float] = {}

    def auc(subset: tuple[str, ...]) -> float:
        key = frozenset(subset)
        if key not in cache:
            values = []
            for policy, x_all in ratios.items():
                y = labels[policy]
                weight = _balance(y)
                columns = [elements.index(element) for element in subset]
                values.append(
                    roc_auc_score(
                        y,
                        _oof(x_all[:, columns], y, weight, n_trees),
                        sample_weight=weight,
                    )
                )
            cache[key] = float(np.mean(values))
        return cache[key]

    singles = {element: auc((element,)) for element in elements}
    matrix = np.full((len(elements), len(elements)), np.nan)
    for i, base in enumerate(elements):
        for j, candidate in enumerate(elements):
            if base == candidate:
                continue
            matrix[i, j] = auc((base, candidate)) - singles[base]
    return singles, matrix


def _heatmap(
    matrix: np.ndarray,
    elements: tuple[str, ...],
    singles: dict[str, float],
    path: Path,
    title: str,
) -> None:
    limit = max(float(np.nanmax(np.abs(matrix))), 0.02)
    fig, axis = plt.subplots(figsize=(15, 13))
    image = axis.imshow(
        matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit
    )
    axis.set_xticks(np.arange(len(elements)))
    axis.set_xticklabels(elements, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(elements)))
    axis.set_yticklabels(
        [f"{element} ({singles[element]:.2f})" for element in elements],
        fontsize=8,
    )
    for i in range(len(elements)):
        for j in range(len(elements)):
            value = matrix[i, j]
            if np.isfinite(value):
                axis.text(
                    j,
                    i,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if abs(value) < 0.6 * limit else "white",
                )
    axis.set_xlabel("added element")
    axis.set_ylabel("single-element base (standalone AUC)")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Delta AUC (base + element - base)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _standalone(
    singles: dict[str, float],
    title: str,
    path: Path,
) -> None:
    order = sorted(singles, key=lambda element: -singles[element])
    fig, axis = plt.subplots(figsize=(7, 0.32 * len(order) + 2.0))
    axis.barh(
        np.arange(len(order)),
        [singles[element] for element in order],
        color=["tab:red" if element == "N" else "tab:blue" for element in order],
    )
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels(order, fontsize=8)
    axis.set_xlim(0.4, 1.0)
    axis.set_xlabel("AUC")
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
    n_trees: int = 30,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "element_conditional_general"
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

    singles_f05, matrix_f05 = _conditional(
        ratios, labels_f05, PRACTICAL, n_trees
    )
    singles_d, matrix_d = _conditional(ratios, labels_d, PRACTICAL, n_trees)

    result = {
        "sigma": sigma,
        "n_trees": n_trees,
        "candidates": list(PRACTICAL),
        "standalone_f05": singles_f05,
        "standalone_D": singles_d,
        "delta_f05": {
            base_element: {
                candidate: float(matrix_f05[i, j])
                for j, candidate in enumerate(PRACTICAL)
                if np.isfinite(matrix_f05[i, j])
            }
            for i, base_element in enumerate(PRACTICAL)
        },
        "delta_D": {
            base_element: {
                candidate: float(matrix_d[i, j])
                for j, candidate in enumerate(PRACTICAL)
                if np.isfinite(matrix_d[i, j])
            }
            for i, base_element in enumerate(PRACTICAL)
        },
    }
    with (outdir / "element_conditional_general.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    for label, singles, matrix in (
        ("f05", singles_f05, matrix_f05),
        ("D", singles_d, matrix_d),
    ):
        with (outdir / f"best_additions_{label}.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["base", "base_auc", "best_addition", "delta"])
            for i, base_element in enumerate(PRACTICAL):
                order = np.argsort(-matrix[i])
                for j in order[:3]:
                    if not np.isfinite(matrix[i, j]):
                        continue
                    writer.writerow(
                        [
                            base_element,
                            f"{singles[base_element]:.4f}",
                            PRACTICAL[j],
                            f"{matrix[i, j]:+.4f}",
                        ]
                    )

    _heatmap(
        matrix_f05,
        PRACTICAL,
        singles_f05,
        outdir / "conditional_general_f05.png",
        "Add one element to a single-element base (f05, sigma=0.2)",
    )
    _heatmap(
        matrix_d,
        PRACTICAL,
        singles_d,
        outdir / "conditional_general_D.png",
        "Add one element to a single-element base (D > 3, sigma=0.2)",
    )
    _standalone(
        singles_f05, "standalone AUC, f05", outdir / "standalone_auc_f05.png"
    )
    _standalone(
        singles_d, "standalone AUC, D > 3", outdir / "standalone_auc_D.png"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-trees", type=int, default=30)
    args = parser.parse_args(argv)
    result = run(
        args.base,
        args.outdir,
        sigma=args.sigma,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
    )
    for label in ("f05", "D"):
        print(label)
        for base_element, row in result[f"delta_{label}"].items():
            best = sorted(row.items(), key=lambda kv: -kv[1])[:3]
            print(
                f"  {base_element:2s} base={result[f'standalone_{label}'][base_element]:.3f} "
                + " ".join(f"{k}:{v:+.3f}" for k, v in best)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
