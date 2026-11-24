"""What does adding element X buy you, given a base set?

Drop-column and add-one-element ablations are misleading when features are
correlated. This script measures the conditional value: fix a base set, add one
element, and report the AUC change. The key question is whether Cu adds anything
once Zn is present, and vice versa.
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

from ..constants import LI_FEATURE_ELEMENTS, REFERENCE
from ..io import load_dataset
from .v20_mega import _balance

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)
LI_ELEMENTS = tuple(
    element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
)
NOISELESS_ELEMENTS = LI_ELEMENTS + ("Cu",)

NOISELESS_BASES: dict[str, tuple[str, ...]] = {
    "none": (),
    "Zn": ("Zn",),
    "Cu": ("Cu",),
    "Zn + Co": ("Zn", "Co"),
    "Zn + Na": ("Zn", "Na"),
    "Zn + Cu": ("Zn", "Cu"),
    "core 7": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni"),
    "core 7 + Zn": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni", "Zn"),
    "full": NOISELESS_ELEMENTS,
}
OBSERVED_BASES: dict[str, tuple[str, ...]] = {
    "none": (),
    "Zn": ("Zn",),
    "Zn + Co": ("Zn", "Co"),
    "Zn + Na": ("Zn", "Na"),
    "core 7": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni"),
    "core 7 + Zn": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni", "Zn"),
    "full": LI_ELEMENTS,
}


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_test: np.ndarray,
    n_trees: int,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    return model.predict_proba(x_test)[:, 1]


def _auc(
    ratios: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    columns: list[int],
    n_trees: int,
) -> float:
    values = []
    for policy, x_all in ratios.items():
        y = labels[policy]
        weight = _balance(y)
        x = x_all[:, columns]
        splits = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=0
        ).split(x, y)
        probability = np.zeros(len(y))
        for train, test in splits:
            probability[test] = _fit_predict(
                x[train], y[train], weight[train], x[test], n_trees
            )
        values.append(
            float(roc_auc_score(y, probability, sample_weight=weight))
        )
    return float(np.mean(values))


def _block(
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    names: tuple[str, ...],
    elements: tuple[str, ...],
    bases: dict[str, tuple[str, ...]],
    n_trees: int,
) -> dict[str, Any]:
    cache: dict[frozenset[str], float] = {}

    def score(subset: tuple[str, ...]) -> float:
        if not subset:
            return 0.5
        key = frozenset(subset)
        if key not in cache:
            columns = [names.index(f"{element}/Fe") for element in subset]
            cache[key] = _auc(matrices, labels, columns, n_trees)
        return cache[key]

    result: dict[str, Any] = {"elements": list(elements), "bases": {}}
    for base_name, base in bases.items():
        base_auc = score(base)
        deltas: dict[str, float] = {}
        for element in elements:
            if element in base:
                deltas[element] = None
                continue
            deltas[element] = score(base + (element,)) - base_auc
        result["bases"][base_name] = {
            "elements": list(base),
            "auc": float(base_auc),
            "delta": deltas,
        }
    return result


def _plot(
    result: dict[str, Any],
    elements: tuple[str, ...],
    path: Path,
    title: str,
    unavailable: tuple[str, ...] = (),
) -> None:
    bases = list(result["bases"])
    matrix = np.full((len(bases), len(elements)), np.nan)
    for i, base in enumerate(bases):
        for j, element in enumerate(elements):
            value = result["bases"][base]["delta"].get(element)
            if value is not None:
                matrix[i, j] = value
    limit = max(float(np.nanmax(np.abs(matrix))), 0.02)
    fig, axis = plt.subplots(figsize=(11, 0.55 * len(bases) + 2.0))
    image = axis.imshow(
        matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit
    )
    axis.set_xticks(np.arange(len(elements)))
    axis.set_xticklabels(
        [f"{element}*" if element in unavailable else element for element in elements],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(bases)))
    axis.set_yticklabels(bases, fontsize=8)
    for i in range(len(bases)):
        for j in range(len(elements)):
            if np.isfinite(matrix[i, j]):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(matrix[i, j]) < 0.06 else "white",
                )
    axis.set_xlabel("added element")
    axis.set_ylabel("base set")
    if unavailable:
        title = title + "  (* not observed in Li+)"
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Delta AUC (added minus base)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    max_samples: int = 5000,
    n_trees: int = 50,
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "element_conditional"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for policy in policies:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data = load_dataset(path)[0]
        n = len(data["y_f_05"])
        if n > max_samples:
            index = np.sort(
                np.random.default_rng(0).choice(n, max_samples, replace=False)
            )
            data = {key: value[index] for key, value in data.items()}
        datasets[policy] = data
    if not datasets:
        raise FileNotFoundError("no v20 datasets")

    labels = {
        policy: data["y_f_05"].astype(int)
        for policy, data in datasets.items()
    }
    noiseless_names = tuple(
        load_dataset(base / policies[0] / "dataset.npz")[1]["feature_names"]
    )
    observed_names = tuple(
        load_dataset(base / policies[0] / "dataset.npz")[1].get(
            "observation_feature_names", noiseless_names
        )
    )

    noiseless = _block(
        {policy: data["features"] for policy, data in datasets.items()},
        labels,
        noiseless_names,
        NOISELESS_ELEMENTS,
        NOISELESS_BASES,
        n_trees,
    )
    observed = _block(
        {policy: data["obs_features"] for policy, data in datasets.items()},
        labels,
        observed_names,
        LI_ELEMENTS,
        OBSERVED_BASES,
        n_trees,
    )
    # Cu is not in the Li+ catalogue. Keep the column visible so the absence is
    # explicit rather than a silent omission.
    observed["elements"] = list(LI_ELEMENTS) + ["Cu"]
    for row in observed["bases"].values():
        row["delta"]["Cu"] = None
    result = {
        "noiseless": noiseless,
        "observed": observed,
        "max_samples": max_samples,
        "n_trees": n_trees,
    }
    with (outdir / "element_conditional.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    _plot(
        noiseless,
        NOISELESS_ELEMENTS,
        outdir / "element_conditional_noiseless.png",
        "Conditional value of adding one element (noiseless)",
    )
    _plot(
        observed,
        LI_ELEMENTS + ("Cu",),
        outdir / "element_conditional_observed.png",
        "Conditional value of adding one element (observed, sigma=0.2)",
        unavailable=("Cu",),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-trees", type=int, default=50)
    args = parser.parse_args(argv)
    result = run(
        args.base,
        args.outdir,
        max_samples=args.max_samples,
        n_trees=args.n_trees,
    )
    for block in ("noiseless", "observed"):
        print(block)
        for base_name, row in result[block]["bases"].items():
            best = sorted(
                (
                    (key, value)
                    for key, value in row["delta"].items()
                    if value is not None
                ),
                key=lambda kv: -kv[1],
            )[:4]
            print(
                f"  {base_name:14s} base={row['auc']:.3f} "
                + " ".join(f"{k}:{v:+.3f}" for k, v in best)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
