"""Which element ratios carry the PISN classifier, and what survives low resolution?

Uses the v20 Li+-free stage-1 datasets. For each element subset, trains a
class-balanced random forest within each yield policy and in a leave-one-policy
out setup, at several generic noise levels. This is the low-resolution proxy:
larger sigma, fewer blue/UV lines.
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

from ..constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE
from ..io import load_dataset
from .v20_mega import _balance

ELEMENTS = tuple(
    element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
)
POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)

SUBSETS: dict[str, tuple[str, ...]] = {
    "all 11": ELEMENTS,
    "no Zn": tuple(e for e in ELEMENTS if e != "Zn"),
    "no Co": tuple(e for e in ELEMENTS if e != "Co"),
    "no Zn, Co": tuple(e for e in ELEMENTS if e not in ("Zn", "Co")),
    "no Zn, Co, Mn": tuple(
        e for e in ELEMENTS if e not in ("Zn", "Co", "Mn")
    ),
    "PFS-like": ("C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Ni"),
    "alpha only": ("Mg", "Si", "Ca", "Ti"),
    "alpha + Na": ("Na", "Mg", "Si", "Ca", "Ti"),
    "alpha + Na, Cr, Ni": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni"),
    "core + Co": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni", "Co"),
    "core + Zn": ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Ni", "Zn"),
}


def _ratios(
    data: dict[str, np.ndarray], sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Rebuild observed ratios at a generic element noise."""
    loga = data["log_abundance"].astype(float)
    noisy = loga + rng.normal(0.0, sigma, loga.shape)
    indices = [ELEMENT_INDEX[element] for element in ELEMENTS]
    ref = ELEMENT_INDEX[REFERENCE]
    return noisy[:, indices] - noisy[:, [ref]]


def _columns(subset: tuple[str, ...]) -> list[int]:
    return [ELEMENTS.index(element) for element in subset]


def _fit_predict_fast(
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


def _within(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, n_trees: int
) -> float:
    splits = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=0
    ).split(x, y)
    probability = np.zeros(len(y))
    for train, test in splits:
        probability[test] = _fit_predict_fast(
            x[train], y[train], weight[train], x[test], n_trees
        )
    return float(roc_auc_score(y, probability, sample_weight=weight))


def _loto(
    datasets: dict[str, dict[str, np.ndarray]],
    ratios: dict[str, np.ndarray],
    subset: tuple[str, ...],
    held_out: str,
    n_trees: int,
) -> float:
    columns = _columns(subset)
    x_train_parts, y_train_parts, w_train_parts = [], [], []
    for policy in datasets:
        if policy == held_out:
            continue
        data = datasets[policy]
        y = data["y_f_05"].astype(int)
        x_train_parts.append(ratios[policy][:, columns])
        y_train_parts.append(y)
        w_train_parts.append(_balance(y))
    x_train = np.vstack(x_train_parts)
    y_train = np.concatenate(y_train_parts)
    w_train = np.concatenate(w_train_parts)
    y_test = datasets[held_out]["y_f_05"].astype(int)
    w_test = _balance(y_test)
    x_test = ratios[held_out][:, columns]
    probability = _fit_predict_fast(
        x_train, y_train, w_train, x_test, n_trees
    )
    return float(roc_auc_score(y_test, probability, sample_weight=w_test))


def run_subsets(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    sigmas: tuple[float, ...] = (0.2, 0.3),
    policies: tuple[str, ...] = POLICIES,
    n_trees: int = 50,
    max_samples: int = 5000,
    compute_loto: bool = False,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "element_subsets"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for policy in policies:
        path = base / policy / "dataset.npz"
        if not path.exists():
            continue
        data = load_dataset(path)[0]
        n = len(data["y_f_05"])
        if n > max_samples:
            rng = np.random.default_rng(0)
            index = np.sort(
                rng.choice(n, size=max_samples, replace=False)
            )
            data = {key: value[index] for key, value in data.items()}
        datasets[policy] = data
    if not datasets:
        raise FileNotFoundError("no v20 stage-1 datasets found")

    result: dict[str, Any] = {
        "policies": list(datasets),
        "sigmas": list(sigmas),
        "subsets": {},
    }
    for subset, elements in SUBSETS.items():
        rows: dict[str, Any] = {"elements": list(elements)}
        for sigma in sigmas:
            print(f"  {subset} sigma={sigma}", flush=True)
            rng = np.random.default_rng(0)
            ratios = {
                policy: _ratios(data, sigma, rng)
                for policy, data in datasets.items()
            }
            columns = _columns(elements)
            within = {}
            for policy, data in datasets.items():
                y = data["y_f_05"].astype(int)
                weight = _balance(y)
                within[policy] = _within(
                    ratios[policy][:, columns], y, weight, n_trees
                )
            loto = {}
            if compute_loto and len(datasets) > 1:
                for policy in datasets:
                    loto[policy] = _loto(
                        datasets, ratios, elements, policy, n_trees
                    )
            rows[str(sigma)] = {
                "within_mean": float(np.mean(list(within.values()))),
                "within_min": float(np.min(list(within.values()))),
                "within_max": float(np.max(list(within.values()))),
                "loto_mean": (
                    float(np.mean(list(loto.values()))) if loto else float("nan")
                ),
                "loto_min": (
                    float(np.min(list(loto.values()))) if loto else float("nan")
                ),
                "loto_max": (
                    float(np.max(list(loto.values()))) if loto else float("nan")
                ),
                "within": within,
                "loto": loto,
            }
        result["subsets"][subset] = rows

    with (outdir / "element_subsets.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    names = list(SUBSETS)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(names) + 2.5))
    for axis, sigma in zip(axes, (0.2, 0.3)):
        within = [
            result["subsets"][name][str(sigma)]["within_mean"]
            for name in names
        ]
        low = [
            result["subsets"][name][str(sigma)]["within_min"]
            for name in names
        ]
        high = [
            result["subsets"][name][str(sigma)]["within_max"]
            for name in names
        ]
        axis.barh(positions, within, color="tab:blue", label="within mean")
        axis.errorbar(
            within,
            positions,
            xerr=[
                np.asarray(within) - np.asarray(low),
                np.asarray(high) - np.asarray(within),
            ],
            fmt="none",
            color="black",
            capsize=2,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [f"{name} ({len(SUBSETS[name])})" for name in names], fontsize=7
        )
        axis.set_xlim(0.80, 1.0)
        axis.set_xlabel("balanced AUC, mean and min/max over policies")
        axis.set_title(rf"$\sigma$ = {sigma} dex")
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("PISN classifier with different element sets", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "element_subsets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--loto", action="store_true")
    parser.add_argument("--sigmas", default="0.2,0.3")
    args = parser.parse_args(argv)
    sigmas = tuple(float(s) for s in args.sigmas.split(",") if s.strip())
    result = run_subsets(
        args.base,
        args.outdir,
        sigmas=sigmas,
        n_trees=args.n_trees,
        max_samples=args.max_samples,
        compute_loto=args.loto,
    )
    for name, row in result["subsets"].items():
        values = row["0.2"]
        print(
            f"{name:24s} n={len(row['elements']):2d} "
            f"within={values['within_mean']:.3f} "
            f"LOTO={values['loto_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
