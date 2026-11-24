"""Support-gate variants: raw, no Ti, offsets, and both.

The current gate uses the raw noiseless mock ratios and the catalogue errors.
This script tests whether the high out-of-model fraction is driven by a global
zero-point offset (especially Ti) rather than by individual odd stars.
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
from scipy.stats import spearmanr

from ..calibration import CORE_SUPPORT_ELEMENTS, _support_distances
from ..constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE
from ..io import load_dataset
from ..observation import load_li_records

RATIO_ELEMENTS = tuple(
    element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
)
RATIO_INDEX = {
    element: LI_FEATURE_ELEMENTS.index(element) for element in RATIO_ELEMENTS
}
MOCK_INDEX = {element: ELEMENT_INDEX[element] for element in RATIO_ELEMENTS}
FE_INDEX = ELEMENT_INDEX[REFERENCE]


DATASETS = {
    "v20/Limongi18_clamp": "src/potluck/tests/v20/application/Limongi18_clamp/dataset.npz",
    "v20/WW95_native": "src/potluck/tests/v20/application/WW95_native/dataset.npz",
    "v20/Nomoto13_blend": "src/potluck/tests/v20/application/Nomoto13_blend/dataset.npz",
    "v20/Nomoto13_clamp": "src/potluck/tests/v20/application/Nomoto13_clamp/dataset.npz",
    "v19/Limongi18": "src/potluck/tests/v19/Limongi18/dataset.npz",
    "v19/Nomoto13": "src/potluck/tests/v19/Nomoto13/dataset.npz",
    "v19/WW95": "src/potluck/tests/v19/WW95/dataset.npz",
}


def _mock_ratios(arrays: dict[str, np.ndarray]) -> np.ndarray:
    loga = arrays["log_abundance"].astype(float)
    return np.column_stack(
        [
            loga[:, MOCK_INDEX[element]] - loga[:, FE_INDEX]
            for element in RATIO_ELEMENTS
        ]
    )


def _real_arrays(records: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [RATIO_INDEX[element] for element in RATIO_ELEMENTS]
    return (
        records.xfe[:, columns],
        records.flags[:, columns],
        records.sigma[:, columns],
    )


def _offsets(
    records: Any, mock: np.ndarray, elements: tuple[str, ...], n_boot: int = 200
) -> dict[str, dict[str, float]]:
    """Median real-minus-mock zero point and its bootstrap interval."""
    real, _, _ = _real_arrays(records)
    rng = np.random.default_rng(0)
    result: dict[str, dict[str, float]] = {}
    for element in elements:
        column = RATIO_ELEMENTS.index(element)
        values = real[:, column]
        values = values[np.isfinite(values)]
        mock_values = mock[:, column]
        mock_values = mock_values[np.isfinite(mock_values)]
        if values.size == 0 or mock_values.size == 0:
            continue
        offset = float(np.median(values) - np.median(mock_values))
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(values, size=values.size, replace=True)
            boots.append(float(np.median(sample) - np.median(mock_values)))
        result[element] = {
            "offset": offset,
            "p16": float(np.percentile(boots, 16)),
            "p84": float(np.percentile(boots, 84)),
            "real_median": float(np.median(values)),
            "mock_median": float(np.median(mock_values)),
            "n_real": int(values.size),
            "trend_rho": float(
                spearmanr(
                    records.feh[np.isfinite(real[:, column])],
                    values,
                    nan_policy="omit",
                ).statistic
            )
            if values.size > 2
            else float("nan"),
        }
    return result


def _support(
    arrays: dict[str, np.ndarray],
    records: Any,
    core: tuple[str, ...],
    offsets: dict[str, float] | None = None,
    n_reference: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Recompute the support gate with a chosen element set and offsets."""
    mock = _mock_ratios(arrays)
    real, real_flags, real_sigma = _real_arrays(records)
    if offsets:
        for element, value in offsets.items():
            if element in core:
                real[:, RATIO_ELEMENTS.index(element)] -= value

    columns = [RATIO_ELEMENTS.index(element) for element in core]
    min_core = min(4, len(columns))

    rng = np.random.default_rng(seed)
    total = arrays["obs_features"].shape[0]
    reference = rng.choice(
        total, size=min(n_reference, max(1, total - 1)), replace=False
    )
    keep = np.ones(total, dtype=bool)
    keep[reference] = False

    mock_flags = np.isfinite(mock)
    mock_sigma = np.ones_like(mock)
    threshold = float(
        np.nanpercentile(
            _support_distances(
                arrays["obs_features"][reference],
                arrays["obs_flags"][reference],
                arrays["obs_sigma"][reference],
                mock[keep],
                mock_flags[keep],
                mock_sigma[keep],
                columns,
                min_core=min_core,
            ),
            99,
        )
    )
    distance = _support_distances(
        real,
        real_flags,
        real_sigma,
        mock,
        mock_flags,
        mock_sigma,
        columns,
        min_core=min_core,
    )
    inside = np.isfinite(distance) & (distance <= threshold)
    return {
        "in_model": int(inside.sum()),
        "out_of_model": int((~inside).sum()),
        "fraction": float(inside.mean()),
        "median_distance": float(np.nanmedian(distance)),
        "threshold": threshold,
        "n_finite": int(np.isfinite(distance).sum()),
        "min_core": int(min_core),
        "n_core": int(len(core)),
    }


def run_support_variants(
    outdir: str | Path = "src/potluck/tests/v20/support_variants",
    datasets: dict[str, str] | None = None,
    n_reference: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the four support variants on every dataset."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = datasets or DATASETS
    records = load_li_records()
    base_core = tuple(CORE_SUPPORT_ELEMENTS)
    no_ti_core = tuple(element for element in base_core if element != "Ti")

    results: dict[str, Any] = {
        "datasets": {},
        "variants": ["raw", "no_ti", "offsets", "no_ti_offsets"],
    }
    for label, path in datasets.items():
        if not Path(path).exists():
            continue
        arrays, _ = load_dataset(path)
        mock = _mock_ratios(arrays)
        offsets = _offsets(records, mock, base_core)
        offset_values = {
            element: info["offset"] for element, info in offsets.items()
        }
        rows = {}
        for name, core, use_offsets in (
            ("raw", base_core, None),
            ("no_ti", no_ti_core, None),
            ("offsets", base_core, offset_values),
            ("no_ti_offsets", no_ti_core, offset_values),
        ):
            rows[name] = _support(
                arrays,
                records,
                core,
                offsets=use_offsets,
                n_reference=n_reference,
                seed=seed,
            )
        results["datasets"][label] = {
            "path": str(path),
            "n_mocks": int(arrays["obs_features"].shape[0]),
            "offsets": offsets,
            "variants": rows,
        }

    with (outdir / "support_variants.json").open("w") as handle:
        json.dump(results, handle, indent=2)

    labels = list(results["datasets"])
    variants = results["variants"]
    fraction = np.array(
        [
            [
                results["datasets"][label]["variants"][variant]["fraction"]
                for variant in variants
            ]
            for label in labels
        ]
    )
    fig, axis = plt.subplots(figsize=(10.5, 0.55 * len(labels) + 2.0))
    image = axis.imshow(fraction, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(variants)))
    axis.set_xticklabels(variants, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(variants)):
            axis.text(
                j,
                i,
                f"{fraction[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if fraction[i, j] < 0.5 else "black",
                fontsize=8,
            )
    axis.set_title("Li+ in-model fraction")
    fig.colorbar(image, ax=axis, label="fraction")
    fig.tight_layout()
    fig.savefig(outdir / "support_variants.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    offset_matrix = np.full((len(labels), len(base_core)), np.nan)
    for i, label in enumerate(labels):
        for j, element in enumerate(base_core):
            info = results["datasets"][label]["offsets"].get(element)
            if info:
                offset_matrix[i, j] = info["offset"]
    fig, axis = plt.subplots(figsize=(9.0, 0.5 * len(labels) + 2.0))
    limit = max(float(np.nanmax(np.abs(offset_matrix))), 0.1)
    image = axis.imshow(
        offset_matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit
    )
    axis.set_xticks(np.arange(len(base_core)))
    axis.set_xticklabels(base_core, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(base_core)):
            if np.isfinite(offset_matrix[i, j]):
                axis.text(
                    j,
                    i,
                    f"{offset_matrix[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    axis.set_title("Real minus mock zero point [dex]")
    fig.colorbar(image, ax=axis, label="offset")
    fig.tight_layout()
    fig.savefig(outdir / "support_offsets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="src/potluck/tests/v20/support_variants")
    parser.add_argument("--n-reference", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    results = run_support_variants(
        args.outdir, n_reference=args.n_reference, seed=args.seed
    )
    for label, payload in results["datasets"].items():
        rows = payload["variants"]
        print(
            f"{label:24s} "
            + " ".join(
                f"{name}={rows[name]['fraction']:.3f}"
                for name in results["variants"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
