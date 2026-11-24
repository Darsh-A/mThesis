"""Whether the Pop II clamp changes the mock chemistry.

The Pop II yield lookup clamps ``Z_star`` to the lowest grid node, so a clamped
mock is drawn from the floor metallicity regardless of its true ``Z_star``.
This diagnostic splits the accepted mocks into clamped and unclamped groups and
reports which ``[X/Fe]`` ratios move between them, with the label rate and mean
importance weight of each group.  If nothing moves, the clamp is harmless.
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

from ..io import load_dataset

FLOOR_DEFAULT = 3.236e-5


def _grid_floor(meta: dict[str, Any]) -> float:
    nodes = meta.get("config", {}).get("yields", {}).get("popii_z_nodes")
    positive = [float(z) for z in (nodes or []) if float(z) > 0.0]
    return min(positive) if positive else FLOOR_DEFAULT


def _clamped(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> np.ndarray:
    if "clamped" in arrays:
        return arrays["clamped"].astype(bool)
    return arrays["z_star"].astype(float) < _grid_floor(meta)


def run_clamp_impact(
    datasets: dict[str, str | Path],
    outdir: str | Path = "src/potluck/tests/v16",
) -> dict[str, Any]:
    """Compare clamped and unclamped mocks element by element."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"datasets": {}}
    deltas: dict[str, np.ndarray] = {}
    names: list[str] = []

    for label, path in datasets.items():
        arrays, meta = load_dataset(path)
        clamped = _clamped(arrays, meta)
        names = list(meta["feature_names"])
        features = arrays["features"].astype(float)
        y = arrays["y_f_05"].astype(bool)
        weight = np.asarray(
            arrays.get("importance_weight", np.ones(features.shape[0])), dtype=float
        )

        delta = np.full(len(names), np.nan)
        elements: dict[str, Any] = {}
        for index, name in enumerate(names):
            values = features[:, index]
            in_clamped = clamped & np.isfinite(values)
            in_free = (~clamped) & np.isfinite(values)
            if in_clamped.sum() and in_free.sum():
                median_clamped = float(np.median(values[in_clamped]))
                median_free = float(np.median(values[in_free]))
                delta[index] = median_clamped - median_free
                elements[name] = {
                    "median_clamped": median_clamped,
                    "median_unclamped": median_free,
                    "delta": float(median_clamped - median_free),
                }
        deltas[label] = delta

        summary["datasets"][label] = {
            "path": str(path),
            "n_clamped": int(clamped.sum()),
            "n_unclamped": int((~clamped).sum()),
            "clamped_fraction": float(np.mean(clamped)),
            "label_rate_clamped": float(np.mean(y[clamped])) if clamped.any() else None,
            "label_rate_unclamped": (
                float(np.mean(y[~clamped])) if (~clamped).any() else None
            ),
            "mean_weight_clamped": float(np.mean(weight[clamped])) if clamped.any() else None,
            "mean_weight_unclamped": (
                float(np.mean(weight[~clamped])) if (~clamped).any() else None
            ),
            "elements": elements,
        }

    order = np.argsort(-np.nan_to_num(np.abs(deltas[list(datasets)[0]]), nan=-1.0))
    position = np.arange(len(names))
    labels = list(datasets)
    width = 0.8 / len(labels)
    fig, axis = plt.subplots(figsize=(9.0, 0.32 * len(names) + 2.0))
    for i, label in enumerate(labels):
        axis.barh(
            position + i * width, deltas[label][order], width, label=label
        )
    axis.set_yticks(position + width * (len(labels) - 1) / 2)
    axis.set_yticklabels([names[j] for j in order], fontsize=8)
    axis.axvline(0.0, color="0.3", lw=0.8)
    axis.set_xlabel(r"median $[X/{\rm Fe}]$: clamped $-$ unclamped (dex)")
    axis.set_title("Element shift between clamped and unclamped mocks")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "clamp_impact.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with (outdir / "clamp_impact.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--outdir", default="src/potluck/tests/v16")
    args = parser.parse_args(argv)

    labels = args.labels or [Path(path).stem for path in args.datasets]
    if len(labels) != len(args.datasets):
        parser.error("--labels must match the number of datasets")
    summary = run_clamp_impact(dict(zip(labels, args.datasets)), args.outdir)
    for label, payload in summary["datasets"].items():
        print(
            f"{label}: clamped {payload['clamped_fraction']:.3f}  "
            f"label rate {payload['label_rate_clamped']:.3f} vs "
            f"{payload['label_rate_unclamped']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
