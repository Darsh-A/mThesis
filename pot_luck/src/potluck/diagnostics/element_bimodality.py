"""Per-element distributions for PISN-dominated vs non-dominated mocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

from ..constants import LI_FEATURE_ELEMENTS, REFERENCE
from ..io import load_dataset

ELEMENTS = tuple(
    element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
)


def _plot(
    matrix: np.ndarray,
    names: tuple[str, ...],
    elements: tuple[str, ...],
    y: np.ndarray,
    outdir: Path,
    prefix: str,
    title: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cols = 4
    rows = int(np.ceil(len(elements) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.7 * cols, 3.0 * rows))
    axes = axes.ravel()
    for axis, element in zip(axes, elements):
        name = f"{element}/Fe"
        if name not in names:
            axis.axis("off")
            continue
        column = names.index(name)
        values = matrix[:, column].astype(float)
        good = np.isfinite(values)
        negative = values[good & (y == 0)]
        positive = values[good & (y == 1)]
        if negative.size == 0 or positive.size == 0:
            axis.axis("off")
            continue
        low, high = np.percentile(values[good], [1, 99])
        pad = 0.05 * (high - low)
        bins = np.linspace(low - pad, high + pad, 45)
        axis.hist(
            negative,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            color="tab:gray",
            label="f_ZP < 0.5",
        )
        axis.hist(
            positive,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color="tab:red",
            label="f_ZP >= 0.5",
        )
        auc_raw = float(roc_auc_score(y[good], values[good]))
        if auc_raw >= 0.5:
            auc = auc_raw
            direction = "higher in PISN"
        else:
            auc = 1.0 - auc_raw
            direction = "lower in PISN"
        median_negative = float(np.median(negative))
        median_positive = float(np.median(positive))
        axis.axvline(median_negative, color="tab:gray", ls=":", lw=1.0)
        axis.axvline(median_positive, color="tab:red", ls=":", lw=1.0)
        axis.set_title(f"[{element}/Fe]", fontsize=9)
        axis.set_xlabel("[X/Fe]")
        axis.set_ylabel("probability density [per dex]")
        axis.grid(alpha=0.25)
        summary[element] = {
            "auc": float(auc),
            "auc_raw": auc_raw,
            "direction": direction,
            "median_negative": median_negative,
            "median_positive": median_positive,
            "delta_median": median_positive - median_negative,
            "n_negative": int(negative.size),
            "n_positive": int(positive.size),
        }

    for unused in axes[len(elements):]:
        unused.axis("off")

    handles, labels = [], []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    fig.savefig(outdir / f"{prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def _ranking_plot(
    summary: dict[str, Any],
    outdir: Path,
    prefix: str,
    title: str,
) -> None:
    """Standalone single-element AUC ranking, no classifier involved."""
    order = sorted(summary, key=lambda key: -summary[key]["auc"])
    values = [summary[element]["auc"] for element in order]
    colors = [
        "tab:red" if summary[element]["auc_raw"] < 0.5 else "tab:blue"
        for element in order
    ]
    fig, axis = plt.subplots(figsize=(7.0, 0.42 * len(order) + 1.7))
    positions = np.arange(len(order))
    axis.barh(positions, values, color=colors, alpha=0.85)
    axis.set_yticks(positions)
    axis.set_yticklabels(order, fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0.5, color="black", ls="--", lw=1.0)
    axis.set_xlim(0.4, 1.0)
    axis.set_xlabel("single-element AUC")
    for position, value in zip(positions, values):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=8)
    axis.legend(
        handles=[
            Patch(color="tab:blue", label="higher in PISN"),
            Patch(color="tab:red", label="lower in PISN"),
        ],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    dataset: str | Path = "src/potluck/tests/v20/Limongi18_clamp/dataset.npz",
    outdir: str | Path | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset)
    outdir = Path(outdir) if outdir is not None else dataset.parent / "element_bimodality"
    outdir.mkdir(parents=True, exist_ok=True)
    arrays, meta = load_dataset(dataset)
    y = arrays["y_f_05"].astype(int)

    noiseless_names = tuple(meta["feature_names"])
    noiseless = _plot(
        arrays["features"],
        noiseless_names,
        ELEMENTS + ("Cu",),
        y,
        outdir,
        "element_bimodality_noiseless",
        "Noiseless [X/Fe]: PISN-dominated vs not",
    )
    observed_names = tuple(meta.get("observation_feature_names", noiseless_names))
    observed = _plot(
        arrays["obs_features"],
        observed_names,
        ELEMENTS,
        y,
        outdir,
        "element_bimodality_observed",
        "Noise-added [X/Fe] (sigma=0.2): PISN-dominated vs not",
    )
    _ranking_plot(
        observed,
        outdir,
        "element_bimodality_single_auc",
        "Single-element AUC ranking (noise-added, sigma=0.2)",
    )
    result = {
        "dataset": str(dataset),
        "label": "y_f_05 = 1[f_ZP >= 0.5]",
        "noiseless": noiseless,
        "observed": observed,
    }
    with (outdir / "element_bimodality.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="src/potluck/tests/v20/Limongi18_clamp/dataset.npz",
    )
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)
    result = run(args.dataset, args.outdir)
    for block in ("noiseless", "observed"):
        print(block)
        for element, row in sorted(
            result[block].items(), key=lambda item: -item[1]["auc"]
        ):
            print(
                f"  {element:2s} AUC={row['auc']:.3f} "
                f"({row['direction']}) "
                f"median {row['median_negative']:+.2f} -> {row['median_positive']:+.2f} "
                f"({row['delta_median']:+.2f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
