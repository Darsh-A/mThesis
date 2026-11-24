"""Histograms of the Li+ posterior and expected f_ZP per v20 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICIES = (
    "Limongi18_clamp",
    "WW95_native",
    "Nomoto13_blend",
    "Nomoto13_clamp",
)
PREVALENCE = 2.5e-3


def _observed_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if text in ("", "[]", "nan"):
        return 0
    return len([part for part in text.strip("[]").replace("'", "").split(",") if part.strip()])


def _detection_count(table: pd.DataFrame) -> pd.Series:
    available = table["available"].map(_observed_count)
    limits = table["limits"].map(_observed_count)
    return available - limits


def _load(
    base: Path, min_elements: int = 0, min_detections: int = 0
) -> dict[str, pd.DataFrame]:
    tables = {}
    for policy in POLICIES:
        path = base / "application" / policy / "li_validation" / "candidates.csv"
        if path.exists():
            table = pd.read_csv(path)
            if min_elements > 0:
                observed = table["available"].map(_observed_count)
                table = table[observed >= min_elements]
            if min_detections > 0:
                table = table[_detection_count(table) >= min_detections]
            tables[policy] = table.reset_index(drop=True)
    return tables


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("p16", "p50", "p84", "p95", "max")}
    return {
        "p16": float(np.percentile(values, 16)),
        "p50": float(np.percentile(values, 50)),
        "p84": float(np.percentile(values, 84)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _grid(
    tables: dict[str, pd.DataFrame],
    column: str,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    vline: float | None,
    path: Path,
    log_y: bool = False,
) -> dict[str, Any]:
    n = len(tables)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows), sharey=log_y)
    axes = np.atleast_1d(axes).ravel()
    all_values = np.concatenate(
        [table[column].to_numpy(dtype=float) for table in tables.values()]
    )
    summary: dict[str, Any] = {}
    for axis, (policy, table) in zip(axes, tables.items()):
        values = table[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        axis.hist(values, bins=bins, color="tab:blue", alpha=0.85)
        if vline is not None:
            axis.axvline(vline, color="tab:red", ls="--", lw=1.1)
        if log_y:
            axis.set_yscale("log")
        q = _quantiles(values)
        summary[policy] = q
        axis.text(
            0.03,
            0.95,
            f"median {q['p50']:.4f}\np95 {q['p95']:.4f}\nmax {q['max']:.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_title(policy, fontsize=9)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Li+ stars")
        axis.grid(alpha=0.25)
    for axis in axes[n:]:
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"bin_low": float(bins[0]), "bin_high": float(bins[-1]), "per_policy": summary}


def _overlay(
    tables: dict[str, pd.DataFrame],
    column: str,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    vline: float | None,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for policy, table in tables.items():
        values = table[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.4,
            label=policy,
        )
    if vline is not None:
        axis.axvline(vline, color="tab:red", ls="--", lw=1.1)
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("density")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_histograms(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path | None = None,
    min_elements: int = 0,
    min_detections: int = 0,
    tag: str = "",
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir) if outdir is not None else base / "li_histograms"
    outdir.mkdir(parents=True, exist_ok=True)
    tables = _load(
        base, min_elements=min_elements, min_detections=min_detections
    )
    if not tables:
        raise FileNotFoundError("no v20 application candidates.csv found")

    post_values = np.concatenate(
        [t["p_posterior"].to_numpy(float) for t in tables.values()]
    )
    post_high = max(0.05, float(np.nanpercentile(post_values, 99)) * 1.2)
    post_bins = np.linspace(0.0, post_high, 45)
    post_summary = _grid(
        tables,
        "p_posterior",
        post_bins,
        "P(f_Z >= 0.5 | x)",
        "Li+ posterior for the PISN-dominated label",
        PREVALENCE,
        outdir / f"li_posterior_hist{tag}.png",
        log_y=True,
    )
    _overlay(
        tables,
        "p_posterior",
        post_bins,
        "P(f_Z >= 0.5 | x)",
        "Li+ posterior by policy",
        PREVALENCE,
        outdir / f"li_posterior_overlay{tag}.png",
    )

    fz_bins = np.linspace(0.0, 1.0, 41)
    fz_summary = _grid(
        tables,
        "expected_fz",
        fz_bins,
        r"expected $f_Z^P$",
        "Li+ expected PISN metal fraction",
        0.5,
        outdir / f"li_expected_fz_hist{tag}.png",
    )
    _overlay(
        tables,
        "expected_fz",
        fz_bins,
        r"expected $f_Z^P$",
        "Li+ expected f_Z by policy",
        0.5,
        outdir / f"li_expected_fz_overlay{tag}.png",
    )

    result = {
        "p_posterior": post_summary,
        "expected_fz": fz_summary,
        "prevalence": PREVALENCE,
        "min_elements": min_elements,
        "min_detections": min_detections,
        "n_per_policy": {policy: int(len(table)) for policy, table in tables.items()},
    }
    with (outdir / f"li_histograms{tag}.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--min-elements",
        type=int,
        default=0,
        help="keep only stars with at least this many observed elements "
        "(counted from the candidates.csv 'available' column)",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=0,
        help="keep only stars with at least this many detected elements "
        "(upper limits excluded)",
    )
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    args = parser.parse_args(argv)
    result = run_histograms(
        args.base,
        args.outdir,
        min_elements=args.min_elements,
        min_detections=args.min_detections,
        tag=args.tag,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
