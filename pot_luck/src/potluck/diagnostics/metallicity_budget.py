"""Mock metallicity budget against the Li+ metallicity from its own abundances.

The mock ``Z_final = M_Ztot / M_dil`` is a real mass fraction.  The Li+ side is
put on the same axis from its **own** abundances, not from a scaled-solar
[Fe/H]: the alpha elements (Mg, Si, Ca) give an [alpha/Fe], and the total metal
mass fraction follows

    Z = Z_sun * 10**[Fe/H] * (f_alpha * 10**[alpha/Fe] + (1 - f_alpha))

with ``f_alpha = 0.638``.  The plain scaled-solar Z is kept as a faint
reference.

Mock histograms are weighted by ``importance_weight``.  ``Z_final`` is shown for
context, but the clamp acts on the pre-enrichment ``Z_star``; the clamped
fraction uses that, and the match is also scored separately for the clamped and
unclamped groups (weighted KS distance and a Q-Q panel).
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

from ..constants import LI_FEATURE_ELEMENTS, Z_SUN_COMPUTED
from ..io import load_dataset
from ..observation import load_li_records

FLOOR_DEFAULT = 3.236e-5
ALPHA_ELEMENTS = ("Mg", "Si", "Ca")
ALPHA_MASS_FRACTION = 0.638


def _grid_floor(meta: dict[str, Any]) -> float:
    nodes = meta.get("config", {}).get("yields", {}).get("popii_z_nodes")
    positive = [float(z) for z in (nodes or []) if float(z) > 0.0]
    return min(positive) if positive else FLOOR_DEFAULT


def _log10_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log10(values[np.isfinite(values) & (values > 0.0)])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    return {
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
    }


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probs, cumulative, values)


def _weighted_ks(x: np.ndarray, weights: np.ndarray, y: np.ndarray) -> float:
    """Max ECDF distance between weighted ``x`` and unweighted ``y``."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cx = np.cumsum(weights)
    cx /= cx[-1]
    y = np.sort(np.asarray(y, dtype=float))
    fy = np.searchsorted(y, x, side="right") / y.size
    return float(np.max(np.abs(cx - fy)))


def li_metallicities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Li+ ``(Z, Z_scaled_solar, [alpha/Fe])`` from the catalogue."""
    records = load_li_records()
    columns = [LI_FEATURE_ELEMENTS.index(element) for element in ALPHA_ELEMENTS]
    alpha = np.nanmean(records.xfe[:, columns], axis=1)
    feh = records.feh.astype(float)
    with np.errstate(invalid="ignore"):
        z_alpha = (
            Z_SUN_COMPUTED
            * 10.0 ** feh
            * (ALPHA_MASS_FRACTION * 10.0 ** alpha + (1.0 - ALPHA_MASS_FRACTION))
        )
        z_scaled = Z_SUN_COMPUTED * 10.0 ** feh
    return z_alpha, z_scaled, alpha


def _clamped(arrays: dict[str, np.ndarray], meta: dict[str, Any]):
    floor = _grid_floor(meta)
    if "clamped" in arrays:
        return arrays["clamped"].astype(bool), floor
    return arrays["z_star"].astype(float) < floor, floor


def run_metallicity_budget(
    datasets: dict[str, str | Path],
    outdir: str | Path = "src/potluck/tests/v16",
) -> dict[str, Any]:
    """Compare mock Z_final with the Li+ metallicity for several datasets."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_alpha, z_scaled, alpha = li_metallicities()
    li_log = _log10_positive(z_alpha)
    li_scaled_log = _log10_positive(z_scaled)
    probs = np.linspace(0.01, 0.99, 99)
    li_quantiles = np.quantile(li_log, probs)

    summary: dict[str, Any] = {
        "z_sun": float(Z_SUN_COMPUTED),
        "alpha_mass_fraction": ALPHA_MASS_FRACTION,
        "li_n": int(li_log.size),
        "li_log_z": _quantiles(li_log),
        "li_scaled_solar_log_z": _quantiles(li_scaled_log),
        "li_alpha_fe": _quantiles(alpha[np.isfinite(alpha)]),
        "datasets": {},
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(datasets), 1)))
    labels = list(datasets)
    for color, label in zip(colors, labels):
        arrays, meta = load_dataset(datasets[label])
        clamped, floor = _clamped(arrays, meta)
        z_final = arrays["z_final"].astype(float)
        z_star = arrays["z_star"].astype(float)
        weight = np.asarray(
            arrays.get("importance_weight", np.ones(z_final.size)), dtype=float
        )
        keep = np.isfinite(z_final) & (z_final > 0.0)
        log_final = np.log10(z_final[keep])
        mock_weight = weight[keep]
        mock_clamped = clamped[keep]

        summary["datasets"][label] = {
            "path": str(datasets[label]),
            "n_mocks": int(z_final.size),
            "grid_floor": float(floor),
            "clamped_fraction": float(np.mean(clamped)),
            "z_final_log10": _quantiles(_log10_positive(z_final)),
            "z_star_log10": _quantiles(_log10_positive(z_star)),
            "weighted_ks_li": _weighted_ks(log_final, mock_weight, li_log),
            "weighted_ks_li_clamped": _weighted_ks(
                log_final[mock_clamped], mock_weight[mock_clamped], li_log
            ),
            "weighted_ks_li_unclamped": _weighted_ks(
                log_final[~mock_clamped], mock_weight[~mock_clamped], li_log
            ),
        }
        axes[0].hist(
            log_final, bins=60, density=True, weights=mock_weight,
            histtype="step", lw=1.6, color=color, label=f"mocks, {label}",
        )
        mock_quantiles = _weighted_quantiles(log_final, mock_weight, probs)
        axes[1].plot(li_quantiles, mock_quantiles, color=color, lw=1.5, label=label)

    axes[0].hist(
        li_scaled_log, bins=60, density=True, histtype="step", lw=1.2,
        color="0.6", ls=":", label="Li+ scaled-solar Z",
    )
    axes[0].hist(
        li_log, bins=60, density=True, histtype="stepfilled",
        color="black", alpha=0.18, label="Li+ Z (from abundances)",
    )
    axes[0].set_xlabel(r"$\log_{10} Z$")
    axes[0].set_ylabel("density (mocks weighted)")
    axes[0].set_title("Mock $Z_{\\rm final}$ versus Li+ $Z$")
    axes[0].legend(fontsize=7)

    span = [float(li_quantiles.min()), float(li_quantiles.max())]
    axes[1].plot(span, span, color="0.4", ls="--", lw=1.0, label="1:1")
    axes[1].set_xlabel(r"Li+ $\log_{10} Z$ quantile")
    axes[1].set_ylabel(r"mock $\log_{10} Z_{\rm final}$ quantile")
    axes[1].set_title("Q-Q against Li+ Z (weighted mocks)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    values = [summary["datasets"][label]["clamped_fraction"] for label in labels]
    kss = [summary["datasets"][label]["weighted_ks_li"] for label in labels]
    axes[2].bar(range(len(labels)), values, color=colors, alpha=0.85, label="clamped fraction")
    axes[2].plot(range(len(labels)), kss, "o-", color="black", lw=1.4, label="weighted KS vs Li+")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=20, fontsize=8)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("fraction / KS distance")
    axes[2].set_title("Clamping and Li+ mismatch")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "z_final_vs_li.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with (outdir / "metallicity_budget.json").open("w") as handle:
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
    summary = run_metallicity_budget(dict(zip(labels, args.datasets)), args.outdir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
