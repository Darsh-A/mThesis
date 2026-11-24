"""One APOGEE star: observed pattern and f_ZP under every Pop II yield table.

The bulk script (``apogee_fzp``) scores the whole catalogue and only draws the
highest-f_ZP stars. This one takes a single ``APOGEE_ID`` and reports it against
each policy, so an individual candidate can be inspected without regenerating
everything.

Outputs (``--outdir``):
    apogee_star_<ID>_fzp.csv
    apogee_star_<ID>_yields.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .apogee_fzp import (
    APOGEE_COLUMNS,
    APOGEE_ERRORS,
    APOGEE_FEATURES,
    _training_data,
    _read_apogee,
)
from .element_influence import POLICIES

# Elements plotted, in a stable order. N is kept out of the figure because its
# APOGEE errors are large; it is still a model feature.
PLOT_ELEMENTS: tuple[str, ...] = tuple(
    element for element in APOGEE_FEATURES if element != "N"
)


def find_star(frame: pd.DataFrame, apogee_id: str) -> pd.Series:
    """Return the highest-SNR row for ``apogee_id``."""
    matches = frame[frame["APOGEE_ID"] == apogee_id]
    if matches.empty:
        raise KeyError(f"{apogee_id} not found in the catalogue")
    return matches.sort_values("SNR", ascending=False).iloc[0]


def predict(
    row: pd.Series, base: Path, policies: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Train one regressor per policy and score this star."""
    values = np.array(
        [row[APOGEE_COLUMNS[element]] for element in APOGEE_FEATURES], dtype=float
    )
    available = int(np.isfinite(values).sum())

    results: dict[str, dict[str, Any]] = {}
    for policy in policies:
        x_train, y_train = _training_data(base, policy)
        regressor = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        )
        regressor.fit(x_train, y_train)
        medians = np.nanmedian(x_train, axis=0)
        imputed = np.where(np.isfinite(values), values, medians)[None, :]
        prediction = float(np.clip(regressor.predict(imputed), 0.0, 1.0)[0])

        # Spread of the mock distribution at the same prediction, for context.
        train_prediction = np.clip(regressor.predict(x_train), 0.0, 1.0)
        results[policy] = {
            "fzp": prediction,
            "train_median": float(np.median(train_prediction)),
            "train_p99": float(np.percentile(train_prediction, 99)),
            "n_features": available,
        }
    return results


def plot_star(
    row: pd.Series,
    results: dict[str, dict[str, Any]],
    apogee_id: str,
    path: Path,
) -> None:
    n = len(results)
    fig, axes = plt.subplots(
        n, 1, figsize=(11.0, 2.7 * n), sharex=True, sharey=True, squeeze=False
    )
    x = np.arange(len(PLOT_ELEMENTS))

    for axis, (policy, info) in zip(axes.ravel(), results.items()):
        observed = np.array(
            [row[APOGEE_COLUMNS[element]] for element in PLOT_ELEMENTS], dtype=float
        )
        error = np.array(
            [row[APOGEE_ERRORS[element]] for element in PLOT_ELEMENTS], dtype=float
        )
        error = np.where(np.isfinite(error), error, 0.0)

        axis.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        axis.errorbar(
            x,
            observed,
            yerr=error,
            fmt="o",
            ms=5,
            color="black",
            ecolor="0.35",
            capsize=2,
            zorder=3,
            label="APOGEE",
        )
        missing = ~np.isfinite(observed)
        for position in np.flatnonzero(missing):
            axis.axvspan(position - 0.4, position + 0.4, color="tab:red",
                         alpha=0.10, zorder=0)
            axis.text(position, 0.02, "n/a", ha="center", va="bottom",
                      fontsize=7, color="tab:red", transform=axis.get_xaxis_transform())

        axis.set_ylabel("[X/Fe]")
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(
            f"{policy}   fZP = {info['fzp']:.3f}"
            f"   (training median {info['train_median']:.2f},"
            f" p99 {info['train_p99']:.2f})",
            fontsize=9,
            loc="left",
        )

    axes.ravel()[-1].set_xticks(x)
    axes.ravel()[-1].set_xticklabels(PLOT_ELEMENTS, rotation=45, ha="right", fontsize=8)
    fig.suptitle(
        f"{apogee_id}   [Fe/H] = {row['FE_H']:.2f}   SNR = {row['SNR']:.0f}   "
        f"{int(results[next(iter(results))]['n_features'])}/{len(APOGEE_FEATURES)} "
        "features measured",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    apogee_id: str,
    fits_path: str | Path = "data/raw/allStarLite-dr17-synspec_rev1.fits",
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/apogee",
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = apogee_id.replace("+", "p").replace("-", "m")

    frame = _read_apogee(Path(fits_path))
    row = find_star(frame, apogee_id)
    results = predict(row, base, policies)

    table = pd.DataFrame(
        [
            {
                "APOGEE_ID": apogee_id,
                "policy": policy,
                "FE_H": float(row["FE_H"]),
                "SNR": float(row["SNR"]),
                "n_features": info["n_features"],
                "fzp": info["fzp"],
                "train_median": info["train_median"],
                "train_p99": info["train_p99"],
            }
            for policy, info in results.items()
        ]
    )
    table.to_csv(outdir / f"apogee_star_{tag}_fzp.csv", index=False)
    plot_star(row, results, apogee_id, outdir / f"apogee_star_{tag}_yields.png")
    return {"star": apogee_id, "row": row, "results": results, "table": table}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apogee_id")
    parser.add_argument("fits", nargs="?",
                        default="data/raw/allStarLite-dr17-synspec_rev1.fits")
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument("--outdir", default="src/potluck/tests/v20/processed/apogee")
    args = parser.parse_args(argv)

    out = run(args.apogee_id, args.fits, args.base, args.outdir)
    row = out["row"]
    print(f"{args.apogee_id}  [Fe/H]={row['FE_H']:.3f}  SNR={row['SNR']:.1f}")
    print(f"{'policy':18s} {'fZP':>7s} {'train p50':>10s} {'train p99':>10s}")
    for policy, info in out["results"].items():
        print(
            f"{policy:18s} {info['fzp']:7.4f} {info['train_median']:10.3f} "
            f"{info['train_p99']:10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
