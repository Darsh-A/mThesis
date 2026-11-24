"""Predict f_ZP for APOGEE DR17 stars and plot the histogram.

The regressor is trained on every APOGEE element that is also modelled by
potluck: C, N, O, Na, Mg, Al, Si, S, K, Ca, Ti, V, Cr, Mn, Co, Ni. P and Cu are
excluded because allStarLite DR17 has no finite values; Ce is not a potluck
feature. Duplicate APOGEE_ID entries are collapsed to the highest-SNR row.
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
import pandas as pd
from astropy.io import fits
from sklearn.ensemble import HistGradientBoostingRegressor

from ..io import load_dataset
from .element_influence import POLICIES
from .element_selection import _ratios

# All potluck feature elements with finite APOGEE allStarLite values.
# P and Cu are all NaN; Ce is not a potluck feature.
APOGEE_FEATURES = (
    "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K",
    "Ca", "Ti", "V", "Cr", "Mn", "Co", "Ni",
)
APOGEE_COLUMNS = {
    "C": "C_FE",
    "N": "N_FE",
    "O": "O_FE",
    "Na": "NA_FE",
    "Mg": "MG_FE",
    "Al": "AL_FE",
    "Si": "SI_FE",
    "S": "S_FE",
    "K": "K_FE",
    "Ca": "CA_FE",
    "Ti": "TI_FE",
    "V": "V_FE",
    "Cr": "CR_FE",
    "Mn": "MN_FE",
    "Co": "CO_FE",
    "Ni": "NI_FE",
}
APOGEE_ERRORS = {element: f"{column}_ERR" for element, column in APOGEE_COLUMNS.items()}


def _training_data(
    base: Path, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    arrays, _ = load_dataset(base / policy / "dataset.npz")
    return _ratios(arrays, APOGEE_FEATURES, 0.2, seed=0), arrays["f_zp"]


def _read_apogee(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as h:
        data = h[1].data
        wanted = [
            "APOGEE_ID",
            "FE_H",
            "SNR",
            "ASPCAPFLAG",
            "STARFLAG",
            "FE_H_FLAG",
        ] + list(APOGEE_COLUMNS.values()) + list(APOGEE_ERRORS.values())
        columns = {}
        for name in wanted:
            if name not in data.names:
                continue
            values = np.array(data[name])
            if values.dtype.kind == "f":
                values = values.astype(np.float64)
            elif values.dtype.kind in "iu":
                values = values.astype(np.int64)
            elif values.dtype.kind == "S":
                values = values.astype("U")
            columns[name] = values
        frame = pd.DataFrame(columns)
    return frame


def _yield_plot(
    selected: pd.DataFrame,
    prediction: np.ndarray,
    outdir: Path,
    policy: str,
    threshold: float = 0.5,
    filename: str = "apogee_fzp_gt05_yields.png",
) -> None:
    stars = selected[prediction >= threshold]
    if stars.empty:
        order = np.argsort(-prediction)[:2]
        stars = selected.iloc[order]
        note = f"no star with fZP >= {threshold}; top 2 candidates"
    else:
        note = f"fZP >= {threshold}"
    stars = stars.iloc[:6]
    ncols = 2
    nrows = int(np.ceil(len(stars) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.4 * nrows), squeeze=False
    )
    plot_elements = tuple(
        element for element in APOGEE_FEATURES if element != "N"
    )
    plot_indices = [APOGEE_FEATURES.index(element) for element in plot_elements]
    x = np.arange(len(plot_elements))
    for axis, (_, row) in zip(axes.ravel(), stars.iterrows()):
        observed = np.asarray(
            [row[APOGEE_COLUMNS[element]] for element in plot_elements],
            dtype=float,
        )
        error = np.asarray(
            [row[APOGEE_ERRORS[element]] for element in plot_elements],
            dtype=float,
        )
        error = np.where(np.isfinite(error), error, 0.0)
        axis.errorbar(
            x,
            observed,
            yerr=error,
            fmt="o",
            color="black",
            capsize=2,
            label="APOGEE",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(plot_elements, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("[X/Fe]")
        axis.set_title(
            f"{row['APOGEE_ID']}  [Fe/H]={row['FE_H']:.2f}  fZP={row['fzp']:.2f}",
            fontsize=8,
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    for axis in axes.ravel()[len(stars):]:
        axis.axis("off")
    fig.suptitle(
        f"APOGEE observed abundance pattern, {policy}: {note}  (N omitted: large error)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    fits_path: str | Path,
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/apogee",
) -> dict[str, Any]:
    fits_path = Path(fits_path)
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = _read_apogee(fits_path)
    total_rows = len(frame)
    unique_ids = frame["APOGEE_ID"].nunique()
    duplicate_rows = total_rows - unique_ids

    # One row per star: highest SNR among the entries with a finite FE_H.
    frame = frame[np.isfinite(frame["FE_H"])]
    frame = frame.sort_values(["APOGEE_ID", "SNR"], ascending=[True, False])
    frame = frame.drop_duplicates("APOGEE_ID", keep="first")
    after_dedup = len(frame)

    quality = (
        (frame["FE_H"] >= -5.0)
        & (frame["FE_H"] <= -1.0)
        & (frame["FE_H_FLAG"] == 0)
        & (frame["SNR"] >= 50.0)
    )
    selected = frame[quality].copy()
    after_quality = len(selected)

    feature_columns = [APOGEE_COLUMNS[element] for element in APOGEE_FEATURES]
    values = selected[feature_columns].to_numpy(dtype=float)
    available = np.isfinite(values).sum(axis=1)
    keep = available >= 8
    selected = selected[keep].copy()
    values = values[keep]
    after_features = len(selected)

    summaries = {}
    for policy in POLICIES:
        x_train, y_train = _training_data(base, policy)
        regressor = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        )
        regressor.fit(x_train, y_train)
        medians = np.nanmedian(x_train, axis=0)
        imputed = np.where(np.isfinite(values), values, medians[None, :])
        prediction = np.clip(regressor.predict(imputed), 0.0, 1.0)

        selected_policy = selected.copy()
        selected_policy["fzp"] = prediction
        selected_policy[
            ["APOGEE_ID", "FE_H", "SNR", "fzp"]
        ].to_csv(outdir / f"apogee_fzp_{policy}.csv", index=False)

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(prediction, bins=50, range=(0.0, 1.0), color="tab:blue")
        axis.set_xlabel(r"predicted $f_Z^P$")
        axis.set_ylabel("APOGEE stars")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        median = float(np.median(prediction))
        high = int((prediction >= 0.5).sum())
        axis.axvline(median, color="tab:orange", ls="--", lw=1.2)
        axis.text(
            0.03,
            0.95,
            f"N = {after_features}\nmedian = {median:.3f}\nfZP >= 0.5: {high}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_title(policy)
        fig.tight_layout()
        fig.savefig(
            outdir / f"apogee_fzp_hist_{policy}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        _yield_plot(
            selected_policy,
            prediction,
            outdir,
            policy,
            filename=f"apogee_fzp_gt05_yields_{policy}.png",
        )

        summary = {
            "policy": policy,
            "total_rows": int(total_rows),
            "unique_apogee_ids": int(unique_ids),
            "duplicate_rows": int(duplicate_rows),
            "after_dedup": int(after_dedup),
            "after_quality_and_metallicity": int(after_quality),
            "after_at_least_8_features": int(after_features),
            "median_fzp": median,
            "max_fzp": float(prediction.max()),
            "fzp_ge_0.5": high,
            "fzp_ge_0.25": int((prediction >= 0.25).sum()),
            "feature_availability": {
                element: float(np.isfinite(values[:, j]).mean())
                for j, element in enumerate(APOGEE_FEATURES)
            },
        }
        with (outdir / f"apogee_fzp_summary_{policy}.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        summaries[policy] = summary

    pd.DataFrame(summaries.values()).to_csv(
        outdir / "apogee_fzp_summary_all.csv", index=False
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fits",
        nargs="?",
        default="data/raw/allStarLite-dr17-synspec_rev1.fits",
    )
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument(
        "--outdir", default="src/potluck/tests/v20/processed/apogee"
    )
    args = parser.parse_args(argv)
    summary = run(args.fits, args.base, args.outdir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
