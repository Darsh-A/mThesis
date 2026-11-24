"""Does upweighting high-fZP mocks change the APOGEE predictions?

Only Limongi18 and Nomoto13 policies. Each policy is trained separately. The
schemes upweight mocks above fZP = 0.1 and/or fZP >= 0.5; APOGEE predictions are
then compared with the unweighted baseline.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from .apogee_fzp import (
    APOGEE_COLUMNS,
    APOGEE_FEATURES,
    _read_apogee,
    _training_data,
)

POLICIES = ("Limongi18_clamp", "Nomoto13_blend", "Nomoto13_clamp")
SCHEMES = {
    "baseline": (1.0, 1.0),
    "gt01_x2": (2.0, 1.0),
    "gt01_x5": (5.0, 1.0),
    "gt05_x5": (1.0, 5.0),
    "gt05_x10": (1.0, 10.0),
    "both": (2.0, 10.0),
}


def _prepare(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = _read_apogee(path)
    frame = frame[np.isfinite(frame["FE_H"])]
    frame = frame.sort_values(["APOGEE_ID", "SNR"], ascending=[True, False])
    frame = frame.drop_duplicates("APOGEE_ID", keep="first")
    quality = (
        (frame["FE_H"] >= -5.0)
        & (frame["FE_H"] <= -1.0)
        & (frame["FE_H_FLAG"] == 0)
        & (frame["SNR"] >= 50.0)
    )
    frame = frame[quality].copy()
    columns = [APOGEE_COLUMNS[element] for element in APOGEE_FEATURES]
    values = frame[columns].to_numpy(dtype=float)
    keep = np.isfinite(values).sum(axis=1) >= 8
    return frame[keep].copy(), values[keep]


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=weight)
    medians = np.nanmedian(x_train, axis=0)
    imputed = np.where(np.isfinite(values), values, medians[None, :])
    return np.clip(model.predict(imputed), 0.0, 1.0)


def run(
    fits_path: str | Path,
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/apogee/reweight",
) -> dict[str, Any]:
    fits_path = Path(fits_path)
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame, values = _prepare(fits_path)

    summaries = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for policy in POLICIES:
        x_train, y_train = _training_data(base, policy)
        finite = np.isfinite(y_train)
        x_train = x_train[finite]
        y_train = y_train[finite]
        for scheme, (w01, w05) in SCHEMES.items():
            weight = np.ones(len(y_train))
            weight[y_train > 0.1] *= w01
            weight[y_train >= 0.5] *= w05
            prediction = _fit_predict(x_train, y_train, weight, values)
            predictions[(policy, scheme)] = prediction
            summaries.append(
                {
                    "policy": policy,
                    "scheme": scheme,
                    "median": float(np.median(prediction)),
                    "max": float(prediction.max()),
                    "ge_0.25": int((prediction >= 0.25).sum()),
                    "ge_0.5": int((prediction >= 0.5).sum()),
                    "mean": float(prediction.mean()),
                }
            )

    table = pd.DataFrame(summaries)
    table.to_csv(outdir / "reweight_summary.csv", index=False)
    with (outdir / "reweight_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)

    schemes = list(SCHEMES)
    x = np.arange(len(POLICIES))
    width = 0.13
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, key, title in zip(
        axes,
        ("ge_0.5", "ge_0.25"),
        ("stars with fZP >= 0.5", "stars with fZP >= 0.25"),
    ):
        for offset, scheme in enumerate(schemes):
            values_scheme = [
                int(
                    table[
                        (table.policy == policy) & (table.scheme == scheme)
                    ][key].iloc[0]
                )
                for policy in POLICIES
            ]
            axis.bar(
                x + (offset - (len(schemes) - 1) / 2) * width,
                values_scheme,
                width,
                label=scheme,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(POLICIES, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "reweight_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Overlay distributions for the Nomoto13 clamp policy.
    fig, axis = plt.subplots(figsize=(8, 5))
    for scheme in ("baseline", "gt01_x5", "gt05_x10", "both"):
        axis.hist(
            predictions[("Nomoto13_clamp", scheme)],
            bins=50,
            range=(0, 1),
            histtype="step",
            label=scheme,
        )
    axis.set_yscale("log")
    axis.set_xlabel("predicted fZP")
    axis.set_ylabel("APOGEE stars")
    axis.set_title("Nomoto13 clamp: training-prior reweighting")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reweight_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"table": table, "predictions": predictions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fits",
        nargs="?",
        default="data/raw/allStarLite-dr17-synspec_rev1.fits",
    )
    parser.add_argument("--base", default="src/potluck/tests/v20")
    parser.add_argument(
        "--outdir",
        default="src/potluck/tests/v20/processed/apogee/reweight",
    )
    args = parser.parse_args(argv)
    result = run(args.fits, args.base, args.outdir)
    print(result["table"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
