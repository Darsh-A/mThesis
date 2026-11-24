"""Is the low APOGEE fZP caused by too few high-fZP mocks?

Checks, for Limongi18 and Nomoto13 only:
1. absolute mock counts in fZP bins,
2. held-out recovery: train on 4/5 of the mocks, predict the held-out fifth,
   and report the predicted fZP in each true-fZP bin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from ..io import load_dataset
from .apogee_fzp import APOGEE_COLUMNS, APOGEE_FEATURES, _training_data

POLICIES = ("Limongi18_clamp", "Nomoto13_blend", "Nomoto13_clamp")
BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01)


def _counts(base: Path, policy: str) -> dict[str, Any]:
    arrays, _ = load_dataset(base / policy / "dataset.npz")
    fzp = arrays["f_zp"].astype(float)
    n_pisn = arrays["n_pisn"]
    return {
        "policy": policy,
        "n": int(len(fzp)),
        "n_pisn_positive": int((n_pisn > 0).sum()),
        "gt_0.1": int((fzp > 0.1).sum()),
        "ge_0.5": int((fzp >= 0.5).sum()),
        "bin_counts": {
            f"{BINS[i]:.2f}-{BINS[i+1]:.2f}": int(
                ((fzp >= BINS[i]) & (fzp < BINS[i + 1])).sum()
            )
            for i in range(len(BINS) - 1)
        },
    }


def _heldout(base: Path, policy: str) -> pd.DataFrame:
    x, y = _training_data(base, policy)
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    prediction = np.zeros(len(y))
    for train, test in KFold(n_splits=5, shuffle=True, random_state=0).split(x):
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        )
        model.fit(x[train], y[train])
        medians = np.nanmedian(x[train], axis=0)
        imputed = np.where(np.isfinite(x[test]), x[test], medians[None, :])
        prediction[test] = model.predict(imputed)

    rows = []
    for i in range(len(BINS) - 1):
        mask = (y >= BINS[i]) & (y < BINS[i + 1])
        if not mask.any():
            continue
        rows.append(
            {
                "policy": policy,
                "true_bin": f"{BINS[i]:.2f}-{BINS[i+1]:.2f}",
                "n": int(mask.sum()),
                "median_pred": float(np.median(prediction[mask])),
                "mean_pred": float(prediction[mask].mean()),
                "frac_pred_ge_0.5": float((prediction[mask] >= 0.5).mean()),
                "frac_pred_lt_0.1": float((prediction[mask] < 0.1).mean()),
            }
        )
    return pd.DataFrame(rows)


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/apogee/coverage_check",
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    counts = [_counts(base, policy) for policy in POLICIES]
    recovery = pd.concat([_heldout(base, policy) for policy in POLICIES])

    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(outdir / "mock_counts.csv", index=False)
    recovery.to_csv(outdir / "heldout_recovery.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy in POLICIES:
        subset = recovery[recovery.policy == policy]
        axes[0].plot(
            range(len(subset)),
            subset["median_pred"],
            marker="o",
            label=policy,
        )
        axes[1].plot(
            range(len(subset)),
            subset["frac_pred_ge_0.5"],
            marker="o",
            label=policy,
        )
    labels = recovery[recovery.policy == POLICIES[0]]["true_bin"].tolist()
    for axis in axes:
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("median predicted fZP")
    axes[1].set_ylabel("fraction predicted >= 0.5")
    axes[0].set_xlabel("true fZP bin")
    axes[1].set_xlabel("true fZP bin")
    axes[0].set_title("Held-out recovery")
    axes[1].set_title("High-fZP recovery")
    fig.tight_layout()
    fig.savefig(outdir / "coverage_check.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    result = {"counts": counts, "recovery": recovery.to_dict("records")}
    with (outdir / "coverage_check.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> int:
    result = run()
    print(pd.DataFrame(result["counts"]).to_string(index=False))
    print(pd.DataFrame(result["recovery"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
