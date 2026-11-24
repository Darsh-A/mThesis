"""What does a predicted fZP actually mean?

Held-out mock predictions for Limongi18 and Nomoto13. Bin by predicted fZP and
report the true fZP distribution in each bin. This is the evidence for why a
0.4 prediction is a candidate, not a confident PISN-dominated detection.
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

from .apogee_fzp import _training_data

POLICIES = ("Limongi18_clamp", "Nomoto13_blend", "Nomoto13_clamp")
PRED_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01)


def _oof(base: Path, policy: str) -> tuple[np.ndarray, np.ndarray]:
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
    return prediction, y


def run(
    base: str | Path = "src/potluck/tests/v20",
    outdir: str | Path = "src/potluck/tests/v20/processed/fzp_bins",
) -> dict[str, Any]:
    base = Path(base)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in POLICIES:
        prediction, truth = _oof(base, policy)
        for i in range(len(PRED_BINS) - 1):
            mask = (prediction >= PRED_BINS[i]) & (prediction < PRED_BINS[i + 1])
            if not mask.any():
                continue
            t = truth[mask]
            rows.append(
                {
                    "policy": policy,
                    "pred_bin": f"{PRED_BINS[i]:.1f}-{PRED_BINS[i+1]:.1f}",
                    "n": int(mask.sum()),
                    "median_true": float(np.median(t)),
                    "p10_true": float(np.percentile(t, 10)),
                    "p90_true": float(np.percentile(t, 90)),
                    "frac_true_ge_0.5": float((t >= 0.5).mean()),
                    "frac_true_lt_0.2": float((t < 0.2).mean()),
                    "frac_true_lt_0.3": float((t < 0.3).mean()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "predicted_bins.csv", index=False)
    with (outdir / "predicted_bins.json").open("w") as handle:
        json.dump(rows, handle, indent=2)

    fig, axis = plt.subplots(figsize=(9, 5))
    for policy in POLICIES:
        subset = table[table.policy == policy]
        axis.plot(
            range(len(subset)),
            subset["median_true"],
            marker="o",
            label=policy,
        )
    labels = table[table.policy == POLICIES[0]]["pred_bin"].tolist()
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_xlabel("predicted fZP bin")
    axis.set_ylabel("median true fZP")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "predicted_bins.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"table": table}


def main() -> int:
    result = run()
    print(result["table"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
