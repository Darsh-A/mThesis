"""Recompute D in existing v20 datasets from the stored noisy observation.

The v20 generator used to store a dummy Detection(D=0, y=f_zp>=0.5). This
script rewrites D, D2, y, detect_reason and n_detected from the stored
`obs_features`, `obs_flags`, `obs_sigma`, `M_tot` and `matrix`, matching the
observed branch of detect_labels: the noisy observed ratios are used, not the
true noiseless ones.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import ELEMENT_INDEX, ELEMENTS, REFERENCE, SOLAR_MASS_FRACTION
from ..ejecta import PISN
from ..io import load_dataset

K = 3.0
MIN_DETECTED = 4
_SOLAR = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENTS])
_REF = ELEMENT_INDEX[REFERENCE]


def recompute(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, np.ndarray]:
    n = len(arrays["M_tot"])
    D = np.zeros(n, dtype=np.float32)
    D2 = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=bool)
    reason = np.full(n, 4, dtype=np.int8)
    n_used = np.zeros(n, dtype=np.int16)

    names = tuple(meta["observation_feature_names"])
    obs = arrays["obs_features"]
    flags = arrays["obs_flags"]
    sigma = arrays["obs_sigma"]
    has_pisn = arrays["matrix"][:, PISN, :].sum(axis=1) > 0.0

    for index in range(n):
        if not has_pisn[index]:
            reason[index] = 3  # no_pisn
            continue
        used = [
            column
            for column in range(len(names))
            if flags[index, column] == 1 and np.isfinite(obs[index, column])
        ]
        n_used[index] = len(used)
        if len(used) < MIN_DETECTED:
            reason[index] = 2  # insufficient
            continue
        used_elements = [names[column].split("/")[0] for column in used]
        element_indices = [ELEMENT_INDEX[element] for element in used_elements]
        element_indices.append(_REF)
        m_minus = arrays["M_tot"][index] - arrays["matrix"][index, PISN, :]
        if np.any(m_minus[element_indices] <= 0.0):
            D[index] = np.inf
            D2[index] = np.inf
            y[index] = True
            reason[index] = 1  # baseline_zero
            continue
        a_minus = np.log10(m_minus[element_indices] / _SOLAR[element_indices])
        x_minus = a_minus[:-1] - a_minus[-1]
        delta = obs[index, used] - x_minus
        sigma_values = sigma[index, used]
        squared = float(np.sum(delta * delta / sigma_values**2))
        squared = max(squared, 0.0)
        D2[index] = squared
        D[index] = np.sqrt(squared)
        y[index] = D[index] > K
        reason[index] = 0  # normal

    arrays["D"] = D
    arrays["D2"] = D2
    arrays["y"] = y
    arrays["detect_reason"] = reason
    arrays["n_detected"] = n_used
    return arrays


def run(paths: list[Path]) -> dict[str, Any]:
    summary = {}
    for path in paths:
        arrays, meta = load_dataset(path)
        arrays = recompute(arrays, meta)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        summary[str(path)] = {
            "n": int(len(arrays["y"])),
            "positive_rate": float(arrays["y"].mean()),
            "median_D": float(np.median(arrays["D"][np.isfinite(arrays["D"])])),
            "reason_counts": {
                str(code): int((arrays["detect_reason"] == code).sum())
                for code in range(5)
            },
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="dataset.npz files to rewrite in place"
    )
    args = parser.parse_args(argv)
    paths = [Path(path) for path in args.paths]
    summary = run(paths)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
