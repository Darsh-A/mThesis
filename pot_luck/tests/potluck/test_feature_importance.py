"""Tests for the feature-importance ranking and selection."""

from __future__ import annotations

import json

from src.potluck.diagnostics.feature_importance import (
    BLOCKS,
    run_feature_importance,
)


def test_run_feature_importance_writes_outputs(small_dataset, tmp_path) -> None:
    arrays, meta = small_dataset
    outdir = tmp_path / "v15"
    summary = run_feature_importance(small_dataset, outdir=outdir)

    assert summary["n_mocks"] == arrays["features"].shape[0]
    assert set(summary["blocks"]) == set(BLOCKS)

    for block in BLOCKS:
        payload = summary["blocks"][block]
        assert payload["n_features"] == len(payload["features"])
        assert set(payload["impurity"]) == set(payload["features"])
        assert set(payload["permutation_mean"]) == set(payload["features"])
        assert 1 <= payload["chosen"]["k"] <= payload["n_features"]
        assert (outdir / f"importance_{block}.png").stat().st_size > 0

    payload = json.loads((outdir / "feature_importance.json").read_text())
    assert payload["n_mocks"] == summary["n_mocks"]
    chosen = json.loads((outdir / "chosen_features.json").read_text())
    assert set(chosen) == set(BLOCKS)
    assert (outdir / "selection_curve.png").stat().st_size > 0
