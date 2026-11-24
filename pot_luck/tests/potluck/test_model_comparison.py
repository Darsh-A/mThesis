"""Tests for the backend comparison (random forest vs gradient boosting vs XGBoost)."""

from __future__ import annotations

import json

from src.potluck.diagnostics.model_comparison import MODELS, run_comparison


def test_run_comparison_writes_outputs(small_dataset, tmp_path) -> None:
    outdir = tmp_path / "model_comparison"
    summary = run_comparison(
        small_dataset, outdir=outdir, models=MODELS, repeats=1
    )

    assert summary["n_mocks"] == small_dataset[0]["features"].shape[0]
    assert set(summary["models"]) == set(MODELS)

    for block, entry in summary["blocks"].items():
        assert set(entry["models"]) == set(MODELS)
        for payload in entry["models"].values():
            assert 0.0 <= payload["auc"] <= 1.0
            assert 0.0 <= payload["brier"] <= 1.0
            assert set(payload["permutation_mean"]) == set(entry["features"])
        assert (outdir / f"importance_{block}.png").stat().st_size > 0

    payload = json.loads((outdir / "comparison.json").read_text())
    assert payload["n_mocks"] == summary["n_mocks"]
    assert (outdir / "metrics.png").stat().st_size > 0
