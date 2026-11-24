"""Tests for the metal-free Nomoto node and the per-table Z grids."""

from __future__ import annotations

import numpy as np

from src.potluck.config import ModelConfig
from src.potluck.diagnostics.yield_cross import yield_variants
from src.potluck.yields.popii import LIMONGI18_Z_NODES, PopIITable
from src.yield_sources import _NOMOTO_Z_GRID, _nomoto_load_for_metallicity


def test_nomoto_grid_holds_the_metal_free_node() -> None:
    assert _NOMOTO_Z_GRID[0] == 0.0
    assert _nomoto_load_for_metallicity(0.0)


def test_nomoto_metal_free_interpolation_is_finite() -> None:
    for z in (1e-6, 1e-4, 1e-3, 3e-3, 5e-2):
        rows = _nomoto_load_for_metallicity(z)
        assert rows
        for row in rows:
            assert all(np.isfinite(value) for value in row["yields"].values())


def test_nomoto_below_first_positive_node_blends_linearly() -> None:
    cold = _nomoto_load_for_metallicity(0.0)
    warm = _nomoto_load_for_metallicity(5.0e-4)
    assert cold and warm
    cold_fe = sum(row["yields"].get("Fe56", 0.0) for row in cold)
    warm_fe = sum(row["yields"].get("Fe56", 0.0) for row in warm)
    assert warm_fe != cold_fe


def test_yield_variants_reset_the_popii_z_grid() -> None:
    variants = yield_variants(ModelConfig())

    limongi = variants["popii_Limongi18"].yields
    assert limongi.popii_z_nodes == LIMONGI18_Z_NODES
    assert limongi.popii_z_grid_size == len(LIMONGI18_Z_NODES)

    for name in ("popii_Nomoto13", "popii_WW95"):
        yields = variants[name].yields
        assert yields.popii_z_nodes is None
        assert yields.popii_z_grid_size == 24


def test_nomoto_table_integrates_at_low_metallicity() -> None:
    table = PopIITable(
        source_name="Nomoto13",
        z_grid_size=4,
        t_grid_size=4,
        cache_dir="data/potluck/cache",
    )
    for z in (1.0e-6, 1.0e-4, 1.3e-2):
        vector = table.integrated(z, 13.0)
        assert np.all(np.isfinite(vector))
        assert vector.sum() > 0.0
