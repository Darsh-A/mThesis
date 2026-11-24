"""Reverse problem: which dilution masses do the yields and Li+ support?

The forward model fixes ``Z_star = M_Zpre / M_dil``; here ``M_dil`` is not
drawn from a prior.  Instead, for every prior draw of the event set and every
Li+ star, we solve for the ``M_dil`` that reproduces the star's [Fe/H], then
weight that solution by how well the resulting [X/Fe] pattern matches the
catalogue.  The result is a prior-predictive, likelihood-weighted inversion of
``M_dil``, reported per yield configuration.

Method
------
1. Draw an enrichment history from the potluck prior (counts, masses, clocks,
   Pop II burst mass) but *not* ``M_dil``.
2. For a log grid of ``M_dil``, compute ``Z_star = M_Zpre / M_dil``, the Pop II
   ejecta at ``(Z_star, t)``, and the total element vector.
3. For each star solve ``[Fe/H]_model(M_dil) = [Fe/H]_star`` on the grid.
4. Weight the solution by the [X/Fe] likelihood, with a per-element offset
   ``delta_X`` shared by the configuration to absorb systematic over/under-
   production, and a fixed intrinsic scatter floor.
5. Aggregate the weighted solutions into a supported ``M_dil`` range.

This is an importance-sampling / ABC-type inversion, not a gradient MCMC:
every sample is a prior draw and the likelihood reweights it, which avoids
imposing a prior on ``M_dil`` (the quantity being inferred).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import ModelConfig
from ..constants import (
    ELEMENTS,
    ELEMENT_INDEX,
    LI_FEATURE_ELEMENTS,
    METALS,
    REFERENCE,
    SOLAR_MASS_FRACTION,
)
from ..ejecta import _pisn_events, _popiii_sums
from ..events import draw_events
from ..observation import DETECTED, load_li_records
from ..yields.pisn import load_pisn_table
from ..yields.popii import PopIITable, z_config_for_table
from ..yields.popiii import PopIIITable

# --- fixed analysis choices ------------------------------------------------

FE_INDEX = ELEMENT_INDEX[REFERENCE]
METAL_INDICES = np.asarray([ELEMENT_INDEX[element] for element in METALS])
SOLAR = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENTS])
KAPPA_FE = float(SOLAR_MASS_FRACTION[REFERENCE])

RATIO_ELEMENTS = tuple(
    element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
)
RATIO_INDICES = np.asarray([ELEMENT_INDEX[element] for element in RATIO_ELEMENTS])
LI_COLUMNS = np.asarray(
    [LI_FEATURE_ELEMENTS.index(element) for element in RATIO_ELEMENTS]
)

# The grid is deliberately far wider than any prior window in use.  The
# supported range is read off the likelihood, not the grid bounds.
MDIL_GRID = np.logspace(3.0, 11.0, 160)

SIGMA_INTRINSIC_DEX = 0.10
MIN_DETECTED_ELEMENTS = 4
STAR_FEH_RANGE = (-4.5, -1.5)
SUPPORT_REDUCED_CHI2 = 4.0
DEFAULT_SCENARIOS = 1200
OFFSET_ITERATIONS = 3
# Structural zeros (an element a source never makes) must not produce
# infinite weights: cap the residual and the missing-[Fe/H] penalty.
RESIDUAL_CAP_DEX = 3.0
FEH_PENALTY = 1.0e4
MASS_FLOOR = 1.0e-20


@dataclass
class ScenarioLibrary:
    """Prior draws of the event set, without a dilution mass."""

    name: str
    M_pisn: np.ndarray
    M_iii: np.ndarray
    M_pre_metals: np.ndarray
    t_myr: np.ndarray
    mstar_ii: np.ndarray
    n_pisn: np.ndarray
    pisn_mass: np.ndarray
    n_iii: np.ndarray
    settings: dict[str, Any]

    def __len__(self) -> int:
        return int(self.t_myr.size)


@dataclass
class ConfigResult:
    """Inversion result for one yield configuration."""

    name: str
    category: str
    settings: dict[str, Any]
    mdil: np.ndarray  # (n_star, n_scenario)
    weights: np.ndarray  # (n_star, n_scenario), rows sum to one
    n_pisn: np.ndarray  # (n_scenario,)
    pisn_mass: np.ndarray  # (n_scenario,)
    n_iii: np.ndarray  # (n_scenario,)
    offsets: np.ndarray  # (n_ratio,)
    scatter: np.ndarray  # (n_ratio,)
    summary: dict[str, Any] = None


# --- catalogue selection ---------------------------------------------------


def select_stars(records, feh_range=STAR_FEH_RANGE) -> np.ndarray:
    """Return indices of Li+ stars with enough detected ratio elements."""
    detected = np.isfinite(records.feh)
    detected &= records.feh >= feh_range[0]
    detected &= records.feh <= feh_range[1]
    count = np.sum(
        (records.flags[:, LI_COLUMNS] == DETECTED)
        & np.isfinite(records.xfe[:, LI_COLUMNS])
        & np.isfinite(records.sigma[:, LI_COLUMNS]),
        axis=1,
    )
    return np.flatnonzero(detected & (count >= MIN_DETECTED_ELEMENTS))


def star_arrays(records, index: np.ndarray):
    """Return observed ratios, errors, availability, and [Fe/H] for stars."""
    xfe = records.xfe[index][:, LI_COLUMNS].astype(float)
    sigma = records.sigma[index][:, LI_COLUMNS].astype(float)
    flags = records.flags[index][:, LI_COLUMNS]
    available = (
        (flags == DETECTED) & np.isfinite(xfe) & np.isfinite(sigma) & (sigma > 0.0)
    )
    sigma = np.where(available, sigma, np.nan)
    sigma = np.where(np.isfinite(sigma), sigma, 0.2)
    feh = records.feh[index].astype(float)
    feh_sigma = records.e_feh[index].astype(float)
    feh_sigma = np.where(np.isfinite(feh_sigma) & (feh_sigma > 0.0), feh_sigma, 0.1)
    return xfe, sigma, available, feh, np.clip(feh_sigma, 0.01, 0.5)


# --- Pop III sampling variants ---------------------------------------------


def popiii_variant(table: PopIIITable, variant: str):
    """Return ``(sampling_mode, weights)`` for a Pop III variant."""
    keys = table.combo_keys
    masses = np.asarray([key[0] for key in keys], dtype=float)
    energies = np.asarray([key[1] for key in keys], dtype=float)

    if variant == "uniform":
        return "combo", None
    if variant == "row":
        return "row", None
    if variant == "low_mass":
        weights = 1.0 / masses
    elif variant == "high_mass":
        weights = masses
    elif variant == "high_energy":
        weights = energies
    elif variant == "low_energy":
        weights = 1.0 / energies
    elif variant == "M11_15":
        weights = (masses <= 15.0).astype(float)
    elif variant == "M25":
        weights = (masses == 25.0).astype(float)
    elif variant == "M40":
        weights = (masses == 40.0).astype(float)
    elif variant == "M100":
        weights = (masses == 100.0).astype(float)
    else:
        raise ValueError(f"unknown Pop III variant {variant!r}")
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0.0:
        raise ValueError(f"Pop III variant {variant!r} has no groups")
    return "combo", tuple((weights / weights.sum()).tolist())


# --- prior draws -----------------------------------------------------------


def build_library(
    config: ModelConfig,
    name: str,
    n_scenarios: int,
    seed: int,
    pisn_table,
    popiii_table: PopIIITable,
    enable_pisn: bool = True,
    enable_iii: bool = True,
    enable_popii: bool = True,
) -> ScenarioLibrary:
    """Draw the event set and compute the pre-enrichment ejecta."""
    rng = np.random.default_rng(seed)
    events = draw_events(n_scenarios, config, rng, pisn_table, popiii_table)

    if not enable_pisn:
        events.n_pisn = np.zeros_like(events.n_pisn)
        events.pisn_masses = np.zeros_like(events.pisn_masses)
    if not enable_iii:
        events.n_iii = np.zeros_like(events.n_iii)
        events.iii_group_counts = np.zeros_like(events.iii_group_counts)

    delta = events.delta_myr
    horizon = delta + events.t_myr

    pisn_vectors, tau_pisn = _pisn_events(events, pisn_table)
    pre_pisn = (tau_pisn <= delta[:, None]) & (events.pisn_masses > 0.0)
    final_pisn = (tau_pisn <= horizon[:, None]) & (events.pisn_masses > 0.0)
    M_pisn = (pisn_vectors * final_pisn[..., None]).sum(axis=1)
    M_pre_pisn = (pisn_vectors * pre_pisn[..., None]).sum(axis=1)

    M_iii, M_pre_iii = _popiii_sums(events, popiii_table, delta, horizon)
    M_pre = M_pre_pisn + M_pre_iii

    # Decouple the Pop II burst mass from M_dil: this study is about M_dil,
    # so M_starII gets its own wide log-uniform prior.
    mstar_ii = 10.0 ** rng.uniform(
        np.log10(config.prior.mstar_ii_range[0]),
        np.log10(config.prior.mstar_ii_range[1]),
        n_scenarios,
    )
    if not enable_popii:
        mstar_ii = np.zeros_like(mstar_ii)

    return ScenarioLibrary(
        name=name,
        M_pisn=M_pisn,
        M_iii=M_iii,
        M_pre_metals=M_pre[:, METAL_INDICES].sum(axis=1),
        t_myr=events.t_myr.astype(float),
        mstar_ii=mstar_ii,
        n_pisn=events.n_pisn.astype(int),
        pisn_mass=events.pisn_masses[:, 0].astype(float).copy(),
        n_iii=events.n_iii.astype(int),
        settings={
            "n_scenarios": int(n_scenarios),
            "seed": int(seed),
            "enable_pisn": bool(enable_pisn),
            "enable_iii": bool(enable_iii),
            "enable_popii": bool(enable_popii),
            "popiii_sampling": config.prior.popiii_sampling,
            "popiii_weights": config.prior.popiii_mass_energy_weights,
            "mstar_ii_range": list(config.prior.mstar_ii_range),
        },
    )


# --- matrix and inversion --------------------------------------------------


def model_grid(
    library: ScenarioLibrary, popii: PopIITable, mdil_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ejecta matrix and model [Fe/H] on the M_dil grid."""
    n = len(library)
    g = mdil_grid.size
    metallicity = library.M_pre_metals[:, None] / mdil_grid[None, :]
    time = np.broadcast_to(library.t_myr[:, None], (n, g))
    yields = popii.integrated(metallicity.ravel(), time.ravel())
    popii_mass = yields.reshape(n, g, len(ELEMENTS))
    popii_mass *= library.mstar_ii[:, None, None]

    matrix = (
        library.M_pisn[:, None, :]
        + library.M_iii[:, None, :]
        + popii_mass
    )
    fe_mass = matrix[..., FE_INDEX]
    with np.errstate(divide="ignore", invalid="ignore"):
        feh = np.log10(fe_mass / (mdil_grid[None, :] * KAPPA_FE))
    return matrix, feh


def solve_mdil(
    feh_grid: np.ndarray, mdil_grid: np.ndarray, target: float
):
    """Solve ``feh_grid(M_dil) = target`` for every scenario row."""
    n, g = feh_grid.shape
    difference = feh_grid - target
    finite = np.isfinite(difference)
    difference = np.where(finite, difference, np.inf)

    left = difference[:, :-1]
    right = difference[:, 1:]
    crossing = (left * right <= 0.0) & np.isfinite(left) & np.isfinite(right)
    has_crossing = crossing.any(axis=1)
    index = np.argmax(crossing, axis=1)

    nearest = np.argmin(np.abs(difference), axis=1)
    index = np.where(has_crossing, index, np.clip(nearest, 0, g - 2))

    rows = np.arange(n)
    d0 = difference[rows, index]
    d1 = difference[rows, index + 1]
    denominator = np.where(d1 != d0, d1 - d0, 1.0)
    fraction = np.clip(-d0 / denominator, 0.0, 1.0)
    fraction = np.where(
        has_crossing,
        fraction,
        np.where(nearest >= g - 1, 1.0, 0.0),
    )

    log_grid = np.log10(mdil_grid)
    log_mdil = log_grid[index] + fraction * (
        log_grid[index + 1] - log_grid[index]
    )
    feh_solution = (
        feh_grid[rows, index] * (1.0 - fraction)
        + feh_grid[rows, index + 1] * fraction
    )
    return 10.0**log_mdil, feh_solution, ~has_crossing, index, fraction


def gather_rows(matrix: np.ndarray, index: np.ndarray, fraction: np.ndarray):
    """Linearly interpolate the matrix at the solved grid index."""
    rows = np.arange(index.size)
    return (
        matrix[rows, index] * (1.0 - fraction)[:, None]
        + matrix[rows, index + 1] * fraction[:, None]
    )


def abundance_ratios(mass: np.ndarray) -> np.ndarray:
    """Return [X/Fe] for the catalogue ratio elements."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abundance = np.log10(np.maximum(mass, MASS_FLOOR) / SOLAR[None, :])
    ratios = (
        log_abundance[:, RATIO_INDICES]
        - log_abundance[:, [FE_INDEX]]
    )
    return ratios


def chi_squared(
    ratios: np.ndarray,
    feh_model: np.ndarray,
    xfe: np.ndarray,
    sigma: np.ndarray,
    available: np.ndarray,
    feh: np.ndarray,
    feh_sigma: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Per-(star, scenario) chi-squared."""
    residual = ratios - xfe[:, None, :] - offsets[None, None, :]
    variance = sigma[:, None, :] ** 2 + SIGMA_INTRINSIC_DEX**2
    residual = np.where(np.isfinite(residual), residual, -RESIDUAL_CAP_DEX)
    residual = np.clip(residual, -RESIDUAL_CAP_DEX, RESIDUAL_CAP_DEX)
    terms = np.where(available[:, None, :], residual**2 / variance, 0.0)
    chi2 = terms.sum(axis=2)
    feh_term = ((feh_model - feh[:, None]) / feh_sigma[:, None]) ** 2
    chi2 += np.where(np.isfinite(feh_term), feh_term, FEH_PENALTY)
    return chi2


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of one-dimensional values."""
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(good) == 0:
        return float("nan")
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(np.interp(0.5, cumulative, values))


def weighted_mad(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median absolute deviation."""
    median = weighted_median(values, weights)
    if not np.isfinite(median):
        return float("nan")
    return 1.4826 * weighted_median(np.abs(values - median), weights)


def fit_offsets(
    ratios: np.ndarray,
    xfe: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Estimate one model-minus-observed offset per element."""
    offsets = np.zeros(len(RATIO_ELEMENTS))
    for element in range(len(RATIO_ELEMENTS)):
        selected = available[:, element]
        if not np.any(selected):
            continue
        values = ratios[selected, :, element] - xfe[selected, element][:, None]
        element_weights = weights[selected, :]
        offsets[element] = weighted_median(values, element_weights)
    return offsets


def invert_configuration(
    name: str,
    category: str,
    library: ScenarioLibrary,
    popii: PopIITable,
    xfe: np.ndarray,
    sigma: np.ndarray,
    available: np.ndarray,
    feh: np.ndarray,
    feh_sigma: np.ndarray,
    mdil_grid: np.ndarray = MDIL_GRID,
) -> ConfigResult:
    """Solve M_dil per star, fit offsets, and return weighted solutions."""
    matrix, feh_grid = model_grid(library, popii, mdil_grid)
    n_star = xfe.shape[0]
    n_scenario = len(library)

    mdil = np.full((n_star, n_scenario), np.nan)
    feh_model = np.full((n_star, n_scenario), np.nan)
    censored = np.zeros((n_star, n_scenario), dtype=bool)
    ratios = np.full((n_star, n_scenario, len(RATIO_ELEMENTS)), np.nan)

    for i in range(n_star):
        solved, feh_solved, censored_row, index, fraction = solve_mdil(
            feh_grid, mdil_grid, feh[i]
        )
        mdil[i] = solved
        feh_model[i] = feh_solved
        censored[i] = censored_row
        ratios[i] = abundance_ratios(gather_rows(matrix, index, fraction))

    offsets = np.zeros(len(RATIO_ELEMENTS))
    weights = np.full((n_star, n_scenario), 1.0 / n_scenario)
    chi2 = chi_squared(
        ratios, feh_model, xfe, sigma, available, feh, feh_sigma, offsets
    )
    chi2_raw = chi2.copy()
    for _ in range(OFFSET_ITERATIONS):
        log_weight = -0.5 * chi2
        log_weight -= log_weight.max(axis=1, keepdims=True)
        weights = np.exp(log_weight)
        total = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights, total, out=np.zeros_like(weights), where=total > 0.0
        )
        offsets = fit_offsets(ratios, xfe, available, weights)
        chi2 = chi_squared(
            ratios, feh_model, xfe, sigma, available, feh, feh_sigma, offsets
        )

    log_weight = -0.5 * chi2
    log_weight -= log_weight.max(axis=1, keepdims=True)
    weights = np.exp(log_weight)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights, total, out=np.zeros_like(weights), where=total > 0.0
    )

    scatter = np.full(len(RATIO_ELEMENTS), np.nan)
    for element in range(len(RATIO_ELEMENTS)):
        selected = available[:, element]
        if not np.any(selected):
            continue
        values = (
            ratios[selected, :, element]
            - xfe[selected, element][:, None]
            - offsets[element]
        )
        scatter[element] = weighted_mad(values, weights[selected, :])

    summary = summarise_config(
        name,
        library,
        mdil,
        weights,
        chi2,
        chi2_raw,
        available,
        censored,
        offsets,
        scatter,
    )
    return ConfigResult(
        name=name,
        category=category,
        settings=library.settings,
        mdil=mdil,
        weights=weights,
        n_pisn=library.n_pisn,
        pisn_mass=library.pisn_mass,
        n_iii=library.n_iii,
        offsets=offsets,
        scatter=scatter,
        summary=summary,
    )


# --- summaries -------------------------------------------------------------


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Weighted quantiles of flattened values."""
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(good) == 0:
        return np.full(len(probabilities), np.nan)
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def star_medians(
    mdil: np.ndarray, weights: np.ndarray, probabilities=(0.16, 0.5, 0.84)
) -> np.ndarray:
    """Return weighted quantiles for every star."""
    result = np.full((mdil.shape[0], len(probabilities)), np.nan)
    for i in range(mdil.shape[0]):
        result[i] = weighted_quantiles(mdil[i], weights[i], np.asarray(probabilities))
    return result


def summarise_config(
    name: str,
    library: ScenarioLibrary,
    mdil: np.ndarray,
    weights: np.ndarray,
    chi2: np.ndarray,
    chi2_raw: np.ndarray,
    available: np.ndarray,
    censored: np.ndarray,
    offsets: np.ndarray,
    scatter: np.ndarray,
) -> dict[str, Any]:
    """Return the supported M_dil range and fit quality for one config."""
    per_star = star_medians(mdil, weights)
    log_mdil = np.log10(np.where(mdil > 0.0, mdil, np.nan))
    probabilities = np.asarray([0.05, 0.16, 0.5, 0.84, 0.95])
    log_pooled = weighted_quantiles(log_mdil, weights, probabilities)

    raw_log_weight = -0.5 * chi2_raw
    raw_log_weight -= raw_log_weight.max(axis=1, keepdims=True)
    weights_raw = np.exp(raw_log_weight)
    raw_total = weights_raw.sum(axis=1, keepdims=True)
    weights_raw = np.divide(
        weights_raw, raw_total, out=np.zeros_like(weights_raw), where=raw_total > 0.0
    )
    per_star_raw = star_medians(mdil, weights_raw)
    log_pooled_raw = weighted_quantiles(log_mdil, weights_raw, probabilities)

    prior_weights = np.full_like(weights, 1.0 / weights.shape[1])
    per_star_prior = star_medians(mdil, prior_weights)
    log_pooled_prior = weighted_quantiles(log_mdil, prior_weights, probabilities)
    best = np.nanmin(chi2, axis=1)
    best_raw = np.nanmin(chi2_raw, axis=1)
    n_elements = available.sum(axis=1)
    reduced = best / np.maximum(n_elements, 1)
    reduced_raw = best_raw / np.maximum(n_elements, 1)
    supported = np.isfinite(reduced) & (reduced <= SUPPORT_REDUCED_CHI2)
    supported_raw = np.isfinite(reduced_raw) & (
        reduced_raw <= SUPPORT_REDUCED_CHI2
    )

    star_log = np.log10(
        np.where(per_star[:, 1] > 0.0, per_star[:, 1], np.nan)
    )
    star_low = np.log10(
        np.where(per_star[:, 0] > 0.0, per_star[:, 0], np.nan)
    )
    star_high = np.log10(
        np.where(per_star[:, 2] > 0.0, per_star[:, 2], np.nan)
    )

    return {
        "name": name,
        "n_scenarios": int(len(library)),
        "n_stars": int(mdil.shape[0]),
        "pooled_log10_mdil": {
            "p05": float(log_pooled[0]),
            "p16": float(log_pooled[1]),
            "p50": float(log_pooled[2]),
            "p84": float(log_pooled[3]),
            "p95": float(log_pooled[4]),
        },
        "pooled_mdil": {
            "p16": float(10.0**log_pooled[1]),
            "p50": float(10.0**log_pooled[2]),
            "p84": float(10.0**log_pooled[3]),
        },
        "pooled_log10_mdil_raw": {
            "p05": float(log_pooled_raw[0]),
            "p16": float(log_pooled_raw[1]),
            "p50": float(log_pooled_raw[2]),
            "p84": float(log_pooled_raw[3]),
            "p95": float(log_pooled_raw[4]),
        },
        "per_star_log10_mdil_raw": {
            "p16": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_raw[:, 1] > 0.0, per_star_raw[:, 1], np.nan)),
                    16,
                )
            ),
            "p50": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_raw[:, 1] > 0.0, per_star_raw[:, 1], np.nan)),
                    50,
                )
            ),
            "p84": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_raw[:, 1] > 0.0, per_star_raw[:, 1], np.nan)),
                    84,
                )
            ),
        },
        "pooled_log10_mdil_prior": {
            "p05": float(log_pooled_prior[0]),
            "p16": float(log_pooled_prior[1]),
            "p50": float(log_pooled_prior[2]),
            "p84": float(log_pooled_prior[3]),
            "p95": float(log_pooled_prior[4]),
        },
        "per_star_log10_mdil_prior": {
            "p16": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_prior[:, 1] > 0.0, per_star_prior[:, 1], np.nan)),
                    16,
                )
            ),
            "p50": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_prior[:, 1] > 0.0, per_star_prior[:, 1], np.nan)),
                    50,
                )
            ),
            "p84": float(
                np.nanpercentile(
                    np.log10(np.where(per_star_prior[:, 1] > 0.0, per_star_prior[:, 1], np.nan)),
                    84,
                )
            ),
        },
        "offset_rms": float(np.sqrt(np.nanmean(offsets**2))),
        "max_abs_offset": float(np.nanmax(np.abs(offsets))),
        "per_star_log10_mdil": {
            "p16": float(np.nanpercentile(star_log, 16)),
            "p50": float(np.nanpercentile(star_log, 50)),
            "p84": float(np.nanpercentile(star_log, 84)),
        },
        "per_star_band_dex": {
            "p16": float(np.nanpercentile(star_low, 16)),
            "p50": float(np.nanpercentile(star_log, 50)),
            "p84": float(np.nanpercentile(star_high, 84)),
        },
        "support_fraction": float(np.mean(supported)),
        "support_fraction_raw": float(np.mean(supported_raw)),
        "median_best_reduced_chi2": float(np.nanmedian(reduced)),
        "median_best_reduced_chi2_raw": float(np.nanmedian(reduced_raw)),
        "median_best_chi2": float(np.nanmedian(best)),
        "median_n_elements": float(np.nanmedian(n_elements)),
        "censored_fraction": float(np.mean(censored)),
        "offsets": {
            element: float(value)
            for element, value in zip(RATIO_ELEMENTS, offsets)
        },
        "scatter": {
            element: float(value)
            for element, value in zip(RATIO_ELEMENTS, scatter)
        },
        "events": {
            "median_n_pisn": float(np.median(library.n_pisn)),
            "median_mstar_ii": float(np.median(library.mstar_ii)),
            "median_n_iii": float(np.median(library.n_iii)),
        },
    }


# --- configurations --------------------------------------------------------


def configuration_list() -> list[dict[str, Any]]:
    """Return the permutation design: combined, separate, and table variants."""
    base = {
        "pisn_table": "HW2002",
        "popii_table": "Limongi18",
        "popiii": "uniform",
        "enable_pisn": True,
        "enable_iii": True,
        "enable_popii": True,
    }
    configs: list[dict[str, Any]] = []

    def add(name, category, **overrides):
        settings = dict(base)
        settings.update(overrides)
        configs.append({"name": name, "category": category, **settings})

    add("all", "combined")
    add("pisn_only", "combined", enable_iii=False, enable_popii=False)
    add("popiii_only", "combined", enable_pisn=False, enable_popii=False)
    add("pisn+popiii", "combined", enable_popii=False)
    add("pisn+popii", "combined", enable_iii=False)
    add("popiii+popii", "combined", enable_pisn=False)

    for table in ("HW2002", "Takahashi-MR", "Takahashi-NM", "Takahashi-NR"):
        add(f"PISN:{table}", "pisn", pisn_table=table)

    for variant in (
        "uniform",
        "row",
        "low_mass",
        "high_mass",
        "high_energy",
        "low_energy",
        "M11_15",
        "M25",
        "M40",
        "M100",
    ):
        add(f"PopIII:{variant}", "popiii", popiii=variant)

    for table in (
        "Limongi18",
        "Limongi18_V150",
        "Limongi18_V300",
        "Nomoto13",
        "Nomoto13_CK20",
        "WW95",
    ):
        add(f"PopII:{table}", "popii", popii_table=table)

    return configs


def variant_config(config: ModelConfig, settings: dict[str, Any]) -> ModelConfig:
    """Apply one configuration's yield and source switches."""
    prior = config.prior
    if not settings["enable_pisn"]:
        prior = replace(prior, n_pisn_probs=(1.0, 0.0, 0.0, 0.0))
    if not settings["enable_iii"]:
        prior = replace(prior, lambda_iii_range=(1.0e-10, 1.0e-9))

    nodes, size = z_config_for_table(settings["popii_table"])
    yields = replace(
        config.yields,
        pisn_table=settings["pisn_table"],
        popii_table=settings["popii_table"],
        popii_z_nodes=nodes,
        popii_z_grid_size=size,
    )
    return replace(config, prior=prior, yields=yields)


# --- plotting --------------------------------------------------------------


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mdil_summary(results: list[ConfigResult], path: Path) -> None:
    """Forest plot of the supported M_dil range per configuration."""
    order = [
        "combined", "pisn", "popiii", "popii",
    ]
    colors = {
        "combined": "tab:blue",
        "pisn": "tab:orange",
        "popiii": "tab:green",
        "popii": "tab:red",
    }
    ordered = [r for category in order for r in results if r.category == category]
    fig, axis = plt.subplots(figsize=(10, 0.34 * len(ordered) + 2.2))
    positions = np.arange(len(ordered))
    for position, result in zip(positions, ordered):
        summary = result.summary
        p16 = summary["pooled_log10_mdil"]["p16"]
        p50 = summary["pooled_log10_mdil"]["p50"]
        p84 = summary["pooled_log10_mdil"]["p84"]
        p05 = summary["pooled_log10_mdil"]["p05"]
        p95 = summary["pooled_log10_mdil"]["p95"]
        color = colors[result.category]
        prior = summary["pooled_log10_mdil_prior"]
        axis.plot(
            [prior["p16"], prior["p84"]], [position, position],
            color="0.75", lw=6.0, alpha=0.5,
        )
        raw = summary["pooled_log10_mdil_raw"]
        axis.plot(
            [raw["p05"], raw["p95"]], [position, position],
            color="0.55", lw=0.8, ls=":", alpha=0.8,
        )
        axis.plot(
            [raw["p16"], raw["p84"]], [position, position],
            color="0.35", lw=1.4, alpha=0.7,
        )
        axis.plot(
            [raw["p50"]], [position], marker="o", ms=4,
            mfc="white", mec="0.2", mew=0.8,
        )
        axis.plot([p05, p95], [position, position], color=color, lw=1.0, alpha=0.45)
        axis.plot([p16, p84], [position, position], color=color, lw=4.0, alpha=0.85)
        axis.plot([p50], [position], marker="o", color="black", ms=3.5)
    axis.axvline(np.log10(2.0e5), color="0.5", ls=":", lw=1.0)
    axis.axvline(np.log10(1.0e9), color="0.5", ls=":", lw=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels([result.name for result in ordered], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
    axis.set_title("Supported dilution mass")
    axis.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([], [], color=colors[key], lw=4, label=key)
        for key in order
    ]
    handles.append(
        plt.Line2D([], [], color="black", lw=4, label="offsets fitted")
    )
    handles.append(
        plt.Line2D([], [], color="0.35", lw=1.4, marker="o", mfc="white",
                   label="no offsets")
    )
    handles.append(
        plt.Line2D([], [], color="0.75", lw=6, alpha=0.6,
                   label="event prior")
    )
    axis.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)
    _save(fig, path)


def plot_category_summary(
    results: list[ConfigResult], category: str, title: str, path: Path
) -> None:
    """Forest plot of one configuration family."""
    selected = [result for result in results if result.category == category]
    fig, axis = plt.subplots(figsize=(8.4, 0.44 * len(selected) + 1.8))
    positions = np.arange(len(selected))
    for position, result in zip(positions, selected):
        summary = result.summary
        prior = summary["pooled_log10_mdil_prior"]
        raw = summary["pooled_log10_mdil_raw"]
        corrected = summary["pooled_log10_mdil"]
        axis.plot(
            [prior["p16"], prior["p84"]], [position, position],
            color="0.78", lw=6.0, alpha=0.55,
        )
        axis.plot(
            [raw["p16"], raw["p84"]], [position, position],
            color="0.35", lw=1.5, alpha=0.75,
        )
        axis.plot(
            [raw["p50"]], [position], marker="o", ms=4,
            mfc="white", mec="0.2", mew=0.8,
        )
        axis.plot(
            [corrected["p16"], corrected["p84"]], [position, position],
            color="tab:blue", lw=4.0, alpha=0.85,
        )
        axis.plot([corrected["p50"]], [position], marker="o", color="black", ms=3.5)
    axis.set_yticks(positions)
    axis.set_yticklabels([result.name for result in selected], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([], [], color="tab:blue", lw=4, label="offsets fitted"),
        plt.Line2D([], [], color="0.35", lw=1.5, marker="o", mfc="white",
                   label="no offsets"),
        plt.Line2D([], [], color="0.78", lw=6, alpha=0.6, label="event prior"),
    ]
    axis.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)
    _save(fig, path)


def plot_raw_vs_corrected(results: list[ConfigResult], path: Path) -> None:
    """Compare the median M_dil with and without element offsets."""
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    colors = {
        "combined": "tab:blue",
        "pisn": "tab:orange",
        "popiii": "tab:green",
        "popii": "tab:red",
    }
    for result in results:
        raw = result.summary["pooled_log10_mdil_raw"]["p50"]
        corrected = result.summary["pooled_log10_mdil"]["p50"]
        axis.scatter(
            [raw], [corrected], color=colors[result.category], s=28,
            label=result.category if result.name == "all" else None,
        )
    low, high = 4.0, 10.0
    axis.plot([low, high], [low, high], color="0.5", ls="--", lw=1.0)
    axis.set_xlabel(r"$\log_{10}M_{\rm dil}$ (no offsets)")
    axis.set_ylabel(r"$\log_{10}M_{\rm dil}$ (offsets fitted)")
    axis.set_title("Systematic offsets and the dilution mass")
    axis.grid(alpha=0.25)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=color, label=key)
        for key, color in colors.items()
    ]
    axis.legend(handles=handles, fontsize=8, frameon=False)
    _save(fig, path)


def plot_mdil_densities(results: list[ConfigResult], names: list[str], path: Path) -> None:
    """Weighted log M_dil densities for selected configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(4.0, 10.5, 90)
    selected = [r for r in results if r.name in names]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(selected), 1)))
    for color, result in zip(colors, selected):
        values = np.log10(np.where(result.mdil > 0.0, result.mdil, np.nan))
        weights = result.weights
        good = np.isfinite(values)
        axes[0].hist(
            values[good],
            bins=bins,
            weights=weights[good],
            density=True,
            histtype="step",
            lw=1.6,
            color=color,
            label=result.name,
        )
        fraction = result.summary["support_fraction"]
        median = result.summary["pooled_log10_mdil"]["p50"]
        axes[1].scatter([median], [fraction], color=color, s=30, label=result.name)
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
    axes[0].set_ylabel("weighted density")
    axes[0].set_title(r"$M_{\rm dil}$ posterior")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"median $\log_{10}(M_{\rm dil}/M_\odot)$")
    axes[1].set_ylabel(r"support fraction ($\chi^2_\nu<4$)")
    axes[1].set_title("Support versus median mass")
    axes[1].grid(alpha=0.25)
    _save(fig, path)


def plot_offsets(results: list[ConfigResult], path: Path) -> None:
    """Heatmaps of per-element offsets and residual scatter."""
    names = [result.name for result in results]
    offsets = np.asarray([result.offsets for result in results])
    scatter = np.asarray([result.scatter for result in results])

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.32 * len(results) + 2.0))
    for axis, matrix, title, label in (
        (axes[0], offsets, "Element offsets", "model - Li+ (dex)"),
        (axes[1], scatter, "Residual scatter", "weighted MAD (dex)"),
    ):
        limit = np.nanpercentile(np.abs(matrix), 98)
        limit = max(limit, 0.1)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r" if matrix is offsets else "magma",
            vmin=-limit if matrix is offsets else 0.0,
            vmax=limit,
        )
        axis.set_xticks(np.arange(len(RATIO_ELEMENTS)))
        axis.set_xticklabels(RATIO_ELEMENTS, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=7)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=label)
    _save(fig, path)


def plot_support(results: list[ConfigResult], path: Path) -> None:
    """Fit quality and support fraction per configuration."""
    names = [result.name for result in results]
    median_chi2 = [result.summary["median_best_reduced_chi2"] for result in results]
    median_chi2_raw = [
        result.summary["median_best_reduced_chi2_raw"] for result in results
    ]
    fractions = [result.summary["support_fraction"] for result in results]
    fractions_raw = [
        result.summary["support_fraction_raw"] for result in results
    ]
    colors = [
        {"combined": "tab:blue", "pisn": "tab:orange", "popiii": "tab:green", "popii": "tab:red"}[
            result.category
        ]
        for result in results
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.3 * len(results) + 2.0))
    positions = np.arange(len(results))
    axes[0].barh(positions - 0.18, median_chi2, height=0.34, color=colors, alpha=0.85)
    axes[0].barh(
        positions + 0.18, median_chi2_raw, height=0.34,
        color="0.55", alpha=0.75, label="no offsets",
    )
    axes[0].axvline(SUPPORT_REDUCED_CHI2, color="black", ls="--", lw=1.0)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"median best $\chi^2_\nu$")
    axes[0].set_title("Fit quality")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].barh(positions - 0.18, fractions, height=0.34, color=colors, alpha=0.85)
    axes[1].barh(
        positions + 0.18, fractions_raw, height=0.34,
        color="0.55", alpha=0.75, label="no offsets",
    )
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("supported star fraction")
    axes[1].set_title(r"Support ($\chi^2_\nu<4$)")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=7, frameon=False)
    _save(fig, path)


def plot_star_mdil(
    result: ConfigResult, feh: np.ndarray, path: Path
) -> None:
    """Per-star median M_dil versus [Fe/H]."""
    per_star = star_medians(result.mdil, result.weights)
    x = feh
    y = np.log10(np.where(per_star[:, 1] > 0.0, per_star[:, 1], np.nan))
    low = np.log10(np.where(per_star[:, 0] > 0.0, per_star[:, 0], np.nan))
    high = np.log10(np.where(per_star[:, 2] > 0.0, per_star[:, 2], np.nan))
    good = np.isfinite(y)
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.errorbar(
        x[good],
        y[good],
        yerr=[y[good] - low[good], high[good] - y[good]],
        fmt="o",
        ms=3,
        lw=0.7,
        alpha=0.65,
        color="tab:blue",
    )
    axis.set_xlabel("[Fe/H]")
    axis.set_ylabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
    axis.set_title(f"Per-star dilution mass: {result.name}")
    axis.grid(alpha=0.25)
    _save(fig, path)


def plot_event_dependence(result: ConfigResult, path: Path) -> None:
    """Weighted M_dil posteriors split by event count."""
    log_mdil = np.log10(np.where(result.mdil > 0.0, result.mdil, np.nan))
    weights = result.weights

    def grouped(axis, counts, bins, label):
        medians, lows, highs, positions = [], [], [], []
        for position, group in enumerate(bins):
            if group.size == 1:
                selected = counts == group[0]
            else:
                selected = (counts >= group[0]) & (counts <= group[1])
            if not np.any(selected):
                continue
            values = log_mdil[:, selected].ravel()
            element_weights = weights[:, selected].ravel()
            quantiles = weighted_quantiles(
                values, element_weights, np.asarray([0.16, 0.5, 0.84])
            )
            positions.append(position)
            lows.append(quantiles[0])
            medians.append(quantiles[1])
            highs.append(quantiles[2])
        axis.errorbar(
            positions,
            medians,
            yerr=[np.asarray(medians) - np.asarray(lows),
                  np.asarray(highs) - np.asarray(medians)],
            fmt="o",
            color="tab:blue",
            capsize=3,
        )
        axis.set_xticks(range(len(bins)))
        axis.set_xticklabels(
            [
                f"{group[0]}" if group.size == 1 else f"{group[0]}-{group[1]}"
                for group in bins
            ]
        )
        axis.set_xlabel(label)
        axis.set_ylabel(r"$\log_{10}(M_{\rm dil}/M_\odot)$")
        axis.grid(axis="y", alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    n_pisn = np.asarray(result.n_pisn)
    grouped(
        axes[0],
        n_pisn,
        [np.asarray([value]) for value in sorted(set(n_pisn.tolist()))],
        r"$n_{\rm PISN}$",
    )
    axes[0].set_title("Dilution mass versus PISN count")

    n_iii = np.asarray(result.n_iii)
    grouped(
        axes[1],
        n_iii,
        [
            np.asarray([0]),
            np.asarray([1]),
            np.asarray([2, 3]),
            np.asarray([4, 10]),
            np.asarray([11, int(max(n_iii.max(), 11))]),
        ],
        r"$n_{\rm III}$",
    )
    axes[1].set_title("Dilution mass versus Pop III count")
    _save(fig, path)


# --- driver ----------------------------------------------------------------


def run_mdil_inverse(
    n_scenarios: int = DEFAULT_SCENARIOS,
    seed: int = 42,
    outdir: str | Path = "src/potluck/tests/v18",
    config: ModelConfig | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full reverse inversion and write figures, tables, and JSON."""
    config = ModelConfig() if config is None else config
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_li_records()
    star_index = select_stars(records)
    xfe, sigma, available, feh, feh_sigma = star_arrays(records, star_index)

    popiii_table = PopIIITable.from_raw()
    pisn_tables: dict[str, Any] = {}
    popii_tables: dict[str, PopIITable] = {}

    results: list[ConfigResult] = []
    for index, settings in enumerate(configuration_list()):
        if verbose:
            print(f"[{index + 1}] {settings['name']}", flush=True)
        pisn_name = settings["pisn_table"]
        if pisn_name not in pisn_tables:
            pisn_tables[pisn_name] = load_pisn_table(pisn_name)
        popii_name = settings["popii_table"]
        if popii_name not in popii_tables:
            nodes, size = z_config_for_table(popii_name)
            popii_tables[popii_name] = PopIITable(
                source_name=popii_name,
                model=config.yields.popii_model,
                m_max=config.yields.popii_m_max,
                fe_scale_ww95=config.yields.fe_scale_ww95,
                z_grid_size=size,
                t_grid_size=config.yields.popii_t_grid_size,
                z_min=config.yields.popii_z_min,
                z_max=config.yields.popii_z_max,
                z_grid=nodes,
                t_min_myr=config.yields.popii_t_min_myr,
                t_max_myr=config.yields.popii_t_max_myr,
                z_floor=config.yields.raiteri_z_floor,
                popii_stochastic=False,
                cache_dir=config.runtime.cache_dir,
                overwrite_cache=config.runtime.overwrite_cache,
            )

        variant = variant_config(config, settings)
        sampling, weights = popiii_variant(popiii_table, settings["popiii"])
        variant = replace(
            variant,
            prior=replace(
                variant.prior,
                popiii_sampling=sampling,
                popiii_mass_energy_weights=weights,
            ),
        )
        library = build_library(
            variant,
            settings["name"],
            n_scenarios,
            seed + index,
            pisn_tables[pisn_name],
            popiii_table,
            enable_pisn=settings["enable_pisn"],
            enable_iii=settings["enable_iii"],
            enable_popii=settings["enable_popii"],
        )
        result = invert_configuration(
            settings["name"],
            settings["category"],
            library,
            popii_tables[popii_name],
            xfe,
            sigma,
            available,
            feh,
            feh_sigma,
        )
        result.settings.update(
            {
                "pisn_table": pisn_name,
                "popii_table": popii_name,
                "popiii_variant": settings["popiii"],
                "category": settings["category"],
            }
        )
        result.summary.update(
            {
                "category": settings["category"],
                "pisn_table": pisn_name,
                "popii_table": popii_name,
                "popiii_variant": settings["popiii"],
            }
        )
        results.append(result)
        if verbose:
            summary = result.summary
            print(
                f"    median logMdil "
                f"{summary['pooled_log10_mdil']['p50']:.2f}  "
                f"support {summary['support_fraction']:.2f}  "
                f"chi2nu {summary['median_best_reduced_chi2']:.2f}",
                flush=True,
            )

    payload = {
        "method": (
            "Prior-predictive likelihood-weighted inversion. For each prior "
            "draw of the event set and each Li+ star, solve the M_dil that "
            "reproduces [Fe/H]; weight by the [X/Fe] likelihood with "
            "configuration-level per-element offsets and a 0.1 dex intrinsic "
            "floor. No prior is placed on M_dil."
        ),
        "mdil_grid": {
            "log10_min": float(np.log10(MDIL_GRID[0])),
            "log10_max": float(np.log10(MDIL_GRID[-1])),
            "n": int(MDIL_GRID.size),
        },
        "selection": {
            "catalogue": "Li+",
            "n_catalogue": int(records.n),
            "n_stars_used": int(star_index.size),
            "feh_range": list(STAR_FEH_RANGE),
            "min_detected_elements": int(MIN_DETECTED_ELEMENTS),
            "ratio_elements": list(RATIO_ELEMENTS),
        },
        "settings": {
            "n_scenarios": int(n_scenarios),
            "seed": int(seed),
            "sigma_intrinsic_dex": SIGMA_INTRINSIC_DEX,
            "support_reduced_chi2": SUPPORT_REDUCED_CHI2,
            "offset_iterations": OFFSET_ITERATIONS,
            "mstar_ii_decoupled": True,
        },
        "configurations": [result.summary for result in results],
    }
    with (outdir / "mdil_inverse.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    _write_tables(results, star_index, feh, outdir)
    _write_figures(results, records, star_index, feh, outdir)
    return payload


def _write_tables(
    results: list[ConfigResult],
    star_index: np.ndarray,
    feh: np.ndarray,
    outdir: Path,
) -> None:
    """Write CSVs for configuration summaries, offsets, and per-star masses."""
    import csv

    with (outdir / "config_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "configuration", "category", "pisn_table", "popii_table",
                "popiii_variant", "n_stars", "log_mdil_p16", "log_mdil_p50",
                "log_mdil_p84", "log_mdil_raw_p50", "log_mdil_prior_p16",
                "log_mdil_prior_p50", "log_mdil_prior_p84",
                "support_fraction", "support_fraction_raw",
                "median_best_reduced_chi2", "median_best_reduced_chi2_raw",
                "offset_rms", "censored_fraction",
            ]
        )
        for result in results:
            summary = result.summary
            writer.writerow(
                [
                    result.name,
                    summary["category"],
                    summary["pisn_table"],
                    summary["popii_table"],
                    summary["popiii_variant"],
                    summary["n_stars"],
                    f"{summary['pooled_log10_mdil']['p16']:.4f}",
                    f"{summary['pooled_log10_mdil']['p50']:.4f}",
                    f"{summary['pooled_log10_mdil']['p84']:.4f}",
                    f"{summary['pooled_log10_mdil_raw']['p50']:.4f}",
                    f"{summary['pooled_log10_mdil_prior']['p16']:.4f}",
                    f"{summary['pooled_log10_mdil_prior']['p50']:.4f}",
                    f"{summary['pooled_log10_mdil_prior']['p84']:.4f}",
                    f"{summary['support_fraction']:.4f}",
                    f"{summary['support_fraction_raw']:.4f}",
                    f"{summary['median_best_reduced_chi2']:.4f}",
                    f"{summary['median_best_reduced_chi2_raw']:.4f}",
                    f"{summary['offset_rms']:.4f}",
                    f"{summary['censored_fraction']:.4f}",
                ]
            )

    with (outdir / "element_offsets.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "element", "offset_dex", "scatter_dex"])
        for result in results:
            for element, offset, scatter in zip(
                RATIO_ELEMENTS, result.offsets, result.scatter
            ):
                writer.writerow(
                    [result.name, element, f"{offset:.4f}", f"{scatter:.4f}"]
                )

    per_star = {result.name: star_medians(result.mdil, result.weights) for result in results}
    with (outdir / "star_mdil.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["star_index", "feh", "configuration", "log_mdil_p16",
             "log_mdil_p50", "log_mdil_p84"]
        )
        for i, star in enumerate(star_index):
            for result in results:
                values = per_star[result.name][i]
                writer.writerow(
                    [
                        int(star),
                        f"{feh[i]:.4f}",
                        result.name,
                        f"{np.log10(values[0]):.4f}" if values[0] > 0 else "",
                        f"{np.log10(values[1]):.4f}" if values[1] > 0 else "",
                        f"{np.log10(values[2]):.4f}" if values[2] > 0 else "",
                    ]
                )


def _write_figures(
    results: list[ConfigResult],
    records,
    star_index: np.ndarray,
    feh: np.ndarray,
    outdir: Path,
) -> None:
    """Write all figures for the inversion."""
    plot_mdil_summary(results, outdir / "mdil_support_by_config.png")

    combined_names = [
        "all", "pisn_only", "popiii_only", "pisn+popiii",
        "pisn+popii", "popiii+popii",
    ]
    plot_mdil_densities(results, combined_names, outdir / "mdil_source_combinations.png")

    for category, title, filename in (
        ("pisn", "PISN table variants", "mdil_pisn_variants.png"),
        ("popiii", "Pop III CCSN sampling variants", "mdil_popiii_variants.png"),
        ("popii", "Pop II table variants", "mdil_popii_variants.png"),
    ):
        plot_category_summary(
            results, category, title, outdir / filename
        )

    plot_offsets(results, outdir / "element_offsets.png")
    plot_support(results, outdir / "configuration_support.png")
    plot_raw_vs_corrected(results, outdir / "mdil_raw_vs_corrected.png")

    main = next(result for result in results if result.name == "all")
    plot_star_mdil(main, feh, outdir / "star_mdil_vs_feh.png")
    plot_event_dependence(main, outdir / "mdil_event_dependence.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="src/potluck/tests/v18")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    payload = run_mdil_inverse(
        n_scenarios=args.n,
        seed=args.seed,
        outdir=args.outdir,
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "n_stars": payload["selection"]["n_stars_used"],
                "configurations": len(payload["configurations"]),
                "all": next(
                    row
                    for row in payload["configurations"]
                    if row["name"] == "all"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
