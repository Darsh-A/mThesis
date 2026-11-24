"""v20: Li+-independent comparison of the potluck yield models.

Everything here uses one prior and one set of physical gates for every yield
table.  M_dil is drawn from a common log-uniform prior and Z* is derived; the
table only changes how the low-Z yields are read, never which samples exist.

Stage 1:
  - common M_dil prior, one z_cr, no table-floor cut,
  - low-Z policies: clamp vs blend (Nomoto), native/clamp for the others,
  - generic element noise, no Li+ masks/errors/weights,
  - direct yield curves at fixed Z*,
  - RF within-table, cross-table and leave-one-table-out transfer,
  - sigma sweep and prior-equality checks.

Stage 2 (optional):
  - apply the Li+ selection/observation layer as a separate application.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..calibration import run_phase_b
from ..config import ModelConfig
from ..constants import (
    ELEMENT_INDEX,
    FE_INDEPENDENT_RATIOS,
    LI_FEATURE_ELEMENTS,
    METALS,
    REFERENCE,
    SOLAR_MASS_FRACTION,
)
from ..dataset import _arrays_from, build_tables, generate_dataset
from ..detect import detect_labels
from ..ejecta import _pisn_events, _popiii_sums, build_ejecta
from ..events import draw_events
from ..features import build_features
from ..io import build_meta, load_dataset, save_dataset
from ..observation import Observation
from ..validation import eq13_check
from ..yields.base import element_vector
from ..yields.popii import z_config_for_table
from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_hw2002
from src.salvadori_funcs import salvadori_Y_X_II, salvadori_yields
from src.yield_sources import get_source

POLICIES: dict[str, dict[str, Any]] = {
    "Limongi18_clamp": {
        "source": "Limongi18",
        "lowest_node": 3.236e-5,
        "policy": "clamp",
    },
    "WW95_native": {
        "source": "WW95",
        "lowest_node": 1.345e-6,
        "policy": "native",
    },
    "Nomoto13_blend": {
        "source": "Nomoto13",
        "lowest_node": 1.0e-3,
        "policy": "blend",
    },
    "Nomoto13_clamp": {
        "source": "Nomoto13_CLAMP",
        "lowest_node": 1.0e-3,
        "policy": "clamp",
    },
}

KEY_ELEMENTS = ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn", "Fe")


def policy_config(
    base: ModelConfig,
    policy: str,
    n_samples: int,
    seed: int,
    outdir: Path,
    mdil_range: tuple[float, float] = (2.0e5, 1.0e9),
) -> ModelConfig:
    """Common v20 configuration, independent of the yield table."""
    source = POLICIES[policy]["source"]
    nodes, size = z_config_for_table(source)
    return replace(
        base,
        prior=replace(
            base.prior,
            mdil_range=mdil_range,
            mdil_slope=0.0,
            z_star_range=None,
            z_cr=1.0e-6,
            z_max=None,
            feh_range=(-5.0, -1.0),
            mstar_ii_range=(1.0e3, 1.0e7),
            mstar_total_max_ratio=0.1,
        ),
        yields=replace(
            base.yields,
            popii_table=source,
            popii_z_nodes=nodes,
            popii_z_grid_size=size,
        ),
        runtime=replace(
            base.runtime,
            n_samples=n_samples,
            chunk_size=max(1000, min(n_samples, 5000)),
            seed=seed,
            max_draw_factor=20,
            output_dir=str(outdir),
        ),
    )


def _generic_observation(
    features: Any,
    ejecta: Any,
    sigma_dex: float,
    rng: np.random.Generator,
) -> Observation:
    """Element-level Gaussian noise, no catalogue masks or weights."""
    log_abundance = features.log_abundance
    noisy = log_abundance + rng.normal(0.0, sigma_dex, log_abundance.shape)
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    ratio_indices = [ELEMENT_INDEX[element] for element in ratio_elements]
    ref_index = ELEMENT_INDEX[REFERENCE]
    ratios = noisy[:, ratio_indices] - noisy[:, [ref_index]]
    n = ratios.shape[0]

    flags = np.ones(ratios.shape, dtype=np.int8)
    sigma = np.full(ratios.shape, sigma_dex, dtype=float)
    mask = np.ones(ratios.shape, dtype=bool)

    independent = np.full((n, len(FE_INDEPENDENT_RATIOS)), np.nan)
    independent_mask = np.zeros_like(independent, dtype=bool)
    independent_sigma = np.full_like(independent, np.nan)
    for column, (numerator, denominator) in enumerate(FE_INDEPENDENT_RATIOS):
        value = (
            noisy[:, ELEMENT_INDEX[numerator]]
            - noisy[:, ELEMENT_INDEX[denominator]]
        )
        independent[:, column] = value
        independent_mask[:, column] = True
        independent_sigma[:, column] = np.sqrt(2.0) * sigma_dex

    return Observation(
        ratios=ratios,
        flags=flags,
        sigma=sigma,
        mask=mask,
        independent_ratios=independent,
        independent_mask=independent_mask,
        independent_sigma=independent_sigma,
        feh_obs=ejecta.feh.copy(),
        feh_err=np.full(n, 0.1),
        logg=np.zeros(n),
        is_giant=np.zeros(n, dtype=bool),
        record_index=np.arange(n, dtype=np.int64),
        weight=np.ones(n),
        feh_true=ejecta.feh.copy(),
        ratio_names=tuple(f"{element}/{REFERENCE}" for element in ratio_elements),
        independent_names=tuple(
            f"{numerator}/{denominator}"
            for numerator, denominator in FE_INDEPENDENT_RATIOS
        ),
    )


def generate_lifree(
    config: ModelConfig,
    n_samples: int,
    seed: int,
    sigma_dex: float,
    policy: str,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate a Li+-independent mock pool."""
    rng = np.random.default_rng(seed)
    pisn, popiii, popii = build_tables(config)
    lowest_node = POLICIES[policy]["lowest_node"]
    target = n_samples
    parts: list[dict[str, np.ndarray]] = []
    ledger = {
        "drawn": 0,
        "r1": 0,
        "r2": 0,
        "r3": 0,
        "r4": 0,
        "accepted": 0,
    }
    max_draw = target * config.runtime.max_draw_factor

    while ledger["accepted"] < target and ledger["drawn"] < max_draw:
        batch = min(config.runtime.chunk_size, max_draw - ledger["drawn"])
        events = draw_events(batch, config, rng, pisn, popiii)
        ejecta = build_ejecta(events, config, pisn, popiii, popii, rng=rng)
        ledger["drawn"] += batch
        ledger["r1"] += int(np.count_nonzero(ejecta.stage == 1))
        ledger["r2"] += int(np.count_nonzero(ejecta.stage == 2))
        ledger["r3"] += int(np.count_nonzero(ejecta.stage == 3))

        keep = (
            ejecta.accepted
            & np.isfinite(ejecta.feh)
            & (ejecta.feh >= -5.0)
            & (ejecta.feh <= -1.0)
        )
        rejected_window = int(
            np.count_nonzero(ejecta.accepted & ~keep)
        )
        ledger["r4"] += rejected_window
        if not np.any(keep):
            continue
        sub = ejecta.subset(keep)
        features = build_features(sub, config)
        observation = _generic_observation(features, sub, sigma_dex, rng)
        detection = detect_labels(
            features, sub, config, observation=observation
        )
        parts.append(
            _arrays_from(sub, features, detection, observation, lowest_node)
        )
        ledger["accepted"] += len(sub)
        if verbose:
            print(
                f"  drawn={ledger['drawn']} accepted={ledger['accepted']}",
                flush=True,
            )

    if ledger["accepted"] < target:
        raise RuntimeError(
            f"only {ledger['accepted']} of {target} accepted; {ledger}"
        )
    arrays = {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
    }
    feature_names = tuple(
        f"{element}/{REFERENCE}"
        for element in config.feature_elements
        if element != REFERENCE
    )
    meta = build_meta(
        config,
        ledger,
        n_drawn=ledger["drawn"],
        n_accepted=ledger["accepted"],
        feature_names=feature_names,
        extra={
            "v20": True,
            "policy": policy,
            "lowest_node": lowest_node,
            "sigma_dex": sigma_dex,
            "feh_window": [-5.0, -1.0],
            "li_free": True,
            "observation_feature_names": tuple(
                f"{element}/{REFERENCE}"
                for element in LI_FEATURE_ELEMENTS
                if element != REFERENCE
            ),
        },
    )
    return arrays, meta


def _abundance_ratio(yields: dict[str, float], element: str) -> float:
    """[X/Fe] from element masses."""
    if element == REFERENCE:
        return 0.0
    x = yields.get(element, 0.0)
    fe = yields.get(REFERENCE, 0.0)
    if x <= 0.0 or fe <= 0.0:
        return float("nan")
    return float(
        np.log10(
            (x / SOLAR_MASS_FRACTION[element])
            / (fe / SOLAR_MASS_FRACTION[REFERENCE])
        )
    )


def yield_curves(outdir: Path) -> dict[str, Any]:
    """Direct IMF-integrated [X/Fe] at fixed Z* and t, no simulation."""
    z_values = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
    t_values = (3.0, 10.0, 20.0, 40.0)
    result: dict[str, Any] = {}
    for policy, settings in POLICIES.items():
        source = get_source(settings["source"])
        result[policy] = {}
        for z_value in z_values:
            data = source.load_for_metallicity(z_value)
            result[policy][f"{z_value:.0e}"] = {}
            for t_myr in t_values:
                turnoff = raiteri_mass_from_lifetime(t_myr * 1.0e6, z_value)
                if turnoff is None:
                    continue
                y_fe = salvadori_Y_X_II(
                    data, REFERENCE, turnoff, sn_input=source, feh=None
                )
                row: dict[str, float] = {}
                for element in KEY_ELEMENTS:
                    if element == REFERENCE:
                        row[element] = 0.0
                        continue
                    y_x = salvadori_Y_X_II(
                        data, element, turnoff, sn_input=source, feh=None
                    )
                    if y_x <= 0.0 or y_fe <= 0.0:
                        row[element] = float("nan")
                    else:
                        row[element] = float(
                            np.log10(
                                (y_x / SOLAR_MASS_FRACTION[element])
                                / (y_fe / SOLAR_MASS_FRACTION[REFERENCE])
                            )
                        )
                result[policy][f"{z_value:.0e}"][f"{t_myr:.0f}"] = row
    with (outdir / "yield_curves.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    # Curves at Z*=1e-5 and 1e-3.
    for z_value, filename in ((1.0e-5, "yield_curves_Z1e-5.png"), (1.0e-3, "yield_curves_Z1e-3.png")):
        fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
        for axis, element in zip(axes.ravel(), KEY_ELEMENTS[:-1]):
            for policy in POLICIES:
                values = []
                times = []
                for t_myr in t_values:
                    row = result[policy].get(f"{z_value:.0e}", {}).get(f"{t_myr:.0f}")
                    if row is None:
                        continue
                    values.append(row[element])
                    times.append(t_myr)
                axis.plot(times, values, marker="o", ms=3, lw=1.3, label=policy)
            axis.set_title(f"[{element}/Fe]")
            axis.grid(alpha=0.25)
        for axis in axes[-1, :]:
            axis.set_xlabel("t [Myr]")
        for axis in axes[:, 0]:
            axis.set_ylabel("[X/Fe]")
        axes[0, 0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"IMF-integrated yields at Z* = {z_value:.0e}", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Spread across policies at t=10 Myr.
    matrix = np.full((len(KEY_ELEMENTS), len(z_values)), np.nan)
    for i, element in enumerate(KEY_ELEMENTS):
        for j, z_value in enumerate(z_values):
            values = [
                result[policy][f"{z_value:.0e}"]["10"][element]
                for policy in POLICIES
                if f"{z_value:.0e}" in result[policy]
                and "10" in result[policy][f"{z_value:.0e}"]
            ]
            values = [v for v in values if np.isfinite(v)]
            if len(values) > 1:
                matrix[i, j] = max(values) - min(values)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    image = axis.imshow(matrix, cmap="magma", aspect="auto")
    axis.set_xticks(np.arange(len(z_values)))
    axis.set_xticklabels([f"{z:.0e}" for z in z_values])
    axis.set_yticks(np.arange(len(KEY_ELEMENTS)))
    axis.set_yticklabels(KEY_ELEMENTS)
    axis.set_xlabel("Z*")
    axis.set_title("Spread across tables at t = 10 Myr [dex]")
    for i in range(len(KEY_ELEMENTS)):
        for j in range(len(z_values)):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="max - min [dex]")
    fig.tight_layout()
    fig.savefig(outdir / "yield_spread.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PISN contrast: pure PISN minus pure Pop II at Z*=1e-5, t=10.
    pisn_entry = min(
        load_hw2002(),
        key=lambda entry: abs(entry["params"]["mass"] - 200.0),
    )
    pisn_vector = element_vector(pisn_entry)
    pisn_ratios = {
        element: _abundance_ratio(
            {e: pisn_vector[ELEMENT_INDEX[e]] for e in KEY_ELEMENTS},
            element,
        )
        for element in KEY_ELEMENTS
    }
    contrast: dict[str, Any] = {"pisn": pisn_ratios, "policies": {}}
    for policy in POLICIES:
        row = result[policy].get("1e-05", {}).get("10", {})
        contrast["policies"][policy] = {
            element: (
                float(pisn_ratios[element] - row[element])
                if np.isfinite(pisn_ratios[element]) and np.isfinite(row.get(element, np.nan))
                else float("nan")
            )
            for element in KEY_ELEMENTS
            if element != REFERENCE
        }
    with (outdir / "pisn_contrast.json").open("w") as handle:
        json.dump(contrast, handle, indent=2)
    return result


def _balance(y: np.ndarray) -> np.ndarray:
    """Class-balanced sample weights."""
    y = np.asarray(y, dtype=int)
    weights = np.zeros(len(y), dtype=float)
    positive = y == 1
    negative = ~positive
    if positive.any():
        weights[positive] = 0.5 / positive.sum()
    if negative.any():
        weights[negative] = 0.5 / negative.sum()
    return weights


def _block(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    return (
        data["obs_features"].astype(float)
        if name == "observed"
        else data["features"].astype(float)
    )


def _impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = train.copy()
    test = test.copy()
    for column in range(train.shape[1]):
        values = train[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if np.any(finite) else 0.0
        train[~finite, column] = fill
        test[~np.isfinite(test[:, column]), column] = fill
    return train, test


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_test: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    return model.predict_proba(x_test)[:, 1]


def _bootstrap_auc(
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        scores.append(
            roc_auc_score(y[index], probability[index], sample_weight=weight[index])
        )
    if not scores:
        return {"low": float("nan"), "high": float("nan")}
    return {
        "low": float(np.percentile(scores, 2.5)),
        "high": float(np.percentile(scores, 97.5)),
    }


def rf_transfer(
    outdir: Path,
    datasets: dict[str, dict[str, np.ndarray]],
    n_boot: int = 100,
    suffix: str = "",
) -> dict[str, Any]:
    """Within-table, cross-table and leave-one-table-out RF transfer."""
    policies = list(datasets)
    result: dict[str, Any] = {"policies": policies, "blocks": {}}

    for block in ("noiseless", "observed"):
        within: dict[str, Any] = {}
        for policy in policies:
            data = datasets[policy]
            x = _block(data, block)
            y = data["y_f_05"].astype(int)
            weight = _balance(y)
            splits = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=0
            ).split(x, y)
            probability = np.zeros(len(y))
            for train, test in splits:
                x_train, x_test = _impute(x[train], x[test])
                probability[test] = _fit_predict(
                    x_train, y[train], weight[train], x_test
                )
            within[policy] = {
                "auc": float(roc_auc_score(y, probability, sample_weight=weight)),
                "brier": float(
                    brier_score_loss(y, probability, sample_weight=weight)
                ),
                "ci": _bootstrap_auc(y, probability, weight, n_boot=n_boot),
                "n": int(len(y)),
            }

        cross_auc = np.full((len(policies), len(policies)), np.nan)
        cross_brier = np.full((len(policies), len(policies)), np.nan)
        cross_ci: dict[str, Any] = {}
        for i, train_policy in enumerate(policies):
            data_train = datasets[train_policy]
            x_train_all = _block(data_train, block)
            y_train = data_train["y_f_05"].astype(int)
            w_train = _balance(y_train)
            for j, test_policy in enumerate(policies):
                data_test = datasets[test_policy]
                x_test_all = _block(data_test, block)
                y_test = data_test["y_f_05"].astype(int)
                w_test = _balance(y_test)
                x_train, x_test = _impute(x_train_all, x_test_all)
                probability = _fit_predict(x_train, y_train, w_train, x_test)
                cross_auc[i, j] = roc_auc_score(
                    y_test, probability, sample_weight=w_test
                )
                cross_brier[i, j] = brier_score_loss(
                    y_test, probability, sample_weight=w_test
                )
                cross_ci[f"{train_policy}->{test_policy}"] = _bootstrap_auc(
                    y_test, probability, w_test, n_boot=n_boot
                )

        loto: dict[str, Any] = {}
        for held_out in (policies if len(policies) > 1 else ()):
            x_train_parts, y_train_parts, w_train_parts = [], [], []
            for policy in policies:
                if policy == held_out:
                    continue
                data = datasets[policy]
                y = data["y_f_05"].astype(int)
                x_train_parts.append(_block(data, block))
                y_train_parts.append(y)
                w_train_parts.append(_balance(y))
            data_test = datasets[held_out]
            x_test = _block(data_test, "observed" if block == "observed" else "noiseless")
            y_test = data_test["y_f_05"].astype(int)
            w_test = _balance(y_test)
            x_train = np.vstack(x_train_parts)
            y_train = np.concatenate(y_train_parts)
            w_train = np.concatenate(w_train_parts)
            x_train, x_test = _impute(x_train, x_test)
            probability = _fit_predict(x_train, y_train, w_train, x_test)
            loto[held_out] = {
                "auc": float(
                    roc_auc_score(y_test, probability, sample_weight=w_test)
                ),
                "brier": float(
                    brier_score_loss(y_test, probability, sample_weight=w_test)
                ),
                "ci": _bootstrap_auc(y_test, probability, w_test, n_boot=n_boot),
            }

        result["blocks"][block] = {
            "within": within,
            "cross_auc": {
                train: dict(zip(policies, row.tolist()))
                for train, row in zip(policies, cross_auc)
            },
            "cross_brier": {
                train: dict(zip(policies, row.tolist()))
                for train, row in zip(policies, cross_brier)
            },
            "cross_ci": cross_ci,
            "loto": loto,
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for axis, matrix, title, cmap, vmin, vmax in (
            (axes[0], cross_auc, "AUC", "viridis", 0.5, 1.0),
            (axes[1], cross_brier, "Brier", "magma", 0.0, 0.3),
        ):
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks(np.arange(len(policies)))
            axis.set_xticklabels(policies, rotation=35, ha="right", fontsize=7)
            axis.set_yticks(np.arange(len(policies)))
            axis.set_yticklabels(policies, fontsize=7)
            axis.set_xlabel("test")
            axis.set_ylabel("train")
            axis.set_title(f"{block}: {title}")
            for i in range(len(policies)):
                for j in range(len(policies)):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] < 0.7 else "black",
                        fontsize=6,
                    )
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            outdir / f"rf_transfer_{block}{suffix}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    with (outdir / f"rf_transfer{suffix}.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_no_floor_transfer(outdir: str | Path) -> dict[str, Any]:
    """Rerun the RF transfer using only mocks above the table floor."""
    outdir = Path(outdir)
    policies = list(POLICIES)
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for policy in policies:
        data, _ = load_dataset(outdir / policy / "dataset.npz")
        mask = ~data["clamped"].astype(bool)
        if np.count_nonzero(mask) < 200:
            continue
        datasets[policy] = {
            key: value[mask] for key, value in data.items()
        }
    if len(datasets) < 2:
        return {"skipped": True, "reason": "fewer than two policies with enough above-floor mocks"}
    return rf_transfer(outdir, datasets, n_boot=100, suffix="_no_floor")


def sigma_sweep(
    datasets: dict[str, dict[str, np.ndarray]],
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
) -> dict[str, Any]:
    """Within-table AUC versus generic element noise."""
    result: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    for policy, data in datasets.items():
        log_abundance = data["log_abundance"].astype(float)
        y = data["y_f_05"].astype(int)
        weight = _balance(y)
        rows = {}
        for sigma in sigmas:
            noisy = log_abundance + rng.normal(0.0, sigma, log_abundance.shape)
            ratios = (
                noisy[:, [ELEMENT_INDEX[e] for e in LI_FEATURE_ELEMENTS if e != REFERENCE]]
                - noisy[:, [ELEMENT_INDEX[REFERENCE]]]
            )
            splits = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=0
            ).split(ratios, y)
            probability = np.zeros(len(y))
            for train, test in splits:
                probability[test] = _fit_predict(
                    ratios[train], y[train], weight[train], ratios[test]
                )
            rows[str(sigma)] = float(
                roc_auc_score(y, probability, sample_weight=weight)
            )
        result[policy] = rows
    return result


def prior_check(
    base: ModelConfig, dataset_meta: dict[str, dict[str, Any]], n: int = 5000
) -> dict[str, Any]:
    """Pre-cut prior distributions and post-cut acceptance."""
    config = policy_config(
        base, "Limongi18_clamp", n, 0, Path(".")
    )
    pisn, popiii, _ = build_tables(config)
    rng = np.random.default_rng(42)
    events = draw_events(n, config, rng, pisn, popiii)
    delta = events.delta_myr
    horizon = delta + events.t_myr
    pisn_vectors, tau = _pisn_events(events, pisn)
    pre = (tau <= delta[:, None]) & (events.pisn_masses > 0.0)
    m_pre_pisn = (pisn_vectors * pre[..., None]).sum(axis=1)
    _, m_pre_iii = _popiii_sums(events, popiii, delta, horizon)
    m_z_pre = (m_pre_pisn + m_pre_iii)[
        :, [ELEMENT_INDEX[e] for e in METALS]
    ].sum(axis=1)
    z_star = np.divide(
        m_z_pre, events.mdil, out=np.zeros(n), where=events.mdil > 0.0
    )
    return {
        "prior": {
            "n_pisn": np.bincount(events.n_pisn, minlength=4).tolist(),
            "n_iii_p16_50_84": np.percentile(events.n_iii, [16, 50, 84]).tolist(),
            "mdil_p16_50_84": np.percentile(events.mdil, [16, 50, 84]).tolist(),
            "m_z_pre_p16_50_84": np.percentile(m_z_pre, [16, 50, 84]).tolist(),
            "z_star_p16_50_84": np.percentile(z_star, [16, 50, 84]).tolist(),
        },
        "post_cut": {
            policy: {"acceptance": meta["acceptance"]}
            for policy, meta in dataset_meta.items()
        },
    }


def quick_transfer(
    datasets: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Fast within-table and leave-one-table-out AUC, class-balanced."""
    policies = list(datasets)
    within: dict[str, float] = {}
    for policy in policies:
        data = datasets[policy]
        x = _block(data, "observed")
        y = data["y_f_05"].astype(int)
        weight = _balance(y)
        splits = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=0
        ).split(x, y)
        probability = np.zeros(len(y))
        for train, test in splits:
            x_train, x_test = _impute(x[train], x[test])
            probability[test] = _fit_predict(
                x_train, y[train], weight[train], x_test
            )
        within[policy] = float(
            roc_auc_score(y, probability, sample_weight=weight)
        )

    loto: dict[str, float] = {}
    for held_out in policies:
        x_train_parts, y_train_parts, w_train_parts = [], [], []
        for policy in policies:
            if policy == held_out:
                continue
            data = datasets[policy]
            y = data["y_f_05"].astype(int)
            x_train_parts.append(_block(data, "observed"))
            y_train_parts.append(y)
            w_train_parts.append(_balance(y))
        data_test = datasets[held_out]
        x_test = _block(data_test, "observed")
        y_test = data_test["y_f_05"].astype(int)
        w_test = _balance(y_test)
        x_train = np.vstack(x_train_parts)
        y_train = np.concatenate(y_train_parts)
        w_train = np.concatenate(w_train_parts)
        x_train, x_test = _impute(x_train, x_test)
        probability = _fit_predict(x_train, y_train, w_train, x_test)
        loto[held_out] = float(
            roc_auc_score(y_test, probability, sample_weight=w_test)
        )
    return within, loto


def run_mdil_sweep(
    outdir: str | Path = "src/potluck/tests/v20/sensitivity",
    n_samples: int = 10000,
    seed: int = 42,
    policies: tuple[str, ...] = tuple(POLICIES),
    priors: dict[str, tuple[float, float]] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Rerun stage 1 with several M_dil priors and compare the metrics."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    priors = priors or {
        "1e5_5e8": (1.0e5, 5.0e8),
        "2e5_1e9": (2.0e5, 1.0e9),
        "1e6_1e9": (1.0e6, 1.0e9),
    }
    base = ModelConfig()
    summary: dict[str, Any] = {"n_samples": n_samples, "seed": seed, "priors": {}}

    for label, mdil_range in priors.items():
        prior_dir = outdir / label
        prior_dir.mkdir(parents=True, exist_ok=True)
        datasets: dict[str, dict[str, np.ndarray]] = {}
        rows: dict[str, Any] = {}
        for policy in policies:
            policy_dir = prior_dir / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            dataset = policy_dir / "dataset.npz"
            config = policy_config(
                base,
                policy,
                n_samples,
                seed,
                policy_dir,
                mdil_range=mdil_range,
            )
            if force or not dataset.exists():
                if verbose:
                    print(
                        f"[{label}/{policy}] generating {n_samples}",
                        flush=True,
                    )
                arrays, meta = generate_lifree(
                    config, n_samples, seed, 0.2, policy, verbose=verbose
                )
                save_dataset(dataset, arrays, meta)
            else:
                arrays, meta = load_dataset(dataset)
            datasets[policy] = arrays
            rows[policy] = {
                "acceptance": float(meta["acceptance"]),
                "zero_pisn_fraction": float(np.mean(arrays["n_pisn"] == 0)),
                "z_star_p16": float(np.percentile(arrays["z_star"], 16)),
                "z_star_p50": float(np.percentile(arrays["z_star"], 50)),
                "z_star_p84": float(np.percentile(arrays["z_star"], 84)),
                "mdil_p16": float(np.percentile(arrays["mdil"], 16)),
                "mdil_p50": float(np.percentile(arrays["mdil"], 50)),
                "mdil_p84": float(np.percentile(arrays["mdil"], 84)),
                "m_z_pre_p50": float(np.percentile(arrays["M_z_pre"], 50)),
                "below_floor_fraction": float(np.mean(arrays["clamped"])),
            }
        within, loto = quick_transfer(datasets)
        for policy in policies:
            rows[policy]["within_auc"] = within[policy]
            rows[policy]["loto_auc"] = loto[policy]
        summary["priors"][label] = {
            "mdil_range": list(mdil_range),
            "policies": rows,
        }

    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    labels = list(summary["priors"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    positions = np.arange(len(labels))

    axis = axes[0, 0]
    for metric, color, marker in (
        ("acceptance", "tab:blue", "o"),
        ("zero_pisn_fraction", "tab:orange", "s"),
    ):
        values = [
            np.mean(
                [
                    summary["priors"][label]["policies"][policy][metric]
                    for policy in policies
                ]
            )
            for label in labels
        ]
        axis.plot(positions, values, marker=marker, color=color, label=metric)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 0.8)
    axis.set_ylabel("fraction")
    axis.set_title("Accepted sample")
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    median_z = [
        np.median(
            [
                summary["priors"][label]["policies"][policy]["z_star_p50"]
                for policy in policies
            ]
        )
        for label in labels
    ]
    below = [
        np.mean(
            [
                summary["priors"][label]["policies"][policy]["below_floor_fraction"]
                for policy in policies
            ]
        )
        for label in labels
    ]
    axis.plot(positions, median_z, "o-", color="tab:green", label="median Z*")
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel(r"median $Z_\star$")
    axis.set_title("Pre-enrichment metallicity")
    axis.grid(alpha=0.25)
    twin = axis.twinx()
    twin.plot(positions, below, "s--", color="tab:red", label="below floor")
    twin.set_ylabel("below-floor fraction", color="tab:red")
    twin.tick_params(axis="y", colors="tab:red")

    for axis, metric, title in (
        (axes[1, 0], "within_auc", "Within-table AUC"),
        (axes[1, 1], "loto_auc", "Leave-one-table-out AUC"),
    ):
        for policy in policies:
            values = [
                summary["priors"][label]["policies"][policy][metric]
                for label in labels
            ]
            axis.plot(positions, values, marker="o", lw=1.3, label=policy)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylim(0.85, 1.0)
        axis.set_ylabel("AUC")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1, 0].legend(fontsize=6, frameon=False, ncol=2)

    fig.suptitle("v20: sensitivity to the M_dil prior", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "mdil_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def run_v20(
    outdir: str | Path = "src/potluck/tests/v20",
    n_samples: int = 20000,
    seed: int = 42,
    sigma_dex: float = 0.2,
    policies: tuple[str, ...] = tuple(POLICIES),
    force: bool = False,
    verbose: bool = True,
    application: bool = False,
) -> dict[str, Any]:
    """Run stage 1 and the optional Li+ application."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = ModelConfig()
    manifest: dict[str, Any] = {
        "n_samples": n_samples,
        "seed": seed,
        "sigma_dex": sigma_dex,
        "policies": list(policies),
        "gates": {
            "mdil_range": [2.0e5, 1.0e9],
            "z_cr": 1.0e-6,
            "feh_window": [-5.0, -1.0],
            "mstar_ii_range": [1.0e3, 1.0e7],
            "mstar_total_max_ratio": 0.1,
        },
        "results": {},
    }
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        policy_dir = outdir / policy
        policy_dir.mkdir(parents=True, exist_ok=True)
        dataset = policy_dir / "dataset.npz"
        config = policy_config(base, policy, n_samples, seed, policy_dir)
        if force or not dataset.exists():
            if verbose:
                print(f"[{policy}] generating {n_samples} Li+-free mocks", flush=True)
            arrays, meta = generate_lifree(
                config, n_samples, seed, sigma_dex, policy, verbose=verbose
            )
            save_dataset(dataset, arrays, meta)
        else:
            arrays, meta = load_dataset(dataset)
        datasets[policy] = arrays
        manifest["results"][policy] = {
            "dataset": str(dataset),
            "source": POLICIES[policy]["source"],
            "policy": POLICIES[policy]["policy"],
            "lowest_node": POLICIES[policy]["lowest_node"],
            "n_accepted": int(meta["n_accepted"]),
            "n_drawn": int(meta["n_drawn"]),
            "acceptance": float(meta["acceptance"]),
            "ledger": meta["ledger"],
            "z_below_floor_fraction": float(np.mean(arrays["clamped"])),
        }

    if verbose:
        print("direct yield curves", flush=True)
    yield_curves(outdir)
    if verbose:
        print("RF transfer", flush=True)
    transfer = rf_transfer(outdir, datasets)
    if verbose:
        print("sigma sweep", flush=True)
    sweep = sigma_sweep(datasets)
    with (outdir / "sigma_sweep.json").open("w") as handle:
        json.dump(sweep, handle, indent=2)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for policy, rows in sweep.items():
        axis.plot(
            [float(k) for k in rows],
            list(rows.values()),
            marker="o",
            lw=1.4,
            label=policy,
        )
    axis.set_xlabel(r"generic element noise $\sigma$ [dex]")
    axis.set_ylabel("balanced within-table AUC")
    axis.set_title("Noise sensitivity")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "sigma_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    prior = prior_check(
        base,
        {
            policy: {"acceptance": manifest["results"][policy]["acceptance"]}
            for policy in policies
        },
        n=5000,
    )
    for policy in policies:
        prior["post_cut"][policy]["zero_pisn_fraction"] = float(
            np.mean(datasets[policy]["n_pisn"] == 0)
        )
        prior["post_cut"][policy]["m_z_pre_p16_50_84"] = np.percentile(
            datasets[policy]["M_z_pre"], [16, 50, 84]
        ).tolist()
    with (outdir / "prior_check.json").open("w") as handle:
        json.dump(prior, handle, indent=2)

    manifest["rf_transfer"] = transfer
    manifest["sigma_sweep"] = sweep
    manifest["prior_check"] = prior

    if application:
        if verbose:
            print("Li+ application", flush=True)
        application_results = {}
        for policy in policies:
            app_dir = outdir / "application" / policy
            app_dir.mkdir(parents=True, exist_ok=True)
            app_dataset = app_dir / "dataset.npz"
            config = policy_config(base, policy, 10000, seed, app_dir)
            config = replace(
                config,
                prior=replace(config.prior, feh_range=(-3.8, -1.7)),
            )
            if force or not app_dataset.exists():
                arrays, meta = generate_dataset(config, verbose=verbose)
                save_dataset(app_dataset, arrays, meta)
            application_results[policy] = run_phase_b(
                app_dataset, app_dir / "li_validation", model="rf", n_boot=50
            )
        manifest["application"] = application_results

    with (outdir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def run_application_only(
    outdir: str | Path,
    n_samples: int = 10000,
    seed: int = 42,
    policies: tuple[str, ...] = tuple(POLICIES),
    force: bool = False,
) -> dict[str, Any]:
    """Apply the Li+ selection/observation layer to the v20 policies."""
    outdir = Path(outdir)
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("application", {})
    base = ModelConfig()
    for policy in policies:
        app_dir = outdir / "application" / policy
        app_dir.mkdir(parents=True, exist_ok=True)
        dataset = app_dir / "dataset.npz"
        config = policy_config(base, policy, n_samples, seed, app_dir)
        config = replace(
            config,
            prior=replace(config.prior, feh_range=(-3.8, -1.7)),
        )
        if force or not dataset.exists():
            arrays, meta = generate_dataset(config, verbose=False)
            save_dataset(dataset, arrays, meta)
        manifest["application"][policy] = run_phase_b(
            dataset, app_dir / "li_validation", model="rf", n_boot=50
        )
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest["application"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="src/potluck/tests/v20")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--application", action="store_true")
    parser.add_argument("--no-floor-transfer", action="store_true")
    parser.add_argument("--application-only", action="store_true")
    parser.add_argument("--mdil-sweep", action="store_true")
    args = parser.parse_args(argv)
    if args.mdil_sweep:
        policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
        summary = run_mdil_sweep(
            args.outdir + "/sensitivity",
            n_samples=args.n,
            seed=args.seed,
            policies=policies,
            force=args.force,
            verbose=not args.quiet,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.no_floor_transfer:
        print(json.dumps(run_no_floor_transfer(args.outdir), indent=2))
        return 0
    if args.application_only:
        policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
        print(
            json.dumps(
                run_application_only(
                    args.outdir, n_samples=10000, seed=args.seed, policies=policies
                ),
                indent=2,
            )
        )
        return 0
    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    manifest = run_v20(
        outdir=args.outdir,
        n_samples=args.n,
        seed=args.seed,
        sigma_dex=args.sigma,
        policies=policies,
        force=args.force,
        verbose=not args.quiet,
        application=args.application,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
