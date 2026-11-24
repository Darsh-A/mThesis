"""
Add a new source by writing its extract/select/mass_from_lifetime hooks and
registering it in SOURCES
"""

import functools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import _combine_elements, _apply_radioactive_decay, limongi_feh_continuous, limongi_mass_from_lifetime, salvadori_select_limongi_feh
from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_limongi18, load_nomoto13, load_ww95
from params import ww95_1Z, ww95_01Z, ww95_001Z, ww95_00001Z, Z_SUN

"""
Three main hooks are needed to define a new source:
- extract(entry: dict) -> dict: given a yield entry, return a dictionary of element yields (after any necessary processing, e.g. radioactive decay)
- select(entry: dict, ctx: dict) -> bool: given a yield entry and a context dictionary, return True if the entry should be used for the current calculation. 
    The context dictionary can contain any relevant information, such as metallicity, model, etc.
- mass_from_lifetime(lifetime: float, ctx: dict)

Other params scalars like: name, m_max, load_for_metallicity can be set as needed.
"""

@dataclass
class YieldSource:
    name: str
    m_max: float
    extract: Callable[[dict], dict] = None
    select: Callable[[dict, dict], bool] = None
    # mass_from_lifetime may mutate ctx to record derived selection keys
    # (e.g. feh) it needed to compute the mass, so select() downstream
    # filters on the same grid point.
    mass_from_lifetime: Callable[[float, dict], Optional[float]] = None
    load_for_metallicity: Optional[Callable[[float], list[dict]]] = None
    grid_label: Optional[Callable[[float], object]] = None


def salvadori_select_ww95_model(Z_rel: float) -> str:
    """Select the nearest WW95 metallicity grid.

    Z_rel = Z_star / Z_sun.
    """
    grids = {
        1.0: ww95_1Z,
        0.1: ww95_01Z,
        0.01: ww95_001Z,
        1e-4: ww95_00001Z,
    }
    return grids[min(grids, key=lambda z: abs(np.log10(Z_rel) - np.log10(z)))]

def _ww95_grid_key(Z_rel: float) -> float:
    grids = (1.0, 0.1, 0.01, 1e-4)
    return min(grids, key=lambda z: abs(np.log10(Z_rel) - np.log10(z)))

def _ww95_grid_label(Z_star: float) -> float:
    return _ww95_grid_key(Z_star)


def _ww95_extract(entry: dict) -> dict:
    yields = _combine_elements(_apply_radioactive_decay(entry["yields"]))
    if "Fe" in yields:
        yields["Fe"] *= 0.5
    return yields


def _ww95_select(entry: dict, ctx: dict) -> bool:
    return entry["params"]["model"] == ctx.get("model", "A")


def _ww95_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    return raiteri_mass_from_lifetime(lifetime=lifetime, Z=ctx["Z_star"])


@functools.lru_cache(maxsize=None)
def _ww95_load_for_metallicity(Z: float) -> list[dict]:
    """Load WW95 yields at absolute metallicity Z (mass fraction).

    Keeps the same contract as the Limongi18/Nomoto13 load_for_metallicity
    hooks (absolute Z); the WW95 grid selector itself works in Z/Z_sun.
    """
    return load_ww95(salvadori_select_ww95_model(Z / Z_SUN))


WW95 = YieldSource(
    name="WW95",
    m_max=40.0,
    extract=_ww95_extract,
    select=_ww95_select,
    mass_from_lifetime=_ww95_mass_from_lifetime,
    load_for_metallicity=_ww95_load_for_metallicity,
    grid_label=_ww95_grid_label,
)


def _limongi_extract(entry: dict) -> dict:
    return entry["yields"]


def _limongi_select_for(velocity: int):
    """Return a selection hook for one Limongi18 rotation velocity."""

    def select(entry: dict, ctx: dict) -> bool:
        feh = ctx.get("feh")
        params = entry["params"]
        return params["velocity"] == velocity and (
            feh is None or params["feh"] == feh
        )

    return select


def _make_limongi_source(name: str, velocity: int) -> YieldSource:
    """Build one Limongi18 source at a fixed rotation velocity."""

    @functools.lru_cache(maxsize=None)
    def load_for_metallicity_fe(feh: int) -> list[dict]:
        all_entries = load_limongi18()
        return [
            e
            for e in all_entries
            if e["params"]["feh"] == feh
            and e["params"]["velocity"] == velocity
        ]

    @functools.lru_cache(maxsize=None)
    def entries_by_feh(feh: int) -> dict[int, dict]:
        return {e["params"]["mass"]: e for e in load_for_metallicity_fe(feh)}

    @functools.lru_cache(maxsize=None)
    def load_for_metallicity_cont(feh: float) -> list[dict]:
        """LC18 yields linearly interpolated to a continuous [Fe/H]."""
        feh = float(np.clip(feh, -3.0, 0.0))
        lo = max(f for f in (-3, -2, -1, 0) if f <= feh)
        hi = min(f for f in (-3, -2, -1, 0) if f >= feh)
        if lo == hi:
            return load_for_metallicity_fe(lo)

        w = (feh - lo) / (hi - lo)
        lo_by_mass = entries_by_feh(lo)
        hi_by_mass = entries_by_feh(hi)

        out = []
        for mass in sorted(lo_by_mass):
            lo_e, hi_e = lo_by_mass[mass], hi_by_mass[mass]
            elems = set(lo_e["yields"]) | set(hi_e["yields"])
            yields = {
                el: (1.0 - w) * lo_e["yields"].get(el, 0.0)
                + w * hi_e["yields"].get(el, 0.0)
                for el in elems
            }
            out.append({
                "label": f"M={mass} v={velocity} [Fe/H]={feh:.4f}",
                "params": {"mass": mass, "velocity": velocity, "feh": feh},
                "yields": yields,
            })
        return out

    def load_for_metallicity(Z_star: float) -> list[dict]:
        """Load LC18 yields interpolated to the continuous [Fe/H] of Z_star."""
        return load_for_metallicity_cont(limongi_feh_continuous(Z_star / Z_SUN))

    def mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
        feh = salvadori_select_limongi_feh(ctx["Z_star"] / Z_SUN)
        ctx["feh"] = feh
        return limongi_mass_from_lifetime(lifetime, feh=feh, velocity=velocity)

    def grid_label(Z_star: float) -> float:
        return limongi_feh_continuous(Z_star / Z_SUN)

    return YieldSource(
        name=name,
        m_max=120.0,
        extract=_limongi_extract,
        select=_limongi_select_for(velocity),
        mass_from_lifetime=mass_from_lifetime,
        load_for_metallicity=load_for_metallicity,
        grid_label=grid_label,
    )


LIMONGI18 = _make_limongi_source("Limongi18", 0)
LIMONGI18_V150 = _make_limongi_source("Limongi18_V150", 150)
LIMONGI18_V300 = _make_limongi_source("Limongi18_V300", 300)


# Nomoto13 CCSN metallicity grid (mass fraction), energy=1 models only.
NOMOTO13_Z_NODES = (0.0, 0.001, 0.004, 0.008, 0.02, 0.05)
_NOMOTO_Z_GRID = NOMOTO13_Z_NODES
_NOMOTO_Z_POSITIVE = tuple(z for z in _NOMOTO_Z_GRID if z > 0.0)


def _nomoto_extract(entry: dict) -> dict:
    """Nomoto13 yields are per-isotope and already post-decay, so just fold
    isotopes into elements (no decay routing needed)."""
    return _combine_elements(entry["yields"])

def _nomoto_select(entry: dict, ctx: dict) -> bool:
    p = entry["params"]
    return p["energy"] == 1.0

def _nomoto_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    return raiteri_mass_from_lifetime(lifetime=lifetime, Z=ctx["Z_star"])


@functools.lru_cache(maxsize=None)
def _nomoto_entries_by_z(z: float) -> list[dict]:
    return [e for e in load_nomoto13() if e["params"]["Z"] == z and e["params"]["energy"] == 1.0]


# --- CK20 edits (temporary variant of the Nomoto13 table) --------------------
# Hypernova fraction applied to progenitors at or above 20 Msun, and the
# constant 2D-jet dex offsets.  The explosion-type mass ranges and the residual
# mismatches are recorded for provenance only; they do not change the sampler.
NOMOTO_CK20_HN_FRACTION: float = 0.5
NOMOTO_CK20_HN_MIN_MASS: float = 20.0
NOMOTO_CK20_OFFSETS: dict[str, float] = {
    "Sc": 1.0,
    "Ti": 0.45,
    "V": 0.3,
    "Co": 0.2,
    "Zn": 0.2,
}
NOMOTO_CK20_EXPLOSION_MASS_RANGES: dict[str, tuple[float, float]] = {
    "ECSN": (8.8, 9.0),
    "SN_II_Ibc": (10.0, 30.0),
    "failed_SN": (30.0, 50.0),
    "HN": (20.0, 50.0),
}
NOMOTO_CK20_RESIDUAL_MISMATCH_DEX: dict[str, float] = {
    "Co": -0.3,
    "Ni": -0.2,
    "Si": 0.2,
    "Ca": -0.1,
}


@functools.lru_cache(maxsize=None)
def _nomoto_hn_entries_by_z(z: float) -> list[dict]:
    """Nomoto13 hypernova rows (explosion energy above 1 B) at one Z."""
    return [
        e for e in load_nomoto13() if e["params"]["Z"] == z and e["params"]["energy"] > 1.0
    ]


@functools.lru_cache(maxsize=None)
def _nomoto_ck20_entries_by_z(z: float) -> list[dict]:
    """Nomoto13 rows with the CK20 edits folded to element yields.

    Two changes are applied at every tabulated metallicity: a 50/50 mix of the
    normal (E=1) and hypernova yields for progenitors at or above 20 Msun, and
    the constant 2D-jet dex offsets for Sc, Ti, V, Co and Zn.
    """
    hypernovae = {e["params"]["mass"]: e for e in _nomoto_hn_entries_by_z(z)}
    out = []
    for entry in _nomoto_entries_by_z(z):
        mass = float(entry["params"]["mass"])
        yields = _combine_elements(entry["yields"])
        if mass >= NOMOTO_CK20_HN_MIN_MASS and mass in hypernovae:
            mix = _combine_elements(hypernovae[mass]["yields"])
            fraction = NOMOTO_CK20_HN_FRACTION
            yields = {
                el: (1.0 - fraction) * yields.get(el, 0.0) + fraction * mix.get(el, 0.0)
                for el in set(yields) | set(mix)
            }
        for element, dex in NOMOTO_CK20_OFFSETS.items():
            if element in yields:
                yields[element] *= 10.0 ** dex
        out.append({
            "label": f"{entry['label']} CK20",
            "params": dict(entry["params"]),
            "yields": yields,
        })
    return out


@functools.lru_cache(maxsize=None)
def _nomoto_ck20_load_for_metallicity(Z_star: float) -> list[dict]:
    """CK20-edited Nomoto13 yields interpolated to a continuous Z."""
    return _nomoto_interpolate(Z_star, _nomoto_ck20_entries_by_z)


def _nomoto_clamp_load_for_metallicity(Z_star: float) -> list[dict]:
    """Nomoto13 yields clamped to the lowest nonzero node below 1e-3."""
    return _nomoto_interpolate(
        Z_star, _nomoto_entries_by_z, blend_below_floor=False
    )


def _nomoto_ck20_clamp_load_for_metallicity(Z_star: float) -> list[dict]:
    """CK20 yields clamped to the lowest nonzero node below 1e-3."""
    return _nomoto_interpolate(
        Z_star, _nomoto_ck20_entries_by_z, blend_below_floor=False
    )


@functools.lru_cache(maxsize=None)
def _nomoto_load_for_metallicity(Z_star: float) -> list[dict]:
    """Nomoto13 (energy=1) yields interpolated to a continuous Z."""
    return _nomoto_interpolate(Z_star, _nomoto_entries_by_z)


def _nomoto_interpolate(
    Z_star: float, entries_by_z, blend_below_floor: bool = True
) -> list[dict]:
    """Interpolate an energy-selected Nomoto13 grid to a continuous Z.

    Z = 0 is tabulated, so the segment between it and the first positive node
    is linear in Z; every other segment is linear in log Z.  Below the first
    positive node the caller can either blend toward Z=0 or clamp to the
    lowest nonzero node.
    """
    if Z_star <= 0.0:
        return entries_by_z(0.0)

    first = _NOMOTO_Z_POSITIVE[0]
    if Z_star < first:
        if not blend_below_floor:
            return entries_by_z(first)
        return _nomoto_blend(entries_by_z(0.0), entries_by_z(first), Z_star / first, Z_star)

    log_grid = np.log10(_NOMOTO_Z_POSITIVE)
    logZ = float(np.clip(np.log10(Z_star), log_grid[0], log_grid[-1]))
    idx = int(np.searchsorted(log_grid, logZ))
    if idx == 0:
        return entries_by_z(first)
    if idx >= len(log_grid):
        return entries_by_z(_NOMOTO_Z_POSITIVE[-1])

    lo_z, hi_z = _NOMOTO_Z_POSITIVE[idx - 1], _NOMOTO_Z_POSITIVE[idx]
    w = (logZ - log_grid[idx - 1]) / (log_grid[idx] - log_grid[idx - 1])
    return _nomoto_blend(entries_by_z(lo_z), entries_by_z(hi_z), w, Z_star)


def _nomoto_blend(
    lo_entries: list[dict], hi_entries: list[dict], w: float, Z_star: float
) -> list[dict]:
    """Blend two tables at weight w towards ``hi_entries``."""
    lo_by_mass = {e["params"]["mass"]: e for e in lo_entries}
    hi_by_mass = {e["params"]["mass"]: e for e in hi_entries}

    out = []
    for mass in sorted(set(lo_by_mass) & set(hi_by_mass)):
        lo_e, hi_e = lo_by_mass[mass], hi_by_mass[mass]
        elems = set(lo_e["yields"]) | set(hi_e["yields"])
        yields = {
            el: (1.0 - w) * lo_e["yields"].get(el, 0.0) + w * hi_e["yields"].get(el, 0.0)
            for el in elems
        }
        out.append({
            "label": f"Z={Z_star:g} M={mass:g} E=1",
            "params": {
                "Z": float(Z_star),
                "mass": mass,
                "energy": 1.0,
                "Mrem": (1.0 - w) * lo_e["params"].get("Mrem", 0.0) + w * hi_e["params"].get("Mrem", 0.0),
            },
            "yields": yields,
        })
    return out


def _nomoto_grid_label(Z_star: float) -> float:
    """log10 of a tabulated metallicity; metal-free maps to -inf."""
    if Z_star <= 0.0:
        return float("-inf")
    return float(
        np.clip(
            np.log10(Z_star),
            np.log10(_NOMOTO_Z_POSITIVE[0]),
            np.log10(_NOMOTO_Z_GRID[-1]),
        )
    )


NOMOTO13 = YieldSource(
    name="Nomoto13",
    m_max=40.0, 
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=_nomoto_mass_from_lifetime,
    load_for_metallicity=_nomoto_load_for_metallicity,
    grid_label=_nomoto_grid_label,
)

# Temporary CK20 variant: Nomoto13 with the 50/50 hypernova mix at M >= 20 and
# the constant 2D-jet offsets.  Same lifetime/mass hooks as Nomoto13.
NOMOTO13_CK20 = YieldSource(
    name="Nomoto13_CK20",
    m_max=40.0,
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=_nomoto_mass_from_lifetime,
    load_for_metallicity=_nomoto_ck20_load_for_metallicity,
    grid_label=_nomoto_grid_label,
)

NOMOTO13_CLAMP = YieldSource(
    name="Nomoto13_CLAMP",
    m_max=40.0,
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=_nomoto_mass_from_lifetime,
    load_for_metallicity=_nomoto_clamp_load_for_metallicity,
    grid_label=_nomoto_grid_label,
)

NOMOTO13_CK20_CLAMP = YieldSource(
    name="Nomoto13_CK20_CLAMP",
    m_max=40.0,
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=_nomoto_mass_from_lifetime,
    load_for_metallicity=_nomoto_ck20_clamp_load_for_metallicity,
    grid_label=_nomoto_grid_label,
)

SOURCES = {
    "WW95": WW95,
    "Limongi18": LIMONGI18,
    "Limongi18_V150": LIMONGI18_V150,
    "Limongi18_V300": LIMONGI18_V300,
    "Nomoto13": NOMOTO13,
    "Nomoto13_CK20": NOMOTO13_CK20,
    "Nomoto13_CLAMP": NOMOTO13_CLAMP,
    "Nomoto13_CK20_CLAMP": NOMOTO13_CK20_CLAMP,
}


def get_source(name: str) -> YieldSource:
    if name not in SOURCES:
        raise ValueError(f"Unrecognized sn_input: {name!r}")
    return SOURCES[name]