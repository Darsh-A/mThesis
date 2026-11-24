# potluck

Inverse-problem generator for metal-poor stellar abundance patterns. It draws a
mixed enrichment history, builds the ejecta of each source, applies physical
cuts, and stores noiseless features, labels, and a counterfactual detectability
label for simulation-based inference.

Three sources are implemented: PISN (HW2002 / Takahashi), Pop III CCSN
(Ishigaki18 grid), and Pop II CCSN (Nomoto13 / Limongi18 / WW95). See
[`src/potluck/README.md`](src/potluck/README.md) for the full model description.

## Layout

```text
src/
  potluck/            the generator package
    config.py         bounds and priors
    constants.py      element list and solar mass fractions
    yields/           pisn.py, popiii.py, popii.py, base.py
    chronology.py     lifetimes and the two-clock gates
    events.py         prior draws
    ejecta.py         source matrix, cuts, Z_*
    features.py       log abundances, ratios, labels
    detect.py         counterfactual D and y
    observation.py    Li+ masks, errors, limits, weights
    dataset.py        rejection sampling and array schema
    io.py             npz + JSON provenance
    diagnostics/      result scripts and plots
    validation.py     Eq. 13, internal checks, ablation
    cli.py            command-line interface
    explanations/     step-by-step prose notes
    learn/            workshop notes
    tests/            versioned experiment outputs (v1–v21)
  formulas.py         lifetimes, IMF, abundance ratios
  load_data.py        yield-table and catalogue readers
  salvadori_funcs.py  Salvadori forward-model functions (validation)
  yield_sources.py    Pop II yield-source registry
  utils.py            isotope/element helpers
params.py             data paths and physical constants
tests/potluck/        pytest suite (self-contained, generates its own data)
data/raw/             raw yield grids and catalogues
data/potluck/         generated caches and analysis products
docs/                 design and validation notes
```

## Setup

```bash
uv sync
```

The code imports `params` and `src.*` as top-level modules, so run everything
from the repository root.

## Quick start

```bash
# Write a tunable configuration
uv run python -m src.potluck config my_config.json

# Generate a dataset (default: 10,000 accepted samples)
uv run python -m src.potluck generate --config my_config.json --n 10000

# Inspect and validate
uv run python -m src.potluck diagnose data/potluck/dataset.npz
uv run python -m src.potluck validate data/potluck/dataset.npz
```

## Tests

```bash
uv run pytest
```

The suite builds small datasets in temporary directories, so it does not need
the stored results under `src/potluck/tests/` or `data/potluck/`.

## Data

Most yield tables and catalogues live under `data/raw/` and are tracked. Two
large inputs are **not** tracked and must be placed locally:

- `data/raw/Ishigaki18/` — the raw Pop III grid. `PopIIITable` raises a clear
  error if it is missing.
- `data/raw/allStarLite-dr17-synspec_rev1.fits` — the APOGEE DR17 catalogue
  used by the `diagnostics/apogee_*` scripts.

Large binary artifacts (`*.npz`, `*.png`) and the `data/potluck/` caches are
also gitignored; the JSON/CSV/MD provenance records beside them are tracked.
