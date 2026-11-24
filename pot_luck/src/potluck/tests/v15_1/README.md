# v15_1 — feature importance with the random forest

Feature ranking and subset selection on the Nomoto13 + CK20 pool using the
**random forest** for the whole pipeline (`--model rf`): the impurity (MDI) and
out-of-fold permutation rankings and the selection-curve scorer are all the
random forest. This is the v15 analysis (`src/potluck/diagnostics/feature_importance.py`)
re-run on the CK20 pool with the RF backend.

## Dataset

```
src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
```
50,000 accepted mocks, seed 42, 27 feature elements (26 ratios).

## What changed versus v15

| item | v15 | **v15_1 (this test)** |
|---|---|---|
| Pop II table (`yields.popii_table`) | `Limongi18` | **`Nomoto13_CK20`** |
| Pop II sampler grid (`yields.popii_z_nodes`) | `[3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2]` | **[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2]** |
| selection-curve model | gradient boosting (`gb`) | **random forest (`rf`)** |

The MDI/permutation ranking always uses the random forest; `--model` selects the
backend for the selection curve. `rf` uses the random forest end to end.

### The CK20 edits (temporary Nomoto13 variant)

Implemented in `src/yield_sources.py` as the `Nomoto13_CK20` source.

1. **Hypernova fraction 0.5 for M >= 20 Msun** — the E = 1 row is mixed 50/50
   with the E = 10/20/30 hypernova row at the same mass and metallicity.
2. **Constant 2D-jet offsets (global):** Sc +1.0, Ti +0.45, V +0.3, Co +0.2,
   Zn +0.2 dex.
3. **Explosion-type mass ranges (metadata only):** ECSN 8.8-9, SN II/Ibc 10-30,
   failed SN 30-50, HN 20-50.
4. **Residual mismatches at [Fe/H] ~ -2 (metadata only):** Co 0.3 low, Ni 0.2
   low, Si 0.2 high, Ca 0.1 low, Ti/Sc/V/K underproduced.

## Results (random forest)

Two blocks are ranked: the 26 noiseless mock ratios and the 11 observed Li+
ratios. The choice is the smallest subset within one standard error of the best
AUC.

Noiseless (26 features) — chosen k = 17, AUC 0.9983, Brier 0.0128:

```
Si/Fe, Al/Fe, S/Fe, P/Fe, Ar/Fe, C/Fe, Cu/Fe, Co/Fe, Zn/Fe, Ca/Fe,
O/Fe, Cr/Fe, Mn/Fe, Cl/Fe, Sc/Fe, Na/Fe, Ti/Fe
```

Observed (11 features) — chosen k = 10, AUC 0.9731, Brier 0.0459:

```
Na/Fe, Ca/Fe, Mg/Fe, Ti/Fe, Si/Fe, Cr/Fe, Co/Fe, Ni/Fe, Zn/Fe, Mn/Fe
```

## Contents

- `importance_noiseless.png`, `importance_observed.png` — MDI and permutation
  importance per feature.
- `selection_curve.png` — out-of-fold AUC against the number of features kept.
- `feature_importance.json`, `chosen_features.json` — machine-readable results.

## Regenerate

```bash
D=src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
python -m src.potluck.diagnostics.feature_importance $D \
    --outdir src/potluck/tests/v15_1 --model rf
```
