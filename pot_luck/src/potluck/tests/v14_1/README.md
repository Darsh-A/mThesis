# v14_1 — per-element contours on the Nomoto13 + CK20 pool

One [X/Fe] histogram/contour panel per element plus the Mg/Si ratio panel,
compared against the Li+ catalogue (`src/potluck/diagnostics/element_contours.py`).
This is the v14 figure set (compare `src/potluck/tests/v14/`) with only the Pop
II table changed.

## Dataset

```
src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
```
50,000 accepted mocks, seed 42, 27 feature elements (26 ratios).

## What changed versus v14

| parameter | v14 (Limongi18 pool) | **v14_1 (this test)** |
|---|---|---|
| Pop II table (`yields.popii_table`) | `Limongi18` | **`Nomoto13_CK20`** |
| Pop II sampler grid (`yields.popii_z_nodes`) | `[3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2]` | **[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2]** |
| everything else | unchanged | unchanged |

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

The elements most affected by the edits are Sc, Ti, V, Co and Zn; their panels
gain the constant offset on top of the 50/50 hypernova mix for M >= 20.

## Contents

- `{Element}_Fe.png` for all 26 feature elements (Al, Ar, C, Ca, Cl, Co, Cr, Cu,
  F, Ga, Ge, K, Mg, Mn, N, Na, Ne, Ni, O, P, S, Sc, Si, Ti, V, Zn).
- `Mg_Si.png` ratio panel.
- `element_contours.json` summary (per-element bin counts and correlations).

Bin counts run 431 (Ni) to 860 (V); the Mg/Si panel has 593 bins.

## Regenerate

```bash
D=src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
python -m src.potluck.diagnostics.element_contours $D --outdir src/potluck/tests/v14_1
```
