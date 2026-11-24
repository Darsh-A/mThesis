# v14_2 — per-element contours on Nomoto13 + CK20 with the revised M_dil window

One `[X/Fe]` histogram/contour panel per element plus the `[Mg/Si]` panel,
compared against the Li+ catalogue (`src/potluck/diagnostics/element_contours.py`).

This is the v14 / v14_1 figure set regenerated on the CK20 Pop II yield set with
the **new default dilution-mass window** `M_dil = 1e5--5e8 Msun`. The only
change versus `v14_1` is the `M_dil` prior.

## Dataset

```
src/potluck/tests/v5/nomoto13_ck20_50k_mdil1e5_5e8_all_elements.npz
```

50,000 accepted mocks, seed 42, 27 feature elements (26 ratios), acceptance
0.385. Realized `M_dil`: p16 = `1.0e6`, median = `9.8e6`, p84 = `7.3e7 Msun`.

## What changed versus v14_1

| parameter | v14_1 | **v14_2 (this test)** |
|---|---|---|
| `prior.mdil_range` | `(2.0e5, 1.0e9)` | **(1.0e5, 5.0e8)** |
| Pop II table | `Nomoto13_CK20` | unchanged |
| `popii_z_nodes` | `[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2]` | unchanged |
| seed, n | 42, 50,000 | unchanged |

The default `PriorConfig.mdil_range` in `src/potluck/config.py` is now
`(1.0e5, 5.0e8)`.

## Contents

- `{Element}_Fe.png` for all 26 ratio elements: Al, Ar, C, Ca, Cl, Co, Cr, Cu,
  F, Ga, Ge, K, Mg, Mn, N, Na, Ne, Ni, O, P, S, Sc, Si, Ti, V, Zn.
- `Mg_Si.png` ratio panel.
- `element_contours.json` summary.

Bin counts run from 439 (Ni) to 811 (V); the `Mg/Si` panel has 596 bins. The
median PISN metal-mass fraction is 0.043.

## Coverage caveat

The same Nomoto13_CK20 limitation as `v7_2` applies: 99.6% of mocks have
`z_star < 1e-3`, the lowest **nonzero** CK20 node, and 77.9% have
`z_star < 1e-4`. They are not clamped — the sampler grid reaches `1e-12` — but
they are evaluated through the linear `Z=0 <-> Z=1e-3` blend, so the Pop II
ejecta are mostly the `Z=0` CK20 row. The contours therefore show the CK20
offsets, not the CK20 metallicity trend, for the bulk of the sample. See
`v7_2/README.md` for the full numbers.

## Regenerate

```bash
D=src/potluck/tests/v5/nomoto13_ck20_50k_mdil1e5_5e8_all_elements.npz
python -m src.potluck.diagnostics.element_contours $D --outdir src/potluck/tests/v14_2
```

Dataset generation used the same CK20 configuration as `v7_2`:

```json
{
  "prior": {"mdil_range": [100000.0, 500000000.0]},
  "yields": {
    "popii_table": "Nomoto13_CK20",
    "popii_z_nodes": [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2],
    "popii_z_grid_size": 9
  },
  "runtime": {"n_samples": 50000, "seed": 42}
}
```
