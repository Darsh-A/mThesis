# 10 — Pop II yield table and the low-metallicity clamp

This note records the `[Fe/H]` mapping bug in the Limongi18 branch, the fix, and
the v9 same-seed comparison between the two Pop II tables. It does not change
the default table; that remains a decision for the user.

## 1. The mapping bug

Limongi18 publishes four metallicities, quoted as `[Fe/H] = 0, -1, -2, -3` with

| [Fe/H] | Z |
|---|---|
| 0 | 1.345e-2 (solar) |
| -1 | 3.236e-3 |
| -2 | 3.236e-4 |
| -3 | 3.236e-5 |

For `[Fe/H] <= -1` these follow `Z = 3.236e-2 * 10^[Fe/H]`, not
`Z = Z_sun * 10^[Fe/H]`. The low-Z Limongi models are alpha-enhanced, so their
Z is 2.4x larger than a solar-scaled conversion would give.

The old code used `[Fe/H] = log10(Z / Z_sun)` (`limongi_feh_continuous` and
`salvadori_select_limongi_feh`). At fixed physical Z that selects a Limongi
model 2.4x too metal-rich, i.e. a **0.381 dex** error in the assigned yield
pattern. Salv's replication inherits the same shortcut.

## 2. The fix

`src/utils.py` now defines `LIMONGI_Z_SOLAR = 0.01345`,
`LIMONGI_Z_ALPHA = 0.03236`, `LIMONGI_FEH_MIN = -3.0` and `limongi_feh_from_z`,
which takes **absolute** Z and is piecewise:

- `Z <= 3.236e-3`: `[Fe/H] = clip(log10(Z / 0.03236), -3, -1)`
- `3.236e-3 < Z < 1.345e-2`: linear in `log10 Z` from -1 to 0
- `Z >= 1.345e-2`: 0

`limongi_feh_continuous(Z_rel)` is kept as a wrapper
`limongi_feh_from_z(Z_rel * Z_sun)`, so the three call sites in
`src/yield_sources.py` did not need to change.

Spot checks: 1e-6 → -3.000, 3.236e-5 → -3.000, 1e-4 → -2.510,
3.236e-4 → -2.000, 1e-3 → -1.510, 3.236e-3 → -1.000, 7e-3 → -0.458,
1.345e-2 → 0.000.

`PopIITable` also gained an explicit `z_grid` argument (with the node list
included in the cache hash), and `YieldConfig` gained `popii_z_nodes`, so the
Limongi branch is built on its four **real** nodes instead of 16 interpolated
ones.

## 3. Why Nomoto13 sits so high

Nomoto13 has five Z nodes: 1e-3, 4e-3, 8e-3, 2e-2, 5e-2. Its floor is
`Z = 1e-3`, which is `[Fe/H] = -1.13`. There are simply no Nomoto models below
that, so every mock with `z_star < 1e-3` is clamped to the 1e-3 yield pattern.

Limongi18 reaches `[Fe/H] = -3` (`Z = 3.236e-5`), 1.5 dex lower. This is a
coverage difference, not a disagreement about the definition of Z.

## 4. v9 comparison

Same seed (42), 20,000 accepted mocks per table, `src/potluck/tests/v9/`.

| quantity | Nomoto13 | Limongi18 |
|---|---|---|
| `[Fe/H]` median [p16, p84] | -2.614 [-3.327, -1.998] | -2.637 [-3.355, -2.025] |
| `z_star` median [p16, p84] | 2.17e-5 [3.62e-6, 1.32e-4] | 2.31e-5 [3.88e-6, 1.34e-4] |
| `f_Z^P` median | 0.0397 | 0.0620 |
| `f_Z^P >= 0.5` | 0.1804 | 0.2340 |
| table floor | 1.0e-3 | 3.236e-5 |
| **clamped fraction** | **0.99965** | **0.56835** |

Median `[X/Fe]` (Nomoto → Limongi, shift):

| element | Nomoto | Limongi | shift |
|---|---|---|---|
| C | -0.250 | -0.110 | +0.141 |
| N | -1.461 | -2.436 | **-0.975** |
| O | +0.575 | +0.425 | -0.150 |
| Na | -0.358 | -0.724 | -0.366 |
| Mg | +0.350 | +0.136 | -0.214 |
| Al | -0.196 | -0.505 | -0.309 |
| Si | +0.556 | +0.425 | -0.131 |
| Ca | +0.316 | +0.262 | -0.054 |
| Ti | -0.176 | -0.284 | -0.108 |
| Cr | +0.196 | -0.005 | -0.202 |
| Mn | -0.444 | -0.665 | -0.222 |
| Co | -0.682 | -0.643 | +0.039 |
| Ni | -0.289 | -0.137 | +0.152 |
| Zn | -0.159 | -0.266 | -0.107 |

A classifier trained only on the noiseless features separates the two pools
with a 5-fold AUC of **0.963** (8,000 mocks per pool). The tables are not
interchangeable: the same events produce different abundance patterns.

## 5. Interpretation

1. `[Fe/H]` is unaffected by the choice of table: the medians differ by
   0.02 dex because `[Fe/H]` is fixed by the Fe mass and the dilution mass.
   Only the `[X/Fe]` pattern changes.
2. The clamp is reduced from essentially all mocks (99.97%) to 57%, but not
   removed. Limongi's floor is `[Fe/H] = -3`, and roughly half of the accepted
   mocks are predicted below that. No published table in this project reaches
   lower.
3. The largest shifts are N (-0.98 dex), Na (-0.37), Al (-0.31), Mn (-0.22),
   Mg (-0.21) and Cr (-0.20). These are comparable to or larger than the PISN
   signature, so the choice of Pop II table is a leading systematic.
4. The positive-label fraction rises from 18.0% to 23.4% with identical events,
   because the Limongi pattern shifts the Fe-independent ratios relative to the
   `f_Z^P = 0.5` threshold.

## 6. Recommendation

Limongi18 on its four real nodes is the better default: it is the only table
with published models below `Z = 1e-3`, it removes the 0.38 dex mapping error,
and it still clamps only half as often. The residual 57% clamp below
`[Fe/H] = -3` should be documented as a known limitation, not hidden.

Because switching the default changes every downstream number (support
fraction, calibration AUC, C2–C4), it was held back until explicitly approved.
**Limongi18 on its four real nodes is now the default** in
`YieldConfig` (`popii_table="Limongi18"`, `popii_z_nodes=(3.236e-5, 3.236e-4,
3.236e-3, 1.345e-2)`, `popii_z_grid_size=4`).

The 50k Ti-off pool was regenerated with it at
`src/potluck/tests/v5/limongi18_50k_ti_off.npz` (accepted 50,000, acceptance
0.370) and the v7 explanation figures were rebuilt on that pool. In the new
pool 56.67% of accepted mocks remain below the table floor, versus 99.95% for
Nomoto13. Any earlier C0–C4 numbers in notes 06–07 refer to the Nomoto13
default and must be re-run before being quoted again.

## 7. Reproduction

```
.venv/bin/python -m src.potluck.diagnostics.popii_metallicity \
    --n 20000 --seed 42 --outdir src/potluck/tests/v9
```

Produces `popii_metallicity.png` and `popii_metallicity.json`.
