# v7_2 — explanation figures on Nomoto13 + CK20 with the revised M_dil window

Mock-only explanation figures (`src/potluck/diagnostics/explanation_plots.py`)
regenerated on the CK20 Pop II yield set with the **new default dilution-mass
window** `M_dil = 1e5--5e8 Msun`.

This is the v7 figure set (compare `v7/` and `v7_1/`). The only change versus
`v7_1` is the `M_dil` prior; the yield table and its sampler grid are identical.

## Dataset

```
src/potluck/tests/v5/nomoto13_ck20_50k_mdil1e5_5e8_all_elements.npz
```

50,000 accepted mocks, seed 42, 27 feature elements (26 ratios), Li+
observation layer with Ti on. Acceptance 0.385.

Realized `M_dil`: p16 = `1.0e6`, median = `9.8e6`, p84 = `7.3e7 Msun`.

## What changed versus v7_1

| parameter | v7_1 | **v7_2 (this test)** |
|---|---|---|
| `prior.mdil_range` | `(2.0e5, 1.0e9)` | **(1.0e5, 5.0e8)** |
| Pop II table | `Nomoto13_CK20` | unchanged |
| `popii_z_nodes` | `[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2]` | unchanged |
| seed, n | 42, 50,000 | unchanged |

The default `PriorConfig.mdil_range` in `src/potluck/config.py` is now
`(1.0e5, 5.0e8)`. The CK20 edits are unchanged from v7_1:

1. 50/50 normal/hypernova mix for progenitors `M >= 20 Msun`.
2. Constant offsets Sc +1.0, Ti +0.45, V +0.3, Co +0.2, Zn +0.2 dex.

## Summary numbers

From `explanation_plots.json`:

| quantity | v7_1 | **v7_2** |
|---|---:|---:|
| positive fraction (`f_Z^P >= 0.5`) | 0.1888 | **0.1946** |
| median PISN share | 0.0414 | **0.0427** |
| median Pop III share | 0.1719 | **0.1720** |
| median Pop II share | 0.3887 | **0.3753** |
| Spearman `f_Z^P`--Na/Mg | -0.440 | **-0.436** |
| Spearman `f_Z^P`--Co/Ni | -0.330 | **-0.321** |
| Spearman `f_Z^P`--Mn/Cr | +0.128 | **+0.143** |
| Spearman `f_Z^P`--Zn/Mg | -0.627 | **-0.617** |
| Pop II grid floor | 1e-12 | 1e-12 |
| fraction below grid floor | 0.0 | 0.0 |
| lowest nonzero CK20 node | 1e-3 | 1e-3 |
| fraction below lowest nonzero node | 0.9995 | **0.9964** |
| fraction inside `[1e-3, 5e-2]` | 0.0005 | **0.0036** |
| `z_star` p16 / p50 / p84 | 3.6e-6 / 2.1e-5 / 1.3e-4 | **4.0e-6 / 2.4e-5 / 1.4e-4** |

The narrower, lower-`M_dil` window shifts the budget slightly towards PISN and
Pop III and away from Pop II, but the normalized abundance trends are almost
unchanged. The accepted `[Fe/H]` distribution is set mostly by the same
`M_Zpre / M_dil` combination.

## Nomoto13 metallicity coverage

This is the important caveat, and the new `M_dil` window does **not** fix it.

The sampler grid now reaches `Z = 1e-12`, so `0%` of mocks are clamped at the
grid floor. But Nomoto13_CK20 only has **nonzero** metallicity nodes at
`Z = 1e-3, 4e-3, 8e-3, 2e-2, 5e-2`, plus the `Z = 0` row. The Li+ mocks live
almost entirely below `1e-3`:

| quantity | old `2e5–1e9` | **new `1e5–5e8`** |
|---|---:|---:|
| median `z_star` | 2.10e-5 | **2.35e-5** |
| fraction `z_star < 1e-3` | 99.95% | **99.64%** |
| fraction in `[1e-3, 5e-2]` | 0.048% | **0.364%** |
| fraction with blend weight `z_star/1e-3 < 0.1` | 79.8% | **77.9%** |

At the median `z_star = 2.35e-5`, the linear `Z=0 <-> Z=1e-3` blend weight is
`w = 0.024`, so the Pop II ejecta are ~98% the **Z = 0 CK20 row**. The CK20
edits (hypernova mix and constant offsets) are still applied, but the
metallicity dependence of the table is essentially not exercised by this mock
sample. The high-`Z` CK20 nodes are only reached by the `0.36%` tail.

Implication: for the Li+ `[Fe/H]` range, CK20 is effectively a metal-free table
plus the CK20 offsets. If true low-metallicity coverage is required, Limongi18
has nonzero nodes at `3.236e-5` and `3.236e-4` (`[Fe/H] ~ -3` and `-2`), which
cover the bulk of these mocks. Nomoto13_CK20 cannot provide that without new
low-`Z` yield calculations.

## Figures

`fig1_priors`, `fig2_timeline`, `fig3_metallicity`, `fig4_yields` (CK20 Pop II
panel), `fig5_trends`, `fig6_ratios`, `fig7_label`, `fig8_zstar`,
`fig9_ca_pisn_fraction`.

`fig8_zstar` marks the CK20 sampler grid floor at `Z = 1e-12`; nothing is
clamped, so the panel annotates that the floor is below the sampled range.

## Regenerate

```bash
D=src/potluck/tests/v5/nomoto13_ck20_50k_mdil1e5_5e8_all_elements.npz
python -m src.potluck.diagnostics.explanation_plots $D --outdir src/potluck/tests/v7_2
```

Dataset generation used:

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

```bash
python -m src.potluck generate --config <config.json> --n 50000 --seed 42 \
    --output src/potluck/tests/v5/nomoto13_ck20_50k_mdil1e5_5e8_all_elements.npz
```
