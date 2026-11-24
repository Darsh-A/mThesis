# 12 — Limongi18 re-run of the observation chain

Notes 06–07 quote support, Ti, calibration, and reweighting numbers that were
computed with Nomoto13. After the default switched to Limongi18 (note 10) the
same chain was re-run on the new 50k pool. Outputs are in
`src/potluck/tests/v11/`.

Commands:

```bash
D=src/potluck/tests/v5/limongi18_50k_ti_off.npz
python -m src.potluck.diagnostics.support_decomposition decompose $D \
    > src/potluck/tests/v11/support_decomposition.json
python -m src.potluck.diagnostics.ti_diagnostic $D --outdir src/potluck/tests/v11
python -m src.potluck.diagnostics.classifier_plots $D src/potluck/tests/v11/calibration
python -m src.potluck.diagnostics.reweighting $D --outdir src/potluck/tests/v11 --verbose
```

The candidate table (support gating of the Li+ stars) is not produced by
`classifier_plots`; it comes from `calibration.run_phase_b`, which was called
directly into `src/potluck/tests/v11/calibration_phase_b/`.

## Tables

### Support

| quantity | Nomoto13 | Limongi18 |
|---|---|---|
| 99th-percentile threshold | 2.6626 | **2.7063** |
| stars in model | 194/385 = 50.4% | **316/385 = 82.1%** |
| available-element subset | 53.8% | 87.5% |
| always-observed subset | 77.9% | 85.5% |
| out of model | 191 | **69** |
| near threshold | 11 | 11 |

### Which element drives rejection

Counts of stars whose largest normalised residual sits on that element:

| element | Nomoto13 | Limongi18 |
|---|---|---|
| Cr | 245 | 145 |
| Ca | 55 | 46 |
| Na | 26 | 62 |
| Mg | 23 | 39 |
| Ni | 9 | 66 |

Per-element outlier fractions:

| element | Nomoto13 | Limongi18 |
|---|---|---|
| Cr | 4.1% | 8.7% |
| Ni | 4.5% | 0.3% |
| Na | 3.6% | 6.9% |
| Ca | 0.5% | 0.8% |
| Mg | 0.0% | 0.0% |

### Calibration (`run_phase_b`, identical code path)

| quantity | Nomoto13 | Limongi18 |
|---|---|---|
| positive rate | 17.9% | 22.8% |
| AUC before -> after | 0.9635 -> 0.9629 | 0.9737 -> **0.9732** |
| Brier after | 0.0526 | 0.0522 |
| candidates | 0 | 0 |
| out of model | 191 | 69 |

`classifier_plots` on the same pool gives AUC 0.9729 and Brier 0.0516 on its
own hold-out split; the two estimates agree to 0.001.

### Reweighting (expressed sample size, 50k mocks)

| scheme | Nomoto13 | Limongi18 |
|---|---|---|
| all | 5704 (11.4%) | **8847 (17.7%)** |
| all, no Cr | 5181 (10.4%) | 7286 (14.6%) |
| all, no Ni | 6487 (13.0%) | 10293 (20.6%) |
| core | 5892 (11.8%) | **9681 (19.4%)** |
| core, no Na | 7984 (16.0%) | 14070 (28.1%) |
| core, no Mg | 3597 (7.2%) | 11474 (22.9%) |
| core, no Ca | 3886 (7.8%) | 10220 (20.4%) |
| core, no Cr | 5825 (11.6%) | 9772 (19.5%) |
| core, no Ni | 7714 (15.4%) | 11274 (22.5%) |
| Cr alone | 24183 (48.4%) | 35202 (70.4%) |
| Ni alone | 26370 (52.7%) | 36765 (73.5%) |
| null | 11437 | 10594 |

## What changed and why

**Support nearly doubles.** The threshold barely moves (2.663 -> 2.706), but
82.1% of the catalogue now lies inside the mock envelope instead of 50.4%. The
cause is the Pop II table itself: Limongi18 spans `[Fe/H] = -3` to `-0.3` on its
four real nodes, so the metal-poor half of the catalogue is no longer compared
against a single clamped yield pattern. The clamp still covers 56.7% of the
mocks, but those mocks now share the same Pop II pattern as the stars they are
compared with, so their ratios agree instead of offsetting.

**The driver moved from Cr to a spread of elements.** Cr is still the single
largest contributor (145 of 316 rejections), but it no longer dominates: Ni (66)
and Na (62) are now comparable. Per element, Cr's own outlier fraction went
*up* (4.1% -> 8.7%) while Ni's collapsed (4.5% -> 0.3%). Limongi18 raises
Cr/Fe in the Pop II pattern relative to Nomoto13, which is why the mock tail at
the metal-poor end now sits above the catalogue's Cr.

**Calibration improves slightly.** AUC 0.9629 -> 0.9732, Brier unchanged at
0.052. The gain is consistent with the larger in-model fraction: fewer stars
are gated out before they can be scored.

**Reweighting improves everywhere.** `ESS_all` rises from 11.4% to 17.7% and
`ESS_core` from 11.8% to 19.4%; no scheme collapses. The largest single change
is `core_no_Mg` (7.2% -> 22.9%), which means Mg is no longer the axis that
forces the weights to concentrate. Cr and Ni alone now retain 70% and 74% of
the sample.

**No candidates either way.** Both runs return `n_candidates = 0`: no Li+ star
passes the support gate *and* the posterior threshold at the adopted prior
`pi = 2.5e-3`. The 11 near-threshold stars are the same objects in both runs.

**Ti cannot be compared like for like.** Ti is excluded from the pool, so the
diagnostic has no mock side. What can be compared is the yield side: integrated
Limongi18 `[Ti/Fe]` is -0.40 to -0.19 across its nodes, versus -0.19 to -0.12
for Nomoto13. The Ti deficit against the Li+ median (+0.44) is therefore
*larger* under Limongi18, which is consistent with keeping Ti out of the
training features. `ti_diagnostic.png` in v11 shows the Li+ side only, with the
Limongi18 single-star Ti/Fe curves.

## Consequence

The Nomoto13-era support and reweighting numbers in notes 06–07 are historical.
The Limongi18 figures are the ones to quote: **82.1% in support at threshold
2.706, AUC 0.973, Brier 0.052, ESS 17.7% of the pool**, with Cr, Ni and Na as
the joint drivers.

The prior screen of note 06 was also run under Nomoto13 and is not re-run here;
it needs a fresh Latin hypercube on this pool before any prior is adjusted.
