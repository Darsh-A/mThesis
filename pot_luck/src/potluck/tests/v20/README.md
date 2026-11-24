# v20 — Li+-independent comparison of the yield models

20,000 accepted mocks per policy, seed 42, generic element noise sigma = 0.2 dex.
No Li+ masks, errors or weights in stage 1.

## Why new datasets

v19 could not be reused. Only the cuts did not change: the prior and the gates
changed too.

- v19 sampled `Z*` first, inside each table's own range, then derived `M_dil`.
- v20 draws `M_dil ~ LogUniform(2e5, 1e9)` for every policy and derives
  `Z* = M_Zpre / M_dil`.
- One `z_cr = 1e-6` for all policies. No table-floor rejection.
- Broad truth `[Fe/H]` window `[-5, -1]` for all policies.
- The table floor is handled in the yield lookup and flagged per sample.

## Gates

Stage 1 (Li+-free):

| gate | v20 definition |
|---|---|
| R1 | `n_PISN + n_III > 0` |
| R2 | `M_Zpre > 0`, `Z* >= 1e-6` (`z_max = None`), and the cloud can host the 1e3 Msun burst under the 10% cap |
| R3 | Fe fraction >= 1e-10 and at least 3 feature elements >= 1e-10 relative to `M_Ztot` |
| R4 | truth `[Fe/H]` in `[-5, -1]` |

There is no table-floor rejection. The floor is carried as a per-sample
`clamped` flag and handled by the clamp/blend lookup policy.

Stage 2 (Li+ application) uses the same R1/R2/R3 and replaces R4 with the
observed Li+ `[Fe/H]` box `[-3.8, -1.7]`, plus the Li+ masks, errors and
importance weights.

## Policies

| policy | source | low-Z handling |
|---|---|---|
| Limongi18_clamp | Limongi18 | clamp at 3.236e-5 |
| WW95_native | WW95 | native (lowest node 1.345e-6) |
| Nomoto13_blend | Nomoto13 | blend to Z=0 below 1e-3 |
| Nomoto13_clamp | Nomoto13_CLAMP | clamp at 1e-3 |
| Nomoto13_CK20_blend | Nomoto13_CK20 | blend to Z=0 below 1e-3 |
| Nomoto13_CK20_clamp | Nomoto13_CK20_CLAMP | clamp at 1e-3 |

### Policy coverage per analysis

The element tables do not all use the same number of tables. Do not compare
values across groups without checking this.

| analysis | tables |
|---|---|
| general selection, conditional map, Cu+Si+Al tests, operating points, simple baselines | Limongi18 clamp, WW95 native, CK20 blend (3) |
| leave-one-out element influence (f05 and D) | all 6 |
| pair ratios | all 6 |
| element subsets | all 6 |
| Li+ conditional map | all 6 |
| element bimodality | one dataset at a time (shown: Limongi18 and CK20) |
| v21 cross-table | all 6 |

The 3-table group excludes Nomoto13 blend/clamp and CK20 clamp. It is a
representative subset, not all six.

## Prior equality

The gates now select the same sample for every table.

| policy | acceptance | zero-PISN fraction | M_Zpre p16 / p50 / p84 [Msun] | below floor |
|---|---:|---:|---:|---:|
| Limongi18_clamp | 0.511 | 0.368 | 87 / 226 / 1090 | 0.520 |
| WW95_native | 0.508 | 0.369 | 88 / 225 / 1098 | 0.038 |
| Nomoto13_blend | 0.512 | 0.366 | 87 / 227 / 1105 | 0.967 |
| Nomoto13_clamp | 0.512 | 0.366 | 87 / 227 / 1105 | 0.967 |
| Nomoto13_CK20_blend | 0.510 | 0.365 | 86 / 225 / 1090 | 0.970 |
| Nomoto13_CK20_clamp | 0.513 | 0.370 | 86 / 226 / 1099 | 0.969 |

The `below floor` column is the fraction of mocks with `Z*` below the table's
lowest nonzero node. It is 52% for Limongi18, 3.8% for WW95, and 97% for
Nomoto/CK20. That is the low-Z policy, not a selection difference.

## M_dil prior sweep

Three common priors, same gates, policies, seed and 10,000 mocks per policy.
Only `mdil_range` changes.

| prior | acceptance | zero-PISN | median Z* | below floor | within AUC | LOTO AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1e5–5e8 | 0.533 | 0.373 | 3.7e-5 | 0.719 | 0.970 | 0.969 |
| 2e5–1e9 | 0.512 | 0.368 | 2.8e-5 | 0.738 | 0.973 | 0.966 |
| 1e6–1e9 | 0.499 | 0.360 | 1.6e-5 | 0.769 | 0.974 | 0.965 |

The accepted sample and the classifier are stable. Acceptance moves by 0.03,
zero-PISN by 0.01, within-table AUC by 0.004, LOTO by 0.004. The absolute
metallicity moves as expected: the median Z* shifts by 0.35 dex and the
below-floor fraction by 0.05. That is the physical prior dependence, and it is
now quantified.

Per-policy LOTO stays in the same band: WW95 is 0.891–0.907, the other policies
are 0.97–0.99 across all three priors. Figure: `sensitivity/mdil_sweep.png`.
Data: `sensitivity/summary.json`.

## Direct yields

IMF-integrated `[X/Fe]` at fixed `Z*` and `t`, no simulation. Spread across the
six policies at `t = 10 Myr`:

| element | Z*=1e-5 | Z*=1e-4 | Z*=1e-3 | Z*=1e-2 |
|---|---:|---:|---:|---:|
| Mg | 0.50 | 0.39 | 0.46 | 0.38 |
| Si | 0.32 | 0.41 | 0.48 | 0.54 |
| Ca | 0.43 | 0.46 | 1.98 | 0.47 |
| Ti | 0.63 | 0.72 | 0.63 | 0.50 |
| Cr | 0.64 | 0.73 | 0.71 | 0.47 |
| Mn | 0.72 | 0.83 | 0.68 | 0.80 |
| Co | **2.77** | 1.12 | 2.08 | 0.21 |
| Ni | 0.55 | 0.77 | 1.00 | 0.56 |
| Zn | **1.03** | 1.01 | 0.82 | 1.04 |

The yield-systematic floor is largest for Co and Zn, moderate for Ti/Cr/Mn,
smaller for Mg/Si/Ca. At `Z*=1e-5`, the low-Z policy changes Co by 2.8 dex and
Zn by 1 dex.

PISN minus Pop II at `Z*=1e-5`, `t=10 Myr`:

| element | Limongi18 | WW95 | Nomoto13 | CK20 |
|---|---:|---:|---:|---:|
| Na | -1.18 | -1.02 | -1.29 | -1.20 |
| Mg | +0.35 | +0.18 | -0.11 | -0.05 |
| Si | +0.76 | +0.43 | +0.47 | +0.56 |
| Ca | +0.49 | +0.19 | +0.42 | +0.55 |
| Ti | +0.29 | -0.28 | +0.11 | -0.32 |
| Cr | +0.74 | +0.11 | +0.27 | +0.39 |
| Mn | +0.59 | -0.13 | +0.28 | +0.41 |
| Co | -2.96 | -0.51 | -2.97 | -3.28 |
| Ni | -1.52 | -1.11 | -0.97 | -1.07 |
| Zn | -7.74 | -8.77 | -8.11 | -8.52 |

The killing elements are Co and Zn; the contrast is large, but its size depends
on the table.

## RF within-table

Observed block, class-balanced weights, 5-fold OOF.

| policy | AUC | Brier | AUC without below-floor |
|---|---:|---:|---:|
| Limongi18_clamp | 0.978 | 0.057 | 0.976 |
| WW95_native | 0.969 | 0.072 | 0.968 |
| Nomoto13_blend | 0.965 | 0.075 | 0.935 |
| Nomoto13_clamp | 0.977 | 0.061 | 0.936 |
| Nomoto13_CK20_blend | 0.975 | 0.062 | 0.938 |
| Nomoto13_CK20_clamp | 0.977 | 0.059 | 0.899 |

## RF transfer

Class-balanced weights, all samples. Observed AUC, train rows, test columns.

| train \ test | Lim | WW95 | N13 blend | N13 clamp | CK20 blend | CK20 clamp |
|---|---:|---:|---:|---:|---:|---:|
| Limongi18_clamp | 0.997 | 0.901 | 0.944 | 0.965 | 0.968 | 0.970 |
| WW95_native | 0.973 | 0.997 | 0.948 | 0.963 | 0.969 | 0.970 |
| Nomoto13_blend | 0.972 | 0.904 | 0.996 | 0.989 | 0.965 | 0.961 |
| Nomoto13_clamp | 0.975 | 0.901 | 0.971 | 0.998 | 0.970 | 0.969 |
| CK20_blend | 0.971 | 0.898 | 0.944 | 0.959 | 0.997 | 0.976 |
| CK20_clamp | 0.971 | 0.899 | 0.942 | 0.960 | 0.975 | 0.997 |

Leave-one-table-out observed AUC:

| held out | all samples | above floor |
|---|---:|---:|
| Limongi18_clamp | 0.975 | 0.971 |
| WW95_native | 0.902 | 0.872 |
| Nomoto13_blend | 0.973 | 1.000 |
| Nomoto13_clamp | 0.986 | 1.000 |
| Nomoto13_CK20_blend | 0.974 | 0.976 |
| Nomoto13_CK20_clamp | 0.973 | 0.943 |

With equal gates the transfer is 0.90–0.99, not 0.62. WW95 is the hardest target.
The above-floor Nomoto entries are based on only ~3% of their sample, so they are
noisy.

Full cross matrices and bootstrap intervals: `rf_transfer.json`,
`rf_transfer_no_floor.json`, `rf_transfer_observed*.png`,
`rf_transfer_noiseless*.png`.

## Noise sweep

Within-table balanced AUC versus generic element noise:

| policy | 0.0 | 0.1 | 0.2 | 0.3 |
|---|---:|---:|---:|---:|
| Limongi18_clamp | 0.995 | 0.990 | 0.978 | 0.961 |
| WW95_native | 0.994 | 0.987 | 0.967 | 0.941 |
| Nomoto13_blend | 0.993 | 0.984 | 0.966 | 0.943 |
| Nomoto13_clamp | 0.996 | 0.989 | 0.977 | 0.957 |
| CK20_blend | 0.995 | 0.989 | 0.976 | 0.952 |
| CK20_clamp | 0.996 | 0.990 | 0.978 | 0.958 |

## Element subsets and low-resolution proxy

Class-balanced within-table AUC, mean over the six policies, 5,000 mocks per
policy and 50 trees. Sigma is a generic element noise; sigma = 0.2–0.3 is a
crude low-resolution proxy, not a full PFS simulation.

| subset | n | AUC sigma=0.2 | AUC sigma=0.3 |
|---|---:|---:|---:|
| all 11 | 11 | 0.968 | 0.944 |
| no Zn | 10 | 0.952 | 0.915 |
| no Co | 10 | 0.967 | 0.942 |
| no Zn, Co | 9 | 0.948 | 0.905 |
| no Zn, Co, Mn | 8 | 0.946 | 0.903 |
| PFS-like (no Zn, Co) | 9 | 0.948 | 0.905 |
| alpha only (Mg, Si, Ca, Ti) | 4 | 0.863 | 0.775 |
| alpha + Na | 5 | 0.929 | 0.877 |
| alpha + Na, Cr, Ni | 7 | 0.941 | 0.896 |
| core + Co | 8 | 0.946 | 0.907 |
| core + Zn | 8 | 0.966 | 0.940 |

Zn is the most valuable single addition. Co is nearly redundant when Zn is
present. Alpha elements alone are not enough. A 7-element set
Na, Mg, Si, Ca, Ti, Cr, Ni keeps AUC 0.941 at sigma=0.2 and 0.896 at sigma=0.3.
Adding Zn gives 0.966/0.940.

![Element-subset ablation](element_subsets/element_subsets.png)

### Conditional value: what does adding X buy, given Y?

Class-balanced within-table AUC, 5,000 mocks per policy, 50 trees, noiseless
features. Delta is the AUC change when the element is added to the base set.

| base | base AUC | add Cu | add Zn | add Co | add Si |
|---|---:|---:|---:|---:|---:|
| none | 0.500 | +0.354 | +0.387 | +0.359 | +0.094 |
| Zn | 0.887 | **+0.032** | — | +0.038 | +0.078 |
| Cu | 0.854 | — | **+0.065** | +0.053 | +0.108 |
| Zn + Co | 0.925 | **+0.013** | — | — | +0.054 |
| Zn + Cu | 0.919 | — | — | +0.019 | +0.058 |
| core 7 | 0.992 | −0.001 | −0.001 | −0.001 | — |

Cu is a strong single element (0.896, 0.854 alone depending on the table), but
conditional on Zn it adds only about 0.03 AUC, and conditional on Zn+Co only
about 0.01. Zn adds more to Cu (+0.065) than Cu adds to Zn (+0.032). In the
noiseless full pattern, the core 7 already reaches 0.992 and the killing
elements are redundant.

In the observed Li+ block (no Cu), Zn alone is 0.845, core 7 is 0.941, and
core 7 + Zn is 0.964. Adding Zn to the core 7 gains +0.023; adding Co gains
about +0.005. Cu cannot be tested in the observed block because Li+ has no Cu.

Figures: `element_bimodality/Limongi18_clamp/element_bimodality_noiseless.png`
(now includes Cu), `element_conditional/element_conditional_noiseless.png`,
`element_conditional/element_conditional_observed.png`.

### Element influence in the full 26-ratio set

Leave-one-out from all 26 ratios, Gaussian noise sigma=0.2, pooled over the
six policies. Delta AUC is full minus without the element; positive means the
element helps. Top-1% turnover is 1 minus the overlap of the top-1% lists.

| element | Delta AUC | 95% CI | top-1% turnover | Delta precision @ FPR 1% |
|---|---:|---|---:|---:|
| Si | +0.0041 | +0.0033 to +0.0050 | 0.65 | +0.003 |
| Zn | +0.0022 | +0.0015 to +0.0031 | 0.60 | -0.000 |
| Al | +0.0018 | +0.0013 to +0.0024 | 0.60 | +0.001 |
| S | +0.0018 | +0.0011 to +0.0026 | 0.68 | +0.001 |
| Co | +0.0005 | +0.0000 to +0.0011 | 0.60 | -0.001 |
| Cu | +0.0002 | -0.0004 to +0.0007 | 0.64 | -0.002 |

No single element is essential when all 26 ratios are present. Every Delta AUC
is at most 0.004, and precision at FPR 1% changes by at most 0.003. The
top-1% turnover is 0.60-0.69 for every element. That turnover is not a loss:
the top 1% is already pure PISN, and precision at top 1% is 1.00 with and
without Zn. The high turnover comes from saturated scores and tie-breaking.

The effect shows up below the top 1%, where the candidate budget is no longer
saturated. For the sparse core 7:

| top K% | core 7 | core 7 + Zn |
|---:|---:|---:|
| 1% | 1.000 | 1.000 |
| 5% | 0.972 | 0.992 |
| 10% | 0.948 | 0.984 |
| 20% | 0.871 | 0.938 |

In the full 26-ratio set, all 26 and all 26 minus Zn are identical at every K.
Zn is redundant when the rest of the pattern is measured. It matters when the
element set is sparse, and it matters at the 5-20% candidate budget, not at the
top 1%.

Figures: `element_influence/leave_one_out_auc.png`,
`element_influence/leave_one_out_top1.png`,
`element_influence/leave_one_out_precision_1pct.png`,
`element_influence/leave_one_out_precision_01pct.png`,
`element_influence/zn_top1_scores.png`,
`element_influence/zn_precision_at_k.png`.

### Label definitions

`y_f05 = 1[f_ZP >= 0.5]` is the physical mixture label: PISN produced at least
half of the metals. It does not depend on the observation.

`y = 1[D > 3]` is the detectability label. The repaired v20 uses the noisy
observed ratios (`obs_features`) for the Li+ 11, matching `detect_labels`.
Positive rate is 0.594 in Limongi18 (f05 0.256), 0.594 in CK20 (f05 0.210),
and 0.599 in WW95 (f05 0.231). Overlap in Limongi18: 6,909 stars are
`D+ / f05-`, 4 are `D- / f05+`.

`k = 3` is a distance cut, not a p-value. With 11 elements, p(D^2 > 9) is about
0.6, so the label is broad and includes noise. A p-value threshold is the next
calibration step.

### Primary-label check: y = 1[D > 3]

The v20 files were repaired in place: `D`, `D2`, `y`, `detect_reason` and
`n_detected` now come from the stored noisy observation. The old D tables are
superseded.

| element | f05 Delta AUC | D Delta AUC | f05 top-1% turnover | D top-1% turnover |
|---|---:|---:|---:|---:|
| Si | +0.0041 | +0.0113 | 0.65 | 0.75 |
| Zn | +0.0022 | +0.0057 | 0.60 | 0.74 |
| S | +0.0018 | +0.0064 | 0.68 | 0.77 |
| Co | +0.0005 | +0.0006 | 0.60 | 0.74 |
| Cu | +0.0002 | +0.0013 | 0.64 | 0.78 |

The repaired D label is harder than f05: positive rate about 0.59 and
classifier AUC lower. Si, S and Zn have the largest D marginal AUC; Co is no
longer negative; Cu is small. The old top-K D table was computed with the
true-pattern reconstruction and is not shown here.

The main f05 conclusion does not change: no single element is essential when
all 26 ratios are present, Zn is the most useful addition to a sparse set, and
the effect is at the 5-20% candidate budget.

Figures: `element_influence_D/leave_one_out_auc.png`,
`element_influence_D/leave_one_out_top1.png`,
`element_influence_D/leave_one_out_precision_1pct.png`,
`element_influence_D/leave_one_out_precision_01pct.png`,
`element_influence_D/zn_top1_scores.png`,
`element_influence_D/zn_precision_at_k.png`.

### Pair ratios: Mg/Si and friends

The production model sees all 26 `[X/Fe]` ratios together, but it does not see
non-Fe pairs such as `[Mg/Si]`. Standalone AUC of selected pairs:

| pair | parent A | parent B | pair | core 7 + pair | full 26 + pair |
|---|---:|---:|---:|---:|---:|
| Si/Al | 0.594 | 0.743 | **0.913** | +0.0192 | +0.0008 |
| Si/Zn | 0.594 | 0.845 | **0.912** | +0.0247 | -0.0012 |
| Mg/Zn | 0.637 | 0.845 | 0.817 | +0.0212 | -0.0009 |
| Mg/Si | 0.637 | 0.594 | 0.752 | -0.0017 | -0.0010 |
| Na/Mg | 0.706 | 0.637 | 0.735 | -0.0026 | -0.0005 |
| Co/Ni | 0.733 | 0.564 | 0.636 | +0.0031 | -0.0003 |
| Cr/Mn | 0.555 | 0.564 | 0.509 | +0.0010 | -0.0004 |

Some pairs beat the best single `[X/Fe]` (Zn 0.845): Si/Al 0.913, Si/Zn 0.912,
Ca/Zn 0.894, Mn/Zn 0.885, Cr/Zn 0.873. Mg/Si 0.752 is better than Mg/Fe 0.637
or Si/Fe 0.594. But the pair is just a linear combination of its parents. Once
both parents are in the feature set, the pair adds nothing: the full 26-ratio
set gains at most +0.0008 from any pair. In the sparse core 7, the best pair is
Si/Zn at +0.0247, essentially the same as adding Zn/Fe alone (+0.022). Mg/Si
adds nothing to core 7 because Mg and Si are already there.

The code's existing independent ratios are mixed: Zn/Mg is strong (0.817),
Na/Mg moderate (0.735), Co/Ni weak (0.636), Mn/Cr is random standalone (0.509).
So no, we are not checking all pair ratios together. We checked the key ones
and added them to the full set; the full set already contains their
information.

Figures: `pair_ratios/pair_standalone_auc.png`,
`pair_ratios/pair_added_to_core7.png`,
`pair_ratios/pair_added_to_full.png`.

### General element subset, no Li+ bias

Greedy forward selection over 20 practical elements (alpha, odd-even,
iron-peak, and light elements with usable lines), Gaussian element-level noise
sigma=0.2, three yield tables, both labels. No Li+ masks, no catalogue
availability.

| label | selection order |
|---|---|
| `y_f_05` | Zn, Si, Al, S, Co, P, Ca, O, Cr, Na |
| `y = 1[D > 3]` (repaired) | Zn, Si, Al, Mn, Cu, S, P, N, V, C |

The general model does not pick Na early. It picks Al and S. Na looked
important in the Li+ analysis because Li+ has no Al, S, O or P. That is an
availability effect, not a physical ranking.

The repaired D label is realised detection with a high positive rate (~0.59),
so its AUCs are lower and not comparable to the old true-pattern D values. The
f05 selection below is the stable one.

Joint candidates, mean f05 AUC over Limongi18, WW95 and CK20:

| subset | f05 |
|---|---:|
| Zn + Si + Al | 0.9573 |
| Zn + Si + Al + S | 0.9652 |
| Zn + Si + Al + Ni | 0.9621 |
| Zn + Si + Al + S + Ni | 0.9684 |
| Zn + Si + Al + S + Ni + Cr | 0.9709 |
| Zn + Si + Al + S + Ni + Cr + Cu | 0.9721 |
| **Zn + Si + Al + S + Ca + Ti + Cr + Ni** | **0.9727** |
| core 7 + Zn | 0.9664 |

Recommended general set: **Zn, Si, Al, S, Ca, Ti, Cr, Ni**. It is tied with
the best greedy set and has a clean physical composition: Zn, Cr and Ni are
iron-peak PISN tracers; Si, S, Ca and Ti are alpha elements; Al is odd-even.
It does not depend on Li+ availability.

Nested general subsets:

| k | subset | f05 |
|---:|---|---:|
| 3 | Zn + Si + Al | 0.957 |
| 5 | Zn + Si + Al + S + Ni | 0.968 |
| 6 | Zn + Si + Al + S + Ni + Cr | 0.971 |
| 8 | Zn + Si + Al + S + Ca + Ti + Cr + Ni | 0.973 |

Noise robustness of the 8-element set, f05:

| sigma | f05 |
|---:|---:|
| 0.1 | 0.987 |
| 0.2 | 0.973 |
| 0.3 | 0.949 |

It beats core 7 + Zn at every noise level. Cu is optional: the repaired D
selection includes it, but the 8-element alpha/odd-even/iron-peak set is
equally good on f05 and does not depend on Cu. If S is not measurable,
substitute Cu or Co; the non-negotiable core is Zn, Si and Al.

Figures: `element_selection/selection_curves_f05.png`,
`element_selection/selection_curves_D.png`.

### General conditional map, single-element bases

Rows are single-element bases, columns are added elements, value is Delta AUC.
The row label gives the standalone AUC of the base. Gaussian sigma=0.2, three
yield tables, both labels. N is included.

Key readings, f05:

- Zn is the best universal add-on: +0.34 to Si, S and Ca, +0.32 to Cr and Mn,
  +0.30 to Ti, +0.23 to Ni, +0.09 to Cu.
- Si is the best partner for Zn (+0.089) and Cu (+0.110), and a strong partner
  for Al (+0.182) and N (+0.122).
- Al and Zn are substitutes when paired with Si or S, not complements:
  Si + Al 0.930, Si + Zn 0.931, Al + Zn 0.819.
- N standalone is 0.745 on f05 and 0.599 on the repaired D label. Its best
  partners are Zn (+0.125), Si (+0.122) and S (+0.114) on f05; Si (+0.106),
  S (+0.095) and Zn (+0.073) on D. N is a useful optional element, not part of
  the core.
- Single-element deltas are large (0.1-0.35) because the base has only one
  feature. In the full 20-element set the same additions are near zero.

Figures: `element_conditional_general/conditional_general_f05.png`,
`element_conditional_general/conditional_general_D.png`,
`element_conditional_general/standalone_auc_f05.png`,
`element_conditional_general/standalone_auc_D.png`,
`element_conditional_general/best_additions_f05.csv`,
`element_conditional_general/best_additions_D.csv`.

### Operating points at survey prevalence

5-fold OOF, 100 trees, full 20k per policy, f05 label. The general 8-element
set is the headline; the Li+ 11 is a comparison. TPR is at a fixed FPR;
precision uses pi = 0.1%, 0.25% and 1%.

| policy | feature set | TPR @ FPR 1% | TPR @ FPR 0.1% | precision @ 0.1% | @ 0.25% | @ 1% | TP / FP per 1e4 @ FPR 0.1% | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Limongi18 | general 8 | 0.741 | 0.493 | 0.331 | 0.553 | 0.833 | 12.3 / 10.0 | 0.050 |
| WW95 | general 8 | 0.673 | 0.405 | 0.288 | 0.504 | 0.804 | 10.1 / 10.0 | 0.058 |
| CK20 | general 8 | 0.727 | 0.450 | 0.310 | 0.530 | 0.820 | 11.2 / 10.0 | 0.050 |
| Limongi18 | Li+ 11 | 0.727 | 0.413 | 0.292 | 0.509 | 0.807 | 10.3 / 10.0 | 0.051 |

The general 8 is better than the Li+ 11 at the low-FPR operating point. WW95
is the weakest, consistent with the cross-table outlier.

FPR split at overall FPR = 0.1%:

| policy | clear negatives (f_Z < 0.25) | near-threshold (0.25-0.5) |
|---|---:|---:|
| Limongi18 | 0.00061 | 0.00365 |
| WW95 | 0.00015 | 0.00618 |
| CK20 | 0.00022 | 0.00611 |

The false positives are dominated by near-threshold negatives, by a factor of
6-40. The model is not confusing clear non-PISN stars; it is confusing
ambiguous mixtures. That is the physically expected failure mode.

At pi = 0.25%, the operating point gives about 11-12 true PISN and 10 false
positives per 10,000 stars, precision about 0.50-0.55. At pi = 0.1%, precision
is 0.29-0.33; at pi = 1%, it is 0.80-0.83. AUC 0.97 is not the same as a clean
detection.

Figures: `operating_points/tpr_at_fpr.png`,
`operating_points/precision_vs_prevalence.png`,
`operating_points/reliability.png`,
`operating_points/fpr_groups.png`.

### Simple-rule baselines at the same FPR

OOF scores, Gaussian sigma=0.2, f05 label, 3 policies. Thresholds are chosen
at the same target FPR on the negative class. TPR at FPR = 0.1%:

| model | Limongi18 | WW95 | CK20 |
|---|---:|---:|---:|
| RF, general 8 | 0.512 | 0.415 | 0.489 |
| logistic, general 8 | 0.486 | 0.418 | 0.418 |
| logistic, [Na/Mg] + [Zn/Mg] | 0.144 | 0.102 | 0.122 |
| RF, [Na/Mg] + [Zn/Mg] | 0.152 | 0.142 | 0.180 |
| single [Zn/Fe] | 0.244 | 0.142 | 0.218 |
| single [Zn/Mg] | 0.113 | 0.080 | 0.112 |
| single [Na/Mg] | 0.043 | 0.047 | 0.038 |

At FPR = 1%: RF general 8 is 0.745 / 0.676 / 0.728, logistic general 8 is
0.726 / 0.648 / 0.700, and the 2-feature logistic is 0.448 / 0.334 / 0.394.

Two conclusions:

1. The 2-feature rule is much worse than the 8-element model. The multi-element
   pattern matters. Meta-learning on two ratios does not recover it.
2. A linear model on the same 8 elements is almost as good as the RF. The RF
   adds only 0.00-0.07 TPR at FPR 0.1%, and 0.02-0.03 at FPR 1%. The value is
   in the 8-element combination, not in the nonlinearity.

The RF is retained as the primary classifier because it is competitive
within-table and not worse on average. The logistic on the general 8 is the
transparent baseline. Do not present the RF as the innovation; the 8-element
pattern is the result. v21 gives the cross-table comparison.

Figures: `simple_baselines/baseline_comparison.png`,
`simple_baselines/simple_baselines.csv`.

## Li+ application

Separate stage: 10,000 mocks per policy, Li+ `[Fe/H]` box, masks, errors,
weights and support. This is an application, not a physics validation.

| policy | out-of-model | in-model | max posterior | median expected f_ZP |
|---|---:|---:|---:|---:|
| Limongi18_clamp | 353 / 385 | 32 | 0.019 | 0.10 |
| WW95_native | 311 / 385 | 74 | 0.006 | 0.08 |
| Nomoto13_blend | 360 / 385 | 25 | 0.022 | 0.06 |
| Nomoto13_clamp | 352 / 385 | 33 | 0.043 | 0.06 |
| Nomoto13_CK20_blend | **256 / 385** | **129** | 0.022 | 0.05 |
| Nomoto13_CK20_clamp | **254 / 385** | **131** | 0.018 | 0.07 |

CK20 has the best Li+ support, then WW95. Limongi and plain Nomoto are the
worst. No star reaches `p_posterior >= 0.5` at the assumed prevalence. The
out-of-model fractions are dominated by the yield offsets, not by the low-Z
policy alone.

### Support-gate variants

The raw gate uses Na, Mg, Ca, Ti, Cr, Ni with no offsets. Four variants:

| dataset | raw | no Ti | offsets | no Ti + offsets |
|---|---:|---:|---:|---:|
| v20 Limongi18 | 0.083 | 0.847 | 0.925 | 0.875 |
| v20 WW95 | 0.192 | 0.683 | 0.948 | 0.891 |
| v20 Nomoto13 blend | 0.065 | 0.397 | 0.927 | 0.865 |
| v20 Nomoto13 clamp | 0.086 | 0.506 | 0.927 | 0.868 |
| v20 CK20 blend | 0.335 | 0.449 | 0.922 | 0.868 |
| v20 CK20 clamp | 0.340 | 0.561 | 0.909 | 0.862 |
| v19 Limongi18 | 0.096 | 0.847 | 0.943 | 0.896 |
| v19 Nomoto13 | 0.083 | 0.545 | 0.940 | 0.909 |
| v19 Nomoto13_CK20 | 0.187 | 0.538 | 0.940 | 0.896 |
| v19 WW95 | 0.138 | 0.647 | 0.938 | 0.883 |

Values are the in-model fraction. The Ti offset is +0.26 to +0.73 dex across
datasets, with a bootstrap interval of about ±0.01 dex and no [Fe/H] trend
(Spearman rho ≈ +0.06). It is a global zero-point, not a per-star anomaly.
Allowing one offset per core element brings every dataset to 91–95% in model.
Removing Ti alone brings 40–85%. The raw gate is therefore a zero-point
diagnostic, not a statement that most Li+ stars are odd.

![Support-gate variants](support_variants/support_variants.png)

![Per-element offsets](support_variants/support_offsets.png)

## Files

```
v20/
  manifest.json
  prior_check.json
  yield_curves.json
  yield_curves_Z1e-5.png
  yield_curves_Z1e-3.png
  yield_spread.png
  pisn_contrast.json
  rf_transfer.json
  rf_transfer_no_floor.json
  rf_transfer_observed.png
  rf_transfer_observed_no_floor.png
  rf_transfer_noiseless.png
  rf_transfer_noiseless_no_floor.png
  sigma_sweep.json
  sigma_sweep.png
  Limongi18_clamp/dataset.npz
  WW95_native/dataset.npz
  Nomoto13_blend/dataset.npz
  Nomoto13_clamp/dataset.npz
  Nomoto13_CK20_blend/dataset.npz
  Nomoto13_CK20_clamp/dataset.npz
  application/<policy>/dataset.npz
  application/<policy>/li_validation/
```

## Regenerate

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --n 20000 --seed 42 --sigma 0.2 \
    --policies Limongi18_clamp,WW95_native,Nomoto13_blend,Nomoto13_clamp,Nomoto13_CK20_blend,Nomoto13_CK20_clamp \
    --force
```

Above-floor transfer and Li+ application:

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --no-floor-transfer

python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --application-only
```

## Caveats

- Stage 1 is Li+-free except for the broad `[-5, -1]` window, which is chosen to
  cover the regime where the model applies.
- The yield spread is a physical model uncertainty, not a selection effect.
  Co and Zn are the worst.
- Nomoto/CK20 still spend 97% of their mocks below 1e-3; clamp and blend differ
  in the abundance pattern, not in the accepted sample.
- The above-floor Nomoto subsets are small and their AUCs are noisy.
- Stage 2 is an application, not a validation of the physics.
