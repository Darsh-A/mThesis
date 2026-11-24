# Caveats and next steps

## Implemented

1. Exact Pop III sums and a binned compound-Poisson Pop II sampler; both seam tests
   pass (explicit-vs-explicit and binned-vs-naive AUC ≈ 0.5).
2. Four Fe-independent ratios: `[Na/Mg]`, `[Co/Ni]`, `[Mn/Cr]`, `[Zn/Mg]`.
3. Li+ observation model: whole-record sampling, ratio errors from `sigma_Total`,
   coverage masks, C upper limits, giant C inflation, observed-`[Fe/H]` R4, and
   Li+ `[Fe/H]` importance weights.
4. Non-C non-detections for Si, Mn, Co, and Zn are forced missing below a
   lower-envelope `[X/H]` threshold. The threshold is a selection boundary, not a
   physical detection model.
5. Weighted OOF FPR at TPR 0.9 and prevalence-weighted precision, with the
   `n_III <= 200` headline and the clear/near-threshold split.

## Current blocker: support mismatch

The 100k Li+ validation run found that only 37 of 385 Li+ stars are inside the mock
support. The support distance is dominated by **Ti**: 353 of 385 stars have Ti as
the largest residual, and the model’s `[Ti/Fe]` is about 0.73 dex below the Li+
median. Removing R4, `Z_cr`, the stellar cap, the 100 M⊙ group, or widening
`lambda_III` does not fix it. The leading explanation is the Pop II yield table, not
the gates. See [Real-star validation](07-real-star-validation.md) for the full
decomposition and ablation.

The matched-prior yield cross (C2) and the direct Ti diagnostic (C2b) confirm this.
Nomoto13 is now the default, but its IMF-integrated `[Ti/Fe]` is −0.19 at low Z,
while Li+ is +0.44. The 0.63 dex offset is constant in `[Fe/H]` and S/N, is only
0.10 dex different between giants and dwarfs, survives with noiseless mocks, and
cannot be closed by the solar scale (needs 0.63 dex) or NLTE (≤ 0.2 dex). Ti is
intrinsically underproduced by the yield tables, not by the gates or the
observation model.

**Superseded by [Every common element](14-all-common-elements.md):** the default
no longer excludes Ti. The text below records the original C2c decision.

The decision was then explicit: **Ti is excluded** from the observation layer and from
the support gate (`ObservationConfig.exclude_elements = ("Ti",)`), and reported
separately. On a 50k Ti-off Nomoto13 pool this raises Li+ in-model coverage from
9.6% to **50.4%** (194/385) at a threshold of 2.663. The remaining exclusion is
distributed across Cr, Ca, Na, Mg, and Ni, and no single element is grossly
discrepant: every per-element outlier fraction is below 5%. The joint
five-dimensional gate is still conservative, and 27 stars with fewer than four
active core elements remain out-of-model by construction.

C3 tested whether any prior range recovers the rest. A 26-point Latin-hypercube
screen varied the PISN mass range, `M_dil`, `Z_cr`, `lambda_III`, and the Pop III
weighting. Six points failed, all at `Z_cr >= 1e-4`, because the cloud is then
pre-enriched too close to the Li+ `[Fe/H]` window. At 3,000 mocks the best point
(p11) reached 70.1% in-model versus 60.5% for the baseline. Re-run at 20,000 mocks
with four seeds, the gain shrinks to **+4.4 points** (57.5% versus 53.1%) and the
seed ranges overlap. The mock-vs-real two-sample AUC stays at 0.992, the classifier
AUC falls by 0.004, FPR rises by 1.3 points, and Cr KS worsens by 0.02. The
element-level residuals are unchanged: **Cr −0.39 dex and Ni +0.31 dex**
(real − mock). No prior range removes the residual, so C3 does not justify a prior
change. See [Real-star validation](07-real-star-validation.md#c3-prior-screen).

C4 then importance-reweighted the 50k Ti-off pool to the full Li+ catalogue. The
effective sample size is **5,704 of 50,000** (11.4%), against a mock-to-mock null
of **11,437** (22.9%, five draws). The pool is therefore more constraining than a
random mock sample but does **not** collapse: thousands of compatible histories
remain. The reweighted posterior favours more Pop III and Pop II activity
(`n_III` 26 → 91, `lambda_III` 26 → 91, `popii_n` 31 → 84), a slightly later `t`
(11.3 → 15.3 Myr), and weaker PISN (`n_PISN` median 1 → 0); `[Fe/H]`, `M_dil`, and
`Delta` barely move. These are broad shifts inside the existing priors, not a new
region. The residual Cr/Ni pattern is distribution-level, not a coverage failure.
See [Real-star validation](07-real-star-validation.md#c4-li-importance-reweighting).

## What is still missing

1. **Probabilistic non-detection.** The hard threshold is a bracket. A logistic
   `P(detected | [X/H], SNR)` fitted with censored data would be better.
2. **N/O/Al/Cu value analysis.** Li+ has none of these elements, so it cannot be
   done on Li+ alone; it needs Xiang or a combined catalogue.
3. **Parent-draw splits.** The dataset currently has one row per independent draw,
   so sample-level CV is parent-level. Add an explicit `parent_id` before any
   augmentation or multiple observations per draw.
4. **Posterior intervals.** Phase B should output `P(f_Z >= 0.5 | x)` with a
   calibrated interval, not only a score.
5. **SNIa is a zero row.** With the current `t` range it is a smaller issue, but it
   is still missing.

## Li+ data caveats

The Li+ files are

```
data/raw/Obs_Li/apjac6514t2_mrt.txt   # stellar parameters
data/raw/Obs_Li/apjac6514t3_mrt.txt   # abundances
```

from Li et al., “Four-hundred Very Metal-Poor Stars Studied with LAMOST and Subaru.
II. Elemental abundances”.

The machine-readable tables contain **no NLTE flag, no C-depletion correction flag,
and no evolutionary-state correction flag**. Treat the abundances as LTE and
uncorrected.

Consequences:

- **C in giants is suspect.** C is detected in 84.6% of giants and 58.5% of dwarfs;
  giants are 55.6% of the sample. Either add a logg-dependent `σ_C` or drop C for
  `logg < 3.5`.
- **Na may have a catalogue-level offset.** `[Na/Mg]` is the broadly available
  odd-even tracer (Na detected in 94.5%), but NLTE is not flagged. Allow a
  catalogue-level Na offset nuisance rather than treating Na as absolute.
- **Zn is giant-biased.** Zn is detected in 60.3% of giants and 23.4% of dwarfs. Any
  Zn-based statement applies mostly to giants.
- **Coverage is pattern-dependent.** The full 11-element pattern is 30.6% of stars and
  96% giants. Dwarfs mostly have partial patterns. Feasibility must be reported
  separately for giants and dwarfs.

## Recommended order

1. C2, C2b, C2c, C3, and C4 are done (all under the old 15-element feature set
   with Ti excluded). Limongi18 is now the default Pop II table;
   Limongi18 and WW95 are retained as comparison tables. Ti is excluded from
   training and support and reported separately. The Ti-off support fraction is
   50.4% at a threshold of 2.663. The prior screen found no range that removes the
   residual: the best point gains only 4.4 points at 20k, with overlapping seeds,
   unchanged two-sample separation, and the same Cr/Ni offsets. C4 reweighting
   gives ESS 5,704/50,000 versus an 11,437 mock-null, so the pool is compatible but
   more constraining than a random mock sample. The next statistical step is a
   **hierarchical or kernel-density comparison of the Li+ and mock distributions**,
   not another prior or threshold change.
2. Replace the hard non-detection threshold with a probabilistic detection model
   and re-run the 100k bracket.
3. Run the N/O/Al/Cu value analysis on Xiang, not Li+.
4. Add `parent_id` and group-aware CV/weights.
5. Produce calibrated `P(f_Z >= 0.5 | x)` with intervals and rank candidates by
   expected `f_Z` and support, not by a posterior cut of 0.5.
6. Add SNIa when the delay-time distribution and W7 table are available.

## What to report with the next run

- Ledger split by PISN presence.
- Per-bin RMSE/MAE, not R² in narrow bins.
- FPR at fixed TPR and precision at the realistic prevalence.
- The no-PISN mimic list, with `n_III`, group composition, Pop II share, `Δ`, and
  `[Fe/H]`.
- Feasibility split by giant/dwarf and by element-availability pattern.

## One unrelated bug

`src/formulas.py::abundance_ratio` double-counts the atomic-mass conversion:

```python
log10((M1/a1)/(M2/a2)) - (A1-A2) - log10(a1/a2)
```

The correct form is `log10(M1/M2) - (A1-A2)`. The potluck features use solar mass
fractions and are unaffected. `salvadori_combined_abundratio` is correct. The legacy
function was not changed.
