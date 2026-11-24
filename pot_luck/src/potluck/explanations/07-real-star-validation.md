# Real-star validation: observation, calibration, and support

This note records the Li+ validation layer and the support diagnostics run after the
100k dataset. It is the current status; the older 2000-sample numbers in
[Diagnostics and results](05-diagnostics-and-results.md) are retained as history.

## Why this layer exists

The generator produces noiseless abundances. A real catalogue has measurement
errors, missing elements, upper limits, and a selection function. Training on
noiseless mocks and applying the result to Li+ would confuse “the model cannot
produce this pattern” with “the catalogue did not measure it well”.

The observation model therefore adds four things:

1. **Element-level noise** drawn from a real Li+ star, not a fixed constant.
2. **Availability masks** copied from a real star.
3. **Censoring**: an element below its detection threshold is reported as a limit,
   not a value.
4. **Selection weights** so the mock `[Fe/H]` distribution matches Li+.

## The Li+ observation model

The catalogue is Li et al. (2022), 385 very metal-poor stars. The model uses 12
elements: C, Na, Mg, Si, Ca, Ti, Cr, Mn, Co, Ni, Fe, Zn.

- **Errors.** `sigma_Total` is the uncertainty in `[X/Fe]`, including the
  temperature, gravity, `[Fe/H]`, and microturbulence terms. It is not an
  element-level abundance error. The ratio covariance is therefore diagonal,
  `diag(sigma_X^2)`, not `diag(sigma_X^2) + sigma_Fe^2 11^T`.
- **Limits.** Only C has upper limits in Li+ (about 18% of stars). All other
  non-detections are missing, not limits. The C limit threshold is fitted from
  `[C/H]` versus `sigma_C`.
- **Record sampling.** Each mock star draws a real record with
  `|[Fe/H]_rec - [Fe/H]_true| <= 0.2` dex, falling back to the nearest record. This
  preserves the correlation between elements at a given signal-to-noise.
- **Giant C.** C in giants is inflated by an extra 0.3 dex because the catalogue
  has no depletion correction.
- **Observed `[Fe/H]`.** R4 is applied to the observed `[Fe/H]`, not the noiseless
  value, because that is what the catalogue selected on.
- **Weights.** Mocks are reweighted to the Li+ `[Fe/H]` histogram separately for
  giants and dwarfs. The weights are clipped and normalized.

On the 100k dataset the effective sample size after weighting is **79,290**, so the
weights are not collapsing the sample.

### Informative missingness

For Si, Mn, Co, and Zn, non-detections are forced missing below a monotone
lower-envelope `[X/H]` threshold versus log SNR. This is a selection boundary, not a
physical detection model. The threshold mode gives FPR 7.08% at TPR 0.9; copying the
mask gives 6.58%. The difference is small, but the threshold mode is the more honest
bracket because it makes missingness depend on abundance.

## 100k evaluation

Dataset: `data/potluck/analysis/v8_threshold_100k.npz`, 100,000 accepted mocks.

| quantity | value |
|---|---|
| draws | 270,000 |
| R1 / R2 / R4 / R3 | 26,863 / 96,094 / 45,932 / 16 |
| accepted | 100,000 |
| Li+ weighted positive rate, `f_Z^P >= 0.5` | 0.230 |
| weighted AUC | 0.9736 |
| weighted Brier | 0.0510 |

At the operating points:

| operating point | threshold | TPR | FPR | precision at `pi = 2.5e-3` |
|---|---|---|---|---|
| TPR = 0.9 | 0.2676 | 0.900 | **0.0708** | 0.0309 |
| FPR = 0.001 | 0.9354 | 0.361 | 0.001 | **0.4747** |

The FPR at TPR 0.9 is too high for a rare-class census. The useful operating point
is the low-FPR end: at FPR 0.1%, about 36% of true positives survive and precision is
about 47% at `pi = 2.5e-3`.

### Where the false positives are

By `n_III`:

| `n_III` | n | positives | AUC | FPR at TPR 0.9 |
|---|---|---|---|---|
| 0 | 11,845 | 7,282 | 0.986 | 0.035 |
| 1–3 | 14,438 | 7,029 | 0.979 | 0.059 |
| 4–10 | 11,918 | 4,435 | 0.969 | 0.093 |
| 11–30 | 13,694 | 3,523 | 0.947 | 0.149 |
| 31–100 | 16,658 | 1,239 | 0.932 | 0.179 |
| >100 | 31,447 | 7 | 0.942 | 0.319 |

`n_III > 200` is 21,846 samples with **zero** positives. They are easy negatives and
flatter the global FPR. Excluding them, FPR at TPR 0.9 is 8.1%.

The 100 M⊙ group matters: `has100` FPR 9.0% versus `no100` 3.7%. Ishigaki et al.
treat 100 M⊙ as a pulsational-pair-instability-bracketed CCSN, with normal and
hypernova energies, so the group is physical but uncertain.

Clear negatives (`f_Z^P < 0.25`) have FPR 3.05%; near-threshold negatives
(`0.25 <= f_Z^P < 0.5`) have FPR 27.3%. Most false positives are near-threshold
negatives, not clear negatives.

The no-PISN mimics are almost pure Pop III: median `n_III = 7`, median Pop II metal
share `1e-4`, median `Delta = 16.8` Myr, median observed `[Fe/H] = -3.06`.

## Calibration and real-star scoring

`src/potluck/calibration.py` uses a 60/20/20 stratified split:

1. train the classifier;
2. fit isotonic calibration on the calibration fold;
3. evaluate on the test fold;
4. shift the probability from the training prevalence to a target prevalence.

On the 100k dataset:

| | AUC | Brier |
|---|---|---|
| raw classifier | 0.97524 | 0.04937 |
| isotonic-calibrated | 0.97509 | **0.04872** |

The reliability curve is close to diagonal at low scores, but still under-confident
in the middle: predicted 0.34 → observed 0.41; predicted 0.43 → observed 0.53.

The prior shift is

```text
odds_post = odds_train * (pi / (1 - pi)) / (pi_train / (1 - pi_train))
```

With `pi_train = 0.230` and `pi = 2.5e-3`, the top calibrated mock probability
0.722 becomes a posterior of **0.021**. To reach a posterior of 0.5 at that
prevalence, the calibrated probability would need to be about 0.997.

Consequently there are **zero candidates at posterior >= 0.5**. This is expected at
a realistic prevalence; it is not evidence that no PISN descendant exists.

The top expected `f_Z^P` values are:

| star | observed `[Fe/H]` | expected `f_Z^P` | posterior | in-model? |
|---|---|---|---|---|
| J0849+1914 | −2.65 | 0.557 | 0.021 | no (`n_core = 3`) |
| J0124+2508 | −3.17 | 0.545 | 0.010 | no |
| J1341+5135 | −3.02 | 0.508 | 0.021 | no |
| J1151−0054 | −3.05 | 0.498 | 0.0096 | **yes** |
| J0409+2009 | −3.37 | 0.464 | 0.021 | no |

No Li+ star is both in-model and expected `f_Z^P >= 0.5`. The highest in-model
expected value is 0.498.

## What “in-model” means

A **mock** is a synthetic star from the generator. The **support** is the region of
abundance-pattern space that the mocks populate.

For each real star, the support distance is the nearest-mock distance after scaling
each element difference by the star’s measurement error. This is a Mahalanobis
distance: 1 means one typical error away, 3 means three typical errors away. The
threshold is the **99th percentile** of mock-to-mock distances.

- **In-model**: the star is closer than the threshold. The classifier is
  interpolating.
- **Out-of-model**: the star is farther than the threshold. The classifier is
  extrapolating.

This is not a PISN/non-PISN classification. A star can be out-of-model and still be a
PISN descendant; it means the simulator has not produced a sufficiently similar
pattern.

The current gate uses six **core elements**: Na, Mg, Ca, Ti, Cr, Ni. A star with
fewer than four core elements gets no distance and is conservatively out-of-model.

On the 100k dataset:

- support threshold: **3.046**
- **37 / 385 in-model**
- **348 / 385 out-of-model**

## C0: decomposing the distance

`src/potluck/diagnostics/support_decomposition.py` identifies the element with the
largest standardized residual to the nearest mock. That element is the **driver**.

| driver | stars |
|---|---|
| **Ti** | **353 / 385** |
| Cr | 15 |
| Na | 3 |
| Mg | 3 |
| Ca | 3 |
| Ni | 2 |

Per-element medians and outlier fractions:

| element | real median `[X/Fe]` | mock median `[X/Fe]` | offset | outlier fraction |
|---|---|---|---|---|
| Ti | **+0.445** | **−0.289** | **+0.73 dex** | **79.5%** |
| Na | −0.15 | −0.70 | +0.55 | 6.0% |
| Mg | +0.37 | +0.13 | +0.24 | 0.5% |
| Ca | +0.33 | +0.24 | +0.09 | 1.3% |
| Cr | −0.20 | −0.01 | −0.20 | 4.9% |
| Ni | +0.02 | −0.11 | +0.12 | 0.3% |

The distance is therefore almost one-dimensional: the model’s Ti/Fe is about
0.7 dex too low. A 0.7 dex offset is too large for NLTE or a solar-scale difference,
which are typically 0.1–0.2 dex.

Alternative support definitions do not remove the mismatch:

| definition | in-model |
|---|---|
| current, require 4 core | 37 / 385 (9.6%) |
| allow 3 core | 40 / 385 (10.4%) |
| compare to observed mocks | 137 / 385 (35.6%) |

The observed-mock comparison improves coverage, but 64% of stars are still outside.
That says part of the problem is the observation model, but not most of it.

## C1: ablating the gates

Each pool has 20,000 mocks; the support threshold is recomputed inside each pool.

| pool | Li+ in-model | Ti KS | mock-vs-real AUC | FPR at TPR 0.9 |
|---|---|---|---|---|
| current | 8.3% | 0.946 | 0.987 | 7.5% |
| no R4 | 10.1% | 0.931 | 0.995 | 8.4% |
| no `Z_cr` | 7.8% | 0.950 | 0.988 | 6.2% |
| cap 0.3 | 7.5% | 0.947 | 0.995 | 7.7% |
| wide `lambda_III` | 7.0% | 0.945 | 0.991 | 8.7% |
| no 100 M⊙ | 6.2% | 0.953 | 0.989 | 6.5% |

No gate change recovers Li+. Removing R4 gives a small increase, but 90% of stars are
still out, and the Ti KS barely moves. Removing the 100 M⊙ group makes support
slightly worse. The gates tested here are not the cause.

## Interpretation

The evidence points to the **yield tables**, specifically Ti.

At low metallicity, for a 20 M⊙ star:

| table | `[Ti/Fe]` |
|---|---|
| Limongi18 | ≈ −0.34 |
| WW95 | ≈ +0.31 |
| Nomoto13 | ≈ −0.09 |
| Li+ data | ≈ +0.45 |

WW95 is much closer for Ti, but its Ni is suspiciously high in the same check, so it
cannot be swapped without checking the other elements.

The result does not mean the priors are irrelevant. It means the gates and ranges
tested so far do not explain the mismatch. PISN mass range, PISN table, and the
relative weight of sources still need a matched-parameter yield cross.

## C2: matched-prior yield-table cross

C2 holds the prior fixed and swaps one table at a time. Each pool has 20,000
accepted mocks and the support threshold is recomputed inside that pool. The
Pop II cross uses PISN HW2002; the PISN cross uses Pop II Nomoto13.

### Pop II cross

| Pop II table | in-model | Ti KS | real − mock Ti | two-sample AUC | FPR@TPR90 |
|---|---|---|---|---|---|
| **Nomoto13 (default)** | **7.0%** | 0.930 | **+0.639** | 0.995 | 0.094 |
| Limongi18 | 8.3% | 0.946 | +0.728 | 0.987 | 0.075 |
| WW95 | 15.3% | 0.787 | +0.601 | 0.997 | 0.108 |

### PISN cross

| PISN table | in-model | Ti KS | real − mock Ti | two-sample AUC | FPR@TPR90 |
|---|---|---|---|---|---|
| HW2002 | 7.0% | 0.930 | +0.639 | 0.995 | 0.094 |
| Takahashi-MR | 9.4% | 0.904 | +0.601 | 0.994 | 0.035 |
| Takahashi-NM | 8.3% | 0.907 | +0.597 | 0.995 | 0.032 |
| Takahashi-NR | 7.8% | 0.905 | +0.588 | 0.995 | 0.033 |

### Reading the cross

- **Nomoto13 is now the default Pop II table.** It is post-decay, unlike WW95, and
  avoids the Limongi18 issue that yields above 25 M⊙ are wind-only. It improves the
  Ti offset relative to Limongi18 (+0.639 vs +0.728 dex), but it does **not** fix
  support: in-model falls from 8.3% to 7.0% and FPR rises from 7.5% to 9.4%.
- **WW95 gets the best Ti KS and the highest in-model fraction**, but still leaves a
  +0.60 dex Ti offset, has a near-perfect mock-vs-real AUC (0.997), and is not
  properly decayed. It is a diagnostic comparison, not the default.
- **Takahashi PISN tables improve in-model to 8–9% and cut FPR to ~3.5%**, but the
  Ti offset remains 0.59–0.60 dex and Ti KS remains ~0.90. PISN is not the main
  driver of the support mismatch.
- Across all seven table combinations, the Ti offset stays in **0.59–0.73 dex**.
  This is not a table-swap problem. It points to a Ti-specific yield or abundance
  normalisation issue, or to missing Ti production physics, not to the gates.

The immediate consequence is that switching to Nomoto13 is justified physically,
but it does not make Li+ stars fit. Ti remains the blocker.

## C2b: direct Ti diagnostic

`src/potluck/diagnostics/ti_diagnostic.py` compares the Li+ `[Ti/Fe]` distribution
with one mock pool. It uses 50,000 accepted Nomoto13 mocks
(`src/potluck/tests/v5/nomoto13_50k.npz`, seed 42; drawn 135,000, R1 13,429,
R2 48,041, R3 27, R4 22,144, discarded 1,359, PISN accepted 31,159).

The mock median uses the Li+ `[Fe/H]` importance weights. Every offset has a
bootstrap 95% interval from 2000 resamples. “KS” is again the two-sample
Kolmogorov–Smirnov distance.

### Overall

| quantity | value |
|---|---|
| Li+ median `[Ti/Fe]` | **+0.440** |
| mock median `[Ti/Fe]` | **−0.191** |
| offset | **+0.631** `[+0.620, +0.650]` |
| KS distance | 0.929 |
| Li+ Ti missing | 1.30% |
| mock Ti missing | 1.66% |
| offset using noiseless mocks | +0.615 |

The observation model is not responsible: removing measurement noise and
censoring moves the offset by only **0.016 dex**. The offset is already present in
the noiseless ratio.

### Versus `[Fe/H]`

| `[Fe/H]` bin | n | offset (dex) |
|---|---|---|
| −4.0 to −3.5 | 17 | +0.475 `[+0.439, +0.566]` |
| −3.5 to −3.0 | 85 | +0.619 `[+0.570, +0.658]` |
| −3.0 to −2.5 | 118 | +0.635 `[+0.606, +0.650]` |
| −2.5 to −2.0 | 130 | +0.655 `[+0.631, +0.675]` |
| −2.0 to −1.5 | 25 | +0.623 `[+0.604, +0.666]` |

The binned slope is 0.065 dex per dex, driven almost entirely by the lowest bin.
Across the range that contains 98% of the stars (−3.5 to −2.0), the offset changes
by only about 0.04 dex. It is therefore a **constant normalisation**, not a
metallicity-dependent one.

### Versus S/N

| S/N bin | n | offset (dex) |
|---|---|---|
| 0–20 | 42 | +0.614 `[+0.481, +0.667]` |
| 20–40 | 127 | +0.610 `[+0.588, +0.652]` |
| 40–60 | 107 | +0.635 `[+0.612, +0.665]` |
| 60–100 | 93 | +0.644 `[+0.634, +0.666]` |
| 100–200 | 11 | +0.615 `[+0.512, +0.647]` |

The binned slope is 0.0001 dex per unit S/N, i.e. flat. The offset is **not a
signal-to-noise artefact**.

### Giants versus dwarfs

| group | n | offset (dex) |
|---|---|---|
| giants (`logg < 3.5`) | 214 | +0.589 |
| dwarfs | 166 | +0.684 |

The dwarf minus giant difference is **−0.095** `[−0.118, −0.057]`: dwarfs are
slightly *worse*. A Ti II ionisation or NLTE problem would normally act more
strongly in giants, so this goes in the wrong direction to explain the offset. It
is also far too small: the two groups differ by 0.1 dex, not 0.6.

### Solar normalisation

The model uses the Asplund scale, `A(Ti) = 4.95 ± 0.05` and `A(Fe) = 7.50 ± 0.04`.
To close the offset by changing only the solar Ti zero point, the Sun would need
`A(Ti) = 5.59`. That is 0.63 dex, more than ten times the stated scale
uncertainty, and inconsistent with every modern solar compilation. The solar
normalisation is **not** the cause.

### Nomoto13 Ti yield

Single-star Nomoto13 `[Ti/Fe]`, 13–40 M⊙ and Z = 0–0.05, spans roughly **−0.33 to
+0.20 dex**. Above 30 M⊙ it reaches ≈ +0.15 to +0.20; at 13–25 M⊙ it is between
−0.33 and +0.09. The IMF-integrated value, which is what the Pop II burst uses, is:

| Z | integrated `[Ti/Fe]` |
|---|---|
| 0.001 | −0.19 |
| 0.004 | −0.15 |
| 0.008 | −0.13 |
| 0.02 | −0.12 |

The mock median (−0.19) agrees with the integrated table value. So the deficit is
intrinsic to the table and is not introduced by interpolation, IMF weighting, or
the sampler.

### Ionisation and NLTE

Li+ Ti is **Ti II**, while the solar reference is total Ti. Ti II is the dominant
ion at these temperatures and gravities, so the ionisation correction is small. A
generous NLTE bracket is about 0.2 dex. That leaves 0.4–0.5 dex unexplained, and
the observed giant/dwarf difference is in the wrong direction.

### Conclusion

The Ti offset is:

- **not** Fe/H-dependent (apart from a small drop in the lowest bin),
- **not** S/N-dependent,
- **not** explained by giants versus dwarfs (0.1 dex, wrong sign),
- **not** caused by the observation model (0.016 dex),
- **not** a solar-scale error (would need 0.63 dex),
- **not** an ionisation or NLTE effect (≤ 0.2 dex).

It is an **intrinsic Ti underproduction in the yield tables**: Nomoto13 predicts
`[Ti/Fe] ≈ −0.19` where Li+ has `+0.44`. All seven table combinations tested in C2
share the same 0.59–0.73 dex deficit.

The physically honest options are:

1. **Use a Ti-specific correction factor** fitted to the data, and record it as a
   calibrated nuisance parameter rather than a prediction. This is a model change
   and needs a decision before it is applied.
2. **Exclude Ti from the core support gate** and report it separately, so that Ti
   does not mask the other elements. This is a diagnostic choice, not a physical
   fix, and must not be presented as improved agreement.
3. **Add the missing Ti physics** (fallback, explosion asymmetry, mixing, or a
   different progenitor mass range). That is beyond the current table-based model.
4. **Do not tune the support threshold** to absorb the offset.

## C2c: Ti excluded from the model

The Ti diagnostic left two defensible choices: apply an explicit Ti correction, or
stop using Ti. The model now **excludes Ti from the observation layer and from the
support gate** and reports Ti separately. Less data is preferable to an element the
yield tables cannot reproduce.

Configuration is `ObservationConfig.exclude_elements = ("Ti",)`, now the default.
The stored array layout is unchanged: excluded elements are stored as missing, so a
pool can still be inspected or regenerated with Ti enabled by setting
`exclude_elements=()`. `active_core_elements` in `calibration.py` selects the core
elements that the observation layer actually populates, so the gate adapts
automatically. With Ti off the core set is **Na, Mg, Ca, Cr, Ni**, and a star needs
four of those five.

The classifier also drops observation columns that are constant or entirely
missing. Without that, the all-missing Ti columns made HistGradientBoosting fail
with `window shape cannot be larger than input array shape`.

### Ti-off pool

`src/potluck/tests/v5/nomoto13_50k_ti_off.npz`, 50,000 accepted Nomoto13 mocks,
seed 42. The seed matches the Ti-on pool, but the RNG stream shifts because Ti noise
is no longer drawn, so the pools are not event-identical.

| ledger | value |
|---|---|
| drawn | 135,000 |
| R1 / R2 / R4 / R3 | 13,390 / 48,057 / 22,347 / 26 |
| accepted | 50,000 |
| discarded | 1,180 |
| PISN accepted | 31,413 |
| no-PISN accepted | 18,587 |

### Support

| quantity | value |
|---|---|
| threshold | **2.663** |
| in-model | **194 / 385 = 50.4%** |
| 3-core variant | 207 / 385 = 53.8% |
| observed-mock reference | 300 / 385 = 77.9% |

By number of active core elements:

| `n_core` | in-model |
|---|---|
| 2 | 0 / 3 |
| 3 | 0 / 24 |
| 4 | 56 / 78 (71.8%) |
| 5 | 138 / 280 (49.3%) |

By evolutionary state: dwarfs 105 / 171 (61.4%), giants 89 / 214 (41.6%).

By `[Fe/H]`:

| `[Fe/H]` bin | in-model |
|---|---|
| −4.0 to −3.5 | 5 / 20 (25.0%) |
| −3.5 to −3.0 | 39 / 88 (44.3%) |
| −3.0 to −2.5 | 58 / 118 (49.2%) |
| −2.5 to −2.0 | 69 / 130 (53.1%) |
| −2.0 to −1.5 | 21 / 24 (87.5%) |

Drivers are now **Cr 245**, Ca 55, Na 26, Mg 23, Ni 9. The per-element outlier
fractions are all below 5%:

| element | threshold | median distance | outlier fraction |
|---|---|---|---|
| Na | 0.0085 | 0.00017 | 3.6% |
| Mg | 0.0058 | 0.00009 | 0.0% |
| Ca | 0.0437 | 0.00004 | 0.5% |
| Cr | 0.4204 | 0.01874 | 4.1% |
| Ni | 0.0070 | 0.00078 | 4.5% |

Dropping Ti more than doubles coverage, from 37 to 194 stars. The remaining
exclusion is no longer driven by a single grossly discrepant element; it is the
joint five-dimensional 99th-percentile gate doing what it was defined to do. The 27
stars with only two or three active core elements are conservatively out-of-model by
construction.

### Calibration

`src/potluck/tests/v5/calibration_ti_off/`:

| | AUC | Brier |
|---|---|---|
| raw classifier | 0.9635 | 0.0530 |
| isotonic-calibrated | 0.9629 | **0.0526** |

Training positive rate is 0.1786. The support threshold is 2.663, with **0
candidates**, **191 out-of-model**, and **11 near-threshold** stars. Compared with
the 100k Ti-on run, dropping Ti costs about 0.01 AUC and 0.002 Brier. The 50k and
100k pools are not a controlled comparison, so this is approximate.

The highest expected `f_Z` values are:

| star | observed `[Fe/H]` | expected `f_Z` | posterior | in-model? | `n_core` |
|---|---|---|---|---|---|
| J0849+1914 | −2.65 | 0.572 | 0.038 | no | 2 |
| J0440+1706 | −3.40 | 0.545 | 0.018 | no | 3 |
| J1341+5135 | −3.02 | 0.525 | 0.035 | no | 3 |
| J1151−0054 | −3.05 | 0.518 | 0.018 | no | 3 |
| J1327+0710 | −3.25 | 0.471 | 0.015 | **yes** | 4 |

There are still **zero stars that are both in-model and have expected
`f_Z >= 0.5`**. The highest in-model expected value is 0.471. That is a statement
about the current pool and support definition, not a detection.

## C3: prior screen

C3 holds the yield tables fixed (Nomoto13 and HW2002) and asks whether any prior
range recovers Li+ support. Ti stays excluded. The screen uses a discrete
Latin-hypercube design: five factors, five levels each, 25 points plus an explicit
baseline (26 total). Each point generates 3,000 accepted mocks, and the support
threshold is recomputed inside that pool.

| factor | levels |
|---|---|
| PISN mass range (M⊙) | (140, 270), (140, 300), (160, 260), (180, 260), (200, 300) |
| `M_dil` | (2e5, 1e9), (1e6, 1e9), (2e5, 1e8), (5e6, 1e9), (2e5, 5e7) |
| `Z_cr` | 1e-6, 1e-5, 1e-4, 1e-3, none |
| `lambda_III` | (0.1, 1000), (0.1, 100), (1, 1000), (0.01, 1000), (0.3, 300) |
| Pop III weighting | uniform, low-mass, high-mass, high-energy, row |

The baseline point uses the first level of every factor, which is exactly the
current `ModelConfig()` prior. The low-mass, high-mass, and high-energy weights test
whether the Pop III mass function or explosion energy, rather than the total
count, drives the abundance pattern. `row` samples the raw table rows instead of
the combination weights.

Each point reports the in-model fraction (the primary metric), per-core-element
KS, the mock-vs-real two-sample AUC, the classifier AUC, FPR at TPR 0.9, the
weighted positive rate, the ledger, and the active core elements. The main effect
of a factor is the mean in-model fraction across the five points at each level. A
single high point is not evidence; the level means are. A range is only interesting
if it raises support without collapsing the classifier or worsening the other core
elements. Choosing a range solely because it maximizes in-model fraction would be
threshold tuning by another name.

One technical caveat: `support_gate` compares 2,000 observed mocks against **all
remaining** noiseless mocks. A 3,000-mock pool therefore has only 1,000 reference
mocks, so its nearest-neighbour distances and 99th percentile are larger than in a
50,000-mock pool. Absolute in-model fractions from the screen are inflated and are
only comparable within the screen. Selected points are re-run at 20,000 mocks for
confirmation.

### Screen result

Twenty of the 26 points completed. Six failed, all with `Z_cr >= 1e-4`: at that
pre-enrichment the cloud is already too close to the Li+ `[Fe/H]` window and the
acceptance collapses. `Z_cr = 1e-3` produced no usable point at all.

| point | in-model | two-sample AUC | classifier AUC | FPR@TPR90 | Cr KS |
|---|---|---|---|---|---|
| **p11 (best)** | **0.701** | 0.993 | 0.950 | 0.135 | 0.791 |
| p23 | 0.652 | 0.994 | 0.942 | 0.158 | 0.789 |
| p16 | 0.629 | 0.995 | 0.966 | 0.098 | 0.813 |
| baseline | 0.605 | 0.996 | 0.958 | 0.127 | 0.781 |

p11 is PISN `(140,300)`, `M_dil` `(2e5,5e7)`, `Z_cr=1e-6`, `lambda_III=(1,1000)`,
and low-mass Pop III weighting.

The main-effect means over the design:

| factor | level | mean in-model | n |
|---|---|---|---|
| PISN mass | (140, 270) | 0.609 | 3 |
| | (140, 300) | 0.586 | 4 |
| | (160, 260) | 0.549 | 5 |
| | (180, 260) | 0.522 | 4 |
| | (200, 300) | 0.546 | 4 |
| `M_dil` | (2e5, 1e9) | 0.578 | 4 |
| | (1e6, 1e9) | 0.480 | 4 |
| | (2e5, 1e8) | **0.617** | 5 |
| | (5e6, 1e9) | 0.487 | 3 |
| | (2e5, 5e7) | 0.603 | 4 |
| `Z_cr` | 1e-6 | 0.562 | 6 |
| | 1e-5 | **0.581** | 5 |
| | 1e-4 | 0.559 | 4 |
| | 1e-3 | — | 0 |
| | none | 0.536 | 5 |
| `lambda_III` | (0.1, 1000) | 0.522 | 4 |
| | (0.1, 100) | 0.558 | 5 |
| | (1, 1000) | **0.635** | 4 |
| | (0.01, 1000) | 0.486 | 4 |
| | (0.3, 300) | 0.608 | 3 |
| Pop III | uniform | 0.590 | 4 |
| | low-mass | **0.623** | 4 |
| | high-mass | 0.526 | 4 |
| | high-energy | 0.535 | 5 |
| | row | 0.519 | 3 |

The screen prefers a narrower `M_dil` upper bound, `lambda_III >= 1`, low-mass Pop III
weighting, and a PISN upper mass of 300 M⊙. But no level separates cleanly from the
baseline, and the 3k pools exaggerate differences.

### 20k confirmation

The baseline, p11, and the opposite-trade-off point p15 were re-run with four seeds
(1–4) at 20,000 accepted mocks each. Values are means over the four seeds with the
seed range in brackets.

| point | in-model | threshold | classifier AUC | FPR@TPR90 | two-sample AUC | Cr KS | Ni KS | positive rate |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.531** [0.491, 0.566] | 2.81 | 0.962 | 0.106 | 0.992 | 0.775 | 0.776 | 0.177 |
| p11 | **0.575** [0.553, 0.626] | 2.78 | 0.958 | 0.119 | 0.991 | 0.796 | 0.665 | 0.197 |
| p15 | **0.431** [0.374, 0.465] | 2.83 | 0.982 | 0.049 | 0.992 | 0.809 | 0.812 | 0.115 |

The p11 in-model gain shrinks from +9.6 points in the 3k screen to **+4.4 points** at
20k, and its seed range overlaps the baseline. The two-sample mock-vs-real AUC does
not move (0.992 in all three). The classifier AUC falls by 0.004 and FPR rises by
1.3 points. Cr KS worsens by 0.02; Ni KS improves by 0.11.

Median offsets (real − mock, dex) at 20k:

| point | Na | Mg | Ca | Cr | Ni |
|---|---|---|---|---|---|
| baseline | +0.207 | +0.022 | +0.022 | **−0.392** | **+0.313** |
| p11 | +0.117 | −0.048 | −0.004 | **−0.409** | **+0.268** |
| p15 | −0.083 | −0.063 | −0.004 | **−0.428** | **+0.338** |

### Reading

The prior screen does not fix the support mismatch. The apparent best point buys a
small geometric improvement by shifting the Pop III mass function and the dilution
range, but the real-vs-mock separation is unchanged, the classifier gets slightly
worse, and the element-level residuals get no better. p15 shows the opposite
trade-off: better separation, less support. The residual is now **Cr overproduction
of about 0.39 dex and Ni underproduction of about 0.31 dex** (mock − real), with Na
and Mg smaller; these are yield-level patterns, not prior-range effects.

`Z_cr >= 1e-4` is not a viable direction: it destroys acceptance rather than
improving support. There is no prior region that both fits Li+ and leaves the
classifier intact. This closes C3. The next step is C4, an importance-reweighting
check, not another prior scan.

## C4: Li+ importance reweighting

C4 asks whether the mock pool contains any region compatible with the **joint**
Li+ pattern, rather than with individual elements. It uses the 50k Ti-off pool and
the 385 Li+ stars. Each mock receives the product likelihood over the stars that
have at least four active ratios; the weights are normalised, and the effective
sample size is

```text
ESS = 1 / sum_i w_i^2.
```

ESS is the number of independent mocks the weighted pool is worth. It is not a
p-value.

The likelihood is deliberately sharp: 385 stars, each with a catalogue `[X/Fe]`
error, so even a small systematic offset is multiplied by the sample size. A
collapse on its own would therefore not prove that no parameter region fits. To
calibrate the scale, the same calculation is repeated with 385 **pseudo-stars
drawn from the pool** as the target, five times, each reweighted against the other
~49,600 mocks. This **mock null** is what ESS looks like when the target really is
drawn from the model.

### Result

| scheme | ESS | ESS / 50,000 | max weight | top 1% share |
|---|---|---|---|---|
| all active ratios | **5,704** | 0.114 | 0.002 | 0.168 |
| core (Na, Mg, Ca, Cr, Ni) | 5,892 | 0.118 | 0.001 | 0.183 |
| all without Cr | 5,181 | 0.104 | 0.003 | 0.183 |
| all without Ni | 6,487 | 0.130 | 0.003 | 0.156 |
| **mock null** | **11,437** | 0.229 | — | — |

The five null values are 10,453–12,119. Real Li+ retains **about half** the
effective sample size of a mock target, not a collapse to one or two mocks. The
single-element ESS values (`min_elements = 1`) are Na 19,637 (39%), Mg 28,962
(58%), Ca 38,359 (77%), Cr 24,183 (48%), and Ni 26,370 (53%). No single element is
catastrophic; the joint pattern is what reduces ESS.

### Reweighted parameters

| parameter | prior median | posterior median | posterior 16–84% |
|---|---|---|---|
| `n_III` | 26 | **91** | 13–457 |
| `lambda_III` | 26 | **91** | 12–460 |
| `popii_n` | 31 | **84** | 14–328 |
| `M_*II` | 2.7e4 | **3.8e4** | 6.3e3–1.5e5 |
| `t` | 11.3 Myr | **15.3 Myr** | 7.7–28.9 |
| `n_PISN` | 1 | **0** | 0–2 |
| `M_dil` | 1.1e7 | 1.1e7 | 1.6e6–9.2e7 |
| `[Fe/H]` | −2.61 | −2.61 | −3.36 to −1.98 |
| `Delta` | 15.9 Myr | 15.5 Myr | 4.6–55.1 |
| `Z_*` | 2.2e-5 | 3.3e-5 | 5.9e-6 to 1.5e-4 |

### Reading

The pool does **not** collapse: 5,704 effective mocks remain, about half the
mock-null. The Li+ pattern is therefore more constraining than a random mock
sample, but the generator still contains thousands of compatible enrichment
histories. C4 does not identify a missing prior region.

The weights prefer **more Pop III events and a larger `lambda_III`**, **more Pop II
stars**, a **slightly later observation time**, and **weaker PISN** (posterior
median `n_PISN = 0`). The `[Fe/H]`, `M_dil`, and `Delta` distributions barely move.
These are broad shifts within the existing priors, not a fine-tuned solution.

The C3 residual (Cr high by ~0.39 dex, Ni low by ~0.31 dex) is a distribution-level
yield mismatch, not an absence of coverage. Reweighting cannot repair it, and ESS
should not be optimized by changing priors.

Caveat: the likelihood treats the 385 stars as independent and ignores the
selection function beyond the `[Fe/H]` weights already in the pool. A hierarchical
or kernel-density comparison of the **distribution** is the next statistical step,
not another single-mock likelihood.

## Next steps

1. **C2, C2b, and C2c are done.** Nomoto13 is the default Pop II table. Ti is
   excluded from training and support and reported separately. The Ti-off support
   fraction is 50.4% at a threshold of 2.663.
2. **C3 is done.** The space-filling screen and the 20k confirmation show that no
   prior range removes the residual mismatch. The best point (p11) raises in-model
   fraction but leaves the mock-vs-real two-sample AUC unchanged and slightly
   worsens the classifier and Cr. The residual is now a yield-level Cr/Ni/Na
   problem, not a prior-range problem.
3. **C4 is done.** Li+ retains 5,704 effective mocks out of 50,000, about half the
   11,437 mock-null. The pool is not empty. The reweighted posterior favours more
   Pop III/Pop II activity, a later `t`, and weaker PISN, but all shifts stay broad.
4. **Distribution-level comparison.** Replace the single-mock product likelihood
   with a hierarchical or kernel-density comparison of the Li+ and mock
   distributions. This is the appropriate next statistical step after C4.
5. **Probabilistic non-detection.** Replace the hard threshold with a censored
   detection model.
6. **N/O/Al/Cu on Xiang** and `parent_id` grouping remain open.
7. Do not tune the support threshold to make stars pass. That would hide the
   mismatch rather than fix it.

## Artifacts

- `src/potluck/tests/v2/candidates.csv`
- `src/potluck/tests/v2/calibration.json`
- `src/potluck/tests/v2/reliability.png`
- `src/potluck/tests/v2/real_stars.png`
- `src/potluck/tests/v3/support_decomposition.csv`
- `src/potluck/tests/v3/support_decomposition.json`
- `src/potluck/tests/v3/support_decomposition.png`
- `src/potluck/tests/v3/support_ablation.json`
- `src/potluck/tests/v3/support_ablation.png`
- `src/potluck/tests/v4/yield_cross.json`
- `src/potluck/tests/v4/yield_cross.png`
- `src/potluck/tests/v5/nomoto13_50k.npz`
- `src/potluck/tests/v5/ti_diagnostic.json`
- `src/potluck/tests/v5/ti_diagnostic.png`
- `src/potluck/tests/v5/nomoto13_50k_ti_off.npz`
- `src/potluck/tests/v5/nomoto13_50k_ti_off.meta.json`
- `src/potluck/tests/v5/support_decomposition.csv`
- `src/potluck/tests/v5/support_decomposition.json`
- `src/potluck/tests/v5/support_decomposition.png`
- `src/potluck/tests/v5/calibration_ti_off/calibration.json`
- `src/potluck/tests/v5/calibration_ti_off/candidates.csv`
- `src/potluck/tests/v5/calibration_ti_off/reliability.png`
- `src/potluck/tests/v5/calibration_ti_off/real_stars.png`
- `src/potluck/tests/v6/prior_screen.json`
- `src/potluck/tests/v6/prior_screen.png`
- `src/potluck/tests/v6/prior_screen_confirmation.json`
- `src/potluck/tests/v6/prior_screen_confirmation.png`
- `src/potluck/tests/v6/reweighting.json`
- `src/potluck/tests/v6/reweighting.png`

Tests: **87 passed** at the time of this note.

## Caveats

- No SNIa; the SNIa row is zero. At `t <= 40` Myr this is a smaller issue, but it is
  still missing.
- No probabilistic non-detection model; the threshold is a bracket.
- No `parent_id`; one row per independent draw, so sample-level CV is parent-level.
- Li+ has no N, O, Al, or Cu; those elements need Xiang.
- C in giants is uncorrected for depletion; Na has no NLTE flag.
- The support gate is conservative: fewer than four core elements means
  out-of-model by default.
