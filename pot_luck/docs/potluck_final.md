# Potluck: model, algorithm and validation

Potluck is a forward simulator for metal-poor stellar abundances. It draws a
mixed enrichment history, builds the ejecta of each source, mixes them in one
gas cloud, applies physical cuts, and produces noiseless abundance patterns,
observed Li+-like patterns, and labels for PISN detection. The code lives in
`src/potluck`. This document describes the algorithm and the results from the
v18–v20 test campaigns.

The short version: the core mixing is validated against Salvadori et al. (2019)
Eq. 13 at machine precision. A PISN-vs-CCSN classifier trained on one yield
table transfers to another at AUC 0.90–0.99 when the selection is matched. The
main uncertainty is not the machine learning; it is the yield tables, especially
Co and Zn, and the low-metallicity treatment. The full model is a legitimate
simulator, not yet a validated physical model.

---

## 1. The physical picture

A gas cloud is a bucket. Each source pours element masses into it. The bucket is
then diluted to a total gas mass `M_dil`, and the observed star freezes in the
mixture.

Three source classes are implemented:

- **PISN** — pair-instability supernova, 140–260 M☉ progenitors.
- **Pop III CCSN** — core-collapse supernovae from the first stars, Ishigaki et
  al. (2018) grid.
- **Pop II CCSN** — a single burst of stars, IMF-integrated from Limongi &
  Chieffi (2018), Nomoto et al. (2013), or Woosley & Weaver (1995).

There is no SNIa and no AGB source yet. The SNIa row exists in the matrix but is
zero.

The chronology uses two clocks. Pop III forms at `T = 0`, Pop II forms at
`T = Δ`, and the observed star forms at `T = Δ + t`. A source enters the
pre-enrichment gas only if its lifetime is shorter than `Δ`; it enters the final
cloud if its lifetime is shorter than `Δ + t`.

---

## 2. Notation and constants

| symbol | meaning |
|---|---|
| `X` | chemical element |
| `F` | feature elements (27), reference `R = Fe` |
| `Z` | metals, `X ≠ H, He` |
| `M_X^(s)` | ejected mass of `X` by source `s`, in M☉ |
| `M_dil` | dilution mass of the cloud |
| `Δ` | time from Pop III to Pop II formation |
| `t` | time from Pop II formation to the observed star |
| `κ_X` | solar mass fraction of `X` |

Constants used throughout:

| quantity | value |
|---|---|
| solar metallicity | `Z_SUN = 0.01345` |
| solar hydrogen mass fraction | `X_H,⊙ = 0.7381` |
| solar abundances | Asplund et al. (2009), `data/asplund.json` |
| IMF | Larson, `φ(m) ∝ m^-2.35`, `m = 0.1–100 M☉` |
| IMF turnover | `m_ch = 0.35 M☉` |
| feature elements | 27, reference Fe |
| validity floor | `10^-10` relative to `M_Z^tot` |
| minimum valid features | 3 |
| default `Z_cr` | `10^-6` |
| Li+ observed [Fe/H] box | `[-3.8, -1.7]` |
| target prevalence | `2.5 × 10^-3` |

---

## 3. Abundances

Solar abundance:

```
A_X = log10(N_X / N_H) + 12
```

Solar mass fraction:

```
κ_X = 10^(A_X - 12) * (m_X / m_H) * X_H,⊙
```

Elemental log abundance and abundance ratio:

```
a_X = log10(M_X / κ_X)
[X/Y] = a_X - a_Y
```

`[X/Fe]` is the feature. `[Fe/H]` is a derived diagnostic, never a feature.

---

## 4. Yield tables

Isotopes are folded into elements:

```
M_X = sum_i m_i
```

All sources use the same 32-element vector in the order of
`src/potluck/constants.py`.

### 4.1 PISN

Ejecta are tabulated at discrete progenitor masses. One event of mass `m_P` is
drawn by interpolation:

```
M_X^(P)(m_P) = I_X[m_P]
```

Per element:

```
I_X[m] = exp(lin-interp(ln m))   if all M_X > 0 and max/min > 1e3
         lin-interp(m)           otherwise
```

Smooth elements are linear; threshold elements that jump orders of magnitude
(Zn, Cu, …) are geometric. Endpoints are clamped.

### 4.2 Pop III CCSN

The Ishigaki grid is a discrete set of rows, one per event:

```
(m_i, E_i, M_X,i^(III))
```

Rows are grouped by `(m, E)`. One event = one row, used verbatim. The group
determines the lifetime.

### 4.3 Pop II CCSN

A single burst of stellar mass `M_*II`. The IMF-averaged yield above the turnoff
mass is

```
Y_X^II(t, Z) = ∫_{m_TO(t,Z)}^{m_max} m_X(m,Z) φ(m) dm
```

The turnoff mass solves `τ_II(m_TO, Z) = t` with the Raiteri et al. (1996)
relation.

For generation, the integral is replaced by a compound-Poisson sum over mass
bins:

```
ν_k = ∫_bin φ(m) dm
n_k ~ Poisson(M_*II ν_k)
M_X^II = sum_k n_k y_X,k
```

The bin mean `y_X,k` is rescaled by a factor in `[0.5, 2]` so that
`E[M_X^II] = M_*II Y_X^II` exactly. With `popii_stochastic = False` the
deterministic integral is returned.

### 4.4 Low-Z policies

The tables have different lowest nonzero nodes:

| table | lowest nonzero Z | policy |
|---|---:|---|
| WW95 | 1.345e-6 | native |
| Limongi18 | 3.236e-5 | clamp |
| Nomoto13 | 1e-3 | clamp or blend to Z=0 |

Below the floor, "clamp" uses the lowest nonzero table; "blend" linearly
interpolates between Z=0 and the first nonzero node. The flag `clamped` marks
the affected samples. Nomoto13 is evaluated below 1e-3 for about 97% of the v20 mocks, so the low-Z
policy is a main systematic.

---

## 5. Prior on the enrichment history

One sample draws:

```
n_P ~ Cat(0.5, 0.3, 0.15, 0.05),   n_P ∈ {0,1,2,3}
m_P,i ~ U[max(140, m_min), min(270, m_max)]
λ_III ~ LogU(0.1, 1000)
n_III ~ Poisson(λ_III)
```

Each Pop III event is assigned to a `(mass, energy)` group and then to one row
inside that group.

```
M_*III = sum_j m_III,j
Δ ~ LogU(1, 100) Myr
t ~ LogU(3, 40) Myr
M_dil ~ LogUniform(2e5, 1e9) M☉   (v20 common prior)
M_*II ~ LogUniform(1e3, 1e7) M☉
```

Two caps tie the stellar budget to the cloud:

```
M_*II ≤ min(1e7, 0.1 M_dil − M_*III)
M_*III + M_*II ≤ 0.1 M_dil
```

The joint prior is the product of the densities and is stored as `log_prior`.

**Why M_dil is a prior, not a measurement.** A star gives ratios, not the total
mass of its birth cloud. `[Fe/H]` fixes the ratio `M_metals / M_dil`, not
`M_dil` on its own. The metal mass also depends on the event amounts and the
Pop II burst. So `M_dil` is latent and must be marginalised. The v20 sweep
shows what this means in practice.

---

## 6. Clocks

Three epochs: Pop III at `T=0`, Pop II at `T=Δ`, observed star at `T=Δ+t`.

Pop III lifetime, Schaerer (2002), log–log interpolation:

```
τ_III(m) from
m/M☉ = 500, 400, 300, 200, 120, 80, 60, 40, 25, 15, 9, 5
τ/Myr = 1.899, 1.974, 2.047, 2.204, 2.521, 3.012, 3.464, 3.864, 6.459, 10.40, 20.22, 61.90
```

PISN events use the same `τ_III(m_P)` as their clock.

Pop II lifetime, Raiteri et al. (1996), with `z = max(Z, 7e-5)`:

```
log10 τ_II(m,Z) = a0 + a1 log10 m + a2 (log10 m)^2   [yr]

a0 = 10.13 + 0.07547 logZ − 0.008084 (logZ)^2
a1 = −4.424 − 0.7939 logZ − 0.1187 (logZ)^2
a2 = 1.262 + 0.3385 logZ + 0.05417 (logZ)^2
```

Gates:

```
pre-enrichment:  τ_i ≤ Δ
final mix:       τ_i ≤ Δ + t
```

A long-lived Pop III CCSN can reach the final mix without raising `Z*`.

![Two-clock timeline](../src/potluck/tests/v19/Limongi18/explanation_plots/fig2_timeline.png)

---

## 7. Ejecta matrix and totals

One row per source, one column per element:

```
M ∈ R≥0^(4 × N_X),   s ∈ {PISN, PopIII, PopII, SNIa}
```

```
M_X^(P)      = sum_i M_X,i^(P)   1[τ_III(m_P,i) ≤ Δ+t]
M_X,pre^(P)  = sum_i M_X,i^(P)   1[τ_III(m_P,i) ≤ Δ]
M_X^(III)    = sum_j M_X,j^(III) 1[τ_III(m_III,j) ≤ Δ+t]
M_X,pre^(III)= sum_j M_X,j^(III) 1[τ_III(m_III,j) ≤ Δ]
M_X^(II)     = compound-Poisson draw at (Z*, t, M_*II)
M_X^(Ia)     = 0
```

Totals:

```
M_X       = sum_s M_X^(s)
M_X,pre   = M_X,pre^(P) + M_X,pre^(III)
M_Z^tot   = sum_{X∈Z} M_X
M_Z^pre   = sum_{X∈Z} M_X,pre
```

The matrix is kept, not just the totals. The PISN share and the counterfactual
without PISN are read from its rows.

---

## 8. Metallicity and dilution

```
Z*      = M_Z^pre / M_dil
Z_final = Z* * M_Z^tot / M_Z^pre = M_Z^tot / M_dil
[Fe/H]  = log10( M_Fe / (M_dil κ_Fe) )
```

`Z*` is the metallicity of the gas the Pop II burst forms from. The observed
metallicity includes what Pop II adds later. `[Fe/H]` is derived, not drawn.

The model has two equivalent parameterisations:

- **M_dil-first**: draw `M_dil`, derive `Z* = M_Z^pre / M_dil`.
- **Z*-first**: draw `Z*`, derive `M_dil = M_Z^pre / Z*`.

They are the same physics but different priors. A uniform prior in one is not a
uniform prior in the other. v19 used Z*-first with a table-dependent range; v20
uses M_dil-first with one common prior for all tables.

---

## 9. Gates

The v20 definitions, which are the current recommended comparison gates:

| gate | definition |
|---|---|
| R1 | `n_P + n_III > 0` |
| R2 | `M_Z^pre > 0`, `Z* ≥ 1e-6`, `z_max = None`, and the cloud can host the 1e3 M☉ burst under the 10% cap |
| R3 | Fe fraction ≥ `1e-10` and at least 3 feature elements ≥ `1e-10` relative to `M_Z^tot` |
| R4 (stage 1) | truth `[Fe/H] ∈ [-5, -1]` |
| R4 (Li+ application) | observed `[Fe/H] ∈ [-3.8, -1.7]`, plus Li+ masks, errors and weights |

There is no table-floor rejection. The floor is carried as a `clamped` flag and
handled by the low-Z lookup policy.

---

## 10. Features and labels

Features:

```
x_X = [X/Fe] = a_X − a_Fe,   X ∈ F \ {Fe}
```

Masked invalid if either `X` or Fe is below the validity floor.

Fe-independent pairs:

```
[Na/Mg], [Co/Ni], [Mn/Cr], [Zn/Mg]
```

Labels:

```
f_Z^P  = sum_{X∈Z} M_X^(P) / M_Z^tot      (regression target)
f_Fe^P = M_Fe^(P) / M_Fe                 (diagnostic)
f_X^P  = M_X^(P) / M_X                   (per-element diagnostic)

y_0.5 = 1[f_Z^P ≥ 0.5]
y_0.8 = 1[f_Z^P ≥ 0.8]
```

`f_Z^P` is the analogue of Salvadori's `f_PISN`. `f_Fe^P` matters because the
features are Fe-normalised.

---

## 11. Counterfactual detectability label

Remove the PISN row and keep everything else fixed:

```
M_X^(-P) = M_X − M_X^(P)
x_X^(-P) = [X/Fe]^(-P)
```

Observed shift and Mahalanobis distance:

```
δ_X = x_X − x_X^(-P)
D^2 = δ^T C^-1 δ
y   = 1[D > k]
```

Noiseless covariance (shared reference):

```
C = diag(σ_X^2) + σ_Fe^2 1 1^T,   σ_X = 0.2, σ_Fe = 0.1, k = 3
```

Observed covariance (ratio errors from the catalogue record):

```
C = diag(σ_X,obs^2)
```

Edge cases:

```
n_used < 4:                 D = 0,    y = 0   (insufficient)
∃X: M_X^(-P) ≤ 0:           D = ∞,    y = 1   (PISN-only element)
no PISN:                    D = 0,    y = 0
```

Here `x_X` is the observed `[X/Fe]`. When an observation model is supplied,
the observed branch uses `observation.ratios` and its errors. That is
**realised detection**: "in this noise realisation, is the no-PISN hypothesis
excluded?" Without an observation model, the noiseless branch uses
`features.ratios`.

The v20 label was repaired to use the noisy Li+ observation. The old
true-pattern version is now a separate diagnostic, not the primary label.

Two labels, two questions:

| | `y_0.5` | `y = 1[D > k]` |
|---|---|---|
| Question | Is PISN the majority metal source? | Can the survey tell that PISN is present? |
| Inputs | Ejecta only | Ejecta + element set + errors + `k` |
| Noise-dependent | No | Yes, through the element mask and `σ` |
| Positive rate, Limongi18 v20 | 0.256 | 0.594 |
| Use | Physical interpretation, literature bridge | Operational detection |

A star can be `y_0.5 = 1` but `y = 0`: PISN-dominated, but the pattern is not
distinguishable at the available errors. A star can be `y_0.5 = 0` but
`y = 1`: a minority PISN contribution that still shifts Zn, Co or Si by more
than the errors. In Limongi18 v20, 6,909 stars are `D+ / f05-` and 4 are
`D- / f05+`.

**What it means.** `y` asks: would I notice if the PISN were not there, given
the measured ratios and their errors? A large `f_Z^P` whose pattern other
sources can mimic is still `y = 0`.

**Planned label choice.** Use `y` as the primary classification target for
detection, `y_0.5` as the bridge to Salvadori/Aguado, and `f_Z^P` as the
continuous regression target. The current production paths still mostly use
`y_0.5`; switching the primary metric to `y` is an open task.

---

## 12. Li+ observation model

The noiseless features are made to look like a real Li+ star.

Noise goes on the elements, not the ratios:

```
â_X = a_X + ε_X,   ε_X ~ N(0, σ_X^2)
x̂_X = â_X − â_Fe
Cov(x̂_X, x̂_Y) = δ_XY σ_X^2 + σ_Fe^2
```

Record sampling: pick a Li+ record with
`|[Fe/H]_r − [Fe/H]_true| ≤ 0.2 dex`, then

```
[Fe/H]_obs = [Fe/H]_true + N(0, e_FeH,r)
```

The record supplies the availability mask, `σ_X`, `log g`, and S/N.

Censoring depends on `[X/H] = [X/Fe] + [Fe/H]`. Non-detection thresholds for
Si, Mn, Co, Zn are a monotone lower envelope in S/N, built from the 0th
quantile of `[X/H]` in six log-S/N bins. Carbon limits use a fitted line
`τ_C = α σ_C + β`. Giants (`log g < 3.5`) get extra C scatter of 0.3 dex.

Mocks are reweighted to the Li+ [Fe/H] histogram, separately for giants and
dwarfs:

```
w_i = p_cat([Fe/H]_i) / p_mock([Fe/H]_i)
w_i ← clip(w_i, 1/20, 20)
⟨w⟩ = 1
```

---

## 13. Generation loop

```
for each batch:
    events  = draw_prior()
    ejecta  = build_matrix(events)
    keep    = R1 & R2 & R3
    obs     = sample_observation(ejecta)
    keep   &= R4
    labels  = detect(ejecta, obs)
    store rows
until n_accepted == n_target or n_drawn == 20 n_target
```

Stored per sample: features, masks, observation columns, events, ejecta
totals, labels, prior density, and the full matrix. Metadata records the config
hash, git commit, cut definitions and rejection ledger.

---

## 14. ML and calibration

Design matrix:

```
z_i = ( x̂_X, flag_X, log10 σ_X, 1[giant], [X/Y], σ_X/Y )_i
```

Constant or all-missing columns are dropped.

Classifier and calibrator:

```
p(y_0.5 = 1 | z)  →  isotonic regression  →  prior shift to π
odds' = odds * [π/(1−π)] / [π_train/(1−π_train)]
```

The prior shift is why `p_posterior` is not the raw score. A model calibrated
on a 40–50% positive mock set must be shifted to a 0.25% prevalence.

**What the target prevalence means.** It is the assumed fraction of real stars
that are PISN-dominated. The default is `π = 2.5 × 10^-3`, about one in 400.
It comes from cosmological and chemical-evolution models (de Bennassuti et al.
2017; Salvadori et al. 2019), not from the Li+ catalogue. It is a prior, not a
measurement.

The classifier is trained on mocks where positives are common, roughly 40–50%.
That is not the real sky. A raw score of 0.8 means “0.8 under the training
mix,” not “0.8 under the real mix.” Bayes’ rule converts one to the other:

```
odds_post = odds_pre × [π/(1−π)] / [π_train/(1−π_train)]
```

Worked example. Take a calibrated score of 0.8 with `π_train = 0.4`:

```
odds_pre   = 0.8 / 0.2 = 4
target odds = 0.0025 / 0.9975 = 0.002506
train odds  = 0.4 / 0.6 = 0.6667
ratio       = 0.002506 / 0.6667 = 0.00376
odds_post   = 4 × 0.00376 = 0.0150
p_post      = 0.0150 / 1.0150 = 0.0148
```

So 0.8 becomes about 1.5%. A raw 0.5 becomes about 0.4%. A raw 0.99 becomes
about 27%. That is the rareness effect: even a confident classifier cannot give
a high posterior when the target is one star in 400.

The shift is monotonic, so the ranking of stars does not change. AUC is
unaffected by prevalence. Precision, the posterior scale and any candidate
threshold do depend on it. The `p_calibrated` column is the probability under
the training mix; `p_posterior` is after the shift. `expected_fz` is not shifted
because it is a regression estimate, not a class probability.

The 0.25% value should be stated as an assumption, and a small sensitivity band
(for example 0.1%, 0.25%, 1%) is worth reporting. If the true prevalence is
different, the posterior changes, but the ranking does not.

Support gate:

```
d_r^2 = min_i sum_{X∈core} ((x_r,X − x_i,X) / σ_r,X)^2
```

with core elements Na, Mg, Ca, Ti, Cr, Ni, at least four available. A star is
out of model if `d_r` exceeds the 99th percentile of the mock-vs-mock
distribution. The gate is a systematics diagnostic, not a physical validation.

Regression: a second tree model predicts `f_Z^P`.

### 14.1 What `P(f_Z^P ≥ 0.5 | z)` is, and what it is not

Three objects live in the same 0-1 range and are easy to conflate.

**(i) The physical fraction.** A property of the enrichment history, known
before any observation:

```
f_Z^P = sum_{X∈Z} M_X^(P) / M_Z^tot  ∈ [0, 1]
```

It is continuous, and it is the regression target.

**(ii) The binary label.** A deterministic function of the fraction:

```
y_0.5 = 1[f_Z^P ≥ 0.5]  ∈ {0, 1}
```

It is not a probability. `f_Z^P = 0.49` gives 0 and `f_Z^P = 0.51` gives 1.

**(iii) The classifier probability.** The quantity reported for a real star:

```
p(z) = P(y_0.5 = 1 | z) = P(f_Z^P ≥ 0.5 | z)
```

The two forms are identical because `y_0.5` is a function of `f_Z^P`, not an
independent random variable. The classifier estimates a conditional probability
over the class, not the value of the fraction. A star with `f_Z^P = 0.51` and one
with `f_Z^P = 0.99` have the same label, so the model cannot separate them
through this target.

**Calculation chain.** All steps are in `src/potluck/calibration.py`:

```
z                  design matrix (§14)
s(z)               HistGradientBoostingClassifier.predict_proba      (126, 401)
p_cal = g(s)       IsotonicRegression on the held-out split           (127-132)
p_post             prior shift to the target prevalence π             (57-68, 403)
```

with the shift written as odds:

```
p_post / (1 - p_post) = [p_cal / (1 - p_cal)] × [π / (1 - π)] / [π_train / (1 - π_train)]
```

Isotonic regression makes `p_cal` a calibrated probability under the training
mix. The odds shift re-expresses the same ranking at the assumed sky prevalence
`π = 2.5 × 10^-3`. Neither step reorders stars; both change the scale.

Isotonic calibration is a **step function**. In the Li+ candidate table it
produces only 9 distinct `p_calibrated` levels for 385 stars (168 stars share
0.0078, 186 share 0.0415, and so on). That is why the scatter shows flat rows:
many stars get the same calibrated probability even though the regressor gives
them different `expected_fz`. The probability scale is coarse; the continuous
regressor is the finer ranking. A smoother calibration (Platt scaling) or more
calibration data would reduce the flat bands.

`p_low` and `p_high` are a 5-95% bootstrap interval on `p_posterior`, obtained
by resampling the calibration set, refitting isotonic calibration, and
reapplying the prior shift. They describe uncertainty in the **probability
estimate**, not uncertainty in the binary label. The true label is still 0 or 1.
The interval does not include yield-table systematics, target-star observational
errors, covariate shift, or prior uncertainty on the prevalence, so it is a
lower bound on the total uncertainty.

**The separate regressor.** A second model estimates the fraction itself:

```
f̂_Z^P(z) = E[f_Z^P | z]  ∈ [0, 1]      (HistGradientBoostingRegressor, 151-155, 404)
```

It is not prior-shifted, because it is not a class probability. Its output is
`expected_fz` in the candidate table.

**How `p` and `f_Z^P` are related.** There is no fixed conversion between them;
they are different functionals of the same conditional distribution. If
`p(f_Z^P | z)` were known,

```
p      = P(f_Z^P ≥ 0.5 | z) = ∫_{0.5}^{1} p(f_Z^P | z) d f_Z^P      (tail mass)
f̂_Z^P  = E[f_Z^P | z]       = ∫_{0}^{1} f_Z^P p(f_Z^P | z) d f_Z^P  (mean)
```

The two coincide only when the conditional distribution collapses to a point.
For a real abundance pattern it does not: the same `[X/Fe]` vector is produced by
several combinations of PISN mass, Pop II/III contribution, dilution mass and
yield table. `p` therefore answers "how much of the probability sits above the
0.5 cut?", while `f̂_Z^P` answers "where is the distribution centred?". Two stars
with the same `f̂_Z^P` can carry different `p` if their conditional widths
differ.

Empirically, on the Li+ candidate table, the two are positively but not
perfectly correlated: Pearson 0.74, Spearman 0.68. A star with
`expected_fz = 0.55` can have `p_posterior = 0.019`, because the posterior is
shifted to the 0.25% prevalence while the regressor is not.

![p_posterior vs expected fZP](../src/potluck/tests/v20/processed/li_p_posterior_vs_expected_fz.png)

| quantity | domain | meaning | prior-shifted | noise-dependent |
|---|---|---|---|---|
| `f_Z^P` | `[0, 1]` continuous | true PISN share of metals | no | no |
| `y_0.5` | `{0, 1}` | PISN-dominated label | no | no (ejecta only) |
| `p(z)` | `[0, 1]` continuous | `P(f_Z^P ≥ 0.5 given z)` | yes (`p_posterior`) | yes |
| `f̂_Z^P(z)` | `[0, 1]` continuous | `E[f_Z^P given z]` | no | yes |

Three practical consequences:

1. `p = 0.6` with `f̂_Z^P = 0.4` is coherent, not contradictory: the best point
   estimate sits below the cut, but more than half of the conditional
   probability lies above it.
2. `p` is not an estimate of the fraction. Reporting `p` as "the star is 60%
   PISN" conflates a tail mass with a mean.
3. `p_post` depends on the assumed `π`; `f̂_Z^P` does not. Changing the
   prevalence rescales every posterior without touching the regressor.

The empirical version of this relation is the §15.5m table: binning held-out
mocks by **predicted** `f_Z^P` and reporting `P(true ≥ 0.5)` in each bin is
exactly the quantity `p` estimates, measured directly instead of by the
classifier.

---

## 15. Validation results

### 15.1 Salvadori Eq. 13

The matched slice is one PISN, no Pop III, one Pop II burst, WW95. It
reproduces Eq. 13 to machine precision:

| m_PISN | f_PISN | max abs difference |
|---:|---|---:|
| 150 | 0.5 / 0.7 / 0.9 | ~1e-15 dex |
| 200 | 0.5 / 0.7 / 0.9 | ~1e-15 dex |
| 250 | 0.5 / 0.7 / 0.9 | ~1e-10 dex |

The `m = 250` residual is floating point and table-selection noise. The test
compares potluck's matrix construction with the legacy Eq. 13 implementation.
It validates the bookkeeping, not the yields.

### 15.2 Direct yields

IMF-integrated `[X/Fe]` at fixed `Z*` and `t`, no simulation. Spread across the
six v20 policies at `t = 10 Myr`:

| element | Z*=1e-5 | Z*=1e-4 | Z*=1e-3 | Z*=1e-2 |
|---|---:|---:|---:|---:|
| Mg | 0.50 | 0.39 | 0.46 | 0.38 |
| Si | 0.32 | 0.41 | 0.48 | 0.54 |
| Ca | 0.43 | 0.46 | 1.98 | 0.47 |
| Ti | 0.63 | 0.72 | 0.63 | 0.50 |
| Cr | 0.64 | 0.73 | 0.71 | 0.47 |
| Mn | 0.72 | 0.83 | 0.68 | 0.80 |
| Co | 2.77 | 1.12 | 2.08 | 0.21 |
| Ni | 0.55 | 0.77 | 1.00 | 0.56 |
| Zn | 1.03 | 1.01 | 0.82 | 1.04 |



Co and Zn are the worst. That is the yield-systematic floor.

PISN minus Pop II at `Z*=1e-5`, `t=10 Myr`:

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

The killing elements are Co and Zn. The contrast is large, but its size depends
on the table.

### 15.3 Equal-gate prior check

v20 uses one M_dil prior, one `z_cr`, and one broad `[Fe/H]` window for every
policy. The accepted sample is the same:

| policy | acceptance | zero-PISN | M_Zpre p16/50/84 | below floor |
|---|---:|---:|---:|---:|
| Limongi18 clamp | 0.511 | 0.368 | 87 / 226 / 1090 | 0.520 |
| WW95 native | 0.508 | 0.369 | 88 / 225 / 1098 | 0.038 |
| Nomoto13 blend | 0.512 | 0.366 | 87 / 227 / 1105 | 0.967 |
| Nomoto13 clamp | 0.512 | 0.366 | 87 / 227 / 1105 | 0.967 |

The `below floor` column is the fraction of mocks below the table's lowest
nonzero node. It is 52% for Limongi18, 3.8% for WW95, and 97% for Nomoto13. That is a low-Z
policy difference, not a selection difference.

### 15.4 M_dil prior sweep

Three common priors, same gates, policies, seed and 10,000 mocks per policy.
Only `mdil_range` changes:

| prior | acceptance | zero-PISN | median Z* | below floor | within AUC | LOTO AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1e5–5e8 | 0.533 | 0.373 | 3.7e-5 | 0.719 | 0.970 | 0.969 |
| 2e5–1e9 | 0.512 | 0.368 | 2.8e-5 | 0.738 | 0.973 | 0.966 |
| 1e6–1e9 | 0.499 | 0.360 | 1.6e-5 | 0.769 | 0.974 | 0.965 |

The accepted sample and the classifier are stable. The absolute metallicity
moves as expected: median `Z*` shifts by 0.35 dex, below-floor by 0.05. This is
the defensible statement: the pattern classification is robust to the M_dil
prior; the absolute Z*, PISN fraction and Li+ support are prior-dependent.


### 15.5 RF performance and transfer

Policy coverage differs by analysis. Do not compare values across groups
without checking this.

| analysis | tables |
|---|---|
| leave-one-out element influence (f05 and D) | 4 |
| pair ratios | 4 |
| element subsets | 4 |
| Li+ conditional map | 4 |
| operating points, simple baselines | 2 |

All diagnostics now run on the same four policies: Limongi18 clamp, WW95 native,
Nomoto13 blend and Nomoto13 clamp. The CK20 variants (blend and clamp) are
excluded from every plot and table pending review of that yield table.

Within-table observed AUC, class-balanced, v20:

| policy | AUC | Brier |
|---|---:|---:|
| Limongi18 clamp | 0.978 | 0.057 |
| WW95 native | 0.969 | 0.072 |
| Nomoto13 blend | 0.965 | 0.075 |
| Nomoto13 clamp | 0.977 | 0.061 |

Cross-table observed AUC, all samples:

|---|---:|---:|---:|---:|---:|---:|
| Limongi18 clamp | 0.997 | 0.901 | 0.944 | 0.965 | 0.968 | 0.970 |
| WW95 native | 0.973 | 0.997 | 0.948 | 0.963 | 0.969 | 0.970 |
| Nomoto13 blend | 0.972 | 0.904 | 0.996 | 0.989 | 0.965 | 0.961 |
| Nomoto13 clamp | 0.975 | 0.901 | 0.971 | 0.998 | 0.970 | 0.969 |

Leave-one-table-out observed AUC: 0.902–0.986. WW95 is the hardest held-out
table at about 0.90. The v19 value of 0.62 was a selection and class-balance
artifact, not a yield effect.



Noise sweep: within-table AUC goes from about 0.995 at σ=0 to 0.94–0.96 at
σ=0.3.


### 15.5b Low-resolution proxy and element subsets

Class-balanced within-table AUC, mean over the six v20 policies, 5,000 mocks
per policy and 50 trees. The generic sigma is a low-resolution proxy, not a full
PFS simulation.

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
Na, Mg, Si, Ca, Ti, Cr, Ni retains 0.941 at sigma=0.2 and 0.896 at sigma=0.3.
Adding Zn gives 0.966/0.940. This is the first pass at the PFS question: a
low-resolution survey without Zn loses about 0.02–0.04 AUC, and one with only
alpha elements loses much more.


### 15.5c Conditional element value

An add-one-element analysis on top of a fixed base set. Class-balanced
within-table AUC, 5,000 mocks per policy, 50 trees, noiseless features.

| base | base AUC | add Cu | add Zn | add Co | add Si |
|---|---:|---:|---:|---:|---:|
| none | 0.500 | +0.354 | +0.387 | +0.359 | +0.094 |
| Zn | 0.887 | +0.032 | — | +0.038 | +0.078 |
| Cu | 0.854 | — | +0.065 | +0.053 | +0.108 |
| Zn + Co | 0.925 | +0.013 | — | — | +0.054 |
| Zn + Cu | 0.919 | — | — | +0.019 | +0.058 |
| core 7 | 0.992 | −0.001 | −0.001 | −0.001 | — |

Cu is a strong standalone discriminant, but conditional on Zn it adds about
0.03 AUC, and conditional on Zn+Co about 0.01. Zn adds more to Cu than Cu adds
to Zn. In the noiseless full pattern, the core 7 already reaches 0.992 and the
killing elements are redundant. In the observed Li+ block (no Cu), Zn alone is
0.845, core 7 is 0.941, and core 7 + Zn is 0.964. Adding Zn to the core 7 gains
about 0.023; adding Co gains about 0.005.

Cu is present in the noiseless 27-element model but absent from the Li+
catalogue. It is a high-priority target for a future survey with Cu coverage,
but it is not part of the current real-star classifier.

![Cu bimodality](../src/potluck/tests/v20/element_bimodality/Limongi18_clamp/element_bimodality_noiseless.png)

![Conditional element value](../src/potluck/tests/v20/element_conditional/element_conditional_noiseless.png)

![Conditional element value, observed](../src/potluck/tests/v20/element_conditional/element_conditional_observed.png)

### 15.5d Element influence with all 26 ratios

Leave-one-out from the full 26-ratio set, Gaussian noise sigma=0.2. Delta AUC is
full minus without the element.

**These numbers are from the earlier six-policy run and have not been
recomputed.** Re-run the diagnostic to refresh them for the current four
policies:

| element | Delta AUC | 95% CI | top-1% turnover |
|---|---:|---|---:|
| Si | +0.0041 | +0.0033 to +0.0050 | 0.65 |
| Zn | +0.0022 | +0.0015 to +0.0031 | 0.60 |
| Al | +0.0018 | +0.0013 to +0.0024 | 0.60 |
| S | +0.0018 | +0.0011 to +0.0026 | 0.68 |
| Co | +0.0005 | +0.0000 to +0.0011 | 0.60 |
| Cu | +0.0002 | -0.0004 to +0.0007 | 0.64 |

No single element is essential when all 26 ratios are present. Every Delta AUC
is at most 0.004, and precision at FPR 1% changes by at most 0.003. The top-1%
turnover is high for every element, but the top 1% is already pure PISN:
precision at top 1% is 1.00 with and without Zn, so the turnover is
tie-breaking, not a loss of detections.

**Three different questions, do not conflate them.** The same element scores very
differently depending on which one is asked.

| question | metric | Zn |
|---|---|---|
| How well does this one ratio separate? | standalone AUC (§15.5, bimodality) | **0.87-0.92, best of all** |
| What does it add to a small set? | core 7 $\to$ core 7 + Zn | **0.950 $\to$ 0.974** |
| What does it add to the full set? | leave-one-out $\Delta$AUC | **+0.001, not resolvable** |

Zn is the strongest single tracer and the most valuable addition to a sparse set.
Its leave-one-out value in the 26-ratio set is small only because
`[Zn/Fe]` is 0.86 correlated with `[Cu/Fe]`, 0.82 with `[Ga/Fe]` and 0.72 with
`[Co/Fe]`, all of which are already present. The information is redundant, not
absent. "Zn is not essential once Cu and Ga are measured" is a statement about
redundancy; "Zn is not important" would be wrong.

The leave-one-out $\Delta$AUC also sits below the retraining noise floor. Across 4
seeds with retraining, the rank correlation between runs is only 0.47
(0.22-0.69), and only Si, S, Ar, Al, P and C clear
$\Delta\mathrm{AUC}/\mathrm{sd} > 2$. Ranks 7-26 are not resolvable. See
`diagnostics/rank_stability.py`.

The effect appears below the top 1%, where the candidate list is not saturated.
For the sparse core 7:

| top K% | core 7 | core 7 + Zn |
|---:|---:|---:|
| 1% | 1.000 | 1.000 |
| 5% | 0.972 | 0.992 |
| 10% | 0.948 | 0.984 |
| 20% | 0.871 | 0.938 |

In the full 26-ratio set, all 26 and all 26 minus Zn are identical at every K.
Zn is redundant when the rest of the pattern is measured. It matters when the
element set is sparse, and at the 5-20% budget, not at the top 1%.

![Element influence, AUC](../src/potluck/tests/v20/element_influence/leave_one_out_auc.png)

![Element influence, precision](../src/potluck/tests/v20/element_influence/leave_one_out_precision_1pct.png)

![Zn top-1% scores](../src/potluck/tests/v20/element_influence/zn_top1_scores.png)

![Zn precision at K](../src/potluck/tests/v20/element_influence/zn_precision_at_k.png)

### 15.5e Primary-label check: y = 1[D > 3] (repaired)

This is still a check, not the production label. The current f05 results are
unchanged. The v20 files were repaired in place: `D`, `D2`, `y`,
`detect_reason` and `n_detected` now come from the stored noisy observation
(`obs_features`), matching `detect_labels`.

Positive rate is 0.594 in Limongi18 (f05 0.256) and 0.599 in WW95 (f05 0.231). Overlap in Limongi18: 6,909 stars are
`D+ / f05-`, 4 are `D- / f05+`.

`k = 3` is a distance cut, not a p-value. With 11 elements, p(D^2 > 9) is about
0.6, so the label is broad and includes noise. A p-value threshold is the next
calibration step.

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

![D-label element influence](../src/potluck/tests/v20/element_influence_D/leave_one_out_auc.png)

![D-label Zn precision at K](../src/potluck/tests/v20/element_influence_D/zn_precision_at_k.png)

### 15.5f Pair ratios: Mg/Si and friends

The model sees all 26 `[X/Fe]` ratios together, but not non-Fe pairs such as
`[Mg/Si]`. Standalone AUC of selected pairs:

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

The existing independent ratios are mixed: Zn/Mg is strong (0.817), Na/Mg
moderate (0.735), Co/Ni weak (0.636), Mn/Cr is random standalone (0.509). The
full `[X/Fe]` set already contains the pair information.

![Pair ratio standalone AUC](../src/potluck/tests/v20/pair_ratios/pair_standalone_auc.png)

![Pair ratio added to core 7](../src/potluck/tests/v20/pair_ratios/pair_added_to_core7.png)

![Pair ratio added to full](../src/potluck/tests/v20/pair_ratios/pair_added_to_full.png)

### 15.5g General element subset, no Li+ bias

Greedy forward selection over 20 practical elements, Gaussian element-level
noise sigma=0.2, two yield tables (Limongi18 and WW95), both labels. No Li+ masks or catalogue
availability.

| label | selection order |
|---|---|
| `y_f_05` | Zn, Si, Al, S, Co, P, Ca, O, Cr, Na |
| `y = 1[D > 3]` (repaired) | Zn, Si, Al, Mn, Cu, S, P, N, V, C |

The general model does not pick Na early; it picks Al and S. Na looked
important in the Li+ analysis because Li+ has no Al, S, O or P. That is an
availability effect, not a physical ranking.

The repaired D label is realised detection with a high positive rate (~0.59),
so its AUCs are lower and not comparable to the old true-pattern D values. The
f05 selection below is the stable one.

Joint candidates, mean f05 AUC over Limongi18 and WW95:

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

![General element selection, f05](../src/potluck/tests/v20/element_selection/selection_curves_f05.png)

![General element selection, D](../src/potluck/tests/v20/element_selection/selection_curves_D.png)

### 15.5h General conditional map, single-element bases

Rows are single-element bases, columns are added elements, value is Delta AUC.
The row label gives the standalone AUC of the base. Gaussian sigma=0.2, two
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

![General conditional map, f05](../src/potluck/tests/v20/element_conditional_general/conditional_general_f05.png)

![General conditional map, D](../src/potluck/tests/v20/element_conditional_general/conditional_general_D.png)

![General standalone AUC, f05](../src/potluck/tests/v20/element_conditional_general/standalone_auc_f05.png)

![General standalone AUC, D](../src/potluck/tests/v20/element_conditional_general/standalone_auc_D.png)

### 15.5i Operating points at survey prevalence

5-fold OOF, 100 trees, full 20k per policy, f05 label. The general 8-element
set is the headline; the Li+ 11 is a comparison. TPR is at a fixed FPR;
precision uses pi = 0.1%, 0.25% and 1%.

| policy | feature set | TPR @ FPR 1% | TPR @ FPR 0.1% | precision @ 0.1% | @ 0.25% | @ 1% | TP / FP per 1e4 @ FPR 0.1% | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Limongi18 | general 8 | 0.741 | 0.493 | 0.331 | 0.553 | 0.833 | 12.3 / 10.0 | 0.050 |
| WW95 | general 8 | 0.673 | 0.405 | 0.288 | 0.504 | 0.804 | 10.1 / 10.0 | 0.058 |
| Limongi18 | Li+ 11 | 0.727 | 0.413 | 0.292 | 0.509 | 0.807 | 10.3 / 10.0 | 0.051 |

The general 8 is better than the Li+ 11 at the low-FPR operating point. WW95
is the weakest, consistent with the cross-table outlier.

FPR split at overall FPR = 0.1%:

| policy | clear negatives (f_Z < 0.25) | near-threshold (0.25-0.5) |
|---|---:|---:|
| Limongi18 | 0.00061 | 0.00365 |
| WW95 | 0.00015 | 0.00618 |

The false positives are dominated by near-threshold negatives, by a factor of
6-40. The model is not confusing clear non-PISN stars; it is confusing
ambiguous mixtures. That is the physically expected failure mode.

At pi = 0.25%, the operating point gives about 11-12 true PISN and 10 false
positives per 10,000 stars, precision about 0.50-0.55. At pi = 0.1%, precision
is 0.29-0.33; at pi = 1%, it is 0.80-0.83. AUC 0.97 is not the same as a clean
detection.

![TPR at fixed FPR](../src/potluck/tests/v20/operating_points/tpr_at_fpr.png)

![Precision vs prevalence](../src/potluck/tests/v20/operating_points/precision_vs_prevalence.png)

![Reliability](../src/potluck/tests/v20/operating_points/reliability.png)

![FPR groups](../src/potluck/tests/v20/operating_points/fpr_groups.png)

### 15.5j Simple-rule baselines at the same FPR

OOF scores, Gaussian sigma=0.2, f05 label, 2 policies. Thresholds are chosen
at the same target FPR on the negative class. TPR at FPR = 0.1%:

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
   pattern matters; a two-ratio cut does not recover it.
2. A linear model on the same 8 elements is almost as good as the RF. The RF
   adds only 0.00-0.07 TPR at FPR 0.1%, and 0.02-0.03 at FPR 1%. The value is
   in the 8-element combination, not in the nonlinearity.

The RF is retained as the primary classifier because it is competitive
within-table and not worse on average. The logistic on the general 8 is the
transparent baseline. Do not present the RF as the innovation; the 8-element
pattern is the result.

![Simple-rule baselines](../src/potluck/tests/v20/simple_baselines/baseline_comparison.png)

### 15.5k Cross-table RF vs logistic (v21)

Train on one yield table, test on the others. General 8, sigma=0.2, f05. Six
policies, 30 cross-table pairs.

| metric | RF | logistic |
|---|---:|---:|
| mean cross-table AUC | 0.961 | **0.964** |
| mean cross-table TPR @ FPR 0.1% | 0.375 | **0.377** |
| AUC wins (of 30) | 17 | 13 |
| TPR wins (of 30) | 15 | 15 |

The models are tied on the mean, and the RF's within-table edge (AUC about
0.975 vs 0.970; TPR about 0.45 vs 0.43) does not survive the table change. On
WW95 as the test table, logistic AUC is 0.954-0.960 against 0.894-0.930 for the
RF, the clearest sign that the RF is fitting table-specific structure.

The RF does not transfer better on average, but it is not worse on average
either. It is retained as the primary classifier, with the logistic on the
general 8 as the transparent baseline and the WW95 transfer as a known weakness.

![v21 cross-table AUC delta](../src/potluck/tests/v21/delta_auc.png)

### 15.5l Why the 0.2-0.4 range is ambiguous

`f_ZP` is continuous. The 0.5 cut is a convention, not a physical boundary.
The model is most reliable at the ends and least reliable near the boundary.

| range | interpretation | model behaviour |
|---|---|---|
| ~0 | no PISN pattern | confident negative |
| 0.05-0.2 | trace contribution | mostly negative |
| 0.2-0.4 | mixed / ambiguous | near the boundary, table-sensitive |
| 0.4-0.6 | boundary region | lowest confidence |
| >0.6 | PISN-dominated | confident positive |

Near 0, the pattern is consistent with no PISN. Above 0.6, the PISN
signature is strong. In between, several mixtures of PISN, Pop II, Pop III and
dilution can produce similar `[X/Fe]`; a small change in one element, the
errors, or the yield table can move the prediction across 0.5. The classifier
probability is therefore close to the decision boundary, and the continuous
regressor has its widest conditional spread there.

Three pieces of evidence:

1. The FPR split. At overall FPR = 0.1%, near-threshold negatives
   (`0.25 < f_ZP < 0.5`) have 6-40 times the false-positive rate of clear
   negatives (`f_ZP < 0.25`). The model is not confusing obvious non-PISN
   stars; it is confusing genuine mixtures.
2. The mock distribution. About 48% of mocks are below `f_ZP = 0.05`, and about
   18% are above 0.6; only about 10% lie in 0.2-0.4. The middle is both
   physically mixed and less densely sampled.
3. The yield-table spread. The per-policy APOGEE medians range from 0.000 to
   0.062, and the number of stars above 0.25 ranges from 2 to 149. Predictions
   in the middle move most between tables.

Practical rule: do not call a 0.2-0.4 prediction "PISN" or "non-PISN". Call it
an **ambiguous or subdominant candidate**. Report the continuous `f_ZP`, not
only the binary label, and require additional elements (Zn, Cu), higher S/N, or
follow-up before treating it as a detection.

### 15.5m What a predicted fZP actually means

Held-out mocks, Limongi18 and Nomoto13. Bin by **predicted** `f_ZP`; report the
true `f_ZP` in that bin. This is the evidence for why a 0.4 prediction is a
candidate, not a confident PISN-dominated detection.

| predicted bin | median true | P(true ≥ 0.5) | P(true < 0.2) | 90% true interval |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 0.00 | 0.3-0.4% | 94% | 0.00-0.14 |
| 0.1-0.2 | 0.05 | 2.6-3.1% | 76% | 0.00-0.35 |
| 0.2-0.3 | 0.20-0.22 | 13-14% | 46-50% | 0.00-0.55 |
| 0.3-0.4 | 0.39-0.40 | 32-33% | 23-26% | 0.00-0.69 |
| 0.4-0.5 | 0.51-0.54 | **53-58%** | 9-12% | 0.15-0.79 |
| 0.5-0.6 | 0.64-0.65 | 72-77% | 4-7% | 0.33-0.87 |
| 0.6-0.8 | 0.77 | 90-92% | 1-2% | 0.51-0.95 |
| 0.8-1.0 | 0.94-0.95 | 99% | <0.3% | 0.75-1.00 |

The 0.5 cut is not resolved at the individual-star level:

- Predicted 0.4-0.5: about half are truly above 0.5, but the 90% interval spans
  0.15-0.79. It is a candidate, not a confident dominated star.
- Predicted 0.3-0.4: only about one third are truly above 0.5; two thirds are
  below. It is ambiguous.
- Predicted 0.5-0.6: still 23-28% are truly below 0.5.
- Predicted >0.6: 90-99% are truly above 0.5. This is the high-confidence tier.

Practical tiers:

| tier | criterion | claim |
|---|---|---|
| high-confidence dominated | predicted > 0.6, or lower CI bound > 0.5 | PISN-dominated |
| ambiguous candidate | predicted 0.3-0.6 | possible subdominant PISN; follow-up |
| low-likelihood | predicted < 0.2 | unlikely to be dominated; not proof of no PISN |

If the science target is **any PISN contribution**, a 0.4 prediction is a valid
candidate and should be reported separately. If the target is **PISN-dominated**,
a 0.4 prediction is not enough: the confidence interval crosses 0.5. That is the
distinction to state explicitly.

### 15.5n Mock coverage and high-fZP recovery

This checks whether the low APOGEE predictions are caused by too few
high-fZP mocks. They are not.

Per policy (20k mocks):

| policy | PISN present | fZP > 0.1 | fZP >= 0.5 |
|---|---:|---:|---:|
| Limongi18 | 12,925 | 9,713 | 5,245 |
| Nomoto13 blend | 13,001 | 8,969 | 4,139 |
| Nomoto13 clamp | 13,001 | 8,958 | 4,141 |

Held-out recovery, train on 4/5 and predict the held-out fifth:

| true fZP | median predicted | fraction predicted >= 0.5 |
|---|---:|---:|
| 0.2-0.3 | 0.16-0.17 | 2-2.4% |
| 0.3-0.4 | 0.22-0.24 | 5-8% |
| 0.4-0.5 | 0.30-0.34 | 14-20% |
| 0.5-0.7 | 0.43-0.47 | 36-44% |
| 0.7-1.0 | 0.81-0.86 | 88-92% |

The high-fZP mocks are abundant and recovered: 88-92% of true 0.7-1.0 mocks
are predicted above 0.5. The 0.5-0.7 bin is the boundary region. The low APOGEE
values are therefore a statement about the APOGEE patterns and the missing
Zn/Cu, not a shortage of high-fZP training data.

Files: `src/potluck/tests/v20/processed/apogee/coverage_check/mock_counts.csv`,
`heldout_recovery.csv`, `coverage_check.png`.

### 15.6 Li+ application

Separate stage: 10,000 mocks per policy, Li+ box, masks, errors, weights and
support. This is an application, not a physics validation.

| policy | out-of-model | in-model | max posterior | median expected f_ZP |
|---|---:|---:|---:|---:|
| Limongi18 clamp | 353 / 385 | 32 | 0.019 | 0.10 |
| WW95 native | 311 / 385 | 74 | 0.006 | 0.08 |
| Nomoto13 blend | 360 / 385 | 25 | 0.022 | 0.06 |
| Nomoto13 clamp | 352 / 385 | 33 | 0.043 | 0.06 |

WW95 has the best Li+ support, then Nomoto13 clamp. Limongi and Nomoto13 blend
are worst. No star reaches `p_posterior ≥ 0.5` at the assumed prevalence. The
out-of-model fractions are dominated by per-element yield offsets (Ti, Co, Zn,
Mn), not by the low-Z policy alone.

**Why nothing reaches 0.5: two separate causes.** They are easy to conflate.

The prior shift does not touch the ranking. Spearman between `p_calibrated` and
`p_posterior` is exactly 1.000 in all four policies, because the shift multiplies
the odds by a positive constant. The candidate order is unchanged. But it does
dominate the histogram: the top of the distribution moves from 0.39-0.79
calibrated to 0.006-0.043 posterior, which is why the plot is a spike at zero.

The shift is not why no star is a candidate. Across the four policies it converts
only a handful of stars from above 0.5 to below it (2 in Limongi18, the rest in
the two Nomoto13 policies); the overwhelming majority were already below 0.5
before any shift.

The real cause is the evidence. Raw classifier scores:

| | median | 90th pct |
|---|---:|---:|
| real Li+ stars | 0.014 | 0.057 |
| mock negatives | 0.020 | 0.200 |
| mock positives | 0.877 | 0.983 |

Real stars are scored like mock negatives, and only 3.6% exceed the mock-negative
90th percentile. In prevalence-free terms the best star in the catalogue carries
a likelihood ratio of 7.9, while a posterior of 0.5 at `π = 2.5e-3` requires
LR ≈ 399. The catalogue is a factor of 50 short of the threshold.

Sweeping the assumed prevalence shows what would have to be true:

| assumed `π` | max posterior | stars ≥ 0.5 |
|---:|---:|---:|
| 0.0025 | 0.019 | 0 |
| 0.05 | 0.293 | 0 |
| 0.10 | 0.466 | 0 |
| 0.20 | 0.663 | 2 |
| 0.50 | 0.887 | 13 |

Roughly 20% of metal-poor stars would have to be PISN-dominated before any Li+
star crosses. So report the likelihood ratio, which is what the data support, or
expected true positives at a fixed threshold; do not report the posterior
histogram as a statement about the stars.

### 15.6a Reading the Li+ histograms

Four figures, and they are not interchangeable.

| figure | quantity | prior-shifted |
|---|---|---|
| `li_histograms/li_posterior_hist.png` | `p_posterior` | yes |
| `li_histograms/li_posterior_overlay.png` | `p_posterior` | yes |
| `application/<policy>/li_validation/real_stars.png` (left) | `p_posterior` | yes |
| `li_histograms/li_expected_fz_hist.png` | `expected_fz` | **no** |
| `li_histograms/li_expected_fz_overlay.png` | `expected_fz` | **no** |

**The posterior plots have a 0-0.05 x-axis, not 0-1.** The limit is set from the
data at `li_histograms.py:149`:

```python
post_high = max(0.05, np.nanpercentile(post_values, 99) * 1.2)
```

so every star is squeezed into the first 5% of a unit interval. Medians are
0.0000-0.0004 and the 95th percentiles are 0.0005-0.0018, i.e. the panel is one
tall bar at the origin.

**That shape is the prior, not the stars.** For small probabilities
`p_posterior ≈ π × LR`, so the whole histogram is the likelihood-ratio
distribution multiplied by 0.0025. Set `π = 0.05` and the same 385 stars would
fill the axis and look like a healthy distribution; nothing about the stars
changed. A spike at zero is what any catalogue produces under a 1-in-400 prior,
however good the classifier is. It is not evidence that the Li+ stars are
PISN-free.

**The `expected_fz` plots are the informative ones** because the regressor is
never shifted. They have real structure: median 0.06-0.10, 84th percentile
0.09-0.16, maximum 0.28-0.55. That is a compressed distribution, but it is a
distribution.

Practical rule: use the posterior plot to state the absolute probability under an
explicit prior, and the `expected_fz` plot to say something about the stars. A
side-by-side works as a teaching figure only if the caption gives the 0-0.05 axis
limit on the first panel.

![Li+ posterior histogram, note the 0-0.05 x-axis](../src/potluck/tests/v20/li_histograms/li_posterior_hist.png)

![Li+ expected f_ZP histogram, not prior-shifted](../src/potluck/tests/v20/li_histograms/li_expected_fz_hist.png)

A direct test of the gate makes this clear. The raw gate uses Na, Mg, Ca, Ti,
Cr, Ni with no offsets. Removing Ti, or allowing one offset per element, changes
the in-model fraction dramatically:

| dataset | raw | no Ti | offsets | no Ti + offsets |
|---|---:|---:|---:|---:|
| v20 Limongi18 | 0.083 | 0.847 | 0.925 | 0.875 |
| v20 WW95 | 0.192 | 0.683 | 0.948 | 0.891 |
| v20 Nomoto13 blend | 0.065 | 0.397 | 0.927 | 0.865 |
| v20 Nomoto13 clamp | 0.086 | 0.506 | 0.927 | 0.868 |
| v19 Limongi18 | 0.096 | 0.847 | 0.943 | 0.896 |
| v19 Nomoto13 | 0.083 | 0.545 | 0.940 | 0.909 |
| v19 WW95 | 0.138 | 0.647 | 0.938 | 0.883 |

The Ti offset is +0.26 to +0.73 dex, with a bootstrap interval of about
±0.01 dex and no [Fe/H] trend (Spearman rho ≈ +0.06). It is a global
zero-point, not a per-star anomaly. Allowing one offset per core element brings
every dataset to 91–95% in model. Removing Ti alone brings 40–85%. The raw gate
is a zero-point diagnostic, not a statement that most Li+ stars are odd.

![Support-gate variants](../src/potluck/tests/v20/support_variants/support_variants.png)

![Per-element offsets](../src/potluck/tests/v20/support_variants/support_offsets.png)



### 15.7 M_dil inverse problem (v18)

v18 solved M_dil per star from `[Fe/H]` under prior event draws and weighted by
`[X/Fe]`. The supported range is about `1e6–3e8 M☉` (16–84 over viable
configurations), with p05–p95 about `9e4–6e8`. That range is prior-predictive,
not a measurement. The median is about `2e7 M☉`. The event prior alone spans a
similar range, so the data only weakly narrow it.

![Supported M_dil by configuration](../src/potluck/tests/v18/mdil_support_by_config.png)

![Element offsets](../src/potluck/tests/v18/element_offsets.png)

### 15.8 APOGEE DR17 application

APOGEE allStarLite DR17, 733,901 rows, 657,135 unique `APOGEE_ID`. After
deduplication (highest SNR), `[Fe/H]` in `[-5, -1]`, `FE_H_FLAG == 0`,
`SNR >= 50`, and at least 8 of 16 features, 18,962 stars remain.

Features are all potluck elements with finite APOGEE values: C, N, O, Na, Mg,
Al, Si, S, K, Ca, Ti, V, Cr, Mn, Co, Ni. Zn is absent; Cu and P are all NaN;
Ce is not a potluck feature. No yield tables are mixed: each policy is trained
and applied separately.

| policy | median fZP | max fZP | fZP >= 0.25 | fZP >= 0.5 |
|---|---:|---:|---:|---:|
| Limongi18 clamp | 0.000 | 0.419 | 10 | 0 |
| WW95 native | 0.027 | 0.492 | 44 | 0 |
| Nomoto13 blend | 0.001 | 0.291 | 2 | 0 |
| Nomoto13 clamp | 0.016 | 0.413 | 14 | 0 |

No policy produces a star above 0.5 in the current set. The highest prediction is
0.492 (WW95 native), followed by 0.419 (Limongi18 clamp) and 0.413 (Nomoto13
clamp).

The yield plots show only the observed APOGEE points; no mock median lines. N is
omitted from the plots because its APOGEE errors are very large, but it is used
by the regressor.

Files and the full cut flow are in
`src/potluck/tests/v20/processed/apogee/README.md`. The main
caveats are: no Zn, no usable Cu, point estimates rather than calibrated
posteriors, and strong per-policy spread.

---

## 16. What the results say

The core physics is validated. Potluck reproduces Salvadori Eq. 13 exactly in
the matched slice. The matrix bookkeeping is right.

The PISN-vs-CCSN pattern is learnable and transfers across yield tables when the
selection is matched. v20 gives 0.90–0.99 cross-table AUC and 0.90–0.99
leave-one-table-out. The v19 0.62 was selection, not yields.

The dominant uncertainty is the yield table. Co varies by up to 2.8 dex at low
Z, Zn by about 1 dex, Ti/Cr/Mn by 0.6–0.8, Mg/Si/Ca by 0.3–0.5. The killing
elements are Co and Zn, and they are also the most table-dependent.

M_dil is not identifiable per star. It is degenerate with `M_Z^pre` and
`M_*II`. The v18 and v20 ranges are prior-dependent. The v20 sweep shows that
the accepted sample and the classifier are stable across M_dil priors, while
the absolute `Z*` and below-floor fraction shift. That is the honest statement.

Li+ is not explained without per-element offsets. The raw support gate flags
254–360 of 385 stars. WW95 is best; Limongi and Nomoto13 blend are worst. The support gate is a systematics diagnostic, not a
validation.

The current model is a legitimate simulator and a credible inference framework.
It is not yet a validated physical model. The remaining gaps are the yield
systematics, the low-Z treatment, SNIa/AGB, NLTE/3D, binarity, rotation, and
external validation against published stars.

---

## 17. Limitations and open items

- No SNIa and no AGB. The SNIa row is zero.
- No binarity. Limongi18 rotation variants exist in the source registry, but the
  v20 policies use `v = 0`; rotation is not part of the baseline.
- Single cloud, instant mixing, no metal loss or spatial structure.
- No NLTE or 3D corrections in the synthetic yields.
- The low-Z policy is a declared extrapolation, not a validated yield model.
- M_dil is latent and prior-dependent; no per-star measurement.
- The primary label is still `y_0.5` in most production paths; switching to the
  detectability label `y` is an open task.
- The classifier is trained and tested on its own simulator; SBC and a held-out
  catalogue are not done.
- The 0.25% target prevalence is a model prior, not a measurement; report a
  sensitivity band.
- Real-star application is Li+ only. The Aguado catalogue reproduction is
  planned but not executed.
- The RF transfer is mock-to-mock, not a validation against real abundances.

---

## 18. Files and how to reproduce

Code:

- `src/potluck/config.py` — priors and gates.
- `src/potluck/events.py` — event draws.
- `src/potluck/ejecta.py` — matrix, clocks, cuts.
- `src/potluck/features.py` — `[X/Fe]`, labels.
- `src/potluck/detect.py` — counterfactual `D` and `y`.
- `src/potluck/observation.py` — Li+ masks, errors, weights.
- `src/potluck/dataset.py` — rejection sampling and arrays.
- `src/potluck/calibration.py` — classifier, calibrator, support gate.
- `src/potluck/yields/` — PISN, Pop III, Pop II tables.
- `src/yield_sources.py` — source registry, clamp/blend variants.
- `src/potluck/diagnostics/v19_mega.py` — v19 test runner.
- `src/potluck/diagnostics/v20_mega.py` — v20 test runner and M_dil sweep.

Results:

- `src/potluck/tests/v18/README.md` — M_dil inverse problem.
- `src/potluck/tests/v19/README.md` — Z*-sampled tables.
- `src/potluck/tests/v20/README.md` — Li+-independent comparison.
- `docs/algorithm.md` — the original algorithm reference.
- `docs/potluck_validation.md` — validation against Salvadori and yield-systematic assessment.
- `docs/catalogue_options.md` — survey catalogue options.
- `docs/aguado2023_reproduction_plan.md` — PISN-explorer reproduction plan.

Reproduce v20:

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --n 20000 --seed 42 --sigma 0.2 \
    --policies Limongi18_clamp,WW95_native,Nomoto13_blend,Nomoto13_clamp \
    --force
```

M_dil sweep:

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --mdil-sweep --n 10000 --seed 42 \
    --policies Limongi18_clamp,WW95_native,Nomoto13_blend,Nomoto13_clamp
```

Li+ application:

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --application-only
```

Above-floor transfer:

```bash
python -m src.potluck.diagnostics.v20_mega \
    --outdir src/potluck/tests/v20 --no-floor-transfer
```

Key readings, f05:

  - Zn is the universal add-on: +0.34 to Si, S and Ca; +0.32 to Cr and Mn; +0.30 to Ti; +0.23 to Ni; +0.09 to Cu.
  - Si is the best partner for Zn (+0.089) and Cu (+0.110), and a strong partner for Al (+0.182) and N (+0.122).
  - Al and Zn are substitutes, not complements: Si + Al reaches 0.930, Si + Zn 0.931, but Al + Zn only 0.819. This is why the general subset has both Si and Al
   but does not rely on Zn–Al.
  - N standalone is 0.745 f05 and 0.697 D. Its best partners are Zn (+0.125), Si (+0.122) and S (+0.114) on f05; Zn (+0.123), Si (+0.111) and S (+0.098) on D.
  N is a useful optional element, not core.
  - Cu standalone is strong (0.792 f05, 0.743 D) but its best additions are only Si (+0.110), S (+0.099) and Ca (+0.089). It is not a universal add-on.
  - The single-element deltas are large (0.1–0.35) because the base has only one feature. In the full 20-element set the same additions are near zero. This
  plot is for pair compatibility, not marginal value in the full set.

  The general subset recommendation does not change: Zn, Si, Al, S, Ca, Ti, Cr, Ni. N is optional; if you want it, add it after Zn/Si/S, not instead of Al.