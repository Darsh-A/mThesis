# 4. The machine-learning layer

This section covers everything after the simulator: what the model eats, what it
outputs, how the probabilities are calibrated, and what is actually doing the work.
It follows `calibration.py` and `diagnostics/ceiling.py`.

All AUCs quoted here are for **label A**, $y_{0.5} = \mathbf{1}[f_Z^P \ge 0.5]$,
because every training site in `diagnostics/v20_mega.py` reads `y_f_05`.

## 4.1 Two design matrices exist; only one is used

**Noiseless** (`features`, `ratio_mask`): the clean $[X/\mathrm{Fe}]$ ratios. Used
only for the "ceiling" diagnostic — how well could you do with perfect data.

**Observed** (`obs_*`): the ratios after §3 has added noise, masks and censoring.
This is the production input, because a real star only ever provides this version.

Mixing them up is the easiest way to produce a number that means nothing. The
ceiling (0.996–0.997) and the observed score (0.965–0.978) are different questions.

## 4.2 What the observed matrix contains

The raw matrix has 42 columns, built block by block:

$$
\mathbf{z}_i = \Big(
\underbrace{\hat x_X}_{11},\
\underbrace{\mathrm{flag}_X}_{11},\
\underbrace{\log_{10}\sigma_X}_{11},\
\underbrace{\mathbf{1}[\text{giant}]}_{1},\
\underbrace{[\mathrm{X}/\mathrm{Y}]}_{4},\
\underbrace{\sigma_{X/Y}}_{4}
\Big)_i .
$$

| block | columns | meaning | when missing |
| --- | ---: | --- | --- |
| $\hat x_X$ | 11 | observed $[X/\mathrm{Fe}]$ | `NaN` |
| $\mathrm{flag}_X$ | 11 | 0 missing / 1 detected / 2 limit | never |
| $\log_{10}\sigma_X$ | 11 | measurement error | `NaN` |
| giant | 1 | $\log g < 3.5$ | never |
| $[\mathrm{X}/\mathrm{Y}]$ | 4 | Na/Mg, Co/Ni, Mn/Cr, Zn/Mg | `NaN` |
| $\sigma_{X/Y}$ | 4 | $\sqrt{\sigma_X^2 + \sigma_Y^2}$ | `NaN` |

Columns that are constant, or entirely missing, are dropped before fitting. The
code keeps columns with at least two distinct finite values, so **41 columns in
practice**. In the Li+ datasets the dropped column is `flag_Mg/Fe`, because Mg is
detected in every catalogue star.

Two choices are worth defending:

**NaNs are passed through, not imputed.** `HistGradientBoosting*` learns a default
direction at each split, which is the honest treatment: imputing would invent data.

**Missingness is a feature.** The flag block makes "no Zn" and "Zn is a limit" two
different states the model can split on. That is information, not a gap.

One caveat: the `giant` column is nearly redundant. Giants receive 0.3 dex extra
carbon scatter (§3.6), so $\log_{10}\sigma_C$ alone predicts giantness at
AUC 0.979. Removing the column costs about 0.0003 AUC. It matters physically (it
drives the carbon treatment and stratifies the reweighting), not as a feature.

## 4.3 Splits and weights

```
train 60 %   calibration 20 %   test 20 %
```

Both splits are stratified on the label, with a fixed seed.

There are two kinds of weight and they must not be confused:

- **Class-balance weights** $w_i \propto 1/N_{y_i}$, used by the diagnostic
  scripts. They make both classes contribute equally.
- **Importance weights** from the observation layer's $[\mathrm{Fe}/\mathrm{H}]$
  reweighting, which carry physical meaning.

`fit_phase_b_models` uses the **importance weights**. A number from
`candidates.csv` and a number from `element_influence.json` are therefore not
weighted the same way and are not directly comparable.

## 4.4 The classifier

Default is `HistGradientBoostingClassifier`:

```python
HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05)
```

Depth 3 is very shallow and deliberately so: the signal is a smooth combination of
about 40 correlated columns on a few thousand rows, and deep trees would memorise
the simulator.

An RF alternative exists (`n_estimators=300, min_samples_leaf=5,
max_features="sqrt"`), which is what the v20 campaign used.

`predict_proba(x)[:, 1]` is **not** a reportable probability. It is the probability
under the training mixture, where positives are roughly 23 %.

## 4.5 Isotonic calibration

Isotonic regression fits the best monotone step function from score to observed
frequency:

$$
\hat g = \operatorname*{arg\,min}_{g\ \text{non-decreasing}}
\sum_i w_i \big(y_i - g(s_i)\big)^2 .
$$

In words: sort the scores, then find the non-decreasing curve closest to the data
in weighted least squares. Ties merge into blocks, and each block takes the
weighted mean of $y$, producing a staircase.

```python
IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
```

It is fitted on the **calibration split only**. Fitting it on training data would
make the calibration look better than it is.

**It slightly lowers AUC on purpose.** Measured on the held-out fold for
`Limongi18_clamp`: AUC $0.9692 \to 0.9659$, Brier $0.059 \to 0.058$. The ranking
gets microscopically worse and the probability scale gets better. That is the
correct trade if the output is a probability, and it is the reason not to quote
post-calibration AUC as a measure of ranking quality.

## 4.6 The prior shift

After calibration you are still in the wrong universe: the training set is about
23 % positive, and the sky is not. In odds form, Bayes' rule is one multiplication:

$$
\frac{p'}{1-p'} = \frac{p}{1-p}\cdot
\frac{\pi/(1-\pi)}{\pi_{\rm train}/(1-\pi_{\rm train})},
\qquad \pi = 2.5\times10^{-3}.
$$

Solved back for a probability:

$$
p' = \frac{\text{odds}'}{1 + \text{odds}'}.
$$

Worked example with $p = 0.8$ and $\pi_{\rm train} = 0.4$:

$$
\frac{0.8}{0.2} = 4.0,
\qquad
4.0 \times
\frac{0.0025/0.9975}{0.4/0.6}
= 4.0 \times 0.00376 = 0.0150,
\qquad
p' = 0.0148 .
$$

A confident 0.8 becomes 1.5 %. A raw 0.99 becomes about 27 %. With one true PISN per
400 stars, even a strong classifier cannot support a high absolute probability.

**What survives the shift:** AUC and the ranking, because the rescale is identical
for every star. **What changes:** precision, recall at a fixed threshold, and the
whole posterior scale.

**The prevalence is an assumption.** $2.5\times10^{-3}$ comes from chemical
evolution models (de Bennassuti et al. 2017; Salvadori et al. 2019), not from the
Li+ catalogue. It should always be quoted with a sensitivity band such as
$(10^{-3},\,2.5\times10^{-3},\,10^{-2})$. Changing it moves every posterior and
moves no ranking.

## 4.7 The regressor

A second tree model on the same matrix, trained on the continuous target $f_Z^P$
over the whole training fold — not restricted to positives:

```python
HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05)
```

The prediction is clipped to $[0,1]$, because a tree ensemble can extrapolate just
outside it.

**It is not prior-shifted.** There is no "prevalence of a continuous variable" to
shift to.

The two outputs answer different questions and are not interchangeable:

$$
p' = P\!\left(f_Z^P \ge 0.5 \mid \mathbf{z}\right)
= \int_{0.5}^{1} p\!\left(f_Z^P \mid \mathbf{z}\right)\mathrm{d}f_Z^P,
$$

$$
\hat f_Z^P = \mathbb{E}\!\left[f_Z^P \mid \mathbf{z}\right]
= \int_0^1 f_Z^P\, p\!\left(f_Z^P \mid \mathbf{z}\right)\mathrm{d}f_Z^P .
$$

One is a tail mass, the other a mean. They agree only if the conditional
distribution is a point. It is not, so `expected_fz = 0.4` together with
`p' = 0.6` is coherent, not contradictory.

### Why a tree regressor can return a value in $[0,1]$

This is a common question, because there is no sigmoid anywhere in the regressor.
The bound comes from two separate places.

**A tree is piecewise constant.** It partitions the feature space into leaves, and
each leaf stores one number: the mean of the training targets that fell into it. So
one tree's prediction is always one of those stored means.

**The target is already bounded.** The regression target is the metal fraction

$$
0 \le f_Z^P
= \frac{\sum_{X\in\mathcal{Z}} M_X^{(\mathrm{PISN})}}
{\sum_{X\in\mathcal{Z}} M_X} \le 1,
$$

so every training value lies in $[0,1]$, and a leaf mean is a weighted average of
such values. A single tree therefore **cannot extrapolate outside the range of its
training targets** — a defining property of trees, and the reason they are safe for
bounded targets (and poor at trend extrapolation).

**Boosting is additive.** With squared loss the model is

$$
F(\mathbf{z}) = F_0 + \eta \sum_{t=1}^{T} h_t(\mathbf{z}),
\qquad \eta = 0.05,\ T = 150,
$$

where $F_0$ is the initial constant and each $h_t$ is fitted to the current
residual. Each $h_t$ is bounded, but the **sum** can overshoot the training range
for points near the edge of the data, so the single-tree guarantee does not
strictly survive.

That is why `calibration.py` clips explicitly:

```python
expected_fz = np.clip(models.regressor.predict(x_real), 0.0, 1.0)
```

The clip is insurance against the additive sum drifting past the edges, not the
mechanism that creates the bound — and the drift is real, not hypothetical. Fitting
the v20 `application/Limongi18_clamp` models and looking at the **unclipped**
predictions gives

$$
\text{raw prediction} \in [-0.049,\ +1.097],
\qquad
P(\text{raw}<0) = 0.019,\quad P(\text{raw}>1) = 0.015 .
$$

So about **3.4 % of rows are clipped**, by at most roughly 0.1 in absolute value.
Typical values are still small (median 0.12, 99th percentile 1.01), which is why
the output histogram looks bounded — but the clip is doing genuine work at both
ends, not just guarding against a corner case.

The contrast with the classifier is the useful part:
`HistGradientBoostingClassifier` predicts **log-odds** and applies a logistic link,
so its $[0,1]$ bound comes from the sigmoid. The regressor has no link function; it
regresses directly on an already-bounded fraction.

## 4.8 The support gate

A probability is meaningless if the star is unlike anything in the training set, so
a second check runs in parallel.

Core elements are Na, Mg, Ca, Ti, Cr and Ni, with at least four measured. The
distance is a nearest-neighbour statistic scaled by the star's own errors:

$$
d_r^2 = \min_i \sum_{X \in \{\mathrm{Na},\mathrm{Mg},\mathrm{Ca},\mathrm{Ti},\mathrm{Cr},\mathrm{Ni}\}}
\left(\frac{x_{r,X} - x_{i,X}}{\sigma_{r,X}}\right)^2 .
$$

The threshold is empirical: draw 2,000 mock references, measure the mock-to-mock
distance distribution, and take its 99th percentile. Measured thresholds are
3.23–3.63 across the v20 application datasets.

A star above the threshold is flagged `out_of_model`.

**This is a systematics diagnostic, not a validation.** Measured in-model fractions
(the raw gate) from `support_variants.json`:

| dataset | raw in-model |
| --- | ---: |
| `v20/Limongi18_clamp` | 0.083 |
| `v20/WW95_native` | 0.192 |
| `v20/Nomoto13_blend` | 0.065 |
| `v20/Nomoto13_clamp` | 0.086 |
| `v20/Nomoto13_CK20_blend` | 0.335 |
| `v20/Nomoto13_CK20_clamp` | 0.340 |

Allow one zero-point offset per core element and every dataset returns to
0.909–0.948. So the gate is mostly measuring a **yield-table zero-point offset**,
not unusual stars. The dominant offender is Ti at $+0.62$ to $+0.73$ dex.

The measured offsets also show which elements are trustworthy indications:

| element | range of offset across 8 datasets | reading |
| --- | ---: | --- |
| Ti | $+0.41$ to $+0.73$ | always large, always positive → real zero-point problem |
| Na | $-0.31$ to $+0.59$ | changes sign with the sampling scheme |
| Mg | $-0.87$ to $+0.23$ | changes sign |
| Ca | $-0.82$ to $+0.07$ | changes sign |
| Cr | $-0.20$ to $-0.67$ | consistently negative but variable |

Because the Na/Mg/Ca offsets move when the $Z_\star$ sampling scheme changes, they
are partly an artefact of the prior. Ti does not move, so the Ti offset is the one
to trust as physical.

> Implementation detail: in `support_gate`, the mock reference set uses unit
> $\sigma$ (`mock_sigma = ones`), while the real star uses its catalogue errors. The
> distance is therefore "in units of the star's errors", which is the intended
> definition, but it is worth knowing when reading the code.

## 4.9 What is actually doing the work

This is the strongest single result in the project, and it is easy to misstate.

From `simple_baselines.json`, `Limongi18_clamp`, same FPR targets:

| model | AUC | TPR @ FPR 1 % | TPR @ FPR 0.1 % | precision @ 0.1 % |
| --- | ---: | ---: | ---: | ---: |
| RF, general 8 | 0.979 | 0.745 | **0.512** | 0.550 |
| logistic, general 8 | 0.974 | 0.726 | **0.486** | 0.536 |
| logistic, [Na/Mg] + [Zn/Mg] | 0.919 | 0.448 | 0.144 | 0.255 |
| single [Zn/Fe] | 0.896 | 0.517 | 0.244 | 0.368 |
| single [Na/Mg] | 0.783 | 0.168 | 0.043 | 0.093 |

Two readings:

1. **The two-feature rule collapses** (0.144 vs 0.512 at FPR 0.1 %). The
   multi-element pattern is essential.
2. **A linear model on the same eight elements is within 0.03 of the forest.**
   The nonlinearity is not the innovation; the eight-element combination is.

And with the full 26-ratio set, no single element matters
(`element_influence.json`):

| element | $\Delta$AUC when removed |
| --- | ---: |
| Al | +0.0018 |
| Ar | +0.0010 |
| C | +0.0007 |
| Ca | +0.0006 |
| Co | +0.0005 |

Full AUC is 0.9807, and the largest removal penalty is 0.0018. The information is
distributed across the pattern, not concentrated in one line.

> **Provenance caveat.** Like `element_subsets.py`, this analysis rebuilds the
> ratios from `log_abundance` plus Gaussian noise and applies **no** availability
> masks or censoring:
>
> ```python
> noisy = loga + rng.normal(0.0, sigma, loga.shape)
> ```
>
> Every element is therefore present in every row, which is not true of the Li+
> catalogue. Repeating the leave-one-out test on the deployed
> `application/Limongi18_clamp` model, where availability *is* realistic, gives a
> different picture (full AUC 0.9732):
>
> | element | $\Delta$AUC |
> | --- | ---: |
> | Mg | +0.0067 |
> | Ca | +0.0051 |
> | Cr | +0.0031 |
> | Ti | +0.0026 |
> | Si | +0.0014 |
> | Ni | +0.0007 |
> | Zn | +0.0001 |
> | Co | −0.0003 |
> | Na | −0.0013 |
>
> The qualitative conclusion survives — **no single element dominates**, the largest
> penalty being 0.0067 — but the specific ranking changes, and Zn falls from
> apparently valuable to negligible once the masks are honest. The "distributed
> pattern" claim is robust; any ranking of individual elements is not.
>
> Two measurement methods also disagree here. Permutation importance (shuffling an
> element's value, flag and $\log\sigma$ columns) gives Ca $+0.0280$, Mg $+0.0268$,
> Cr $+0.0128$, Zn $+0.0021$; retraining gives the table above. Permutation is
> biased upwards for redundant columns, and all 11 ratios share Fe, so it
> understates each element's independent role. Trust the retrained numbers, and do
> not present any ranking as stable.

## 4.10 Known limitations

1. **The target is a label, not a measurement.** $y_{0.5}$ is defined by the
   simulator's own $f_Z^P$, so the classifier learns the simulator's decision
   boundary.
2. **Label choice is unresolved.** Production trains on $y_{0.5}$; the
   detectability label $y = \mathbf{1}[D>3]$ is more honest and gives a harder
   problem.
3. **The bootstrap interval covers calibration only.** `bootstrap_posterior`
   refits isotonic regression 200 times but never retrains the classifier, so
   `p_low`/`p_high` are optimistic.
4. **Hyperparameters were never tuned.** Fixed a priori. Defensible with ~40
   features, but not evidence of optimality.
5. **AUC is prevalence-independent.** It is not a statement about detection
   performance; §5.6 gives the operating points that are.

## 4.11 What to remember from this section

1. The production input is the observed matrix, 41 columns after dropping constants.
2. NaNs are kept and flags encode missingness — that is information, not a defect.
3. Score → isotonic → prior shift. Ranking survives, scale does not.
4. Two outputs: a tail probability and a conditional mean. Do not conflate them.
5. The support gate measures yield offsets, not odd stars; Ti is the robust offender.
6. The eight-element pattern is the result. The forest is a convenience.
