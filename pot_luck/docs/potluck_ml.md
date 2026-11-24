# Potluck: the machine-learning layer

The simulator produces labelled mocks. This document covers everything after that:
what the models eat, what they output, how the probabilities are calibrated, and
which hyperparameters are actually in use.

Three objects are fitted, and they are easy to confuse:

| object | target | output | file |
| --- | --- | --- | --- |
| classifier | $y_{0.5}$ (binary) | $p(y_{0.5}=1\mid z)$ | `calibration.py` |
| isotonic calibrator | — | recalibrated version of the above | `calibration.py` |
| regressor | $f_Z^P$ (continuous) | $\mathbb{E}[f_Z^P\mid z]$ | `calibration.py` |

The classifier answers "is this star PISN-dominated?". The regressor answers "how
much of its metal came from PISN?". They are trained separately on the same inputs
and are never combined into one number.

---

## 1. Inputs: the design matrix

Two matrices exist, and only one is used for training.

**Noiseless** (`features`, `ratio_mask`): the clean $[X/\mathrm{Fe}]$ ratios and their
validity mask. Used for the "chemistry ceiling" diagnostics, never for the
production classifier.

**Observed** (`obs_*`): the ratios after the Li+ observation model has added noise,
applied element masks and censoring. This is the production input, because a real
star only ever provides the noisy version.

The observed matrix is built column-block by column-block:

$$
z_i = \Big(
\underbrace{\hat x_X}_{11},\
\underbrace{\mathrm{flag}_X}_{11},\
\underbrace{\log_{10}\sigma_X}_{11},\
\underbrace{\mathbf{1}[\text{giant}]}_{1},\
\underbrace{[\mathrm{X}/\mathrm{Y}]}_{4},\
\underbrace{\sigma_{X/Y}}_{4}
\Big)_i .
$$

That is 42 raw columns. One is dropped when it has fewer than two distinct finite
values, leaving **41 columns in practice**. In the application datasets the dropped
column is `flag_Mg/Fe`, because Mg is detected in every Li+ star in the catalogue, so
that flag is constant at 1. Which column drops is data-dependent, not hard-coded.

The blocks in detail:

| block | columns | meaning | missing values |
| --- | --- | --- | --- |
| $\hat x_X$ | 11 | observed $[X/\mathrm{Fe}]$, $X \in$ C, Na, Mg, Si, Ca, Ti, Cr, Mn, Co, Ni, Zn | `NaN` when not measured |
| $\mathrm{flag}_X$ | 11 | 0 missing, 1 detected, 2 upper limit | never |
| $\log_{10}\sigma_X$ | 11 | measurement error, only where a flag exists | `NaN` where missing |
| giant | 1 | $\log g < 3.5$ | never |
| $[\mathrm{X}/\mathrm{Y}]$ | 4 | Na/Mg, Co/Ni, Mn/Cr, Zn/Mg | `NaN` unless both parents measured |
| $\sigma_{X/Y}$ | 4 | $\sqrt{\sigma_X^2+\sigma_Y^2}$ | same |

Two design choices matter here.

**Missingness is a feature, not a problem.** The flag block carries it explicitly.
An upper limit and a non-detection are different states, so they enter as flags 2 and
0 rather than as a numeric substitute. The tree can therefore split on "this star has
no Zn measurement", which is informative.

**NaNs are passed through, not imputed.** `HistGradientBoosting*` handles `NaN`
natively by learning a default direction at each split. The random-forest paths
cannot, so they median-impute on the training fold only (`v20_mega._impute`). This is
a real difference between the two model families, not a cosmetic one.

The $[X/\mathrm{Fe}]$ ratios are deliberately **not** standardised for the tree
models, since splits are rank-based. The logistic baseline in `simple_baselines.py`
does wrap the input in `StandardScaler`.

### The giant column is redundant

`giant = 1[\log g < 3.5]`, threshold `GIANT_LOGG = 3.5` in `observation.py`. It is not
derived from abundances: `load_li_records` reads `logg` from the Li+ catalogue, and
`sample_observation` gives each mock the `log g` of the donor star it matched within
0.2 dex in $[\mathrm{Fe}/\mathrm{H}]$. So it describes the borrowed record, not the
simulated cloud.

The split is nearly even — 54.5% of mocks, and 55.6% of the Li+ catalogue itself
(214/385 stars, $\log g$ 0.55–4.76, median 3.06).

It contributes nothing to the classifier. Measured on the Limongi18 application set,
5-fold OOF with importance weights:

| model | with | without | $\Delta$ |
| --- | ---: | ---: | ---: |
| `gb` | 0.971916 | 0.971884 | +0.000032 |
| `rf` | 0.971850 | 0.971565 | +0.000285 |

RF impurity importance ranks it 34th of 41. Shuffling the column also changes nothing.

The reason is redundancy, not irrelevance. Giants get 0.3 dex extra carbon scatter in
quadrature, so `log10 sigma_C` alone predicts giantness at **AUC 0.979** (median
$\sigma_C$ 0.331 giant vs 0.160 dwarf), and the C/Fe flag distribution differs too
(dwarfs hit upper limits 25% of the time, giants 9%). The tree recovers the information
from columns it already has. The label rate is also flat across the split (0.242 giant,
0.231 dwarf), so there is no class-correlated signal to exploit.

It still matters physically, just not as a feature: it drives the carbon treatment
(`inflate` adds 0.3 dex, `drop` removes C for giants) and it stratifies the
$[\mathrm{Fe}/\mathrm{H}]$ reweighting, which is done separately for the two classes.
Treat it as a provenance variable. A continuous `log g` would be strictly more useful
than the binary if the column is to be kept at all.

---

## 2. Splits and weights

```
train 60%   cal 20%   test 20%
```

`_split` in `calibration.py` uses `train_test_split` twice with `stratify=y`, so the
positive fraction is preserved in all three folds. The seed defaults to 0.

Weights are class-balanced:

$$
w_i = \begin{cases}
\dfrac{0.5}{N_+}, & y_i = 1,\\[6pt]
\dfrac{0.5}{N_-}, & y_i = 0,
\end{cases}
\qquad \sum_i w_i = 1 .
$$

Two kinds of weight exist and they are easy to mix up:

* **Class-balance weights** (`_balance`) — only in the diagnostics. They make the two
  classes contribute equally, so a metric is not dominated by the majority class.
* **Importance weights** (`importance_weight`) — from the observation layer's Li+
  $[\mathrm{Fe}/\mathrm{H}]$ reweighting. These carry physical meaning.

`fit_phase_b_models` uses the **importance weights**. The diagnostic scripts use
`_balance`. If you compare a number from `candidates.csv` against one from
`element_influence.json`, check which weighting was in play.

The importance weights are not free: the effective sample size is about 7,500–8,000
out of 10,000 mocks (Kish), so the reweighting costs roughly a fifth of the data.

---

## 3. The classifier

### Which model

Two are implemented, selected by the `model` argument:

```python
model="gb"   # HistGradientBoostingClassifier — default
model="rf"   # RandomForestClassifier
```

| parameter | `gb` (default) | `rf` |
| --- | --- | --- |
| `max_iter` / `n_estimators` | 150 | 300 |
| `max_depth` | 3 | unlimited |
| `learning_rate` | 0.05 | — |
| `min_samples_leaf` | — (default 20) | 5 |
| `max_features` | — | `"sqrt"` |
| `n_jobs` | — | -1 |
| `random_state` | seed | seed |

`gb` is a histogram-based gradient-boosted tree ensemble. `max_depth=3` keeps it
strongly regularised, which is appropriate because the signal is a smooth
combination of ~40 correlated columns and the training set is a few thousand rows.

`rf` is a bagged forest. It is the primary classifier in the v20/v21 campaign, where
`v20_mega.py` uses **300 trees, `min_samples_leaf=5`, `max_features="sqrt"`**.
The class-balance weights are passed as `sample_weight`.

### What it outputs

`predict_proba(x)[:, 1]` — the raw score. This is **not** a probability you should
report. It is the probability under the training mixture, where positives are roughly
20–25% of the sample (measured: 0.237 for Limongi18, 0.216 for WW95 in the
application datasets).

Everything after this point exists to turn that raw score into something interpretable.

---

## 4. Isotonic calibration

### What it is

Isotonic regression fits the best **monotone** step function from score to observed
frequency. Given calibration pairs $(s_i, y_i)$ with weights, it solves

$$
\hat g = \arg\min_{g\ \text{non-decreasing}} \sum_i w_i\,\big(y_i - g(s_i)\big)^2 .
$$

In words: sort the scores, then find the non-decreasing curve that is closest to the
data in weighted least squares. Ties are merged into blocks, and each block takes the
weighted mean of $y$ inside it. That produces a staircase.

### Why monotone matters

A calibration map must not reorder stars. If star A scores above star B, A must not
become less likely after calibration. Isotonic is the *most flexible* map with that
guarantee. Logistic (Platt) scaling is the alternative — smoother, but it imposes a
sigmoid shape that may not fit.

### How it is fitted here

```python
calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(cal_score, y[cal], sample_weight=weight[cal])
```

Three details worth knowing:

* It is fitted on the **cal split only**, never on train. Fitting it on train would
  make the calibration look better than it is.
* `out_of_bounds="clip"` means a test score outside the calibration range snaps to
  the nearest endpoint instead of extrapolating wildly.
* `y_min=0.0, y_max=1.0` keeps the output in the valid probability range.

The result is a lookup: score in, calibrated probability out. After this step,
"predicted 0.3" means "about 30% of mocks that scored this way are positive".

### It slightly lowers AUC, on purpose

Isotonic is monotone but not strictly monotone: it merges tied scores into flat
blocks, and within a block the ordering is lost. Measured on held-out data:

| policy | AUC before | AUC after | Brier before | Brier after |
| --- | --- | --- | --- | --- |
| Limongi18_clamp | 0.9692 | 0.9659 | 0.059 | 0.058 |
| WW95_native | 0.9623 | 0.9601 | 0.059 | 0.056 |

So the ranking gets very slightly worse while the probability scale gets better. That
is the correct trade for a calibrated probability, and it is a reason not to quote
post-calibration AUC as if it measured ranking quality. If ranking is what you care
about, use the raw score.

### What it does not fix

Isotonic calibrates the *training* mixture. It does nothing about the fact that the
real sky is not 23% positive. That is the next step.

---

## 5. The prior shift

### The formula

The classifier is trained where positives are common. The sky is not. Bayes' rule
converts one to the other, and in odds form it is a single multiplication:

$$
\frac{p'}{1-p'} = \frac{p}{1-p}\cdot
\frac{\pi/(1-\pi)}{\pi_{\rm train}/(1-\pi_{\rm train})}.
$$

Solved for $p'$:

$$
p' = \frac{\text{odds}'}{1+\text{odds}'}.
$$

Implementation (`prior_shift`):

```python
p = np.clip(p, 1e-12, 1 - 1e-12)      # avoid log(0) / division by zero
odds = p / (1 - p)
odds *= (prevalence / (1 - prevalence)) / (train_rate / (1 - train_rate))
return odds / (1 + odds)
```

The clipping at $10^{-12}$ is not cosmetic: without it a calibrated 0.0 produces an
infinite odds ratio and the whole vector becomes `NaN`.

### The numbers

| quantity | value |
| --- | --- |
| $\pi$ (target prevalence) | $2.5\times10^{-3}$ |
| $\pi_{\rm train}$ | about 0.23, recomputed from the train fold |
| default source | `DEFAULT_PREVALENCE = 2.5e-3` |

Worked example with $\pi_{\rm train}=0.4$:

```
p        = 0.8
odds     = 4.0
ratio    = (0.0025/0.9975) / (0.4/0.6) = 0.00376
odds'    = 4.0 × 0.00376 = 0.0150
p'       = 0.0150/1.0150 = 0.0148
```

A raw 0.8 becomes 1.5%. A raw 0.99 becomes about 27%. This is the rareness effect:
with one true PISN per 400 stars, even a confident classifier cannot support a high
absolute probability.

### What survives the shift

The map is monotone, so:

* **AUC is unchanged.** The shift is a rescaling of odds, identical for every star.
* **Ranking is unchanged.** The candidate list order is identical.
* **Precision, recall at a fixed threshold, and the posterior scale all change.**

This is why the AUC numbers in the repo are prevalence-independent, and why quoting
them as evidence of detection performance is misleading. At $\pi=0.25\%$ and FPR 0.1%,
precision is about 0.50–0.55; the same model at $\pi=1\%$ gives 0.80–0.83.

### The prevalence is an assumption

$2.5\times10^{-3}$ comes from chemical-evolution models (de Bennassuti et al. 2017;
Salvadori et al. 2019), not from the Li+ catalogue. It should be reported as a prior
with a sensitivity band (0.1%, 0.25%, 1%). Changing it moves every posterior and does
not move the ranking.

---

## 6. The regressor

A separate model on the same input matrix, trained on the continuous target:

```python
HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05)
# or RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features="sqrt")
```

Fitted on `arrays["f_zp"]` over the entire train fold — not restricted to positives.

**Output:** `expected_fz = np.clip(regressor.predict(x_real), 0.0, 1.0)`. The clip is
enforced because a tree ensemble can extrapolate slightly outside $[0,1]$.

**It is not prior-shifted.** There is no "prevalence of a continuous variable" to
shift to. It is a regression estimate, and it appears in `candidates.csv` as
`expected_fz` next to `p_posterior` without any adjustment.

### Why not just threshold the regressor

Because they answer different questions, and because the regressor does not have to be
calibrated as a probability. If the conditional distribution of $f_Z^P$ given $z$ were
known,

$$
p' = P(f_Z^P \ge 0.5\mid z) = \int_{0.5}^{1} p(f_Z^P\mid z)\,\mathrm{d}f_Z^P,
\qquad
\hat f_Z^P = \mathbb{E}[f_Z^P\mid z] = \int_0^1 f_Z^P\,p(f_Z^P\mid z)\,\mathrm{d}f_Z^P .
$$

One is a tail mass, the other a mean. They agree only if the conditional distribution
is a point. For real abundance patterns it is not, so `expected_fz = 0.4` with
$p'=0.6$ is coherent rather than contradictory.

---

## 7. Uncertainty on the posterior

The interval on `p_posterior` is a **bootstrap over the calibration step**, not over
the classifier:

```python
bootstrap_posterior(..., n_boot=200, seed=0)
```

Each iteration:

1. resample the calibration set with probability proportional to its weight,
2. skip the draw if it contains only one class,
3. refit isotonic regression on the resample,
4. apply the prior shift to the target scores,
5. store the shifted probabilities.

The interval is the 5th–95th percentile across draws, giving `p_low` and `p_high`.

**What this captures:** uncertainty in the calibration curve, which is genuinely
large when the calibration set is small.

**What it does not capture:** uncertainty from retraining the classifier itself. The
classifier is fitted once and frozen. So the interval is optimistic — a point worth
stating, and the same limitation as `element_influence._paired`, which reuses fixed
score arrays. A fully honest interval would refit the classifier inside the bootstrap.

---

## 8. The support gate

A probability is meaningless if the star is unlike anything in the training set, so a
second check runs in parallel.

Core elements: Na, Mg, Ca, Ti, Cr, Ni. At least four must be measured. The distance is
a nearest-neighbour Mahalanobis-style statistic with the star's own errors as the
scale:

$$
d_r^2 = \min_i \sum_{X \in \mathrm{core}}
\left(\frac{x_{r,X} - x_{i,X}}{\sigma_{r,X}}\right)^2 .
$$

The threshold comes from the mocks themselves: draw 2,000 mock references, measure the
mock-to-mock distribution, take its 99th percentile. A star with $d_r$ above that
threshold gets `out_of_model = True`.

This is a **systematics diagnostic**, not a physical validation. In the Li+
application the raw gate keeps only **6.5–34% of stars in model**, i.e. it flags
66–94% as out of model. Allowing one zero-point offset per core element raises the
in-model fraction to 91–95% for every dataset. So the gate is mostly measuring a
yield-table offset — Ti in particular, by about +0.26 to +0.73 dex with no
$[\mathrm{Fe}/\mathrm{H}]$ trend — not unusual stars.

---

## 9. Outputs

`score_real_stars` returns one row per Li+ star, written to `candidates.csv`:

| column | meaning |
| --- | --- |
| `star` | catalogue identifier |
| `feh` | observed $[\mathrm{Fe}/\mathrm{H}]$ |
| `n_core` | number of core elements measured |
| `score` | raw `predict_proba` |
| `p_calibrated` | after isotonic, before prior shift |
| `p_posterior` | after prior shift — the headline number |
| `p_low`, `p_high` | 5–95% bootstrap bounds |
| `expected_fz` | regressor output, clipped to $[0,1]$ |
| `support_distance` | the gate statistic $d_r$ |
| `support_threshold` | 99th percentile of the mock distribution |
| `out_of_model` | `d_r` above threshold or undefined |
| `near_threshold` | $0.25 \le$ `expected_fz` $< 0.5$ |
| `candidate` | `p_posterior >= 0.5` |
| `is_giant` | $\log g < 3.5$ |
| `available`, `missing`, `limits` | comma-separated element lists |

Sorted by `p_posterior` descending. A `calibration.json` report is written alongside,
containing train rate, prevalence, before/after AUC and Brier, the support threshold,
and the three counts (candidates, out of model, near threshold).

In practice **no Li+ star reaches `p_posterior >= 0.5`** at $\pi=0.25\%$. Across the
six policies the maximum is 0.0055–0.0432 (385 stars each; largest for
Nomoto13_clamp at 0.043), and zero candidates are flagged. That is the prevalence
effect, not a failure of the model.

---

## 10. Which model where

The hyperparameters differ by script, deliberately — diagnostics trade accuracy for
speed.

| script | model | settings | purpose |
| --- | --- | --- | --- |
| `calibration.py` | `gb` / `rf` | 150 iters / 300 trees | production scoring |
| `v20_mega.py` | RF | 300 trees, leaf 5, sqrt | AUC + transfer matrix |
| `ceiling.py` | `gb` | 150 iters, depth 3, lr 0.05 | 5-fold OOF ceiling |
| `v19_mega.py` | RF | 300 / 200 trees | legacy campaign |
| `element_influence.py` | RF | 50 trees (default) | leave-one-out deltas |
| `rank_stability.py` | RF | 40 trees | retraining noise floor |
| `operating_points.py` | RF | configurable | TPR at fixed FPR |
| `simple_baselines.py` | RF + logistic | 100 trees / `max_iter=1000`, 5-fold | non-ML baseline |

All use `random_state=seed` and `n_jobs=-1`. None of them tune hyperparameters — the
values are fixed a priori, which is defensible given the small feature count but means
no search evidence exists that they are optimal.

---

## 11. Known limitations

**The target is a label, not a measurement.** $y_{0.5}$ is defined by the simulator's
own $f_Z^P$, so the classifier learns the simulator's decision boundary. A star the
simulator calls "dominant" is not necessarily one an observer would.

**Label choice is unresolved.** Most production paths train on $y_{0.5}$. The
detectability label $y=1[D>3]$ is more operationally honest and gives a harder problem
(AUC drops from about 0.98 to 0.81). Switching is listed as an open task.

**Precision is the weak point.** AUC 0.98 reads well, but at $\pi=0.25\%$ and FPR 0.1%
precision is roughly 0.5, and the false positives are concentrated in genuine
near-threshold mixtures ($0.25 < f_Z^P < 0.5$), not in obvious non-PISN stars.

**The bootstrap interval covers calibration only.** See §7. Retraining variance is not
included, and for element rankings that variance turned out to be larger than the
effects being ranked.

**Hyperparameters are not tuned.** Fixed values, no cross-validated search. The cost of
that is small: a logistic regression on the same 8 general elements reaches mean AUC
0.969 against 0.975 for the forest, and the TPR gap at FPR 0.1% is +0.027
(Limongi18) and −0.003 (WW95) — the forest wins one and ties one.
At FPR 1% the gap is a consistent +0.019 to +0.028. So the 8-element combination, not
the nonlinearity, is doing the work. That is an argument from results, not from a
search.
