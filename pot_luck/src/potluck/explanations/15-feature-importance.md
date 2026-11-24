# 15 — Which elements actually carry the label

`src/potluck/diagnostics/feature_importance.py` ranks all 26 mock features and
then asks how many are needed. Outputs: `src/potluck/tests/v15/`
(`importance_noiseless.png`, `importance_observed.png`, `selection_curve.png`,
`feature_importance.json`, `chosen_features.json`).

## Method

Two matrices are ranked separately, because they answer different questions:

| block | columns | question |
|---|---|---|
| `noiseless` | the 26 mock ratios | what the physics encodes, with no measurement noise |
| `observed` | the 11 Li+ ratios, with the catalogue noise | what a star actually delivers |

On each block:

1. **Impurity importance** (MDI) from a 300-tree random forest, averaged over five folds.
2. **Permutation importance**: shuffle one column in the held-out fold, five
   repeats, and record the drop in weighted AUC. This is the arbiter when the
   two disagree, because MDI is biased towards columns with many split points or
   many missing values.
3. **Selection curve**: keep the top `k` columns by permutation, refit the
   production classifier (gradient boosting, 150 iterations, depth 3) on five
   folds, and record the out-of-fold AUC and Brier for every `k`.
4. **Choice**: the smallest `k` whose AUC is within one standard error of the
   best `k`.

## Noiseless ranking

| feature | permutation | MDI |
|---|---|---|
| Si/Fe | **0.0345** | 0.074 |
| Al/Fe | 0.0056 | 0.062 |
| P/Fe | 0.0026 | 0.058 |
| S/Fe | 0.0023 | 0.054 |
| Zn/Fe | 0.0022 | 0.106 |
| F/Fe | 0.0010 | 0.073 |
| Cu/Fe | 0.0010 | 0.103 |
| Co/Fe | 0.0010 | 0.163 |
| O/Fe, Ar/Fe, C/Fe, N/Fe, Cr/Fe, Na/Fe, Cl/Fe, Mn/Fe, Mg/Fe, Ti/Fe, Ca/Fe, Ga/Fe, Ne/Fe, Ni/Fe, Sc/Fe, K/Fe, Ge/Fe, V/Fe | 0.0009 → 0.0000 | 0.016 → 0.003 |

`Si/Fe` alone is worth six times the next feature. The curve says the rest is
almost free: `k = 1` gives AUC 0.794, `k = 2` jumps to 0.996, `k = 5` reaches
0.9975, and all 26 give 0.9976. **Five noiseless features capture the entire
noiseless ceiling**, and the chosen set is `Si, Al, P, S, Zn`.

That is the physics, not an artefact: the label asks whether removing the PISN
ejecta changes the pattern by more than the noise floor, and the elements with
the largest PISN-versus-Pop II contrast per unit noise are the silicon-group and
odd-iron elements, of which Si dominates. `Zn` ranks fifth on permutation but
has the highest MDI of the low-ranked group, which is the missingness effect
described below, not extra physics.

## Observed ranking

| feature | permutation | MDI | Li+ detections |
|---|---|---|---|
| Na/Fe | **0.0606** | 0.246 | 364 |
| Mg/Fe | 0.0296 | 0.144 | 385 |
| Ca/Fe | 0.0295 | 0.154 | 384 |
| Cr/Fe | 0.0150 | 0.160 | 369 |
| Si/Fe | 0.0104 | 0.049 | 177 |
| Ti/Fe | 0.0101 | 0.056 | 380 |
| Co/Fe | 0.0051 | 0.054 | 265 |
| Ni/Fe | 0.0046 | 0.069 | 288 |
| Zn/Fe | 0.0024 | 0.018 | 169 |
| C/Fe | 0.0011 | 0.028 | 281 + 70 limits |
| Mn/Fe | 0.0010 | 0.024 | 178 |

Curve: `k = 1` 0.771, `k = 3` 0.951, `k = 5` 0.964, `k = 6` 0.969, `k = 8`
0.9711, `k = 10` 0.9716, all 11 0.9716.

**Chosen: eight elements** — `Na, Mg, Ca, Cr, Si, Ti, Co, Ni`. The three dropped
(`Zn`, `C`, `Mn`) are also the three with the fewest useful measurements
(Zn 169, Mn 178, C 281 with 70 upper limits), so the statistics and the
performance ranking agree.

Note that `Ti` is now a *strong* contributor (0.0101, sixth place). Excluding it
was the wrong call for the classifier even though its yield offset is real.

## Confirmation on the production matrix

The ranking above uses the ratio columns alone. The production classifier also
sees detection flags, log uncertainties, the giant indicator and the independent
ratio features. Refitting that full matrix with and without the three dropped
elements, same folds and seed:

| columns | AUC | Brier |
|---|---|---|
| all 11 ratios (+ flags, σ, giant, independent) | 0.9739 | 0.0511 |
| chosen 8 (+ the same extras) | **0.9738** | 0.0512 |

So the selection costs 0.0001 AUC and buys three fewer columns, three fewer
catalogue failure modes, and a narrower `NaN` surface for the real stars.

## Caveats

1. **MDI is not trustworthy alone.** `Zn/Fe` is missing whenever the Zn yield
   falls below the floor, and that missingness correlates with the source mix.
   Median imputation turns it into a split that MDI rewards (0.106) while
   permutation, which is measured on the same imputed data but against the
   actual AUC, gives it only 0.0022. Trust the permutation column.
2. **The observed block imputes missing ratios with the column median.** The
   production matrix carries the missingness explicitly as a flag instead; the
   ranking is a guide to the columns, not a reproduction of the model.
3. **This is mock cross-validation, not real-star validation.** It says which
   columns carry the label in the generator. The stars decide whether the
   generator is right, and that question is still open.
4. **The subset is not yet the default.** `ObservationConfig.feature_elements`
   still holds all 12 Li+ elements; applying the choice means setting it to the
   eight above and regenerating the pool, after which the calibration and
   reweighting numbers must be recomputed.
