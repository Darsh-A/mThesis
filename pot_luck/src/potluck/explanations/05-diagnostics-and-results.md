# Diagnostics and results

This file records the 2000-sample generator diagnostics. The later 100k Li+
observation, calibration, and support results are in
[Real-star validation](07-real-star-validation.md); where the two disagree, the
newer note is authoritative.

## What the diagnostics test

| diagnostic | question |
|---|---|
| ledger | which cut removes which samples, and does the order hide a prior? |
| presence split | does selection change the PISN fraction? |
| labels | are `f_Z^P` and `f_Fe^P` spread over the full range? |
| class balance | is `y` tied to `Z*`, `n_III`, `t`, `M_dil`, or `M*II`? |
| mixture cells | are all source combinations represented? |
| features | are the ratios collinear, and which are masked? |
| clock | does `Delta` behave as the two-clock model predicts? |
| dilution | does `Z* = M_Zpre/M_dil` behave, and is the cap active? |
| event masses | how much metal does each source actually add? |
| detection | is `D` monotone in `f_Z^P`, and where is the threshold? |
| limit sweep | how sensitive is `y` to `k`, `z_max`, feature floor, `[Fe/H]`, cap? |
| ceiling | what is the best chemistry-only performance? |
| nuisance | can the label be predicted without any abundances? |
| stratified | where is the ceiling lost? |
| transfer | does a model trained on the base prior survive an ablation prior? |
| mimic list | which no-PISN samples look most like PISN? |

All figures are written by

```bash
python -m src.potluck plots data/potluck/dataset.npz
```

and the scalar summaries by `diagnose`, `ceiling`, `stratified`, `reliability`,
`transfer`.

## Latest 2000-sample run

Generated with the current defaults:

```bash
python -m src.potluck generate --n 2000 --output data/potluck/analysis/dataset_presence.npz --verbose
```

### Ledger

10,000 draws, 2000 retained, 3673 accepted before truncation, 1673 discarded.

| stage | count | fraction of draws |
|---|---|---|
| R1 | 964 | 0.096 |
| R2 | 3572 | 0.357 |
| R4 | 1791 | 0.179 |
| R3 | 0 | 0.000 |
| accepted before truncation | 3673 | 0.367 |
| retained | 2000 | 0.200 |

R2 reasons: `no_pre_enrichment = 1910`, `below_z_cr = 1662`,
`above_z_max = 0`.

### Presence split

| | all | PISN present | no PISN |
|---|---|---|---|
| drawn | 10000 | 4972 (0.497) | 5028 (0.503) |
| R1 | 964 | 0 | 964 |
| R2 | 3572 | 1561 | 2011 |
| R4 | 1791 | 1150 | 641 |
| accepted before truncation | 3673 | 2261 (0.455) | 1412 (0.281) |
| retained | 2000 | 1214 | 786 |

The accepted sample is **61.6% PISN**, up from 49.7% drawn. R1 drives the shift:
a no-PISN draw needs a Pop III event, so about 19% of no-PISN draws fail R1. PISN
draws lose more at R2 because they push `Z*` up, but not enough to offset R1.

Important: the 61.6% is a **training prevalence**, not a target prevalence. The
likelihood-ratio conversion should use the measured `pi_train`; do not reweight the
sample to a target presence unless you are deliberately changing the prior.

### Labels and sample properties

- `y = 0.3545`.
- `f_Z^P` exactly zero: 0.393; near one (>0.99): 0.059.
- Detection reasons: `normal = 1161`, `baseline_zero = 53`, `no_pisn = 786`.
- `[Fe/H]` range: −3.799 to −1.701 (R4 box).
- `n_III` quantiles: 0, 0, 3, 24, 157, 689, 932, 1025.
- `M*III/M_dil` p99 = 0.093, max = 0.0995: the joint cap is active.

### Ceiling and calibration

| model | AUC | Brier |
|---|---|---|
| chemistry (features + mask) | 0.99435 | 0.02421 |
| nuisance only | 0.84489 | 0.15606 |

Regression:

| target | R² | MAE |
|---|---|---|
| `f_Fe^P` | 0.94667 | 0.0521 |
| `f_Z^P` | 0.94985 | 0.0490 |

The nuisance AUC fell from 0.991 before the λ_III/R4 refactor to 0.845. Chemistry
stayed at 0.994. This is the main success of the refactor: the label is no longer
almost entirely a function of the prior.

### Stratified results

By `f_Z^P`:

| bin | n | y | AUC | f_Fe R² | f_Z R² | f_Z RMSE / MAE |
|---|---|---|---|---|---|---|
| 0–0.01 | 819 | 0 | — | — | — | — |
| 0.01–0.25 | 450 | 0.078 | 0.989 | 0.714 | 0.566 | 0.0455 / 0.0330 |
| 0.25–0.50 | 179 | 0.682 | 0.897 | 0.839 | −1.05 | **0.1063 / 0.0818** |
| 0.50–0.75 | 184 | 1.000 | — | 0.887 | −1.72 | **0.1239 / 0.0898** |
| 0.75–0.99 | 249 | 1.000 | — | 0.892 | — | 0.0867 / 0.0611 |
| 0.99–1.01 | 119 | 1.000 | — | — | — | 0.0373 / 0.0289 |

Negative R² in a narrow bin is a within-bin variance artifact, not a failure. The
RMSE is the useful number: about 0.10–0.12 in the mixed middle, consistent with the
irreducible spread.

By PISN mass:

| mass bin | n | y | AUC | f_Fe R² | f_Z R² | f_Z RMSE |
|---|---|---|---|---|---|---|
| 140–160 | 172 | 0.430 | 0.968 | 0.960 | 0.844 | 0.1325 |
| 160–180 | 170 | — | 0.9947 | 0.932 | 0.946 | 0.0786 |
| 180–200 | 194 | — | 0.998 | 0.958 | 0.972 | 0.0622 |
| 200–220 | 190 | — | 0.9966 | 0.934 | 0.969 | 0.0649 |
| 220–240 | 194 | — | 0.998 | 0.940 | 0.970 | 0.0620 |
| 240–270 | 294 | — | 0.994 | 0.906 | 0.963 | 0.0704 |

The light-PISN bin improved from R² 0.38 / RMSE 0.248 to 0.844 / 0.133 after the
refactor, but it is still the worst bin. This is consistent with the Fe-reference
degeneracy: a light PISN makes almost no Fe.

By `[Fe/H]`:

| bin | n | AUC | f_Fe R² | f_Z R² | f_Z RMSE |
|---|---|---|---|---|---|
| −4…−3.5 | 226 | 0.992 | 0.923 | 0.916 | 0.1099 |
| −3.5…−3 | 435 | 0.996 | 0.951 | 0.944 | 0.0876 |
| −3…−2.5 | 476 | 0.997 | 0.952 | 0.960 | 0.0729 |
| −2.5…−2 | 565 | 0.993 | 0.944 | 0.959 | 0.0709 |
| −2…−1.5 | 298 | 0.995 | 0.954 | 0.952 | 0.0755 |

Bins outside −4…−1.5 are empty by R4.

### No-PISN mimic list

The top 50 no-PISN samples by out-of-fold chemistry probability have:

- median `n_III = 12.5` (p90 = 39.5);
- median Pop II metal share = 0.033;
- median `[Fe/H] = -2.92`;
- median `Δ = 14.2` Myr.

They are dominated by high-mass Pop III groups (100 and 40 M☉) with little Pop II.
That is the expected confuser: a few massive, short-lived Pop III events can mimic
the PISN pattern without a PISN.

The top mimic has probability 0.979, `n_III = 5`, composition
`100M1×2, 40M1, 15M1, 13M1`, Pop II share 0.18, `Δ = 22.1` Myr, `[Fe/H] = -3.43`.

## How to read the ceiling

AUC 0.994 is the **easy-problem** number. It is computed on the training prior, with
placeholder noise and no catalogue selection. At a realistic PISN prevalence of
0.1–0.25% in halo stars, precision is low even at high TPR:

| TPR | FPR | prevalence | precision |
|---|---|---|---|
| 0.9 | 0.01 | 1e-3 | ~8% |
| 0.9 | 0.02 | 1e-3 | ~4% |

The next metrics should be FPR at fixed TPR and prevalence-weighted precision, not
AUC. The mimic list is the first step: it shows which no-PISN samples would become
false positives.
