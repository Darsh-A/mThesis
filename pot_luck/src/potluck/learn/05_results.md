# 5. The results

All numbers are read from the JSON and `.npz` outputs under
`src/potluck/tests/v20/`, and are for **label A**, $y_{0.5}$, unless stated
otherwise.

## 5.1 Two sanity checks first

Before any ML number means anything, two things have to be true.

### The sampler is not silently reweighting

From `prior_check.json`, the drawn PISN counts are

| $n_P$ | count | fraction | prior |
| ---: | ---: | ---: | ---: |
| 0 | 2,545 | 0.509 | 0.50 |
| 1 | 1,471 | 0.294 | 0.30 |
| 2 | 741 | 0.148 | 0.15 |
| 3 | 243 | 0.049 | 0.05 |

So the categorical draw behaves.

### Every yield table selects the same sample

This is the fix for the old v19 problem, and it is what makes cross-table
comparison legitimate.

| policy | acceptance | zero-PISN | $M_{Z,\rm pre}$ p16 / p50 / p84 [$M_\odot$] |
| --- | ---: | ---: | --- |
| `Limongi18_clamp` | 0.511 | 0.368 | 87 / 226 / 1,090 |
| `WW95_native` | 0.508 | 0.369 | 88 / 225 / 1,098 |
| `Nomoto13_blend` | 0.512 | 0.366 | 87 / 227 / 1,105 |
| `Nomoto13_clamp` | 0.512 | 0.366 | 87 / 227 / 1,105 |
| `Nomoto13_CK20_blend` | 0.510 | 0.365 | 86 / 225 / 1,090 |
| `Nomoto13_CK20_clamp` | 0.513 | 0.370 | 86 / 226 / 1,099 |

Acceptance varies by 0.005 across tables and zero-PISN by 0.005. The accepted
clouds are the same. **Any remaining difference between tables is a yield
difference, not a selection difference.**

## 5.2 Within-table discrimination

5-fold out-of-fold, 20,000 mocks per policy, observed block:

| policy | observed AUC | noiseless AUC | Brier |
| --- | ---: | ---: | ---: |
| `Limongi18_clamp` | 0.978 | 0.997 | 0.057 |
| `WW95_native` | 0.969 | 0.996 | 0.072 |
| `Nomoto13_blend` | 0.965 | 0.997 | 0.075 |
| `Nomoto13_clamp` | 0.977 | 0.997 | 0.061 |
| `Nomoto13_CK20_blend` | 0.975 | 0.997 | 0.062 |
| `Nomoto13_CK20_clamp` | 0.977 | 0.997 | 0.059 |

The gap between 0.997 and 0.97 is the **cost of observation**: noise, missing
elements, censoring. Chemistry alone is nearly perfect; measuring it costs about
three AUC points.

This is the easy question, though. Within one table the classifier has seen that
table's quirks in training, so it can exploit them.

## 5.3 Cross-table transfer

Train on one table, test on another. This is where the physics shows through.

Leave-one-table-out, observed AUC:

| held out | AUC |
| --- | ---: |
| `Limongi18_clamp` | 0.975 |
| `WW95_native` | **0.902** |
| `Nomoto13_blend` | 0.973 |
| `Nomoto13_clamp` | 0.986 |
| `Nomoto13_CK20_blend` | 0.974 |
| `Nomoto13_CK20_clamp` | 0.973 |

The full matrix tells the same story from every direction: **models trained on any
other table score 0.898–0.904 on WW95.** Everything else transfers at 0.94–0.99.

So the defensible claim is:

> PISN-vs-CCSN discrimination transfers between yield tables at AUC 0.94–0.99,
> except for WW95, which is the systematic outlier at about 0.90.

That is a strong result because it says the classification is not an artefact of
one table's choices — except where a table is genuinely different.

## 5.4 What the PISN fingerprint actually is

If someone asks what the model is keying on, this is the answer. PISN minus Pop II
at $Z_\star = 10^{-5}$ and $t = 10\,\mathrm{Myr}$, pooled over tables
(`pisn_contrast.json`):

| element | contrast (dex) |
| --- | ---: |
| Zn | **−8.16** |
| Co | **−3.25** |
| Na | −1.39 |
| Ni | −1.31 |
| Mn | −0.11 |
| Ti | −0.08 |
| Mg | +0.41 |
| Cr | +0.52 |
| Ca | +0.81 |
| Si | +1.13 |

A PISN **strongly underproduces Zn and Co** relative to ordinary core-collapse, and
overproduces Si, Ca, Cr, Mg. Zn is off by eight orders of magnitude — a cliff, not
a subtle statistical signal.

The caveat is that the *size* of the contrast is table-dependent:

| element | across policies (dex) |
| --- | ---: |
| Zn | −7.74 to −8.77 |
| Co | **−0.51 to −2.97** |

Zn is the robust tracer. Co is the fragile one.

## 5.5 Which elements are actually needed

Mean within-table AUC over the six policies (`element_subsets.json`), generic
element noise:

| subset | $n$ | $\sigma = 0.2$ | $\sigma = 0.3$ |
| --- | ---: | ---: | ---: |
| all 11 Li+ elements | 11 | 0.968 | 0.944 |
| no Zn | 10 | 0.952 | 0.915 |
| no Co | 10 | 0.967 | 0.942 |
| no Zn, no Co | 9 | 0.948 | 0.905 |
| alpha only (Mg, Si, Ca, Ti) | 4 | **0.863** | 0.775 |
| alpha + Na | 5 | 0.929 | 0.877 |
| core 7 + Zn | 8 | 0.966 | 0.940 |

Three conclusions:

1. **Dropping Zn costs about 0.015 AUC; dropping Co costs about 0.001.** Zn is the
   valuable element; Co is nearly redundant once Zn is present.
2. **Alpha elements alone are not enough** (0.863). That is the most useful
   negative result in the project: an argument built only on alpha enhancement will
   not work.
3. **Seven or eight elements recover almost everything** (0.966 vs 0.968).

> **Critical limitation on conclusion 1.** This analysis does **not** use the Li+
> observation layer. `element_subsets.py` rebuilds the ratios directly from
> `log_abundance` plus Gaussian noise:
>
> ```python
> noisy = loga + rng.normal(0.0, sigma, loga.shape)
> ```
>
> There are no availability masks and no censoring, so **Zn is present in 100 % of
> rows**. In the real Li+ application it is present in 28 %.
>
> Repeating the leave-one-out test on the deployed model tells a different story.
> With retraining on `application/Limongi18_clamp` (full AUC 0.9732):
>
> | element | $\Delta$AUC |
> | --- | ---: |
> | Mg | +0.0067 |
> | Ca | +0.0051 |
> | Cr | +0.0031 |
> | Ti | +0.0026 |
> | Si | +0.0014 |
> | Ni | +0.0007 |
> | Zn | **+0.0001** |
> | Co | −0.0003 |
> | Na | −0.0013 |
>
> Restricted to the 2,818 rows where Zn is actually measured, dropping it gives
> $\Delta\mathrm{AUC} = +0.0025$.
>
> So the correct statement is: **in a Li+-free idealised survey Zn is the most
> valuable single element; in the Li+ catalogue its contribution is small, because
> it is measured in 44 % of stars and survives censoring in 28 % of mocks.** The
> Zn result is about an idealised survey, not about Li+.

With the full 26-ratio set, no single element matters at all: the largest removal
penalty is Al at $\Delta\mathrm{AUC} = 0.0018$ (§4.9).

## 5.6 Operating points — the honest headline

AUC is prevalence-independent, so it does not answer the question a survey actually
asks. The relation that does is

$$
\mathrm{precision}
= \frac{\mathrm{TPR}\cdot\pi}
{\mathrm{TPR}\cdot\pi + \mathrm{FPR}\cdot(1-\pi)} ,
$$

which shows immediately why a low prevalence destroys precision: with
$\pi = 2.5\times10^{-3}$, even a small FPR contributes as many false positives as
TPR contributes true ones.

From `operating_points.json`, 5-fold OOF, f05 label:

| policy | feature set | TPR @ FPR 1 % | TPR @ FPR 0.1 % | precision @ 0.1 % | @ 0.25 % | @ 1 % | TP / FP per $10^4$ @ FPR 0.1 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Limongi18` | general 8 | 0.741 | **0.493** | 0.331 | 0.553 | 0.833 | 12.3 / 10.0 |
| `Limongi18` | observed 11 | 0.727 | **0.413** | 0.292 | 0.509 | 0.807 | 10.3 / 10.0 |
| `WW95` | general 8 | 0.673 | 0.405 | 0.288 | 0.504 | 0.804 | 10.1 / 10.0 |
| `WW95` | observed 11 | 0.653 | 0.398 | — | 0.500 | — | 10.0 / 10.0 |

At FPR 0.1 % and $\pi = 0.25\%$, that is roughly **10 true PISN and 10 false
positives per 10,000 stars**, precision about 0.51.

`operating_points.json` currently contains only `Limongi18` and `WW95`. The CK20
row that appears in the v20 README is not in the output file and should not be
quoted.

### The failure mode is the right one

Splitting the false positives by how ambiguous the object is, at overall
FPR $= 0.1\%$:

| policy | FPR on clear negatives ($f_Z^P < 0.25$) | FPR on near-threshold ($0.25 \le f_Z^P < 0.5$) | ratio |
| --- | ---: | ---: | ---: |
| `Limongi18` | 0.00062 | 0.00365 | 5.9× |
| `WW95` | 0.00015 | 0.00618 | **41×** |

False positives are **6× to 41× concentrated in genuine near-threshold mixtures**,
not in obvious non-PISN stars. The model is not confused by clean negatives; it is
confused by the physically ambiguous ones. That is the expected failure mode and it
is worth stating explicitly.

## 5.7 The Li+ application

Separate stage: 10,000 mocks per policy, Li+ iron box, masks, errors, weights and
support gate.

| policy | out of model | in model | max posterior | median $\mathbb{E}[f_Z^P]$ |
| --- | ---: | ---: | ---: | ---: |
| `Limongi18_clamp` | 353 / 385 | 32 | 0.019 | 0.10 |
| `WW95_native` | 311 / 385 | 74 | 0.006 | 0.08 |
| `Nomoto13_blend` | 360 / 385 | 25 | 0.022 | 0.06 |
| `Nomoto13_clamp` | 352 / 385 | 33 | **0.043** | 0.06 |
| `Nomoto13_CK20_blend` | 256 / 385 | **129** | 0.022 | 0.05 |
| `Nomoto13_CK20_clamp` | 254 / 385 | **131** | 0.018 | 0.07 |

Three separate messages, which should be kept separate.

**1. Zero candidates.** No Li+ star reaches $p' \ge 0.5$ at
$\pi = 2.5\times10^{-3}$. The largest posterior anywhere is 0.043. This is the
prevalence effect, not a modelling failure, but say it before anyone finds it.

**2. The raw support gate is broken, and Ti is why.** In-model fractions are
6.5–34 %. Allowing one offset per core element brings every dataset to 91–95 %.
Ti's offset is $+0.62$ to $+0.73$ dex with no $[\mathrm{Fe}/\mathrm{H}]$ trend
($\rho \approx 0.06$). Present the raw gate as a diagnostic of the yields, never as
"most Li+ stars are unusual".

**3. CK20 fits Li+ best**, at 129–131 of 385 in model versus 25–74 for the others.
That is a mild preference for CK20, but it rests on extrapolated yields, because
97 % of CK20 mocks sit below that table's own floor.

## 5.8 What is strong and what is fragile

**Strong**

- The generator reproduces Salvadori et al. (2019) Eq. 13 to machine precision —
  a bookkeeping check, not a yield validation.
- Selection is identical across all six tables.
- Transfer is 0.94–0.99 except WW95.
- The eight-element pattern is the result; a linear model nearly matches the forest.
- False positives concentrate in near-threshold mixtures.

**Fragile**

- 97 % of Nomoto/CK20 mocks use extrapolated yields.
- Ti is off by $0.6$–$0.7$ dex against Li+, unresolved.
- The headline classifier predicts label A, which no observer can measure.
- Zero candidates at the assumed prevalence.
- The observation layer is Li+-specific and inherits its biases.
- Zn, the key element, exists in only 28 % of the mocks.

## 5.9 What to remember from this section

1. All tables select the same sample, so cross-table differences are physical.
2. Within-table AUC 0.965–0.978 observed, 0.996–0.997 noiseless.
3. Transfer 0.902–0.986, with WW95 the outlier.
4. Zn is the signal; alpha elements alone are not enough.
5. The honest headline is ~10 true / 10 false per $10^4$ at FPR 0.1 %, not AUC 0.97.
6. No Li+ star is a candidate at $\pi = 0.25\%$, and the support gate measures
   yields rather than stars.
