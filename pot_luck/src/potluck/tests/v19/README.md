# v19 — Z* sampling across the Pop II tables

50,000 accepted mocks per table, seed 42.



| table | table Z* range | sampled Z* range |
|---|---:|---:|
| Limongi18 | 3.236e-5 – 1.345e-2 | 3.236e-5 – 1e-2 |
| Nomoto13 | 1e-3 – 5e-2 | 1e-3 – 1e-2 |
| Nomoto13_CK20 | 1e-3 – 5e-2 | 1e-3 – 1e-2 |
| WW95 | 1.345e-6 – 1.345e-2 | 1.345e-6 – 1e-2 |

## Main results

| table | acceptance | Z* p16 / p50 / p84 | M_dil p16 / p50 / p84 [Msun] | [Fe/H] p16 / p50 / p84 | RF AUC | Brier | positive rate | precision @ FPR 0.1% | Li+ out-of-model | Eq. 13 max |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Limongi18 | 0.286 | 4.9e-5 / 1.2e-4 / 3.9e-4 | 1.7e5 / 1.7e6 / 9.1e6 | -2.65 / -2.19 / -1.85 | 0.957 | 0.081 | 0.421 | 0.285 | 348 / 385 | 7.8e-16 |
| Nomoto13 | 0.043 | 1.2e-3 / 1.9e-3 / 4.0e-3 | 1.9e4 / 4.2e4 / 8.3e4 | -2.42 / -2.00 / -1.74 | 0.938 | 0.064 | 0.868 | 0.348 | 353 / 385 | 1.3e-15 |
| Nomoto13_CK20 | 0.040 | 1.2e-3 / 1.9e-3 / 4.0e-3 | 1.9e4 / 4.2e4 / 8.4e4 | -2.40 / -1.97 / -1.73 | 0.943 | 0.059 | 0.875 | 0.424 | 313 / 385 | 7.2e-16 |
| WW95 | 0.370 | 5.8e-6 / 3.6e-5 / 2.2e-4 | 4.0e5 / 6.5e6 / 6.1e7 | -3.44 / -2.77 / -2.06 | 0.969 | 0.057 | 0.235 | 0.468 | 332 / 385 | 7.8e-16 |

Zero mocks fall below or above the sampled `Z*` range for every table.

## Per-table notes

**Limongi18.** Covers the Li+ range best after WW95. Final `[Fe/H]` median
-2.19. Pop II share 0.16, zero Pop II 7%. 39% of accepted mocks have zero
PISN, close to the prior.

**Nomoto13.** Lowest nonzero node is 1e-3. Final `[Fe/H]` median -2.00, so the
Li+ median -2.64 is not reachable. Small clouds: median `M_dil` 4.2e4, median
`M_starII` 1.7e3, median `popii_n` 1, 39% zero Pop II. The accepted sample is
PISN-rich: only 13% have zero PISN and median `f_ZP` is 0.84.

**Nomoto13_CK20.** Same low-Z floor as Nomoto13. The offsets improve the Li+
pattern a little (313/385 out-of-model vs 353/385), but the final `[Fe/H]`
median is still -1.97. It is a pattern correction, not a low-Z extension.

**WW95.** Best `[Fe/H]` match (-2.77), best RF AUC (0.969) and lowest Brier
(0.057). Pop II share 0.39, zero Pop II 17%. Positive rate 0.24, the most
Li+-like class balance.

## Why the accepted M_Zpre differs

The prior is the same. Before cuts, `M_Zpre` p16 / p50 / p84 is
0 / 96.8 / 581.6 Msun for every table; after `n_PISN + n_III > 0` it is
0 / 115.5 / 678.7.

Accepted values:

| table | accepted M_Zpre p16 / p50 / p84 | accepted zero-PISN fraction |
|---|---:|---:|
| Limongi18 | 68 / 178 / 938 | 0.386 |
| Nomoto13 | 66 / 80.6 / 152 | 0.134 |
| Nomoto13_CK20 | 66 / 80.7 / 152 | 0.132 |
| WW95 | 72 / 200 / 1021 | 0.388 |

The Nomoto `Z*` range is higher, so for the same `M_Zpre` the cloud is smaller.
The burst-floor and `[Fe/H]` cuts then remove the high-`M_Zpre` / large-burst
tail. The prior is unchanged; the selection is different.

## RF transfer

Within-table observed AUC is 0.995–0.996. Cross-table transfer (train on one
table, test on another) is the honest number:

| train \ test | Limongi18 | Nomoto13 | CK20 | WW95 |
|---|---:|---:|---:|---:|
| Limongi18 | 0.995 | 0.871 | 0.885 | 0.857 |
| Nomoto13 | 0.881 | 0.996 | 0.926 | 0.618 |
| CK20 | 0.889 | 0.913 | 0.996 | 0.626 |
| WW95 | 0.923 | 0.851 | 0.832 | 0.995 |

Cross-table AUC drops to 0.62–0.93. Noiseless features transfer better
(0.78–0.999), but WW95 is still the most distinct. Within-table numbers are
optimistic.

## Confusion-matrix cuts

- `TPR = 0.9` cut: FPR 8.7–19%. High recall, many false alarms.
- `FPR = 0.001` cut: TPR 16–35%. High purity, most true positives predicted
  negative.
- The `FPR = 0.001` matrix looks bad on positives because the cut is chosen to
  keep only 0.1% of negatives. The misses are the price of that cut.


## Regenerate

```bash
python -m src.potluck.diagnostics.v19_mega \
    --outdir src/potluck/tests/v19 --n 50000 --seed 42 \
    --tables Limongi18,Nomoto13,Nomoto13_CK20,WW95 --force
```

Summary only:

```bash
python -m src.potluck.diagnostics.v19_mega \
    --outdir src/potluck/tests/v19 --summary-only
```

