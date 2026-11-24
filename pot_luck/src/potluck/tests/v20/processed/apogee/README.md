# APOGEE DR17 fZP predictions

Input: `data/raw/allStarLite-dr17-synspec_rev1.fits` (733,901 rows).

## Features

The regressor is trained separately for each yield policy, using every potluck
element with finite APOGEE values:

C, N, O, Na, Mg, Al, Si, S, K, Ca, Ti, V, Cr, Mn, Co, Ni.

Excluded:

- Zn: column absent.
- Cu: `CU_FE` is present but has 0 finite values; only `CU_FE_ERR` is populated.
- P: `P_FE` has 0 finite values.
- Ce: not a potluck feature.
- Fe: reference element.

`FE_H` is used only for the metallicity cut, not as a feature.

## Duplicates and cuts

- 733,901 rows, 657,135 unique `APOGEE_ID`, 76,766 duplicate rows.
- One row per star: highest `SNR`.
- Finite `FE_H`.
- `[Fe/H]` in `[-5, -1]`.
- `FE_H_FLAG == 0`.
- `SNR >= 50`.
- At least 8 of the 16 features finite; remaining missing ratios are median
  imputed from the training mocks.

Counts:

| step | count |
|---|---:|
| unique stars with finite `FE_H` | 585,579 |
| `[Fe/H]` in `[-5, -1]` | 24,850 |
| `FE_H_FLAG == 0`, `SNR >= 50` | 19,352 |
| after >= 8 features | 18,962 |

## Per-policy results

Each yield policy is trained on its own 20k mocks. No tables are concatenated.

| policy | median fZP | max fZP | fZP >= 0.25 | fZP >= 0.5 |
|---|---:|---:|---:|---:|
| Limongi18 clamp | 0.000 | 0.419 | 10 | 0 |
| WW95 native | 0.027 | 0.492 | 44 | 0 |
| Nomoto13 blend | 0.001 | 0.291 | 2 | 0 |
| Nomoto13 clamp | 0.016 | 0.413 | 14 | 0 |
| CK20 blend | 0.016 | 0.358 | 6 | 0 |
| CK20 clamp | 0.062 | 0.504 | 149 | 2 |

Only CK20 clamp has stars above 0.5:

| APOGEE_ID | [Fe/H] | SNR | fZP |
|---|---:|---:|---:|
| 2M01265151-7748239 | -1.07 | 58.0 | 0.504 |
| 2M10090230+2518224 | -1.35 | 69.1 | 0.501 |

## Files

Per policy:

- `apogee_fzp_{policy}.csv`
- `apogee_fzp_hist_{policy}.png`
- `apogee_fzp_summary_{policy}.json`
- `apogee_fzp_gt05_yields_{policy}.png`

Summary:

- `apogee_fzp_summary_all.csv`

Other:

- `apogee_feh_hist.png` — APOGEE [Fe/H] distribution.
- `coverage_check/` — mock counts and held-out fZP recovery.
- `reweight/` — training-prior upweighting experiment.
- `../fzp_bins/` — predicted-vs-true fZP bins.

## Caveats

- The output is a point estimate, not a calibrated posterior.
- APOGEE has no Zn and no usable Cu.
- The yields are never mixed; each policy is separate.
- The yield plots show observed APOGEE points only. N is omitted from the plots
  because its APOGEE errors are very large; it is still in the model.
- The 0.2-0.4 range is ambiguous; see `docs/potluck_final.md` sections 15.5l
  and 15.5m.
