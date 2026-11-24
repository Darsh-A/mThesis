# Potluck — learning notes

Working notes for understanding the potluck simulator and its results, written
topic by topic for the Li+ workshop.

## Reading order

| file | topic |
| --- | --- |
| [`01_big_picture.md`](01_big_picture.md) | The question, the bucket, simulator vs forward fit, the two labels |
| [`02_physics_model.md`](02_physics_model.md) | Prior draws, three sources, the two clocks, dilution, the cuts, the low-Z floor |
| [`03_observation_layer.md`](03_observation_layer.md) | Record borrowing, element-level noise, missing vs limits, reweighting, catalogue coverage |
| [`04_ml_layer.md`](04_ml_layer.md) | Design matrix, classifier, isotonic calibration, prior shift, regressor, support gate |
| [`05_results.md`](05_results.md) | Sanity checks, within-table AUC, cross-table transfer, element importance, operating points, Li+ application |
| [`06_pushback.md`](06_pushback.md) | Likely questions from the Li+ authors and the answers that hold |

## Sourcing

Numbers are read from the code and from the stored outputs under
`src/potluck/tests/v20/`, not from the older prose documents in `docs/`, which are
stale in several places:

- `config.py` defaults `mdil_range` to $[10^5, 5\times10^8]\,M_\odot$; the v20
  manifest records $[2\times10^5, 10^9]$.
- `config.py` defaults `popii_table` to `Limongi18`; the package README says
  Nomoto13.
- `config.py` defaults `exclude_elements = ()`, so Ti is *not* excluded by default.
- The package README says "14 noiseless ratios"; the default feature set gives 26.
- Every v20 training site uses `y_f_05`, so all reported AUC, transfer and
  operating-point numbers are label A.
- The v20 README's CK20 operating-point row is not present in
  `operating_points.json`.

## Scope of the reported results

Unless stated otherwise:

- label A, $y_{0.5} = \mathbf{1}[f_Z^P \ge 0.5]$
- 20,000 accepted mocks per yield policy, seed 42
- stage-1 datasets: no Li+ masks, errors or importance weights
  ($N_{\rm eff} = N$)
- the Li+ application is a separate stage with masks, errors and weights
  ($N_{\rm eff} \approx 7{,}986$ of 10,000)

## Yield policies

| policy | source | low-$Z$ handling |
| --- | --- | --- |
| `Limongi18_clamp` | Limongi & Chieffi (2018) | clamp at $3.236\times10^{-5}$ |
| `WW95_native` | Woosley & Weaver (1995) | native, floor $1.345\times10^{-6}$ |
| `Nomoto13_blend` | Nomoto et al. (2013) | blend to $Z=0$ below $10^{-3}$ |
| `Nomoto13_clamp` | Nomoto et al. (2013) | clamp at $10^{-3}$ |
| `Nomoto13_CK20_blend` | Nomoto + CK20 | blend to $Z=0$ below $10^{-3}$ |
| `Nomoto13_CK20_clamp` | Nomoto + CK20 | clamp at $10^{-3}$ |
