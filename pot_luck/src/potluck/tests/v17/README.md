# v17 — M_dil window sweep against the Li+ metallicity

Sweeps the **`M_dil` window bounds** (α held at 0) using the `prior_screen`
levels, and scores each against the Li+ metallicity computed from its own
abundances (not scaled-solar).  If any window were clearly better, it would show
up here.  None is.

## Windows and pools

| label | `mdil_range` [Msun] | pool |
|---|---|---|
| 2e5-1e9 | (2.0e5, 1.0e9) | `limongi18_50k_a0_all_elements.npz` (default) |
| 1e6-1e9 | (1.0e6, 1.0e9) | `limongi18_50k_w1_all_elements.npz` |
| 2e5-1e8 | (2.0e5, 1.0e8) | `limongi18_50k_w2_all_elements.npz` |
| 5e6-1e9 | (5.0e6, 1.0e9) | `limongi18_50k_w3_all_elements.npz` |
| 2e5-5e7 | (2.0e5, 5.0e7) | `limongi18_50k_w4_all_elements.npz` |

All in `src/potluck/tests/v5/`, 50,000 accepted mocks, seed 42, Limongi18 on its
own z nodes.

## Scoring

Li+ Z is from its own abundances: `[alpha/Fe]` = mean of measured `[Mg/Fe]`,
`[Si/Fe]`, `[Ca/Fe]`, then `Z = Z_sun 10**[Fe/H] (0.638 10**[alpha/Fe] + 0.362)`.
Mock `Z_final` is weighted by `importance_weight`.  The match is a weighted
two-sample KS distance against the Li+ Z, plus a Q-Q panel.  Clamped mocks are
`Z_star < 3.236e-5`.

## Result

```
 window   clamp   Zfin med    KS   KS_clamp  KS_free   label_cl  label_free
2e5-1e9   0.569    -4.202   0.075    0.239    0.368      0.185     0.309
1e6-1e9   0.629    -4.305   0.048    0.240    0.358      0.193     0.244
2e5-1e8   0.502    -4.102   0.080    0.241    0.367      0.221     0.307
5e6-1e9   0.784    -4.521   0.076    0.211    0.349      0.195     0.066
2e5-5e7   0.448    -4.043   0.082    0.251    0.370      0.238     0.313
```
(Li+ abundance Z log10 median = -4.228; scaled-solar = -4.513.)

- The overall KS moves only over 0.048-0.082 across the whole sweep.  The window
  barely matters at the 50k level.
- **Raising the lower bound** (1e6, 5e6) removes the small-cloud / high-Z tail:
  clamping climbs (0.63, 0.78) and the Z median drops to -4.31 / -4.52.
- **Lowering the upper bound** (1e8, 5e7) removes the large-cloud / low-Z tail:
  clamping falls (0.50, 0.45) but the Z median rises to -4.10 / -4.04, away from
  Li+.
- The default `2e5-1e9` is the only window whose Q-Q curve tracks the 1:1 line
  across the full range.  `1e6-1e9` has the lowest KS (0.048) but a 0.08 dex low
  median, so KS and location disagree.

`KS_clamp` (~0.21-0.25) and `KS_free` (~0.35-0.37) exceed the overall KS because
each group spans only part of the Li+ Z range; the overall number is the one to
quote.

For scale, a two-sample KS against 385 Li+ stars has a 5% critical distance of
about `1.36/sqrt(385) = 0.069`.  The default (0.075) is marginally above it,
`1e6-1e9` (0.048) below — but the mock sample is weighted, so the effective
count is not 50,000 and this is indicative only.

## Conclusion

**No window is established as "proper".**  The default `(2.0e5, 1.0e9)` is as
consistent with the Li+ metallicity as any other and is the only one that tracks
the full Q-Q line; the alternatives trade clamping for a Z offset in one
direction or the other.  This is expected: `Z_final = M_Ztot / M_dil` is
degenerate with the drawn event amounts, so the observed Z constrains only the
ratio, not `M_dil` on its own.

Report the window as a prior with a sensitivity band (this table), not as a
fitted value.

## Figures and data

- `z_final_vs_li.png` — weighted histograms, Q-Q vs Li+, clamped fraction + KS.
- `clamp_impact.png` — per-element shift between clamped and unclamped mocks.
- `metallicity_budget.json`, `clamp_impact.json`.

## Regenerate

```bash
SP=<scratchpad>; D=src/potluck/tests/v5
declare -A W=([w1]="1000000 1000000000" [w2]="200000 100000000" [w3]="5000000 1000000000" [w4]="200000 50000000")
for tag in w1 w2 w3 w4; do read lo hi <<< "${W[$tag]}"
  echo "{\"prior\": {\"mdil_range\": [$lo.0, $hi.0]}, \"runtime\": {\"n_samples\": 50000, \"seed\": 42}}" > $SP/v17_$tag.json
  python -m src.potluck generate --config $SP/v17_$tag.json --n 50000 --seed 42 \
      --output $D/limongi18_50k_${tag}_all_elements.npz --verbose
done
python -m src.potluck.diagnostics.metallicity_budget \
    $D/limongi18_50k_a0_all_elements.npz $D/limongi18_50k_w1_all_elements.npz \
    $D/limongi18_50k_w2_all_elements.npz $D/limongi18_50k_w3_all_elements.npz \
    $D/limongi18_50k_w4_all_elements.npz \
    --labels "2e5-1e9" "1e6-1e9" "2e5-1e8" "5e6-1e9" "2e5-5e7" --outdir src/potluck/tests/v17
python -m src.potluck.diagnostics.clamp_impact \
    $D/limongi18_50k_a0_all_elements.npz $D/limongi18_50k_w1_all_elements.npz \
    $D/limongi18_50k_w2_all_elements.npz $D/limongi18_50k_w3_all_elements.npz \
    $D/limongi18_50k_w4_all_elements.npz \
    --labels "2e5-1e9" "1e6-1e9" "2e5-1e8" "5e6-1e9" "2e5-5e7" --outdir src/potluck/tests/v17
```
