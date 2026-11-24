# v16 — M_dil slope as a sensitivity knob, and where the clamp actually bites

This test does three things:

1. Compares the mock metallicity budget (`Z_final`) with the **Li+ metallicity
   computed from its own abundances**, not from a scaled-solar `[Fe/H]`.
2. Reruns the Limongi18 pool with a steeper `M_dil` prior and reports `alpha = 0,
   0.5, 1` as a **sensitivity**, not a tuning target.
3. Splits the accepted mocks into clamped and unclamped groups and asks which
   element ratios move between them.

The conclusion is: **do not steepen `alpha` to remove the clamp.** Once the Li+
metallicity is corrected for alpha enhancement, the default `alpha = 0` already
matches it; steepening overshoots. And the clamp is not harmless — it shifts the
label rate and several element ratios.

## The knob

`p(log M_dil) ∝ M_dil^(-alpha)` over `[2e5, 1e9] Msun`, implemented as
`PriorConfig.mdil_slope` and used in `events._tilted_loguniform`. `alpha = 0` is
the original log-uniform prior; `alpha > 0` tilts it toward smaller clouds.

| alpha | median M_dil | share < 1e7 Msun |
|---|---|---|
| 0   | 1.4e7 | 46% |
| 0.5 | 7.8e5 | 87% |
| 1   | 4.0e5 | 98% |

## Pools

```
src/potluck/tests/v5/limongi18_50k_a0_all_elements.npz    (alpha = 0)
src/potluck/tests/v5/limongi18_50k_a05_all_elements.npz   (alpha = 0.5)
src/potluck/tests/v5/limongi18_50k_a1_all_elements.npz    (alpha = 1)
```
50,000 accepted mocks each, seed 42, Limongi18 Pop II on its own z nodes. The
alpha = 0 pool reproduces the original v5 Limongi18 ledger exactly.

## Corrected Li+ comparison

The previous comparison used `Z = Z_sun 10**[Fe/H]`, which is biased low for
alpha-enhanced halo stars. Here `[alpha/Fe]` is the mean of the measured
`[Mg/Fe]`, `[Si/Fe]`, `[Ca/Fe]` (median +0.375), and

```
Z_Li = Z_sun * 10**[Fe/H] * (0.638 * 10**[alpha/Fe] + 0.362)
```

| quantity | log10 Z median |
|---|---|
| Li+ scaled-solar (biased low) | -4.513 |
| **Li+ from abundances** | **-4.228** |
| mock `Z_final`, alpha = 0 | -4.202 |
| mock `Z_final`, alpha = 0.5 | -3.844 |
| mock `Z_final`, alpha = 1 | -3.673 |

The alpha correction moves the Li+ peak by +0.286 dex, onto the `alpha = 0`
mocks. `alpha = 0.5` and `alpha = 1` overshoot by 0.38 and 0.55 dex.

Mock histograms are weighted by `importance_weight` (the Li+ [Fe/H] reweighting)
and are already R4-accepted. Figure: `z_final_vs_li.png`.

## Clamping and its impact

The clamp acts on the pre-enrichment `Z_star` (lookup floor 3.236e-5), not on
`Z_final`. `clamped = Z_star < floor`; stored as a per-sample flag in new pools
(`arrays["clamped"]`) and derived from `z_star` for these older pools.

| alpha | clamped fraction | label rate clamped | label rate unclamped |
|---|---|---|---|
| 0   | 0.569 | 0.185 | 0.309 |
| 0.5 | 0.317 | 0.171 | 0.397 |
| 1   | 0.202 | 0.088 | 0.446 |

The clamped group carries the PISN label less often, so the clamp is not
neutral. At `alpha = 0` the element ratios that move most between the groups
(median clamped minus unclamped) are:

```
Ga/Fe -0.50   Na/Fe +0.31   F/Fe -0.27   K/Fe +0.23
Sc/Fe +0.23   N/Fe -0.22    Ge/Fe -0.20  ...
```

`Na/Fe` is an observed Li+ element, so the clamp is visible in the measured
chemistry, not just trace species. Figure: `clamp_impact.png`.

Caveat: "clamped" is correlated with low `Z_star`, so this split confounds the
clamp with the low-metallicity end. A clean clamp test would compare clamped
mocks against a counterfactual evaluated at the same `Z_star` without the floor.

## Consequences

1. **Keep `alpha = 0`.** It matches the corrected Li+ metallicity; the steeper
   priors move the mock peaks away from the Li+ distribution.
2. **Do not present any alpha as derived.** It is a prior, and `0, 0.5, 1` are a
   sensitivity ladder.
3. **Report metrics split by clamp group** (or at least the clamped fraction),
   since the label rate and `Na/Fe` differ between them.
4. If a compromise is ever wanted, `alpha ~ 0.25` sits between the two; it was
   not generated here because the corrected comparison does not call for it.

## Regenerate

```bash
SP=<scratchpad>
for pair in "a0:0.0" "a05:0.5" "a1:1.0"; do
  tag=${pair%%:*}; slope=${pair##*:}
  cat > $SP/v16_$tag.json <<EOF
{"prior": {"mdil_slope": $slope}, "runtime": {"n_samples": 50000, "seed": 42}}
EOF
  python -m src.potluck generate --config $SP/v16_$tag.json --n 50000 --seed 42 \
      --output src/potluck/tests/v5/limongi18_50k_${tag}_all_elements.npz --verbose
done

D=src/potluck/tests/v5
python -m src.potluck.diagnostics.metallicity_budget \
    $D/limongi18_50k_a0_all_elements.npz \
    $D/limongi18_50k_a05_all_elements.npz \
    $D/limongi18_50k_a1_all_elements.npz \
    --labels "alpha=0" "alpha=0.5" "alpha=1" --outdir src/potluck/tests/v16
python -m src.potluck.diagnostics.clamp_impact \
    $D/limongi18_50k_a0_all_elements.npz \
    $D/limongi18_50k_a05_all_elements.npz \
    $D/limongi18_50k_a1_all_elements.npz \
    --labels "alpha=0" "alpha=0.5" "alpha=1" --outdir src/potluck/tests/v16
```

Outputs: `z_final_vs_li.png`, `clamp_impact.png`, `metallicity_budget.json`,
`clamp_impact.json`.
