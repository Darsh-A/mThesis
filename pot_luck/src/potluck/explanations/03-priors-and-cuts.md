# Priors, limits, and cuts

Every number below is a field in `src/potluck/config.py`. Change it there; do not
edit the sampler or the ejecta code.

## Prior table

| parameter | default | meaning |
|---|---|---|
| `n_pisn_probs` | `(0.5, 0.3, 0.15, 0.05)` | probability of 0, 1, 2, 3 PISN |
| `pisn_mass_range` | `(140, 270)` M☉ | uniform PISN progenitor mass |
| `lambda_iii_range` | `(0.1, 1000)` | log-uniform Poisson rate for Pop III |
| `n_iii_max` | `4096` | safety cap on one draw |
| `delta_range_myr` | `(1, 100)` | log-uniform Pop III → Pop II delay |
| `t_range_myr` | `(1, 100)` | log-uniform Pop II → star delay |
| `mdil_range` | `(2e5, 1e9)` M☉ | log-uniform cloud mass |
| `mstar_ii_range` | `(1e3, 1e7)` M☉ | log-uniform Pop II stellar mass |
| `mstar_total_max_ratio` | `0.1` | `M*III + M*II <= 0.1 M_dil` |
| `z_cr` | `1e-6` | minimum `Z*` |
| `z_max` | `None` | optional maximum `Z*` (off) |
| `feh_range` | `(-3.8, -1.7)` | R4 box on final `[Fe/H]` |
| `popiii_sampling` | `"combo"` | uniform over the 10 Ishigaki groups |
| `popiii_mass_energy_weights` | `None` | optional group weights |

Detection (`DetectConfig`):

| parameter | default |
|---|---|
| `sigma_dex` | `0.2` |
| `sigma_fe_dex` | `0.1` |
| `k` | `3.0` |
| `min_detected` | `4` |

Feature validity (`ModelConfig`):

| parameter | default |
|---|---|
| `min_element_fraction` | `1e-10` |
| `min_fe_fraction` | `1e-10` |
| `min_valid_features` | `3` |

## Why `M_dil` is 2e5–1e9

The paper's dilution derivation, with `<E51> = 1`, `f_kin = 0.1`, `f_* = 1e-4`,
and `z = 30`, gives

```
M_dil^min ≈ 30 * M_g^0.6 M_sun
```

For `M_g = 1e8 M_sun` this is about `1.9e6`, which reproduces `f_dil > 0.02`.
The smallest cloud that can host one ~200 M☉ PISN at `f_* >= 1e-4` is about
`2e6 M_sun`, giving `M_dil^min ≈ 2e5 M_sun`. Larger `f_*` or `E51 ≈ 20` raises the
floor; the scaling is `f_*^0.6 E51^0.6`, about a factor of 6 for `E51 = 20`.

The upper end comes from `f_dil <= 10` at `M_g <= 1e8`, about `1e9 M_sun`.

The ceiling matters as much as the floor. A PISN-dominated cloud with
`M_dil = 1e7 M_sun` reaches only `[Fe/H] ≈ -3`. Li+ has about 5% of stars below
-3.5, which needs clouds of `1e7–1e8 M_sun`. A 1e7 ceiling cannot populate that tail.

**Assumption to state:** the paper's `f_* <= 0.1` is relative to the gas mass `M_g`.
The code applies `0.1 M_dil`. These are the same only if `f_dil ≈ 1`.

## Why the joint stellar cap

The cap is

```
M*III + M*II <= 0.1 * M_dil
```

`M*III` is not a free parameter: each Pop III event has a fixed progenitor mass, so

```
M*III = sum_g n_g m_g
```

With `n_III ~ 1000` at tens of M☉, `M*III` is `1e4–1e5 M_sun`, comparable to a small
cloud. The joint cap ties a large Pop III population to a large cloud, which keeps
the gas mass and the stellar budget consistent. It also prevents the `M*II` window
from collapsing at the `M_dil` floor: at `M_dil = 2e5`, the window is `[1e3, 2e4]`.

## Why `lambda_III` is log-uniform

A single PISN ejects a median of about 96 M☉ of metals. A Pop III event ejects a mean
of 3.54 M☉ (median 0.34). The sum of `n` events has expectation `3.54 n`, not
`0.34 n`. Therefore:

| `f_Z` | required `n_III` at the mean |
|---|---|
| 0.9 | ~3 |
| 0.5 | ~27 |
| 0.1 | ~245 |
| 0.05 | ~515 |

A log-uniform rate from 0.1 to 1000 covers the regime where a PISN dominates through
the regime where Pop III dilutes it completely. A fixed count prior over 0–3 cannot.

At `n = 27`, the sum standard deviation is about 41 M☉, so `f_Z` scatters roughly
0.35–0.65 at fixed `n`. That spread is physical, not noise.

## Two-level Pop III sampler

For each sample:

1. Draw `lambda_III` and `n_III ~ Poisson(lambda_III)`.
2. Draw group counts `(n_1, ..., n_10) ~ Multinomial(n_III, w_g)`.
3. Sample `n_g` explicit rows from group `g` for every group.
4. Sum the rows into the sample's Pop III vector with `np.add.at`.

The group lifetime is fixed by the group mass, so the `Delta` gate is a deterministic
step: all `n_g` events pass or none do. There is no binomial thinning.

The explicit sum is exact. At `n = 31` the vectorised path matches a naive per-event
loop to machine precision, and a classifier trained to separate two independent
explicit draws gives AUC = 0.5 within the sampling tolerance.

## Cut order

```
R1  n_P + n_III == 0
R2  M_Zpre == 0 or Z* < z_cr or Z* > z_max
R4  [Fe/H] outside feh_range
R3  Fe or too few features below the relative floor
```

R4 is applied to the final `[Fe/H]`, not to `Z*`. In the current run it uses the
noiseless value. The intended version uses the observed value and importance weights;
that is not implemented.

## Current acceptance

From a 10,000-draw run:

| stage | count | fraction |
|---|---|---|
| R1 | 964 | 0.096 |
| R2 | 3572 | 0.357 |
| R4 | 1791 | 0.179 |
| R3 | 0 | 0.000 |
| accepted (before truncation) | 3673 | 0.367 |
| retained | 2000 | 0.200 |
| discarded | 1673 | — |

R2 reasons: `no_pre_enrichment = 1910`, `below_z_cr = 1662`,
`above_z_max = 0`.

The presence split is in the results document. In short, R1 removes no PISN samples
(by construction they have no PISN, so they need a Pop III event), while R2 removes
more PISN samples because a PISN pushes `Z*` up.
