# Algorithm, step by step

This is the full pipeline implemented in `src/potluck/`. Each section names the
module that does the work.

## 0. Element bookkeeping

`constants.py` defines the common element vector:

```
Al Ar B Be C Ca Cl Co Cr Cu F Fe Ga Ge H He K Li Mg Mn N Na Ne Ni O P S Sc Si Ti V Zn
```

Every yield table is folded onto these 32 elements. Isotopes are summed by
`utils._combine_elements`, and radioactive decay is applied by the table adapter
before this point.

The solar mass fraction of element `X` is

```
X_sun,X = 10^(A_X - 12) * m_X / m_H * X_H_sun
```

with `X_H_sun = 0.7381`. The log abundance and abundance ratio are

```
a_X = log10(M_X / X_sun,X)
[X/Y] = a_X - a_Y
```

The default feature set is every element that all three yield tables
synthesise above the floor below — 27 elements:

```
Al Ar C Ca Cl Co Cr Cu F Fe Ga Ge K Mg Mn N Na Ne Ni O P S Sc Si Ti V Zn
```

and the default ratios are the remaining 26 against Fe:

```
Al/Fe Ar/Fe C/Fe Ca/Fe Cl/Fe Co/Fe Cr/Fe Cu/Fe F/Fe Ga/Fe Ge/Fe K/Fe
Mg/Fe Mn/Fe N/Fe Na/Fe Ne/Fe Ni/Fe O/Fe P/Fe S/Fe Sc/Fe Si/Fe Ti/Fe
V/Fe Zn/Fe
```

Nothing is dropped by hand. `H` and `He` are the reference reservoir rather than
enrichment products, `B` and `Be` never clear the floor, and `Li` is the label
species with no Li+ catalogue column. Feature selection is left to the classifier,
which can rank or drop columns by importance.

An element is usable only if

```
M_X / M_Ztot >= min_element_fraction = 1e-10
```

Otherwise its ratio is `NaN` and the mask is false. This is deliberate: real yield
tables have structural zeros and near-zeros, and a fake floor would create an
artificial abundance. Fe has its own floor, `M_Fe / M_Ztot >= 1e-10`.

## 1. Draw the enrichment history

`events.py` draws one `Events` object per sample.

### PISN

The number of PISN is categorical:

```
n_P ~ Categorical(0.5, 0.3, 0.15, 0.05)
```

Each mass is uniform in the intersection of the prior range and the table range:

```
m_P ~ Uniform(140, 270) M_sun
```

The HW2002 table gives a 32-element ejecta vector for each mass. The median metal
mass is about 96 M_sun, and PISN ejecta are almost entirely metals.

### Pop III CCSN

The rate is log-uniform:

```
lambda_III ~ LogUniform(0.1, 1000)
n_III ~ Poisson(lambda_III)
```

The Ishigaki grid has 10 `(mass, energy)` groups. The group counts are

```
(n_1, ..., n_10) ~ Multinomial(n_III, w_g)
```

The default weights are uniform over groups. Each group has a fixed progenitor mass,
so

```
M*III = sum_g n_g m_g
```

is exact. Every event is an explicit row from the table, including groups with large
`n_g`. The group counts are expanded into a flat list of row indices, and the ejecta
are summed with a vectorised `np.add.at` over the 10 groups. There is no Gaussian
shortcut: the explicit sum preserves the structural zeros and the heavy tails of the
Ishigaki grid.

The lifetime gate is deterministic per group, because every event in a group has the
same progenitor mass:

```
pre_g   = 1[tau_g <= Delta]
final_g = 1[tau_g <= Delta + t]
```

There is no binomial thinning: the gate either passes all events in the group or
none.

At the mean `M_Z ≈ 3.54 M_sun` per Pop III event, reaching `f_Z = 0.5` against a
median PISN needs about 27 events. The median per-event mass is only 0.34 M_sun, but
the sum of `n` events has expectation `3.54 n`; the median is not the right scale for
a sum.

### Pop II

The Pop II stellar mass is log-uniform:

```
M*II ~ LogUniform(1e3, min(1e7, 0.1 M_dil - M*III))
```

The joint cap is

```
M*III + M*II <= 0.1 M_dil
```

This is a cap on the total stellar budget, not just on `M*II`. It matters when
`n_III` is large: 1000 events at tens of M☉ are 1e4–1e5 M☉, comparable to a small
cloud.

### Cloud and clocks

```
Delta ~ LogUniform(1, 100) Myr
t     ~ LogUniform(1, 100) Myr
M_dil ~ LogUniform(2e5, 1e9) M_sun
```

The `M_dil` floor comes from the paper's dilution derivation: a cloud must be massive
enough to host the PISN progenitors and to have `f_dil > 0.02`. The ceiling comes from
`f_dil <= 10`. The exact values are a prior choice and are recorded in `config.py`.

## 2. Chronology

`chronology.py` supplies the lifetimes.

Pop III uses the Schaerer (2002) zero-metallicity table:

| mass (M☉) | 500 | 200 | 120 | 80 | 60 | 40 | 25 | 15 | 9 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| lifetime (Myr) | 1.90 | 2.20 | 2.52 | 3.01 | 3.46 | 3.86 | 6.46 | 10.4 | 20.2 | 61.9 |

The code interpolates log lifetime against log mass and clips outside the table.

Pop II uses the Raiteri relation at `Z*`, with a floor `Z >= 7e-5` because the
relation is not reliable at zero metallicity. The turnoff mass `m_TO(t, Z*)` is the
mass whose lifetime equals `t`. The IMF integral runs from
`max(m_TO, table_min_mass)` to the table maximum. Stars below the table's lowest mass
are treated as having no CCSN yield; they are not assigned the lowest-mass yield.

## 3. Ejecta matrix

`ejecta.py` builds a `(n_samples, 4, 32)` matrix. The four rows are PISN, Pop III,
Pop II, and SNIa. The SNIa row is zero for now.

For each sample:

```
M_X      = sum_s M_X^(s)
M_pre    = PISN_pre + III_pre
M_Ztot   = sum_{X in metals} M_X
M_Zpre   = sum_{X in metals} M_pre,X
```

The pre-enrichment is what Pop II sees. The final cloud also contains the Pop II
ejecta and any PISN or Pop III event that arrived after `Delta` but before
`Delta + t`.

The metallicities are derived, not drawn:

```
Z*      = M_Zpre / M_dil
Z_final = Z* * M_Ztot / M_Zpre
[Fe/H]  = log10(M_Fe / (M_dil * X_sun,Fe))
```

`Z_final >= Z*` because the final metals include the pre-enrichment. `[Fe/H]` is
stored for diagnostics and for R4, but it is not a feature: it is collinear with the
overall scale.

## 4. Cuts

The cuts are applied in a fixed order, and the ledger records each stage.

### R1: nothing to enrich the cloud

```
reject if n_P + n_III == 0
```

This happens before any yield is computed.

### R2: no pre-enrichment or metallicity out of range

```
reject if M_Zpre == 0
reject if Z* < z_cr = 1e-6
reject if Z* > z_max        (currently z_max = None)
```

`z_cr = 1e-6` is both the Pop II table floor and about `1e-4.1 Z_sun`. It prevents
the Pop II yields from being evaluated far below the table range.

### R4: final iron abundance

```
reject if [Fe/H] < -3.8 or [Fe/H] > -1.7
```

The box is chosen to match the Li+ sample. R4 is applied to the **observed**
`[Fe/H]` after the Li+ observation layer, and the survivors are importance-weighted
to the Li+ `[Fe/H]` histogram within giants and dwarfs.

### R3: feature validity

```
reject if M_Fe / M_Ztot < min_fe_fraction
reject if n_valid_features < min_valid_features = 3
```

Elements below their relative floor are masked, not floored. The sample survives if
Fe and at least three feature elements are valid.

The stage codes are `1 = R1`, `2 = R2`, `4 = R4`, `3 = R3`, `0 = accepted`. R4 is
applied before R3 so that the ledger separates the abundance-box rejection from the
feature-validity rejection.

## 5. Features and labels

`features.py` computes

```
a_X = log10(M_X / X_sun,X)
x_X = a_X - a_Fe
```

and the labels

```
f_Z^P  = sum_{X in metals} M_X^P / M_Ztot
f_Fe^P = M_Fe^P / M_Fe
f_X^P  = M_X^P / M_X
```

`f_Z^P` is the metal-mass fraction made by the PISN. `f_Fe^P` is the iron fraction;
it is small for light PISN even when `f_Z^P` is large. `f_X^P` is stored per element
for diagnostics.

The feature matrix contains the 14 ratios. Invalid ratios are `NaN` and the mask is
stored beside them.

## 6. Detection label

`detect.py` asks a counterfactual question: if the PISN were removed, would the
observable pattern change by more than the noise?

```
M_X^(-P) = M_X - M_X^P
x_X^(-P) = a_X^(-P) - a_Fe^(-P)
Delta_X  = x_X - x_X^(-P)
C        = diag(sigma_X^2) + sigma_Fe^2 * 11^T
D^2      = Delta^T C^{-1} Delta
y        = 1[D > k]
```

The covariance has a shared Fe term because every ratio uses the same reference
element. The placeholder noise is `sigma_X = 0.2` dex, `sigma_Fe = 0.1` dex, and
`k = 3`. These are not the final observational model.

Edge cases:

- No PISN: `y = 0`, `D = 0`.
- Fewer than `min_detected = 4` usable ratios: `y = 0`.
- Removing the PISN leaves an element at zero: `D = inf`, `y = 1`. This is recorded
  as `baseline_zero`; it is physical but inflates positives.

## 7. Dataset

`dataset.py` runs chunked rejection sampling. It stops when the requested number of
accepted samples is reached, or when `n_samples * max_draw_factor` draws have been
made. If a batch overshoots the target, the extra accepted samples are counted as
`discarded`, so the ledger still sums to the number of draws.

`io.py` writes a compressed `.npz` plus a `.meta.json` with the config hash, git
commit, element and feature names, cut definitions, ledger, and seed.

Stored arrays include:

- noiseless features and their mask;
- log abundances for all 32 elements;
- `f_Z^P`, `f_Fe^P`, `f_X^P`, `y`, `D`, `D2`, detection reason;
- the 4×32 ejecta matrix and its totals;
- `Z*`, `Z_final`, `[Fe/H]`, `Delta`, `t`, `M_dil`, `M*II`;
- `n_P`, `n_III`, PISN masses, `lambda_III`, `M*III`, group counts;
- the log prior.

## 8. Diagnostics

`diagnostics/` contains the plots and the noiseless ceiling.

- `ledger.py`: cut fractions, including the presence split.
- `priors.py`: label distributions, class balance, mixture cells, feature coverage.
- `plots.py`: 11 figures covering labels, features, clock, dilution, event masses,
  detection, and the limit sweep.
- `ceiling.py`: HistGradientBoosting classifier and regressor on features plus mask,
  compared with a nuisance-only model using only `Z*`, `t`, `M_dil`, `M*II`, and
  `Delta`. Also reliability/Brier, stratified AUC and RMSE, noise sweep.
- `validation.py`: Eq. 13 cross-check, dataset consistency checks, ablation ladder,
  and train-on-base/test-on-variant transfer.

The noiseless ceiling is not a performance guarantee. It is the best a model can do
with chemistry alone under the current prior. The nuisance comparison tests whether
the label is actually chemical or mostly prior.
