# Features, labels, and detection

## From ejecta to abundances

For element `X` with total ejecta mass `M_X` (M☉), the log abundance relative to the
Sun is

```
a_X = log10(M_X / X_sun,X)
```

where `X_sun,X` is the solar mass fraction. The abundance ratio is

```
[X/Y] = a_X - a_Y
```

The default reference is Fe, and every element common to the three yield tables
above the validity floor becomes a feature, so the default set is

```
Al/Fe Ar/Fe C/Fe Ca/Fe Cl/Fe Co/Fe Cr/Fe Cu/Fe F/Fe Ga/Fe Ge/Fe K/Fe
Mg/Fe Mn/Fe N/Fe Na/Fe Ne/Fe Ni/Fe O/Fe P/Fe S/Fe Sc/Fe Si/Fe Ti/Fe
V/Fe Zn/Fe
```

`H`, `He`, `B`, `Be` and `Li` are the only exclusions, for the reasons given in
[the algorithm note](02-algorithm.md).

Keeping `a_X` as an intermediate means the observation model can add noise to each
element before forming ratios. Noise on `X` and on the reference then produces the
correct correlation.

## Validity masks

A feature element is valid if

```
M_X / M_Ztot >= min_element_fraction = 1e-10
```

Fe is valid if

```
M_Fe / M_Ztot >= min_fe_fraction = 1e-10
```

Invalid ratios are stored as `NaN` and the mask is stored separately. The mask is not
a leak by itself: a mask-only classifier has AUC ≈ 0.536. The mask matters because
real catalogues will not have the same availability pattern, and the observation
model must reproduce it.

A sample is accepted only if Fe is valid and at least

```
min_valid_features = 3
```

feature elements are valid.

## Regression labels

For every accepted sample:

```
f_Z^P  = sum_{X in metals} M_X^P / M_Ztot
f_Fe^P = M_Fe^P / M_Fe
f_X^P  = M_X^P / M_X
```

- `f_Z^P` is the fraction of the cloud's metal mass made by the PISN. It is the
  quantity most comparable to Salvadori's `f_PISN`, but it is oxygen-dominated.
- `f_Fe^P` is the fraction of iron made by the PISN. It is small for light PISN even
  when `f_Z^P` is large, because a light PISN makes very little Fe.
- `f_X^P` is a per-element diagnostic. It is not a training target by default.

None of these is a one-to-one function of the abundance pattern. Different mixtures
can produce the same ratios with different `f`. That irreducible spread is the honest
posterior width; it is not a model failure.

`[Fe/H]` is stored but is not a feature. It is derived from `M_Fe` and `M_dil`, and it
is strongly collinear with the overall metal scale.

## Detection label

The label is not "PISN present". It is "would removing the PISN change the observable
pattern by more than the noise?"

Remove the PISN row:

```
M_X^(-P) = M_X - M_X^P
```

Recompute the log abundances and ratios without the PISN:

```
a_X^(-P) = log10(M_X^(-P) / X_sun,X)
x_X^(-P) = a_X^(-P) - a_Fe^(-P)
```

The standardised shift is

```
Delta_X = x_X - x_X^(-P)
```

with covariance

```
C = diag(sigma_X^2) + sigma_Fe^2 * 11^T
```

The shared Fe term is the correlation induced by using the same reference element in
every ratio. Then

```
D^2 = Delta^T C^{-1} Delta
D   = sqrt(D^2)
y   = 1[D > k]
```

The placeholder values are `sigma_X = 0.2` dex, `sigma_Fe = 0.1` dex, and `k = 3`.

### Edge cases

| case | result | reason code |
|---|---|---|
| sample not accepted | `y=0`, `D=NaN` | `undefined` |
| no PISN contribution | `y=0`, `D=0` | `no_pisn` |
| fewer than 4 usable ratios | `y=0`, `D=0` | `insufficient` |
| removing PISN leaves an element at zero | `y=1`, `D=inf` | `baseline_zero` |
| normal | `y=1[D>k]` | `normal` |

`baseline_zero` is physical but can inflate positives. It occurred 53 times in the
latest 2000-sample run. The intended fix is to treat those elements as missing from
`D` rather than assigning `D=inf`, or to record them separately and exclude them from
the main positive class.

## Why remove the PISN, not just check `f_Z^P`

A PISN with a high `f_Z^P` can still produce a pattern that other sources mimic. The
counterfactual asks whether the PISN is *observable*, not whether it is present. That
is the label the classifier needs for class balance. It also handles the Fe problem: a
tiny `f_Fe^P` can still produce a large shift in `[X/Fe]` if the PISN dominates `X`.

The caveat is that the counterfactual does not measure degeneracy with other
mixtures. That is the classifier's job, and the diagnostics measure it.

## Feature set caveats

The current 14 ratios are all Fe-normalised. This is a problem for light PISN, which
make almost no Fe: every `[X/Fe]` is then set by the contaminants' Fe, so the ratios
do not encode the PISN share. The 140–160 M☉ bin shows this: its `f_Z^P` R² is 0.844
and RMSE 0.133, worse than the 0.06–0.08 of heavier bins.

The Fe-independent ratios are now included:

```
[Na/Mg]  [Co/Ni]  [Mn/Cr]  [Zn/Mg]
```

These are odd-even diagnostics and do not divide by Fe. Li+ supports Na, Co, Ni, Mn,
Cr, and Zn (Zn in 44% of stars); it lacks N, O, Al, and Cu. Xiang has poor Na/Mg
coverage and no Zn, so it is a stress test rather than the primary search sample.

## Current limitations

- Non-C non-detections use a hard lower-envelope threshold, not a probabilistic
  detection model.
- Only C has catalogue upper limits; Si/Mn/Co/Zn non-detections are forced missing.
- `[Fe/H]` is observed for R4 and weighted to Li+, but the weights are not yet
  propagated into every downstream metric.
- N, O, Al, and Cu are absent from Li+, so their value cannot be measured on the
  primary catalogue.
