# 14 — Every common element as a feature

The hand-written feature list is gone. The model now uses **every element that
all three yield tables synthesise above the validity floor**, and feature
selection is left to the classifier.

## What is in, and what is not

The three tables in use are HW2002 (PISN), the Ishigaki Pop III table, and
Limongi18 (Pop II). All three carry all 32 entries of `ELEMENTS`, so the
intersection is decided by the validity floor `M_X / M_Ztot >= 1e-10`, not by
element availability. That leaves 27 elements and therefore 26 features:

```
Al Ar C Ca Cl Co Cr Cu F Fe Ga Ge K Mg Mn N Na Ne Ni O P S Sc Si Ti V Zn
```

Four groups are excluded, each for a structural reason rather than a
performance one:

| excluded | reason |
|---|---|
| `H`, `He` | the reference reservoir, not enrichment products; `[H/Fe]` would encode the dilution, not the source mix |
| `B`, `Be` | always below the floor: 0% of mocks are valid, in all three tables |
| `Li` | the label species; the Li+ catalogue has no Li/Fe column, so the feature exists only in the mocks and cannot be measured at prediction time |

`Li` is the one debatable exclusion. It is a one-line change in
`DEFAULT_FEATURE_ELEMENTS` if a mock-only column is wanted, but it would train
the classifier on a quantity that is unavailable for the stars it is applied to.

## The pool

```
.venv/bin/python -m src.potluck generate --n 50000 --seed 42 \
    --output src/potluck/tests/v5/limongi18_50k_all_elements.npz --verbose
```

50,000 accepted at acceptance 0.370, R4 fail 22,984 (0.170), R2 fail 48,050
(0.356). The cuts are unchanged, so the acceptance rate is the same as the
Ti-off pool; only the stored feature matrix is wider (26 columns instead of 14).

## What moved in v7

`src/potluck/tests/v7/` was regenerated on the new pool.

| quantity | 14 features, Ti off | 26 features, all common | Δ |
|---|---|---|---|
| positive fraction (`f_Z^P >= 0.5`) | 0.2345 | **0.2385** | +0.004 |
| median PISN metal share | 0.0610 | 0.0631 | +0.002 |
| median Pop III metal share | 0.2687 | 0.2706 | +0.002 |
| median Pop II metal share | 0.1771 | 0.1733 | −0.004 |
| ρ(`f_Z^P`, [Na/Mg]) | −0.450 | −0.449 | — |
| ρ(`f_Z^P`, [Co/Ni]) | −0.567 | −0.570 | — |
| ρ(`f_Z^P`, [Mn/Cr]) | −0.010 | −0.013 | — |
| ρ(`f_Z^P`, [Zn/Mg]) | −0.598 | −0.603 | — |
| below table floor | 0.5667 | 0.5687 | +0.002 |

The shifts are sampling noise, not physics. Widening the feature matrix changes
how many random draws the observation layer consumes per accepted mock (Ti is
now an active observation element), so the accepted sample is not identical even
at the same seed. The enrichment model, the cuts, and every correlation are
unchanged.

## What moved in v14

`src/potluck/tests/v14/` now holds **27 contour panels**: the 26 features plus
the `Mg_Si.png` ratio panel. Bin counts run 388–835, all with median
`f_Z^P ≈ 0.063`, and `|rho([X/Fe], [Fe/H])| <= 0.17` for every panel
(the largest is Ge at +0.174, the smallest Mg at −0.096). The `Mg/Si` panel is
unchanged in construction and still gives −0.012.

## Consequences

1. **Ti is now trained on and enters the support gate.** Everything in
   [note 12](12-limongi-rerun.md) — support 82.1%, AUC 0.973, ESS 17.7% —
   was computed with Ti excluded and is stale. The Ti deficit (note 11) now
   costs support instead of being hidden, which is the point of the change.
2. **The classifier sees 26 features, most of which are `NaN` for real stars.**
   Only the 12 Li+ catalogue elements are measured observationally. The
   classifier must not lean on mock-only columns, so the real-star comparison is
   the arbiter, not the mock AUC.
3. **Feature selection is deferred, not skipped.** The natural next step is to
   train with all 26 and read the permutation/impurity importances, then keep the
   columns that matter for the real stars. That replaces the old hand exclusion
   with an evidence-based one.
