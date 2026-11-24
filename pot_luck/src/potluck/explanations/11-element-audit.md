# 11 — Per-element audit: mocks, yields, and Li+ stars

`src/potluck/diagnostics/element_audit.py` compares, element by element:

1. the noiseless mock distribution (50k Limongi18 Ti-off pool),
2. the pure-source yield fingerprints (PISN at 220 `M_sun`, mean Pop III ejecta,
   Pop II integrated at `Z = 3.236e-3`, `t = 10 Myr`),
3. the observed Li+ distribution, where the element is measured.

Outputs: `src/potluck/tests/v10/element_audit.png` and
`src/potluck/tests/v10/element_audit.json`.

## Table

Mock and Li+ values are medians; brackets are the 16-84% range for the mock.
`d` is median(mock) - median(Li+).

| el | mock [16-84] | Li+ [16-84] | d | PISN | Pop III | Pop II |
|---|---|---|---|---|---|---|
| C | -0.11 [-0.50, 0.41] | +0.30 [-0.27, 1.14] | **-0.41** | -0.67 | -0.39 | +0.61 |
| N | -2.44 [-3.10, -2.10] | — | — | -5.17 | -2.25 | -0.01 |
| O | +0.42 [0.13, 0.66] | — | — | -0.01 | +0.35 | +0.79 |
| Na | -0.72 [-1.24, -0.08] | -0.15 [-0.33, 0.10] | **-0.57** | -1.91 | -1.05 | +0.45 |
| Mg | +0.14 [-0.09, 0.36] | +0.37 [0.22, 0.49] | -0.23 | -0.08 | +0.09 | +0.36 |
| Al | -0.50 [-0.92, -0.24] | — | — | -1.40 | -0.56 | +0.30 |
| Si | +0.42 [0.29, 0.70] | +0.49 [0.33, 0.64] | -0.07 | +0.69 | +0.30 | +0.59 |
| Ca | +0.26 [0.12, 0.46] | +0.33 [0.23, 0.43] | -0.07 | +0.46 | +0.13 | +0.11 |
| Ti | -0.28 [-0.40, -0.17] | +0.45 [0.30, 0.57] | **-0.73** | -0.30 | -0.20 | -0.23 |
| Cr | -0.01 [-0.14, 0.18] | -0.20 [-0.34, -0.07] | **+0.19** | +0.35 | -0.00 | -0.18 |
| Mn | -0.67 [-0.73, -0.44] | -0.38 [-0.49, -0.20] | -0.29 | -0.28 | -0.71 | -0.44 |
| Co | -0.64 [-0.93, -0.36] | +0.22 [0.08, 0.37] | **-0.86** | -1.68 | -0.66 | -0.16 |
| Ni | -0.14 [-0.30, 0.15] | +0.01 [-0.08, 0.11] | -0.15 | -0.74 | -0.18 | +0.20 |
| Zn | -0.26 [-1.05, 0.08] | +0.25 [0.13, 0.40] | **-0.51** | -2.76 | +0.08 | -0.14 |

Mean absolute offset over the 11 measured elements: **0.37 dex**.

## What is consistent

For **every** element the mock median lies between the lowest and the highest
pure-source fingerprint. That is the expected arithmetic: the mock distribution
is a mixture of the three tables, so its median must be bracketed by them. This
is a direct check that the ejecta are summed and converted correctly, with no
missing term.

## What is not consistent

The mock is **below** the Li+ median for ten of eleven elements (Cr is the only
exception). The four largest offsets are Co (-0.86), Ti (-0.73), Na (-0.57) and
Zn (-0.51), followed by C (-0.41).

- **Ti** is the known yield-table deficit; it is excluded from training and
  support, so the offset does not affect the classifier, but it stays visible
  here for completeness.
- **Co** is the largest remaining problem. Limongi18 predicts Co/Fe ~-0.64 for
  the mixture while the catalogue sits at +0.22. Within the tables, only the
  most massive PISN reach Co/Fe ~-0.9; Pop II stays at -0.16.
- **Zn** is similar in sign but smaller: the Pop III fingerprint (+0.08) is
  close to the data (+0.25), while PISN drags the mixture down (-2.76 at
  220 `M_sun`).
- **C** is the one case where the observation model matters: the catalogue
  median is inflated by giants, and this audit uses raw Li+ values with no
  giant correction.
- **Na** is low because Limongi18 Pop II gives Na/Fe ~+0.45 while the data sit
  at -0.15; the mock mixture lands at -0.72.

## Caveats

1. The Li+ medians are not selection-corrected; the catalogue is a targeted EMP
   sample and no importance weights are applied here.
2. The mocks are noiseless, so their 16-84% ranges are much wider than the
   observed ones. The audit compares centres, not widths.
3. N, O and Al have no Li+ measurement, so they are shown for the yield check
   only.
4. Ti is excluded from the model's feature set but is plotted because it is a
   useful diagnostic.

## Consequence

The audit does not justify retuning anything. It identifies Co, Zn, Na and C as
the elements where the yield tables and the catalogue disagree most, and it
confirms that the mock mixture is arithmetically consistent with the tables.
Any change should come from a different yield source or from the observation
model, not from adjusting priors to close the offsets.
