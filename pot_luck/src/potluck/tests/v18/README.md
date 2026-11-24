# v18 — Reverse problem: what dilution mass do the yields and Li+ support?

The question: the `M_dil` window (`2e5–1e9 Msun`) is a hand-chosen prior. Can the
yield tables and the Li+ catalogue tell us which `M_dil` they actually support?
This test inverts the forward model instead of drawing `M_dil` from a prior.

The default was subsequently changed to the conservative supported band
`(1.0e5, 5.0e8)` in `src/potluck/config.py`; the core 16–84 range remains
`1e6–3e8`.

**Short answer.** Yes, but only weakly. The yields plus Li+ support

```
log10(M_dil / Msun) = 6.09 to 8.20  (16-84 median across viable configs)
                     (1.2e6 to 1.6e8 Msun, median 2.2e7)
```

with a conservative 5-95% span `9e4–6e8 Msun`. The event prior alone already
spans `6e5–1.2e8`, so the data mainly shift the distribution up by ~0.25 dex
rather than narrowing it. `M_dil` is degenerate with the absolute event amounts:
the catalogue constrains the ratio `M_Fe/M_dil` through `[Fe/H]`, not `M_dil`
alone.

Two robust conclusions survive that degeneracy:

1. **PISN-only is excluded.** Raw support is 2% and the required offsets reach
   2 dex (Zn -2.07, Na -1.63, Co -1.34). PISN alone does not make the catalogue
   pattern.
2. **Nomoto13 + CK20 edits is the most self-consistent Pop II table.** Raw
   support 83%, offset RMS 0.18 dex. Baseline Limongi18 is 66% / 0.34 dex;
   Nomoto13 without the hypernova edits is 31% / 0.35 dex.

The old window `2e5–1e9` contains the supported range but is wider than the
bulk, mostly at the low-mass end. `1e6–3e8` is the core working window;
`1e5–5e8` is the conservative band now used as the default.

---

## Data

| item | choice |
| --- | --- |
| Catalogue | Li+ very metal-poor stars (`load_li_records`) |
| Stars in catalogue | 385 |
| Stars used | 377 |
| Selection | finite `[Fe/H]` in `[-4.5, -1.5]`, ≥4 detected ratio elements |
| Ratio elements | C, Na, Mg, Si, Ca, Ti, Cr, Mn, Co, Ni, Zn |
| Fe | reference element, not fitted as a ratio |
| C upper limits | stored but excluded from the fit |

Eight stars are excluded: outside the `[Fe/H]` box or fewer than four detected
elements. Fe is available for all 385.

---

## Forward model being inverted

For one prior draw of the event set, the potluck forward model gives a source
matrix `M` and

```
M_X       = sum_s M_X^(s)                       total ejecta of X
M_Zpre    = sum_{X in metals} M_X^(PISN) + M_X^(PopIII, pre)
Z_star    = M_Zpre / M_dil
Z_final   = M_Ztot / M_dil
[Fe/H]    = log10( M_Fe / (M_dil * kappa_Fe) )
[X/Fe]    = log10[(M_X/kappa_X)/(M_Fe/kappa_Fe)]
```

`Z_star` enters the Pop II table, so `M_Fe` depends on `M_dil` through the Pop II
ejecta. See `docs/algorithm.md` for the full forward chain.

---

## Method: prior-predictive likelihood-weighted inversion

This is **not** a gradient MCMC. It is a simulation-based (ABC-type) inversion:
every sample is a prior draw, and the catalogue likelihood reweights it.

1. **Draw the event set, not M_dil.** Sample PISN counts/masses, Pop III counts
   and `(mass, energy)` rows, `Delta`, `t`, and `M_starII` from the potluck
   prior. `M_starII` is decoupled from `M_dil` (log-uniform `1e3–1e7 Msun`) so
   the study does not inherit the old `M_dil` cap.
2. **Put M_dil on a grid.** Use 160 log-spaced points from `1e3` to `1e11 Msun`,
   far wider than any prior window. For every grid point compute
   `Z_star = M_Zpre/M_dil`, the Pop II ejecta at `(Z_star, t)`, and the total
   ejecta vector.
3. **Solve M_dil per star.** For each Li+ star and each prior draw, find the
   `M_dil` where the model `[Fe/H]` equals the observed `[Fe/H]` (linear
   interpolation on the log grid). This uses the star's measured iron, so `M_dil`
   is an output, not a prior.
4. **Weight by the pattern.** Compute a chi-squared over the detected `[X/Fe]`
   ratios with the catalogue errors plus a 0.1 dex intrinsic floor.
   Structural zeros are capped at 3 dex instead of producing infinite weights.
5. **Absorb over/under-production.** One offset `delta_X` per element is shared
   by the configuration and estimated by weighted median residual, iterated
   three times. Positive `delta_X` means the model overproduces `X`; negative
   means it underproduces. Report `delta_X` and the weighted residual MAD.
6. **Aggregate.** Each star contributes equally. Report three Mdil
   distributions:
   * **event prior** — uniform weights over the prior draws (no data);
   * **raw** — likelihood weights with `delta_X = 0`;
   * **offsets fitted** — likelihood weights after step 5.

The raw distribution is the strict physical match. The offsets-fitted
distribution is the shape match after allowing a global yield-normalisation
error per element. The difference between them is the systematic budget.

**Why not an MCMC?** The inverted quantity `M_dil` is one-dimensional per star
and is solved exactly from `[Fe/H]`; the expensive part is the prior draw of the
event set, which an MCMC would have to re-sample anyway. Importance reweighting
of prior draws gives the same posterior with no arbitrary `M_dil` prior and no
chain convergence issues. The remaining prior dependence (`M_starII`, `t`,
event amounts) is reported explicitly through the "event prior" baseline.

### Support definition

A star is supported if its best chi-squared per detected element is below 4
(`chi2_nu < 4`). This is a pattern-shape test. It does **not** mean the absolute
abundances are right; that is what the raw support and the offsets measure.

---

## Configurations

26 configurations, 1200 prior draws each, seed 42. "Separately" means the
source class is switched off; "variants" means one table family is permuted
while the others stay at default.

### Source combinations

| configuration | PISN | Pop III | Pop II |
| --- | :-: | :-: | :-: |
| `all` | on | on | on |
| `pisn_only` | on | off | off |
| `popiii_only` | off | on | off |
| `pisn+popiii` | on | on | off |
| `pisn+popii` | on | off | on |
| `popiii+popii` | off | on | on |

Pop II-only is not defined in this model: without a pre-enriching source,
`M_Zpre = 0` and `Z_star` has no meaning.

### Yield-table permutations

| family | variants |
| --- | --- |
| PISN | HW2002, Takahashi-MR, Takahashi-NM, Takahashi-NR |
| Pop III CCSN (Ishigaki) | uniform groups, row sampling, low-mass, high-mass, high-energy, low-energy, M=11-15, M=25, M=40, M=100 |
| Pop II CCSN | Limongi18 v=0, v=150, v=300, Nomoto13, Nomoto13_CK20 (hypernova + 2D-jet offsets), WW95 |

`Limongi18_V150` and `Limongi18_V300` are new sources added in
`src/yield_sources.py`; they use the same LC18 table at the tabulated rotation
velocities.

---

## Results

### 1. Consensus dilution range

Median of the 20 "viable" configurations (raw support ≥50%, offset RMS ≤0.35
dex). Values are `log10(M_dil/Msun)`.

| distribution | p05 | p16 | p50 | p84 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| event prior | 4.38 | 5.79 | 7.09 | 8.08 | 8.66 |
| raw likelihood | 3.00 | 3.09 | 5.29 | 7.04 | 8.04 |
| **offsets fitted** | **4.96** | **6.09** | **7.34** | **8.20** | **8.75** |
| linear (offsets fitted) | 9e4 | 1.2e6 | 2.2e7 | 1.6e8 | 5.6e8 |

The prior range `2e5–1e9` corresponds to `5.30–9.00`. It brackets the supported
range, but the supported bulk is about 0.8 dex narrower at the low end and
0.8 dex narrower at the high end.

**The raw row is deliberately biased low.** It is what happens if the
element-by-element yield normalisation is taken literally: the fit drives
`M_dil` down to 1e5 to compensate for underproduced elements. The offsets row is
the recommended inversion; the raw row is the control.

### 2. Source combinations

`M_dil` is `p16–p84` with median; `raw support` is `chi2_nu<4` with no offsets.

| configuration | log p16 | log p50 | log p84 | raw support | offset RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 6.31 | 7.36 | 8.26 | 0.66 | 0.34 |
| `pisn_only` | 6.36 | 6.83 | 7.43 | **0.02** | **0.98** |
| `popiii_only` | 4.51 | 5.53 | 6.39 | 0.50 | 0.29 |
| `pisn+popiii` | 5.04 | 5.97 | 6.73 | 0.33 | 0.33 |
| `pisn+popii` | 5.58 | 7.07 | 8.18 | 0.34 | 0.37 |
| `popiii+popii` | 5.63 | 7.03 | 8.11 | 0.76 | 0.31 |

Reading: Pop III-only supports a much smaller cloud (`~3e5 Msun`) than the full
mix (`~2e7 Msun`). Pop II with either pre-enricher gives `~1e7 Msun`. PISN alone
can be forced to a `M_dil` but cannot produce the catalogue pattern.

### 3. Yield-table permutations

#### PISN table

| table | log p16 | log p50 | log p84 | raw support | offset RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| HW2002 | 6.34 | 7.34 | 8.19 | 0.59 | 0.33 |
| Takahashi-MR | 6.00 | 7.47 | 8.36 | 0.54 | 0.35 |
| Takahashi-NM | 5.93 | 7.37 | 8.29 | **0.75** | 0.32 |
| Takahashi-NR | 6.21 | 7.47 | 8.31 | 0.57 | 0.34 |

The PISN table changes the median by <0.15 dex. Takahashi-NM is the easiest to
match; HW2002 is typical.

#### Pop III CCSN sampling

| variant | log p16 | log p50 | log p84 | raw support | offset RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 6.35 | 7.47 | 8.34 | 0.57 | 0.34 |
| row | 6.30 | 7.33 | 8.20 | 0.61 | 0.33 |
| low_mass | 5.96 | 7.04 | 7.97 | 0.70 | 0.31 |
| high_mass | 6.28 | 7.52 | 8.39 | 0.54 | 0.35 |
| high_energy | 6.56 | 7.61 | 8.47 | 0.58 | 0.33 |
| low_energy | 6.01 | 7.31 | 8.29 | 0.63 | 0.36 |
| M11-15 | 5.55 | 6.81 | 8.01 | 0.54 | 0.32 |
| M25 | 5.83 | 6.89 | 7.80 | 0.70 | 0.30 |
| M40 | 6.19 | 7.30 | 8.16 | 0.66 | 0.31 |
| M100 | 6.41 | 7.63 | 8.49 | 0.48 | 0.36 |

Direction: low-mass and M25 Pop III progenitors push `M_dil` down to
`~1e7 Msun`; M100 and high-energy push it up to `~4e7 Msun`. More Fe per Pop III
event needs a larger cloud to reach the same `[Fe/H]`.

#### Pop II table

| table | log p16 | log p50 | log p84 | raw support | offset RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Limongi18 v=0 | 6.20 | 7.42 | 8.28 | 0.66 | 0.34 |
| Limongi18 v=150 | 5.85 | 7.16 | 8.10 | 0.66 | 0.31 |
| Limongi18 v=300 | 6.13 | 7.16 | 8.05 | 0.62 | 0.32 |
| Nomoto13 | 6.30 | 7.49 | 8.36 | 0.31 | 0.35 |
| **Nomoto13_CK20** | 6.05 | 7.50 | 8.50 | **0.83** | **0.18** |
| WW95 | 5.89 | 6.88 | 7.58 | 0.58 | 0.29 |

Rotation lowers the median by ~0.25 dex and slightly improves the raw match.
The CK20 edits are the dominant improvement: raw support rises from 0.31 to
0.83 and the offset RMS halves.

### 4. Over- and under-production

`delta_X = model - Li+` (dex), `all` configuration versus `Nomoto13_CK20`.

| element | `all` | CK20 | reading |
| --- | ---: | ---: | --- |
| C | +0.15 | -0.06 | near solar scale |
| Na | +0.01 | -0.02 | good |
| Mg | -0.12 | +0.13 | good |
| Si | -0.02 | +0.14 | good |
| Ca | +0.02 | +0.01 | good |
| Ti | **-0.73** | **-0.18** | baseline underproduces Ti |
| Cr | +0.10 | **+0.36** | CK20 overproduces Cr |
| Mn | **-0.29** | -0.11 | baseline underproduces Mn |
| Co | **-0.62** | **-0.25** | baseline underproduces Co |
| Ni | +0.09 | -0.27 | CK20 slightly low |
| Zn | **-0.43** | +0.10 | baseline underproduces Zn |

The baseline underproduces the same elements that motivated the CK20 edits
(Ti, Co, Zn), and CK20 largely removes them. The cost is Cr (+0.36) and Ni
(-0.27). PISN-only is far worse: Zn -2.07, Na -1.63, Co -1.34, Na -1.63.

### 5. Catalogue support

* 377/385 Li+ stars are used.
* With offsets fitted, 93-98% of stars pass `chi2_nu<4` for most configurations.
* Without offsets, support spans 0.02 (PISN-only) to 0.83 (CK20).
* The full pool's corrected support is 0.95; its raw support is 0.66.

The high corrected support is expected: eleven global offsets absorb the mean
normalisation error. It should not be quoted as "the yields explain the
catalogue". Raw support and offset RMS are the discriminating metrics.

---

## Figures

| file | content |
| --- | --- |
| `mdil_support_by_config.png` | forest: event prior (grey), raw (open), offsets fitted (filled), all 26 configs |
| `mdil_source_combinations.png` | weighted Mdil densities for the six source combinations |
| `mdil_pisn_variants.png` | forest for the four PISN tables |
| `mdil_popiii_variants.png` | forest for the ten Pop III sampling variants |
| `mdil_popii_variants.png` | forest for the six Pop II tables |
| `element_offsets.png` | heatmaps of `delta_X` and residual MAD, config × element |
| `configuration_support.png` | median `chi2_nu` and support fraction, raw vs offsets |
| `mdil_raw_vs_corrected.png` | median Mdil without versus with offsets |
| `star_mdil_vs_feh.png` | per-star Mdil (16-84) versus [Fe/H] for `all` |
| `mdil_event_dependence.png` | Mdil posterior versus PISN count and Pop III count |

All axes and legends are labelled; no explanatory text is placed inside the
figures.

---

## Files

| file | content |
| --- | --- |
| `mdil_inverse.json` | full results: selection, settings, per-config summaries, offsets, event medians |
| `config_summary.csv` | one row per configuration: prior/raw/corrected Mdil quantiles, support, offset RMS |
| `element_offsets.csv` | `delta_X` and residual scatter per configuration and element |
| `star_mdil.csv` | per-star Mdil 16/50/84 per configuration |
| `run.log` | verbose generation log |
| `*.png` | figures above |

---

## Regenerate

```bash
python -m src.potluck.diagnostics.mdil_inverse \
    --n 1200 --seed 42 --outdir src/potluck/tests/v18 --verbose
```

Runtime is a few minutes. The routine is in
`src/potluck/diagnostics/mdil_inverse.py`.

---

## Caveats

1. **Offsets are global per element, not per star or per metallicity.** They
   absorb a systematic yield-normalisation error. Large offsets mean the
   corrected support is not physical; use the raw support for viability.
2. **`M_dil` is solved exactly from the measured `[Fe/H]`.** The `[Fe/H]` error
   is not propagated; at 0.1 dex it is small compared with the ~2 dex scenario
   spread, but it is not zero.
3. **`M_starII` is sampled independently of `M_dil`.** This was deliberate, to
   avoid inheriting the old window, but it means the event prior still sets the
   width of the result. The "event prior" row quantifies that dependence.
4. **No SNIa, Li+ only.** The catalogue is the Li+ sample; 8 stars are excluded.
   C upper limits are not used in the fit.
5. **The answer is a supported range, not a measurement.** The degeneracy
   `M_dil <-> M_Zpre` is physical: the ratio `M_Fe/M_dil` is what `[Fe/H]`
   constrains. Quoting a single `M_dil` requires fixing the event amounts.
6. **Ti remains a known problem.** Even CK20 leaves Ti at -0.18 dex and Cr at
   +0.36 dex.
7. **The absolute Fe yield is degenerate with M_dil.** A global Fe offset moves
   the inferred `M_dil` directly (it is solved from `[Fe/H]`), not the `[X/Fe]`
   offsets. The result is therefore conditional on the Fe normalisation of each
   table, including the WW95 `Fe x 0.5` scaling and the CK20 edits. This is the
   main reason different source combinations return different `M_dil`.
