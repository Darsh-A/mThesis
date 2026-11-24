# Explanation figures

These eight figures explain the generator with **mock data only**. They never read
the Li+ catalogue, so they show what the model does rather than how well it fits
real stars. They are built for a talk and each one stands alone.

## How to run

```bash
python -m src.potluck.diagnostics.explanation_plots \
    src/potluck/tests/v5/limongi18_50k_ti_off.npz --outdir src/potluck/tests/v7
```

The script writes `fig1_priors.png` … `fig8_zstar.png` plus
`explanation_plots.json`, a small summary of the pool (mock count, positive
fraction, median source shares, and the ratio correlations). It uses the
*noiseless* `features`, `feh`, `f_zp`, and `matrix` arrays, never `obs_*`.

`Ti` is now one of the 26 features (see
[Every common element](14-all-common-elements.md)); the v7 figures below were
regenerated after that change. `Mn/Cr` is shown in figure 6 precisely because it
is *not* a PISN diagnostic.

The figure command needs the HW2002 PISN table, the Ishigaki Pop III table, and
the Limongi18 Pop II table. The first two are read from the repository data; the
Limongi18 table is built on first use and cached in `data/potluck/cache/`.

## The figures

| File | Message |
|---|---|
| `fig1_priors.png` | What the generator samples: PISN count, Pop III rate, dilution mass, Pop II burst mass, the two clocks, the final metallicity, and the PISN mass fraction. |
| `fig2_timeline.png` | The two clocks: an event contributes if its lifetime is shorter than `Delta` (pre-enrichment) or `Delta + t` (final cloud). Eight example histories and the source metal budget by `n_PISN`. |
| `fig3_metallicity.png` | Dilution sets the `[Fe/H]` scale, the Pop II burst mass raises it, and PISN events barely move it. |
| `fig4_yields.png` | The three yield sources have different fingerprints; the HW2002 PISN changes character above ~148 `M_sun` when the Fe-peak ejecta turn on. |
| `fig5_trends.png` | `[X/Fe]` versus `[Fe/H]`, coloured by `f_Z^P`: PISN-dominated mocks sit at low Na, Co, Zn and high Si, Ca. |
| `fig6_ratios.png` | Three Fe-independent ratios track `f_Z^P`; `[Mn/Cr]` does not. This is why the classifier can work at all. |
| `fig7_label.png` | The counterfactual label, its PISN driver, and how well two ratios separate the two classes. |
| `fig8_zstar.png` | The sampled `z_star` and `z_final` against the Pop II yield grid: the majority of the pool lies below the first Limongi18 node (`3.24e-5`). |

### Figure 1 — priors

Eight histograms with the median and 16–84% interval in each title. The sampled
quantities are `n_PISN` (0–3), `n_III` (Poisson), `M_dil`, `M_star,II`, `Delta`,
`t`, `f_Z^P`, and `Z_star`. The panel makes two points: the priors are broad and
log-uniform where the physics allows it, and the PISN fraction is **not** uniform
— only 23.5% of the 50k pool has `f_Z^P >= 0.5`.

### Figure 2 — clocks and bookkeeping

The top panel is a schematic, not data. Pop III forms at `t = 0`; a PISN and the
Pop III core-collapse events explode after their own lifetimes; the Pop II burst
forms at `Delta`; the observed star forms at `Delta + t`. The blue band is the gas
that can be pre-enriched (`tau <= Delta`), the orange band the final cloud
(`tau <= Delta + t`).

The bottom-left panel draws eight real histories from the pool with filled markers
for the sources that actually contribute. The bottom-right panel shows the mean
metal-mass share by `n_PISN`: PISN dominates the budget once it is present, but
the *median* share over the whole pool is only 6.1% because a PISN contributes
nothing when its progenitor is too long-lived to explode before the star forms.

### Figure 3 — the metallicity scale

`[Fe/H]` in the generator is exactly
`log10(M_Fe^ej / (M_dil * X_Fe,sun))`; the correlation with that ratio is 1.000 to
double precision (residual ~1e-7 dex). So the panels are not independent
discoveries, they are the definition:

- more dilution lowers `[Fe/H]`;
- a larger Pop II burst raises it;
- **PISN events barely move it**: the median `[Fe/H]` is −2.65, −2.64, −2.61,
  −2.61 for `n_PISN = 0, 1, 2, 3`. The reason is that `M_dil` varies by more than
  an order of magnitude while a PISN adds only a fraction of the Fe budget.
- The grey histogram shows where the adopted `[-3.8, -1.7]` window sits.

This is the figure to show when someone asks why `[Fe/H]` cannot be used to infer
the PISN contribution.

### Figure 4 — yields

The first three panels are fingerprints at fixed source parameters: PISN at 220
and 260 `M_sun`, four Pop III `(mass, energy)` combinations, and Pop II at four
metallicities with `t = 10 Myr`. Note the PISN pattern is *not* monotonic in mass.

The bottom-right panel is the reason the model is interesting. The HW2002 table
has a sharp threshold at ~148 `M_sun`: at 140 `M_sun` the ejecta is 65 `M_sun` of
mostly O, C, and Ne with essentially no Fe-peak elements; at 150 `M_sun` Si, Ca,
and the Fe peak appear. Above the threshold the Fe-peak yields plateau. A PISN
below the threshold therefore produces a very different abundance pattern from
one above it, and the model samples the full 140–270 `M_sun` range.

### Figure 5 — abundance trends

Six `[X/Fe]` panels versus `[Fe/H]`, with every mock coloured by `f_Z^P` and two
median curves split at `f_Z^P = 0.5`. The colour, not `[Fe/H]`, carries the
structure: at fixed `[Fe/H]` the red (PISN-dominated) curve is offset in Na, Co,
and Zn, and mildly in Si and Ca. The scatter is large because the mixture is
stochastic — one PISN event of a different mass moves the pattern — so the
separation is statistical, not one-to-one.

`[Fe/H]` is on the x-axis only because it is the observable axis of a stellar
catalogue. The model does not claim that `[Fe/H]` drives the ratios.

### Figure 6 — the Fe-independent ratios

Four ratios that do not use Fe, so they are immune to the dilution uncertainty
and to the Fe yield. Spearman rho with `f_Z^P`:

| ratio | rho |
|---|---|
| `[Na/Mg]` | −0.45 |
| `[Co/Ni]` | −0.57 |
| `[Zn/Mg]` | −0.60 |
| `[Mn/Cr]` | −0.01 |

`[Mn/Cr]` is flat, which is the useful null result: not every ratio is a PISN
diagnostic, and the classifier has to learn *which* ones matter. The three
informative ratios are the ones that separate PISN from Pop II ejecta (PISN is
Na-, Co-, and Zn-poor relative to Mg/Ni).

### Figure 7 — the label

The label is `y = 1[f_Z^P >= 0.5]`, a counterfactual: "would removing the PISN
contribution change the abundance pattern by more than the measurement noise?".
The panels show the label is 23.5% positive, that it rises with `n_PISN`, that
`[Na/Mg]` versus `[Co/Ni]` already separates the bulk of the two classes, and
that positive mocks are PISN-dominated in their metal budget.

The separation is good but not perfect: the grey and red clouds overlap. That
overlap is the irreducible part of the inverse problem and is the reason the
downstream classifier is probabilistic.

### Figure 8 — metallicity sampling

The left panel shows `z_star` (the pre-enrichment metallicity, `M_z_pre / M_dil`)
with the `z_cr = 1e-6` cut and the first Limongi18 node at `3.24e-5`. The right panel
shows the final metallicity after the Pop II burst. In the current pool
**56.7% of accepted mocks have `z_star < 3.24e-5`**, so for more than half of the
sample the Pop II table is clamped to its lowest node. Nothing is outside the configured
range; the figure makes the saturation explicit.

## What these figures do not show

- No measurement noise, missing elements, or upper limits. Those belong to the
  observation model in [Real-star validation](07-real-star-validation.md).
- No Li+ stars. The figures are deliberately blind to the catalogue so that the
  model structure is judged on its own.
- No Ti. Its offset is an intrinsic yield-table deficit, documented separately.
- No classifier performance. Figure 7 shows the geometric separability, not the
  AUC; the calibrated numbers are in
  [Real-star validation](07-real-star-validation.md).

## Numbers quoted above

All from the 50k Limongi18 Ti-off pool and the default configuration:

- positive fraction 0.2345;
- median shares of the metal mass: PISN 0.061, Pop III 0.269, Pop II 0.177
  (medians of three shares need not sum to one);
- the PISN threshold in HW2002 is at 148 `M_sun`.

The summary JSON is the authoritative record for a given run; the numbers in this
note are the values from the run above.
