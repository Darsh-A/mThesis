# 12 — Per-element contours coloured by the PISN share

`src/potluck/diagnostics/element_contours.py` draws one figure per feature
element. In each figure the 50k Limongi18 mocks are binned in the
(`[Fe/H]`, `[X/Fe]`) plane and every bin is coloured by its **median `f_Z^P`**,
the fraction of the metal mass that comes from PISN ejecta. The Li+ catalogue
is overlaid as white circles with its `[Fe/H]` and `[X/Fe]` 1σ error bars;
upper limits are white triangles.

Outputs: `src/potluck/tests/v14/` — 14 element PNGs (`C_Fe.png` … `Zn_Fe.png`),
one ratio PNG (`Mg_Si.png`), plus `element_contours.json`.

## How to read a panel

- **Colour** — 0 (dark) means the metals in that bin come from Pop II / Pop III
  only; 1 (yellow) means essentially all of them come from PISN.
- **Grey lines** — mock density contours at 5%, 20% and 50% of the peak bin.
- **White circles** — Li+ detections, with catalogue errors. **Triangles** —
  Li+ upper limits.
- Bins with fewer than 12 mocks are blank, so the coloured area is the region
  where the model actually predicts something.

Axes cover the central 99% of the mocks, widened to include the Li+ range. The
extreme PISN-only tail is visible where it is dense enough to fill a bin (for
example `N/Fe ≈ -4`), and blank where it is not.

## What the panels show

The colour gradient is the physics: as `[Fe/H]` rises at fixed `[X/Fe]`, the
Pop II share grows and `f_Z^P` falls. Elements whose yield comes mostly from
massive progenitors reverse this and light up at low `[X/Fe]`:

| element | structure |
|---|---|
| N | strongest gradient: `f_Z^P → 1` where `N/Fe ≲ -3.5`, because only PISN produce so little N |
| O, Al, Zn | same direction, weaker: PISN are poor in these elements |
| C | `f_Z^P` rises towards low `C/Fe`; the Li+ sample sits above the mock band |
| Co, Na | Li+ sits clearly above the mock band — the offsets measured in note 11 |
| Mg, Si, Ca, Ni, Cr | compact bands, Li+ inside or near the mock 16–84% range |

The `[Fe/H] → [X/Fe]` Spearman correlation is small for every element
(|ρ| < 0.10), which is the intended design: abundance ratios are set by the
source mixture, `[Fe/H]` by the dilution mass. The colour gradient is therefore
driven by the ratio, not by `[Fe/H]` itself.

## Ratio panel: `Mg_Si.png`

`[Mg/Si]` is drawn with the same layout, with `[Mg/Si] = [Mg/Fe] - [Si/Fe]` on
the vertical axis; the Li+ errors are the quadrature sum of the two catalogue
errors and only stars with both elements detected are shown (here 177 of 385 —
neither Mg nor Si has an upper limit in the sample).

This panel is the cleanest single diagnostic of the mixture, because Mg and Si
respond oppositely to the source mix: Pop II yields produce `[Mg/Si] ≈ 0`, while
PISN ejecta are Mg-poor and Si-rich. The result is a near-horizontal band —
colour depends on `[Mg/Si]` alone (ρ with `[Fe/H]` = -0.005) and runs smoothly
from `f_Z^P ≈ 1` at `[Mg/Si] ≈ -0.75` to `f_Z^P ≈ 0` at `[Mg/Si] ≈ 0`. A vertical
cut through the panel therefore reads the PISN share almost directly. The Li+
stars sit inside this band, near its `f_Z^P → 0` edge.

## Reproduce

```bash
python -m src.potluck.diagnostics.element_contours \
    src/potluck/tests/v5/limongi18_50k_ti_off.npz \
    --outdir src/potluck/tests/v14
```

## Caveats

1. The mocks are noiseless, so their bins are much tighter than the Li+
   error bars; compare centres, not widths.
2. `f_Z^P` is heavily bimodal (many mocks sit at exactly 0 or 1), so a bin
   median jumps between the two modes. The colour is a summary of the mixture,
   not a continuous physical variable.
3. Ti is excluded from training and support but is plotted here because the
   table still predicts it.
4. N, O and Al have no Li+ measurements; their panels are pure model.
