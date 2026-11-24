# v7_1 — explanation figures on the Nomoto13 + CK20 pool

Mock-only explanation figures (`src/potluck/diagnostics/explanation_plots.py`),
regenerated on the temporary **CK20** Pop II yield set. This is the v7 figure set
(compare `src/potluck/tests/v7/`) with the Pop II table and its sampler grid
changed.

## Dataset

```
src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
```
50,000 accepted mocks, seed 42, 27 feature elements (26 ratios), Li+ observation
layer with Ti on. Acceptance 0.370.

## What changed versus v7

| parameter | v7 (v14/v15 pool) | **v7_1 (this test)** |
|---|---|---|
| Pop II table (`yields.popii_table`) | `Limongi18` | **`Nomoto13_CK20`** |
| Pop II sampler grid (`yields.popii_z_nodes`) | `[3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2]` | **[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2]** |
| everything else | unchanged | unchanged |

### The CK20 edits (temporary Nomoto13 variant)

Implemented in `src/yield_sources.py` as the `Nomoto13_CK20` source
(`_nomoto_ck20_entries_by_z`). Two yield changes are applied, plus recorded-only
metadata.

1. **Hypernova fraction 0.5 for M >= 20 Msun.** For every progenitor at or above
   20 Msun the normal (E = 1) row is mixed 50/50 with the hypernova row
   (E = 10/20/30) tabulated at the same mass and metallicity.
2. **Constant 2D-jet offsets (global, all masses):**
   Sc +1.0, Ti +0.45, V +0.3, Co +0.2, Zn +0.2 dex
   (applied as a multiplicative factor 10^dex to the element yield).
3. **Explosion-type mass ranges (metadata only, do not change the sampler):**
   ECSN 8.8-9 Msun, SN II/Ibc 10-30, failed SN 30-50, HN 20-50.
4. **Residual mismatches at [Fe/H] ~ -2 (metadata only):**
   Co ~0.3 dex low, Ni ~0.2 low, Si ~0.2 high, Ca ~0.1 low, Ti/Sc/V/K
   underproduced.

Note on the hypernova mixing: the 50/50 blend is applied at the tabulated mass
nodes (20, 25, 30, 40 Msun), so the transition from 0 to 0.5 hypernova fraction
ramps linearly across the 18-20 Msun segment of the IMF-weighted bin average
rather than as a hard step at 20 Msun.

## Figures

`fig1_priors`, `fig2_timeline`, `fig3_metallicity`, `fig4_yields` (Pop II panel
now shows the CK20 table), `fig5_trends`, `fig6_ratios`, `fig7_label`,
`fig8_zstar`, and `fig9_ca_pisn_fraction`.

`fig8_zstar` marks the Pop II sampler grid floor (lowest `popii_z_nodes` entry)
when it falls inside the sampled range. The floor is now 1e-12, below the
sampled range (log10 z_star ~ -6 to -3), so no vertical line is drawn and the
panel annotates that nothing is clamped. The right panel keeps its fixed
-6.2..-2.0 x-range.

`fig9_ca_pisn_fraction`: the fraction of mocks whose Ca is >90% from the PISN
(per-element PISN fraction `f_xp` at Ca), in bins of `[Ca/H]` (computed as
`[Ca/Fe] + [Fe/H]`) and `[Fe/H]`, over the full mock sample. Each panel is a bar
plot of the per-bin fraction with a continuous line drawn through the bar tops.
Overall 11.5% of mocks clear the 90% threshold; the fraction is flat at ~0.10
across `[Ca/H]` up to ~`-1.5` then rises steeply (to ~0.36), and declines with
`[Fe/H]` from ~0.18 at the metal-poor end.

Note on grids: `popii_z_nodes` is the *sampler* grid (where the Pop II table is
evaluated), distinct from the yield table's own Z nodes. The grid resolves the
Nomoto13 low-metallicity transition: the source is linear in Z below Z=1e-3
while the sampler interpolates in log10(Z), so a coarse grid clamps nearly all
the metal-poor mocks to the Z=1e-3 table (99.9% before this fix). Nodes at
1e-4, 1e-6, 1e-9 and a 1e-12 proxy for the metal-free Z=0 node let the
low-metallicity mocks use the Z=0 pattern instead. The grid cannot contain Z=0
directly (the table builder requires positive, strictly increasing nodes).

## Summary numbers (explanation_plots.json)

- positive fraction (`f_Z^P >= 0.5`): 0.1888
- median source share: PISN 0.0414, Pop III 0.1719, Pop II 0.3887
- Spearman rho(f_Z^P, ratio): Na/Mg -0.440, Co/Ni -0.330, Mn/Cr +0.128, Zn/Mg -0.627
- Pop II table: Nomoto13_CK20; sampler grid 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 2e-2, 5e-2
- Pop II grid floor Z = 1e-12; fraction of mocks below it: 0.0 (nothing clamped)

## Regenerate

```bash
D=src/potluck/tests/v5/nomoto13_ck20_50k_all_elements.npz
python -m src.potluck.diagnostics.explanation_plots $D --outdir src/potluck/tests/v7_1
```
