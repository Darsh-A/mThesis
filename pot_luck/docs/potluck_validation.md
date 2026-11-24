# Potluck validation against Salvadori et al. (2019)

This is a scientific validation note, not a code review. It asks four questions:

1. Does potluck reproduce the published Salvadori+2019 PISN + Pop II model?
2. Is it a legitimate model?
3. What holes or unexplained behaviour remain?
4. Can it survive community criticism as a publishable model?

Short verdict:

- **The core physics is validated.** In the matched slice (one PISN, no Pop III,
  one Pop II burst, WW95 yields), potluck reproduces Salvadori+2019 Eq. 13 to
  machine precision (`max |Δ| ≈ 10⁻¹⁵–10⁻¹⁰` dex).
- **Potluck is a legitimate forward simulator, not yet a validated physical
  model.** The equations are right; the dominant uncertainties are the yield
  tables, the low-metallicity coverage of the Pop II tables, and the
  observational/statistical framing.
- **It is on a good road to publishable, conditionally.** The main blockers are
  the label definition, the yield-systematic treatment, the low-Z clamp, and the
  lack of real-star controls/SBC — not the core model.
- **It can hold community criticism only if it is framed as a simulation-based
  inference framework with explicit assumptions.** The Eq. 13 match is necessary
  but not sufficient; it validates bookkeeping, not yields.

---

## 1. What was compared

Salvadori+2019 model:

```
one PISN of mass mPISN
one Pop II burst
fPISN = fraction of ISM metals from the PISN
f*/fdil = star-formation efficiency / dilution factor
[X/Fe] from Eqs. 12–13
```

Potluck matched slice:

```
n_PISN = 1
no Pop III CCSN
one Pop II burst
Pop II table = WW95
M*II set by fPISN
M_dil set by Z* = M_Zpre/M_dil
```

Parameter mapping:

| Salvadori | potluck |
|---|---|
| `fPISN` | `f_ZP = M_ZPISN / M_Ztotal` (identical in the matched slice) |
| `mPISN` | PISN event mass |
| `f*/fdil` | `Z* = M_Zpre/M_dil`, with `M_dil = mPISN fPISN / (f*/fdil)` in the one-PISN slice |
| `tpopII` | fixed in Salvadori's grid; potluck samples `t` |
| one PISN | 0–3 PISN events |
| no Pop III | Pop III CCSW events (Ishigaki) |
| deterministic Pop II | binned compound-Poisson sampler |

The matched slice is the correct comparison. The full potluck model is a
superset, not a drop-in replacement.

---

## 2. Eq. 13 cross-check

`src/potluck/validation.py::eq13_check` compares potluck's matrix bookkeeping
with the legacy Salvadori implementation. Results for `f_ratio = 1e-3`,
`tpop2 = 10 Myr`, all 26 feature elements:

| Pop II table | mPISN | fPISN | max abs difference (dex) | mean abs difference |
|---|---:|---:|---:|---:|
| WW95 | 150 | 0.5 / 0.7 / 0.9 | 1.6e-15 / 1.2e-15 / 9.4e-16 | ~2e-16 |
| WW95 | 200 | 0.5 / 0.7 / 0.9 | 7.8e-16 / 8.9e-16 / 1.1e-15 | ~2e-16 |
| WW95 | 250 | 0.5 / 0.7 / 0.9 | 6.5e-11 / 6.8e-11 / 6.9e-11 | ~3e-11 |
| Limongi18 | 150/200 | all | ≤2.3e-15 | ≤8e-16 |
| Nomoto13 | 150/200 | all | ≤1.4e-15 | ≤6e-16 |
| Nomoto13_CK20 | 150/200 | all | ≤1.2e-15 | ≤4e-16 |

**Conclusion:** the linear mixing, PISN normalisation, dilution, and Pop II mass
bookkeeping are algebraically identical to Salvadori Eq. 13. This holds for
every Pop II table tested. The `m = 250` residuals (~1e-10 dex) are floating
point/table-selection noise, not a model difference.

**What this does not prove:** it does not validate the yield tables, the choice
of `tpopII`, the low-metallicity regime, or the stochastic/observational layers.
It is a bookkeeping test, not an independent physical validation.

---

## 3. The dominant uncertainty: yield-table spread

At a matched point (`mPISN = 200`, `fPISN = 0.5`, `f*/fdil = 1e-3`,
`tpopII = 10 Myr`), the predicted `[X/Fe]` varies across Pop II tables by:

| element | spread across WW95 / Limongi18 / Nomoto13 / CK20 (dex) |
|---|---:|
| Co | **2.24** |
| N | **1.19** |
| Cu | **1.06** |
| Ni | **0.92** |
| Zn | **0.88** |
| Ca | **0.85** |
| Na | 0.53 |
| C | 0.47 |
| Ti | 0.27 |
| Mn | 0.18 |
| Mg | 0.16 |
| Cr | 0.15 |
| Si | 0.12 |
| Al | 0.11 |
| O | 0.15 |

The PISN-table spread (HW2002 vs Takahashi MR/NM/NR) is smaller for most
elements (0.25–0.43 dex), except N (~1.6 dex). So the **Pop II table is the
dominant systematic**, and it is largest exactly for the killing elements
Co, Cu, Zn, and for N.

This is the central scientific limitation: the PISN signal in a single element
can be smaller than the table-to-table spread. Potluck cannot quote a unique
`f_ZP` or a single-element “smoking gun” without marginalising over yields.

---

## 4. The second hole: low-metallicity coverage

The default Pop II table, Limongi18, has its lowest node at
`Z = 3.236e-5` (`[Fe/H] = -3`). The default mock sample has:

| quantity | value |
|---|---:|
| median `Z*` | 2.28e-5 |
| fraction `Z* < 3.236e-5` | **56.9%** |
| fraction `Z* < 1e-4` | **78.9%** |

So **more than half of the default mocks sit below the lowest Limongi18 node**
and use the `[Fe/H] = -3` table. The model is not sampling the table's
metallicity dependence for the bulk of the sample.

The CK20 variant is worse: its lowest **nonzero** node is `Z = 1e-3`, and
99.6% of mocks are below it; 77.9% are below `1e-4`. They are evaluated through
the `Z=0 ↔ Z=1e-3` blend, i.e. mostly the metal-free row.

The Salvadori slice avoids this because WW95 has a `Z = 1.345e-6` node. The
matched Eq. 13 check is therefore valid across the relevant range, but the
**default potluck configuration is not**: its bulk lies below the Limongi18
grid. This is a model-coverage hole, not an equation error.

Options:

1. Use WW95 or a low-Z table for the default low-metallicity regime.
2. Restrict the `Z*` prior to the table range.
3. Extend the table with low-Z yields (not available for Limongi18/Nomoto13).
4. Document the clamped fraction and report results split by `Z*` regime.

---

## 5. Mdil and the Salvadori parameter space

In the one-PISN slice,

```
M_dil = mPISN fPISN / (f*/fdil)
```

For Salvadori's grid (`mPISN = 140–260`, `fPISN = 0.5–1.0`,
`f*/fdil = 1e-4–1e-1`), this maps to roughly `M_dil ≈ 7e2–2.6e6 Msun`.
Potluck's default prior is `1e5–5e8`, and the v18 data-driven range is
`1e6–1e8`. The relevant physical variable is `Z*`, not `M_dil`; larger
`M_Zpre` (more events) allows larger `M_dil` at the same `Z*`.

The v18 study found that `M_dil` is only weakly constrained: the event prior
already spans `6e5–1.2e8`, and the data shift it up by ~0.25 dex. This is not a
bug, but it means:

- `M_dil` is not a prediction of the model; it is a degenerate parameter.
- The supported range `1e6–3e8` is a prior-predictive statement, not a
  measurement.
- The Salvadori slice lives at the low-`M_dil` end of potluck's prior; the
  default prior explores a different region.

---

## 6. What is validated

| item | status |
|---|---|
| Linear mixing of source ejecta | validated by Eq. 13 |
| PISN yield normalisation (`Y_X = M_X/mPISN`) | validated |
| Dilution and `Z*` definition | validated |
| Pop II mass tied to `fPISN` | validated |
| `f_ZP = fPISN` in the matched slice | validated |
| Duplicate/independent implementation | partial: the check uses the same repo functions; an external cross-check is still needed |
| Stellar yields | **not** validated: table spread 0.1–2.2 dex |
| Low-Z regime | **not** validated: 57% of default mocks below the Limongi18 grid |
| Chronology/two-clock gates | internally consistent; not externally validated |
| Observation layer | realistic for Li+; no NLTE/3D |
| Labels/classifier | internally consistent; label ambiguity remains |

---

## 7. Holes and unexplained behaviour, ranked

1. **Yield-table systematics (critical).** Co 2.24 dex, N 1.19, Cu 1.06,
   Ni 0.92, Zn 0.88, Ca 0.85 across tables at a matched point. This is larger
   than the single-element PISN signature.
2. **Low-Z coverage/clamping (critical).** 57% of default mocks below the
   lowest Limongi18 node; 99.6% of CK20 mocks below its lowest nonzero node.
3. **`M_dil` degeneracy (major).** v18 shows the data only weakly constrain it;
   the prior still matters.
4. **Label ambiguity (major).** The production paths train on
   `y_f_05 = 1[f_ZP ≥ 0.5]`, while the stated target is the counterfactual
   detectability label `y = 1[D > k]`. A subdominant PISN is the negative class
   in the production classifier.
5. **Precision at realistic prevalence (major).** Mock AUC ≈ 0.97, but at
   `π = 2.5e-3` and 90% recall, precision ≈ 3%. Report expected true positives,
   not just AUC.
6. **Missing physics (major).** No SNIa, no AGB, no rotation/binarity for Pop
   III, no metal loss, no spatial/kinematic structure; single cloud, instant
   mixing.
7. **Observation model (moderate).** Li+ only; non-detection threshold uses a
   0th quantile (sample minimum); global per-element offsets; no NLTE/3D;
   σ-normalised support gate.
8. **No population-level inference (moderate).** Event counts are independent
   of an IMF; `f_ZP` is model-dependent; converting to IMF constraints needs a
   population step.
9. **No independent stellar validation (moderate).** Eq. 13 is a bookkeeping
   check. Potluck has not yet reproduced BD+80°245, SDSS J0018−0939,
   Sextans S58, Draco 361/3053, or the Aguado fits.
10. **No SBC/held-out catalogue (moderate).** The classifier is trained and
    tested on its own simulator; real-star calibration is limited to Li+.

---

## 8. Publishability scorecard

| criterion | status | action |
|---|---|---|
| Clear model description | yes (`docs/algorithm.md`) | keep updated |
| Core equations validated | yes (Eq. 13) | add table to paper |
| Independent code/yield validation | partial | reproduce 2–3 published stars with an external implementation |
| Yield-systematic treatment | **no** | hierarchical per-element offsets / multi-table marginalisation |
| Low-Z coverage | **no** | use WW95 or low-Z table, or restrict `Z*` |
| Label definition | **no** | fix `y` vs `y_f_05`; one primary metric |
| Realistic prevalence reporting | **no** | expected true positives, precision/recall at `π` |
| Real-star controls | **no** | matched control sample; held-out catalogue |
| SBC/calibration | partial | add SBC and injection tests |
| Missing physics documented | yes | state SNIa/AGB/rotation/binary limitations |
| Reproducibility | yes | config hashes, seeds, versioned data |
| Falsifiable predictions | partial | define Zn/Cu/Co/Mn vs alpha, Mg/Al, Mg/Ca tests |

---

## 9. Can it hold community criticism?

| likely criticism | current evidence | response | remaining gap |
|---|---|---|---|
| “It is just Salvadori with more knobs.” | Eq. 13 exact in the matched slice | the extras are physical and the Salvadori slice is a submodel | need an ablation ladder showing each knob's effect |
| “The yields are uncertain.” | Pop II spread up to 2.24 dex | multiple tables, offset fits | need hierarchical yield uncertainty, not just table switching |
| “`M_dil` is arbitrary.” | v18 supports `1e6–3e8` | data-driven range; report prior-predictive | still weakly constrained; state it |
| “AUC is meaningless at 0.25% prevalence.” | precision ≈3% at 90% recall | report expected true positives/precision | add control sample and prevalence-aware metrics |
| “You train on your own simulator.” | no SBC yet | add SBC, injection, held-out catalogue | not done |
| “You ignore NLTE/3D/SNIa/AGB.” | documented limitations | add where possible; state impact | SNIa/AGB not modelled |
| “The model is not falsifiable.” | labels are model-defined | define observable predictions (Zn/Cu/Co/Mn, odd-even, Mg/Al, Mg/Ca) and test on real stars | needs a real-star test that can fail |
| “The candidate list is selection-biased.” | no controls yet | matched controls | not done |
| “`f_ZP` is model-dependent.” | true | report per-table; no single number | needs a model-averaged posterior |

The model can hold these criticisms **if** it is presented as a simulator with
explicit assumptions and a validation/ablation program, not as a discovery
claim. The strongest unresolved criticism is the yield-table spread plus the
low-Z clamp: those are physical/model-systematic issues, not code bugs.

---

## 10. Priority fixes before submission

1. **Fix the label and the primary metric.** Decide whether the paper is about
   `f_ZP ≥ 0.5` (PISN-dominated) or `y` (counterfactual detectability). Do not
   mix them.
2. **Treat yields hierarchically.** Add a per-element offset/scatter term,
   marginalise over the tables, and report the posterior spread. A single table
   is not defensible.
3. **Fix low-Z coverage.** Use WW95 or a low-Z table, or restrict `Z*` to the
   table range; report the clamped fraction and split results by regime.
4. **Validate against published stars.** Reproduce BD+80°245,
   SDSS J0018−0939, Sextans S58, Draco 361/3053, and the Aguado fits in the
   matched slice. Eq. 13 alone is not enough.
5. **Add SBC and a held-out catalogue/control sample.** The classifier must be
   tested on data it did not generate.
6. **Report precision/expected true positives at realistic prevalence.**
   AUC alone will not survive a survey-statistics referee.
7. **Add SNIa/AGB or justify their omission quantitatively.** At
   `[Fe/H] > -2`, AGB/SNIa can matter; at the PISN peak they may not, but show it.
8. **Replace the 0th-quantile censoring** with a survival model.
9. **Write the paper as a simulator + inference framework**, with the
   Salvadori slice as a validation, not as the model's only claim.

---

## 11. Bottom line

- **Legitimate model?** Yes, as a forward simulator: the core physics matches
  Salvadori Eq. 13 exactly, the components are explicit, and the code is
  reproducible.
- **Validated physical model?** Not yet. The yield-table spread, low-Z clamp,
  `M_dil` degeneracy, label ambiguity, and lack of real-star controls are
  unresolved.
- **Publishable?** Conditionally yes, after the priority fixes. The core is
  strong enough; the framing and systematics are the blockers.
- **Can it hold criticism?** Yes, if it is honest about the yields and the
  degeneracies, and if it reports realistic detection metrics. The strongest
  criticism is not “the equations are wrong” — they are right — but “the
  physical uncertainties are larger than the signal you claim.”
