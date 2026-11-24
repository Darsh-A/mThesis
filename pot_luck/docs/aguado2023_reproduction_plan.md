# Reproducing the Aguado+2023 PISN-explorer catalogue for potluck

Goal: rebuild the input database and candidate catalogue of
**Aguado et al. 2023, MNRAS 520, 866** ("PISN-explorer"), then use it as a
real-star test set for potluck.

References:

- Paper: arXiv:2301.03604, DOI 10.1093/mnras/stad164, bibcode
  `2023MNRAS.520..866A`.
- Zotero item: `YZ8BXS7Y` (PDF attachment `33C2K4QD`).
- Public status: the 166-candidate and 45-object golden catalogues are **not
  public**; the paper says "the catalogue will be published in a forthcoming
  paper" and the reduced data are available on request. The online material
  contains only the UVES line table for 2M13593064+3241036. Plan for both
  "request the list" and "reconstruct from the same inputs".

---

## 0. What the paper actually did (verified from the text)

### Input catalogues

| survey | version used | stars analysed | solar scale | notes |
|---|---|---:|---|---|
| APOGEE | DR17 `allStar` | 879,437 spectra / 733,901 objects | Asplund+2005 | H-band R≈22,500; Ti I/Ti II discarded; Cu IR weak, only useful for [Fe/H] > −1; **no Zn** |
| GALAH | DR3 `main_allspec_v2` | 588,571 | Grevesse+2007 | R≈28,000; up to 30 elements incl. Cu, Zn, Co, Mn |
| Gaia-ESO | DR5 (iDR6-era) | 114,324 | Asplund+2009 | GIRAFFE + UVES; up to 31 elements incl. Cu, Zn, Co |
| MINCE | year-1 sample | 43 | Lodders+2009 (O, S: Caffau+2011) | O, Na, Mg, Al, Si, S, Ca, Sc, Ti, V, Cr, Mn, Co, Ni, Cu, Zn |
| JINA | full database | 1,658 VMP stars ([Fe/H] < −2) | Asplund+2009 | literature compilation; includes ~50 dwarf-galaxy stars (Sextans, Draco, Fornax) |

Total analysed: **1,438,497 stars**.

### Model grid

- Salvadori et al. (2019) parametric model.
- PISN yields: Heger & Woosley (2002).
- Pop II yields: Woosley & Weaver (1995), initial masses 8–40 M☉,
  Z = 10⁻⁴–1 Z☉, Larson IMF with m_ch = 0.35 M☉.
- Stars 40–140 M☉ collapse to black holes and are not included.
- 25 elements from C (Z=6) to Zn (Z=30).
- Grid axes:
  - `fPISN` = 0.50–1.00, step 0.10;
  - `log10(f*/fdil)` = −4 to −1, step 1 (4 nodes);
  - `mPISN` = 140–260 M☉, step 9 M☉.
- The grid is written in FERRE format; fitting uses quadratic interpolation
  and Nash's truncated Newton algorithm.
- Weights in the fit: `ω = 0` for missing, **1 for Cu and Zn**, 0.5 for all
  other elements. Cu and Zn therefore dominate χ² when present.
- Reduced χ² = χ²/(N − M), with the χ² of Salvadori+2019 Eq. 14.
- The paper's three axes do not include `t_popII`; it is fixed by the
  Salvadori model. The potluck `eq13_check` uses `t_popII = 10 Myr`; confirm
  this from Salvadori+2019 and by reproducing the published fits.

### Selection

1. **Pre-cuts:** discard bad flags (APOGEE `STAR_BAD`, GALAH `flag_sp != 0`);
   only `x_fe_flag == 0` abundances; require >5 available abundances;
   4000 K < Teff < 7000 K; [Fe/H] < −0.7.
   This leaves **385,202 stars** from 1,438,497.
2. **Fit + hard cuts:** discard χ² > 0.35, `mPISN < 145 M☉`,
   `fPISN < 55%`.
3. **Individual cuts:** if Cu/Zn available, discard `[Cu,Zn/Fe] > 0`;
   prefer `[Mg/Al] > 0.7`; prefer `[Mg/Ca] < 0`.
   Result: **166 candidates**.
4. **Golden catalogue:** χ² < 0.35 (two exceptions with [Zn/Fe] < −0.8),
   remove Ba/Ce s-process-enriched stars, prefer N ≥ 10 abundances,
   Vmag ≤ 16. Result: **45 objects**.

### Published validation targets

Use these to check any reimplementation:

| star | published fit |
|---|---|
| BD+80°245 | fPISN = 50%, log f*/fdil = −2.0, mPISN = 213.8 M☉, χ² = 0.17 |
| SDSS J0018−0939 | fPISN = 50%, log f*/fdil = −2.0, mPISN = 256 M☉, χ² = 0.44 |
| 2M13593064+3241036 (APOGEE) | fPISN = 66.1%, log f*/fdil = −3.1, mPISN = 193.8 M☉ |
| Sextans S58 | fPISN = 55%, mPISN = 253 M☉, χ² = 0.10 |
| Draco 361 | χ² = 0.20, [Zn/Fe] ≈ −0.45 |
| Draco 3053 | fPISN = 50%, log f*/fdil = −2.0, mPISN = 218 M☉, χ² = 0.19 |
| TYC 1118-595-1 | fPISN = 50%, log f*/fdil = −2.3, mPISN = 193 M☉ |
| J1124+4535 | fPISN = 50%, log f*/fdil = −2.4, mPISN = 231 M☉, χ² = 0.29 |

The candidate MDF peaks at [Fe/H] ≈ −1.7, with a secondary peak at −3.0 driven
by JINA.

---

## 1. Success criteria

The reproduction is acceptable if:

1. The pre-cut count is close to **385,202** (within catalogue-version
   differences) and the input total is close to **1,438,497**.
2. The 166-candidate and 45-golden counts are reproduced within a documented
   tolerance, and the per-survey contributions and MDF shape match.
3. The eight validation stars above are reproduced within the published
   uncertainties (fPISN ±0.1, mPISN ±10 M☉, χ² within ~0.05).
4. Cross-calibration offsets are measured, not assumed; candidates that depend
   on a single catalogue's scale are flagged.
5. A matched control sample exists, so potluck is not evaluated on a
   pre-selected candidate list alone.

---

## 2. Phase A — acquire and unify the input catalogues

### A1. Download

| survey | access | file / table |
|---|---|---|
| APOGEE DR17 | SDSS SAS/CAS (`https://www.sdss4.org/dr17/irspec/spectro_data/`) | `allStar-…fits` or CAS `apogeeAllStar` |
| GALAH DR3 | Data Central (`https://cloud.datacentral.org.au/teamdata/GALAH/`) | `main_allspec_v2` |
| GALAH DR4 (requested extension) | `https://www.galah-survey.org/dr4/the_catalogues/` | `galah_dr4_allstar_240705.fits` |
| Gaia-ESO DR5.1 | ESO Phase 3 (`safcat.GES_DR5_1_V1`) | 350-column FITS/TAP |
| MINCE year 1 | Cescutti et al. 2022 tables / MINCE release | 43-star table |
| JINA | `https://jinabase.pythonanywhere.com/` | full SQL/CSV |

**Important:** Aguado used GALAH **DR3**, not DR4. For exact reproduction use
DR3. Run DR4 as a separate "updated backbone" variant and report differences
rather than mixing them.

### A2. Unified schema

One row per star:

```
survey, survey_id, gaia_dr3_source_id, ra, dec, vmag,
teff, e_teff, logg, e_logg, feh, e_feh,
X_fe[elem], e_X_fe[elem], flag_X[elem], n_elem,
original_solar_scale, quality_flag, parent_id/cluster, load_version
```

Elements: C, N, O, Na, Mg, Al, Si, S, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni,
Cu, Zn, plus Ba/Ce for the s-process cut. Store missing as `NaN`, not zero.

### A3. Flags and quality

- APOGEE: `ASPCAPFLAG & STAR_BAD == 0`; per-element `X_FE_FLAG == 0`.
  Drop Ti I/Ti II. Treat Cu as missing for dwarfs.
- GALAH: `flag_sp == 0`; per-element `flag_X_fe == 0`. Do not use
  `flag_fe_h` (documented broken in DR4).
- GES: use the release flags (`SSA`, `SSP`, `NIA`, `SNR`) and per-element
  `NL_X`/`NN_X`; do not trust a non-blank Zn/Cu/Co cell without checking the
  node count.
- MINCE/JINA: keep literature flags; JINA has upper-limit markers.

### A4. Solar scale and cross-calibration

Each survey uses a different solar abundance set. Convert every value to a
common scale before fitting:

1. Reconstruct [X/Fe] using the survey's own solar abundances (as Aguado did).
2. Measure per-element offsets against a common scale (Asplund+2009) using
   stars in common:
   - APOGEE–GALAH overlap is large;
   - GES–GALAH overlap;
   - Li+ overlap with all three.
3. Fit offset + slope versus [Fe/H], Teff, logg; apply the offset and report
   the residual.
4. For elements with <30 common stars, do not fit a slope; apply a constant
   offset and widen the uncertainty.

The literature already reports >0.1 dex offsets between APOGEE, GALAH and GES,
so this is not optional.

### A5. Deduplicate

- Join on Gaia DR3 `source_id` first.
- Fall back to RA/Dec < 1–2″.
- Add cluster/stream/parent flags; count independent enrichment sites, not
  just stars.
- Keep the provenance of the retained row (which survey supplied each
  element).

### A6. Reproduce the pre-cut count

Apply the Aguado pre-cuts and compare to 385,202. Differences are expected
from DR4, flag changes, and duplicated stars; report the ledger by survey and
by cut.

---

## 3. Phase B — rebuild the PISN-explorer model grid

### B1. Model

Use the existing `src/salvadori_funcs.py` (Salvadori+2019 Eq. 12/13) as the
starting point. It already provides:

- `salvadori_yields` (PISN yield normalisation),
- `salvadori_Y_X_II`, `salvadori_Y_Z_II` (IMF-integrated Pop II),
- `salvadori_combined_abundratio` (Eq. 13),
- `salvadori_combined_abundratio_WrtH`.

Build a dedicated grid generator that matches Aguado's assumptions:

```
fPISN     = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
log10(f*/fdil) = -4, -3, -2, -1
mPISN     = 140, 149, ..., 260   (step 9)
elements  = C ... Zn (Z=6..30)
t_popII   = fixed (verify; likely 10 Myr)
```

Use HW2002 PISN yields and WW95 Pop II yields, not the potluck
Limongi18/Nomoto13 defaults, so the grid matches Aguado. If the grid is later
re-run with Limongi18/CK20, label it as a separate model variant.

### B2. Fit engine

Two options:

1. **FERRE** (`https://github.com/callendeprieto/ferre`): exact reproduction,
   but a Fortran dependency and a new build step.
2. **Python reimplementation**: interpolate the grid (quadratic or linear),
   minimize χ² with `scipy.optimize.least_squares` / `differential_evolution`,
   and use `emcee` for the MCMC cross-check. This is more transparent and
   easier to integrate with potluck.

Recommended: implement the Python fitter first, validate against the eight
published stars; only fall back to FERRE if the Python fit cannot reproduce
the published χ²/parameters.

### B3. χ² and weights

Match Aguado:

```
χ² = Σ_X ω_X [( [X/Fe]_obs - [X/Fe]_model ) / σ_X ]²
ω_X = 0 missing, 1 for Cu and Zn, 0.5 otherwise
χ²_red = χ² / (N - M),  M = 3
```

Store the best fit, the interpolated model, and the per-element residuals.

### B4. Grid validation

Before running 1.4M stars, reproduce the eight validation stars. This catches:

- wrong `t_popII`,
- wrong solar scale,
- wrong PISN/Pop II yield version,
- wrong interpolation/weighting,
- wrong definition of fPISN.

---

## 4. Phase C — apply the selection and build the candidate database

Run the same three stages and store every intermediate flag:

1. pre-cuts (flags, >5 abundances, Teff, [Fe/H]);
2. fit quality and grid-limit cuts (χ² > 0.35, mPISN < 145, fPISN < 55%);
3. individual cuts (`[Cu,Zn/Fe] > 0`, `[Mg/Al]`, `[Mg/Ca]`);
4. golden cuts (s-process, N ≥ 10, Vmag ≤ 16).

Output schema:

```
id, survey, ra, dec, gaia_dr3, vmag, teff, logg, feh,
n_elem, elements_used, fPISN, e_fPISN, log_fstar_fdil, mPISN, e_mPISN,
chi2_red, residuals, selection_stage, golden_flag,
zn_fe, cu_fe, mg_al, mg_ca, ba_ce_flag, provenance
```

Compare:

- total 166 and golden 45;
- per-survey counts;
- MDF peak at −1.7 and JINA peak at −3.0;
- fPISN–mPISN distribution (Fig. 5);
- which candidates are APOGEE-only and therefore lack the Cu/Zn killing
  elements.

If the exact 166/45 list is obtained from the authors, store it as the
reference and compare the reconstructed list to it (overlap fraction, missed
and extra candidates).

---

## 5. Phase D — validation and controls

### D1. Cross-calibration robustness

Re-run the fit after applying ±1σ of every cross-calibration offset. Report:

- how many of the 166 survive,
- which candidates are driven by a single survey's scale,
- which candidates have no Cu/Zn measurement (and are therefore weak).

### D2. Matched control sample

Aguado's 166 are pre-selected, so potluck cannot be scored on them alone.
Build a control sample from the same input catalogues:

- same pre-cuts (flags, Teff, [Fe/H], >5 abundances);
- same survey mixture;
- matched in [Fe/H], Teff, logg, N_elem, and Vmag;
- excluding the 166 candidates.

Use 5–10 controls per candidate.

### D3. Null tests

- Shuffle the element labels within each star (destroys the joint pattern,
  preserves marginals). The classifier must collapse to the prior.
- Fit the control sample with the PISN-explorer; the candidate rate should be
  consistent with the selection threshold, not with a PISN signal.
- Split the candidates by survey; a real signal should not be driven by one
  catalogue.

---

## 6. Phase E — test potluck on the catalogue

### E1. Loader

Add `src/potluck/catalogues/aguado2023.py` with:

- `load_aguado_candidates()`,
- `load_aguado_controls()`,
- a `ModelConfig`-compatible observation layer,
- per-element masks/errors and flags.

### E2. Match the model before comparing

Aguado's model is **PISN + Pop II (WW95)** with a single PISN and a fixed
fPISN grid. Potluck's full model adds Pop III CCSN and stochastic events.
Run two potluck variants:

1. **Matched slice:** PISN + Pop II only, WW95, no Pop III. Compare potluck's
   `f_ZP` posterior to Aguado's `fPISN`.
2. **Full model:** all sources. Ask whether the added Pop III/CCSN freedom
   changes the candidate ranking or makes the Aguado candidates out-of-model.

Do not compare the full potluck model directly to Aguado's fPISN without
stating the model difference.

### E3. Metrics

On candidates + controls:

- AUC / ROC for `y_f_05` and for `f_ZP ≥ 0.5`;
- precision at fixed recall, at Aguado's prevalence;
- calibration (reliability curve, Brier);
- confusion between potluck's PISN label and Aguado's candidate/golden flag;
- support-gate / out-of-model fraction;
- per-element residual offsets versus the Li+ scale;
- feature importance: does potluck rank Zn/Cu, Na/Mg, Co/Ni, Mg/Al, Mg/Ca as
  Aguado does?

### E4. Expected outcomes

- If potluck and Aguado agree on the golden 45, the models are consistent.
- If potluck flags many Aguado candidates as out-of-model, the difference is
  the Pop III/stochastic model, not necessarily a failure.
- If potluck's top features differ, report it: Aguado's selection is
  Cu/Zn-weighted; potluck may rely more on Na/Mg, Co/Ni.

---

## 7. Deliverables and layout

```
data/aguado2023/
  raw/                 APOGEE, GALAH, GES, MINCE, JINA downloads
  processed/
    unified.parquet    one row per star, all elements, flags, provenance
    candidates.csv     166 reconstructed candidates
    golden.csv         45 reconstructed golden
    controls.csv       matched controls
    cross_calibration.json
  README.md

src/potluck/catalogues/
  aguado2023.py        loader + schema

scripts/
  build_aguado_database.py
  fit_pisn_explorer.py
  cross_calibrate_aguado.py
  test_potluck_on_aguado.py

src/potluck/tests/v19/
  README.md
  candidate_overlap.png
  potluck_vs_aguado_fpisn.png
  feature_importance_comparison.png
  support_and_calibration.png
  metrics.json
```

Use `parquet` for the unified table; CSV for the candidate lists.

---

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| 166/45 catalogue not public | request from authors; reconstruct from the same inputs; compare overlap if obtained |
| FERRE vs Python fit differences | validate on 8 published stars; document χ²/parameter differences; fall back to FERRE if needed |
| GALAH DR3 vs DR4 | exact reproduction uses DR3; DR4 is an extension; never mix versions in one fit |
| Solar-scale offsets | measure per-element offsets on common stars; propagate; report candidates that depend on one scale |
| APOGEE has no Zn, Cu unreliable | flag APOGEE-only candidates as weak; rely on GALAH/GES/JINA for the killing elements |
| Pre-selected candidates | matched control sample; do not compute precision on candidates alone |
| Duplicates/cluster members | Gaia DR3 join + positional dedup + parent flags |
| Potluck model mismatch (WW95 vs Limongi/CK20; Pop III) | run matched PISN+PopII/WW95 slice first, then full model |
| t_popII unknown | reproduce published fits; if they fail, scan t_popII and document |
| Compute | fit only pre-cut stars (≈385k) with a small grid (≤336 models); cache interpolants |

---

## 9. Immediate next steps

1. **Request the 166/45 catalogue** from the Aguado authors; meanwhile
   reconstruct it.
2. **Download** APOGEE DR17, GALAH DR3, Gaia-ESO DR5.1, MINCE year 1, JINA.
3. **Build the unified schema** and reproduce the 1,438,497 / 385,202 counts.
4. **Build the 336-model grid** and reproduce the eight validation stars.
5. **Run the selection** and compare the 166/45 counts and MDF.
6. **Build matched controls.**
7. **Run potluck** on candidates + controls, matched slice first.
8. **Write v19** with metrics, overlap plots, and a README.

The first go/no-go is step 4: if the Python fitter cannot reproduce
BD+80°245, SDSS J0018−0939, and Sextans S58, stop and switch to FERRE before
running the full sample.
