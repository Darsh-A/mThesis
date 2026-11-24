# Survey catalogue options for PISN-descendant searches

Question: is there a public survey catalogue that is clearly better than the
current 385-star Li+ sample for finding pair-instability supernova (PISN)
descendants in very metal-poor stars?

**Bottom line.** No single public catalogue is better on *all* axes. The current
Li+ sample is already a homogeneous, high-resolution (R = 36,000) VMP sample;
the large surveys win on sample size and element breadth but lose on VMP density,
resolution, or the two killing elements (Cu, Zn). The best realistic strategy is
a **two-tier combination**, not a swap:

1. **GALAH DR4** — the only large, homogeneous, public catalogue that carries
   N, Cu, Zn, Co, Mn, Si, Ca, Cr, Ni, Na, Mg, Ti, C simultaneously.
2. **APOGEE DR17/DR18** — the strongest homogeneous source for N, Co, Mn, Cr,
   Ni, Si, Ca in giants, but **no Zn and Cu is declared unusable**.
3. **Gaia-ESO DR5.1** — the best per-star quality (UVES R = 47,000, Co/Cu/Zn on
   subsets) but a small VMP tail and weak metal-poor homogenisation.
4. **SAGA / JINAbase** — heterogeneous literature compilations; use them for
   cross-checks and candidate assembly, not as an abundance scale.
5. **LAMOST / Pristine / SkyMapper** — target selection only.

The physics says the decisive PISN discriminants are the odd-Z / Fe-peak
elements (Zn, Cu, Co, Mn) against the alpha elements (Mg, Si, Ca), with
Fe-independent ratios such as Zn/Mg and Co/Ni. Those are exactly the elements
that are hardest to measure in large surveys, so "more stars" does not
automatically mean "more detectable PISNs".

---

## 1. Current catalogue: Li+ (Li et al. 2022)

| property | value |
|---|---|
| Reference | Li H. et al. 2022, ApJ 931, 147, arXiv:2203.11529 |
| Stars | 385 |
| [Fe/H] | about −1.7 to −4.3; >110 stars below −3.0 |
| Spectra | Subaru/HDS, R = 36,000, 4000–6800 Å |
| Analysis | 1D LTE, homogeneous single programme |
| Elements | C, Na, Mg, Si, Ca, Ti, Cr, Mn, Fe, Co, Ni, Zn (12 in our feature set) |
| Coverage | Mg/Ca/Fe nearly complete; C 281 detections + 70 limits; Co 265; Mn 178; Si 177; Zn 169 |
| Machine-readable | SAGA TSV: `sagadatabase.jp/Data/txt_table_LAMSUB_20231106.tsv` |
| Main weakness | sparse Zn/Mn/Si/Co, so the highest-signal ratios (Zn/Mg, Co/Ni) are the least measured |

This is a high-resolution, homogeneous VMP sample. It is not obviously beaten by
a 10^5–10^6-star survey whose VMP subset is smaller, lower resolution, or
missing Zn.

---

## 2. Ranked shortlist

| rank | catalogue | size | VMP density | R | Zn | Cu | Co/Mn/N | public | verdict |
|---|---|---:|---|---:|---|---|---|---|---|
| 1 | **GALAH DR4** (Buder+2024, PASA 42, 51) | 917,588 | thin (~2% at [Fe/H]≤−1 in DR3) | ~28,000 | ✓ | ✓ | ✓ | FITS/TAP, Data Central | best single homogeneous source; strict flagging needed |
| 2 | **Li+ 2022 + Roederer+2014** | 385 + 313 | dense VMP | 36k / ~40k | ✓ | ✓ | ✓ | SAGA TSV / paper tables | best homogeneous high-res backbone |
| 3 | **Gaia-ESO DR5.1** (iDR6) | 114,916 | small VMP tail | 20k (GIRAFFE) / 47k (UVES) | ⚠ | ⚠ | ⚠/✓ | ESO Phase 3, TAP | best per-star quality; use for calibration |
| 4 | **APOGEE DR17/DR18** | ~650k | ~289 likely below −2.5 (Montelius+2026) | ~22,500 (H) | ✗ | ✗ | ✓ | SDSS SAS/CAS | N/Co/Mn/Cr/Ni anchor only; cannot supply Zn/Cu |
| 5 | **SAGA** | 6,742 MW objects | dense VMP | heterogeneous | ✓ | ✓ | ✓ | TSV, updated 2026 | cross-check / candidate assembly, not a scale |
| 6 | **JINAbase** | 1,658 | dense VMP | heterogeneous | ✓ | ✓ | ✓ | SQL/web | same caveat as SAGA |
| 7 | **Hypatia** | 14,085 | mostly local, few VMP | heterogeneous | ✓ | ✓ | ✓ | JSON API | corroboration only |
| 8 | **LAMOST DD-Payne** (Xiang+2019) | 5.98M | large | ~1,800 | ✗ | ⚠ (0.2–0.3 dex) | ⚠ | FITS | target selection, not Zn/Co science |
| 9 | **RAVE DR6** | 451,783 | large | ~7,500 | ✗ | ✗ | ✗ | FITS/SQL | Fe/Al/Ni/α only; not useful here |
| 10 | **Pristine DR1 / SkyMapper DR4** | 10^8–10^9 | huge | photometric | ✗ | ✗ | ✗ | VizieR / MAST | target selection only |
| — | **4MOST / WEAVE / SDSS-V MWM** | future | — | high | — | — | — | not public yet | revisit at first public abundance DR |

Notes:

- **"MINES"** did not resolve in ADS/arXiv. The likely intended survey is
  **MINCE** (Measuring at Intermediate metallicity Neutron-Capture Elements),
  used by Aguado et al. (2023). Confirm before citing.
- **PISN-explorer** (Aguado et al. 2023, MNRAS 520, 866) already screened >1.4M
  objects from APOGEE, GALAH, GES, MINCE and JINA and produced 166 candidates.
  That pipeline is the closest existing competitor and should be read first.
- **Xing et al. (2023, Nature 618, 712)** reported a metal-poor star with a PISN
  abundance pattern; their element set is a useful template for what a credible
  candidate needs.

---

## 3. Element coverage

Legend: ✓ measured/recommended, ⚠ measured but flagged unreliable or
subset-dependent, ✗ absent/unusable.

| catalogue | N | Cu | Zn | Co | Mn | Si | Ca | Cr | Ni | Na | Mg | Ti | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GALAH DR4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| APOGEE DR17 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ⚠ | ✓ |
| Gaia-ESO DR5.1 | ⚠ | ⚠ | ⚠ | ⚠ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SAGA / JINAbase | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LAMOST DD-Payne | ⚠ | ⚠ | ✗ | ⚠ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RAVE DR6 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Pristine / SkyMapper | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

Practical caveats:

- **GALAH DR4:** all elements exist, but the VMP tail is thin and many
  `flag_X_fe` are non-zero there. Zn/Cu/Co survive least often at low
  metallicity. Do not use `flag_fe_h` (documented broken); use `flag_sp` plus
  per-element flags.
- **APOGEE DR17:** Zn is not in the 27 considered species; Cu is officially
  "unsuccessful". Co is reliable in giants but unsuccessful in dwarfs. Ti is
  rated "deviant". Zn cannot be recovered from APOGEE.
- **Gaia-ESO:** `ZN1`, `CU1`, `CO1` exist but are populated only for subsets;
  filter on `NL_ZN1`/`NN_ZN1` rather than trusting a non-blank cell. The ESO
  column table mislabels `CO1` as "Copper"; `CO1` is Co I, `CU1` is Cu I.
- **SAGA/JINAbase:** abundances are [X/H] on heterogeneous scales; convert to
  [X/Fe] and carry per-element uncertainties. Useful for literature variance,
  not for a homogeneous classifier.
- **LAMOST DD-Payne:** no Zn, Cu σ ≈ 0.2–0.3 dex (0.3–0.5 in some papers), Co
  trend inverted for dwarfs, and everything below [Fe/H] ≈ −1.5 is extrapolation
  from a solar-metallicity-dominated training set.

---

## 4. Physics: which elements actually matter

Ranked diagnostic power for separating PISN from Pop III and Pop II CCSN. This
is a synthesis of the model results and standard nucleosynthesis; it is not a
claim that any single ratio is a clean “smoking gun”.

### Essential

| element/ratio | why |
|---|---|
| **Fe** | reference; [X/Fe] is the observable |
| **Mg** | clean alpha element; sets the alpha baseline; nearly complete in Li+ |
| **Si, Ca** | alpha elements strongly produced by PISN; shape of the alpha pattern |
| **Zn** | PISN produces little Zn relative to alpha/Fe; **Zn/Mg is the strongest single ratio** in our model (−0.62) |
| **Co** | odd-Z Fe-peak; PISN low, CCSN higher; **Co/Ni** is an odd-even discriminant (−0.32) |
| **Ni** | even-Z Fe-peak denominator for Co/Ni; useful even when Co is noisy |

### Useful but systematics-limited

| element/ratio | caveat |
|---|---|
| **Na** | odd-Z; Na/Mg useful (−0.44), but NLTE and giant/evolution sensitivity |
| **Mn** | odd-Z Fe-peak; PISN low; strong NLTE and metallicity dependence |
| **Cu** | PISN low; a key “killing” element in the literature, but hyperfine structure, blends, NLTE, and hard in VMP giants |
| **Cr** | even-Z Fe-peak; PISN high in our model; NLTE and yield systematics; CK20 overproduces it |
| **Ti** | alpha/Fe-peak; historically problematic (NLTE, 3D, yield tables); baseline underproduces it by 0.73 dex |

### Confounded / optional

| element | caveat |
|---|---|
| **N** | PISN low, but CNO cycling, mixing, binaries, giants, and a ~0.1 dex GALAH-vs-APOGEE scale offset |
| **C** | strong evolutionary mixing and CEMP-no selection; mostly upper limits in Li+ |

**Minimum credible set for a PISN candidate:** Fe + Mg + Si + Ca + Na + Co (or
Ni) + Zn (or Cu) + Mn + Cr. In practice: at least 6–8 detected ratios including
**one of Zn/Cu** and **one of Co/Mn**. A candidate with no Zn, Cu, Co or Mn is
not a PISN test; it is only an alpha-enhanced star.

### Systematics hierarchy

1. **Yield-table offsets (0.3–0.7 dex)** dominate: Ti, Co, Zn, Mn. These are
   larger than most measurement errors and are not removed by more stars.
2. **NLTE/3D/hyperfine (0.1–0.3 dex)** for Mn, Co, Cu, Ti, Na.
3. **Catalogue-to-catalogue scale offsets (0.1–0.3 dex)** between APOGEE, GALAH,
   Gaia-ESO; the literature explicitly finds offsets >0.1 dex in absolute
   abundance scales, with parameter-dependent trends in giants.
4. **Measurement noise (0.05–0.2 dex)** — the smallest term for the best
   elements, but not for Zn/Cu/Co in low-S/N VMP stars.

**Implication:** the survey choice is secondary to cross-calibration and
yield-systematic control. A larger catalogue with a 0.3 dex Zn offset is worse
than a smaller catalogue with a common scale.

---

## 5. What “better” requires

A larger catalogue is not automatically better. Before adopting one:

1. **Define the target.** If the goal is a subdominant PISN, the production
   label must be the counterfactual detection label, not the `f_Z^P ≥ 0.5`
   proxy. The current code trains most production models on `y_f_05`, which
   makes a subdominant PISN the negative class.
2. **Estimate the numerator, not just N.** Precision at fixed true count scales
   roughly as 1/N. A deeper re-observation of the same volume adds false
   positives, not PISNs. Adopt a catalogue only if the expected true PISN count
   in the new footprint rises faster than the false-positive rate.
3. **Count VMP stars with the required elements.** Report, for [Fe/H] < −1.5:
   fraction with ≥4 of the eight key ratios, and fraction with Zn and Co. If
   Zn+Co coverage is below Li+, the catalogue cannot deliver the two strongest
   diagnostics.
4. **Cross-calibrate.** Require ≥500 stars in common with Li+, per-element
   offsets <0.1 dex, no [Fe/H] trend, and publish the offsets. Use Gaia DR3
   `source_id` as the join key.
5. **Deduplicate and group.** Positional cross-match (<1–2″), cluster/stream
   flags, and an independent-enrichment-site count. Ten members of one cluster
   are one draw, not ten.
6. **Refit censoring.** The current non-detection threshold uses a 0th quantile
   (sample minimum), which changes with N; replace it with a survival model and
   validate the predicted non-detection fraction per element per S/N bin.
7. **Reproduce Li+ first.** Apply the refactored pipeline to the existing Li+
   sample and reproduce the published AUC, positive rate and support. If it
   cannot reproduce Li+, it cannot be trusted on new data.
8. **Hold out a catalogue or sky region.** All model choices so far were made on
   Li+; a new catalogue evaluated on the same choices is not an independent test.

---

## 6. Recommended plan

### Short term (usable now)

1. **GALAH DR4 query** for [Fe/H] < −1.5 with `flag_sp == 0`, SNR > 30, and
   per-element `flag_X_fe == 0`. Count stars with Zn, Cu, Co, Mn, N. Expect
   heavy attrition; report the surviving sample before any modelling.
2. **APOGEE DR17 cross-match** for N, Co, Mn, Cr, Ni, Si, Ca in the giants that
   survive. Do not use APOGEE for Zn or Cu.
3. **Gaia-ESO UVES subset** for high-resolution Zn/Cu/Co confirmation of the
   best GALAH/APOGEE candidates; use it as a calibration/validation set.
4. **SAGA/JINAbase** to assemble literature Cu/Zn/Co values and to cross-check
   the Li+ stars; the SAGA LAMOST-Subaru TSV is the machine-readable Li+ source.

### Medium term

- Cross-calibrate GALAH, APOGEE, Gaia-ESO and Li+ on common stars; publish
  per-element offsets and slopes.
- Retrain the mock observation layer on the new catalogue's actual masks and
  error vectors, not on Li+.
- Pre-register one primary metric: expected true PISN descendants in the
  top-K followable candidates, with K stated.

### Long term

- **4MOST** and **WEAVE** high-resolution VMP surveys will be the real gain;
  neither has a public science abundance release yet.
- Dedicated follow-up of the PISN-explorer 166 candidates (Aguado et al. 2023)
  is likely a better use of telescope time than a brute-force catalogue swap.

---

## 7. Sources

Primary papers:

- Li, H. et al. 2022, ApJ 931, 147, arXiv:2203.11529 — the current Li+ catalogue.
- Buder, S. et al. 2021, MNRAS 506, 150, arXiv:2011.02505 — GALAH DR3.
- Buder, S. et al. 2025, PASA 42, 51, arXiv:2409.19858 — GALAH DR4.
- Abdurro'uf et al. 2022, ApJS 259, 35, arXiv:2112.02026 — SDSS DR17 / APOGEE.
- Gilmore, G. et al. 2022, arXiv:2208.05432; Worley, C. C. et al. 2024,
  arXiv:2402.06076; Hourihane, A. et al. 2023, A&A 676, 129, arXiv:2304.07720 —
  Gaia-ESO DR5.1 / homogenisation.
- Suda, T. et al. 2008, PASJ 60, 1159, arXiv:0806.3697 — SAGA.
- Abohalima, A. & Frebel, A. 2018, ApJS 238, 36, arXiv:1711.04410 — JINAbase.
- Roederer, I. U. et al. 2014, AJ 147, 136, arXiv:1403.6853 — 313 VMP stars,
  48 elements.
- Xiang, M. et al. 2019, ApJS 245, 34, arXiv:1908.09727 — LAMOST DD-Payne.
- Steinmetz, M. et al. 2020, AJ 160, 83, arXiv:2002.04512 — RAVE DR6.
- Aguado, D. S. et al. 2023, MNRAS 520, 866, arXiv:2301.03604 — PISN-explorer.
- Xing, Q.-F. et al. 2023, Nature 618, 712 — PISN-pattern star.
- Hegedűs, V. et al. 2023, A&A 670, A107, arXiv:2211.03416 — cross-survey
  abundance-scale offsets.
- Montelius, M. et al. 2026, arXiv:2602.10859 — APOGEE metal-poor tail.

Data access:

- GALAH DR4: https://www.galah-survey.org/dr4/the_catalogues/
  (FITS: `galah_dr4_allstar_240705.fits`; TAP via Data Central)
- APOGEE DR17: https://www.sdss4.org/dr17/irspec/aspcap/ and
  https://www.sdss4.org/dr17/irspec/abundances/
- Gaia-ESO DR5.1: https://www.eso.org/qi/catalogQuery/index/411
  (`safcat.GES_DR5_1_V1`)
- SAGA: https://sagadatabase.jp/ ; TSV:
  https://sagadatabase.jp/Data/txt_table_recommended.tsv
- Hypatia: https://www.hypatiacatalog.com/ ; API:
  https://www.hypatiacatalog.com/api
- LAMOST DD-Payne: https://dr5.lamost.org/doc/vac
- RAVE DR6: https://www.rave-survey.org/
- Pristine DR1: VizieR J/A+A/692/A115
- SkyMapper DR4: https://skymapper.anu.edu.au/data-release/

---

## 8. Caveats from the red-team

These are the main reasons a catalogue switch could fail even with a “better”
sample:

- The production classifier is trained on `f_Z^P ≥ 0.5`, which makes a
  subdominant PISN the negative class. More data cannot fix a label mismatch.
- Precision at fixed true count scales ~1/N; a larger catalogue can lower
  precision. The numerator is not currently estimated.
- Catalogue-scale offsets (0.1–0.3 dex) are comparable to the subdominant-PISN
  signal; without cross-calibration the classifier can learn provenance.
- Duplicates and cluster members inflate N without adding independent
  enrichment sites.
- The current censoring threshold is a sample minimum (0th quantile), so it
  changes with N; it must be refit.
- The σ-normalised support gate is insensitive to a uniform noise change and
  can make a noisier catalogue look better.
- All model choices were made on Li+; a new catalogue evaluated with the same
  choices is not an independent test.

The strongest counter-argument to a catalogue swap: the binding limitation is
the yield-model offset, not the number of stars. A larger catalogue exposes that
offset over more stars and may guarantee model rejection, while the two
highest-signal diagnostics (Zn/Mg, Co/Ni) are the ones most likely to be missing
in a multi-object survey. The realistic gain is **cross-calibrated GALAH DR4 +
APOGEE + Gaia-ESO + targeted follow-up**, not a single bigger catalogue.
