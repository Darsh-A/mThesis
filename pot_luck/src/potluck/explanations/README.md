# Potluck model: explanations

This folder explains the inverse-problem generator in `src/potluck/` step by step.
The code is authoritative; these notes explain the physics, the choices, and the
current limitations.

## Reading order

1. [Overview](01-overview.md) — what the model is and why it differs from Salvadori.
2. [Algorithm](02-algorithm.md) — the full pipeline, one stage at a time.
3. [Priors and cuts](03-priors-and-cuts.md) — every tunable bound and the rejection order.
4. [Features, labels, detection](04-features-labels-detection.md) — how ejecta become
   abundances, regression targets, and the counterfactual label.
5. [Diagnostics and results](05-diagnostics-and-results.md) — what the plots test and
   what the 2000-sample run found.
6. [Real-star validation](07-real-star-validation.md) — the Li+ observation model,
   the 100k evaluation, calibration, and the support/Ti finding.
7. [Explanation figures](08-explanation-figures.md) — eight mock-only figures that
   explain the model structure without the catalogue.
8. [Number-of-PISN prior](09-npisn-prior.md) — how much of the `f_Z^P`
   distribution is set by the `n_PISN` prior.
9. [Pop II table and the low-Z clamp](10-popii-metallicity.md) — the Limongi18
   metallicity mapping fix and the Nomoto13/Limongi18 comparison.
10. [Per-element audit](11-element-audit.md) — mock, yield, and Li+ medians
    element by element, with the offsets.
11. [Limongi18 re-run](12-limongi-rerun.md) — support, calibration, and
    reweighting recomputed on the new default table.
12. [Per-element contours](13-element-contours.md) — `[X/Fe]` versus `[Fe/H]`
    coloured by the PISN share, with the Li+ stars overlaid.
13. [Every common element](14-all-common-elements.md) — the feature set is now all
    26 elements the three tables share; what that changed in v7 and v14.
14. [Feature importance](15-feature-importance.md) — which of the 26 elements carry
    the label, and the eight-element observed subset.
15. [Caveats and next steps](06-caveats-and-next-steps.md) — what is not implemented yet.

## One-paragraph summary

The model draws a metal-enrichment history for a single cloud: zero or more PISN,
a Poisson number of Pop III core-collapse events, and one Pop II burst. It decides
which events exploded before the Pop II burst and which exploded before the observed
star formed, sums their ejecta element by element, applies physical and selection
cuts, and turns the result into noiseless abundance ratios. It then defines a
detectability label by removing the PISN contribution and asking whether the
abundance pattern would change by more than the measurement noise. The output is a
catalogue-like dataset for simulation-based inference, not a fit to any one star.

## Quick start

```bash
python -m src.potluck generate --n 2000 --output data/potluck/dataset.npz --verbose
python -m src.potluck diagnose data/potluck/dataset.npz
python -m src.potluck validate data/potluck/dataset.npz
python -m src.potluck plots data/potluck/dataset.npz
python -m src.potluck ceiling data/potluck/dataset.npz
python -m src.potluck.diagnostics.support_decomposition decompose data/potluck/analysis/v8_threshold_100k.npz
python -m src.potluck.diagnostics.support_decomposition ablate data/potluck/analysis/v8_threshold_100k.npz --n 20000
python -m src.potluck.diagnostics.yield_cross --n 20000 --outdir src/potluck/tests/v4
python -m src.potluck.diagnostics.ti_diagnostic src/potluck/tests/v5/nomoto13_50k.npz --outdir src/potluck/tests/v5
python -m src.potluck.diagnostics.prior_screen --n 3000 --points 25 --outdir src/potluck/tests/v6
python -m src.potluck.diagnostics.prior_screen --confirm --labels baseline,p11,p15 --seeds 1,2,3,4 --n 20000 --outdir src/potluck/tests/v6
python -m src.potluck.diagnostics.reweighting src/potluck/tests/v5/nomoto13_50k_ti_off.npz --outdir src/potluck/tests/v6 --verbose
python -m src.potluck.diagnostics.explanation_plots src/potluck/tests/v5/nomoto13_50k_ti_off.npz --outdir src/potluck/tests/v7
python -m src.potluck.diagnostics.npisn_prior --n 20000 --seed 42 --outdir src/potluck/tests/v8
python -m src.potluck.diagnostics.popii_metallicity --n 20000 --seed 42 --outdir src/potluck/tests/v9
python -m src.potluck.diagnostics.element_audit src/potluck/tests/v5/limongi18_50k_ti_off.npz --outdir src/potluck/tests/v10
python -m src.potluck.diagnostics.element_contours src/potluck/tests/v5/limongi18_50k_all_elements.npz --outdir src/potluck/tests/v14
python -m src.potluck.diagnostics.feature_importance src/potluck/tests/v5/limongi18_50k_all_elements.npz --outdir src/potluck/tests/v15
python -m src.potluck.diagnostics.support_decomposition decompose src/potluck/tests/v5/limongi18_50k_ti_off.npz
python -m src.potluck.diagnostics.ti_diagnostic src/potluck/tests/v5/limongi18_50k_ti_off.npz --outdir src/potluck/tests/v11
python -m src.potluck.diagnostics.classifier_plots src/potluck/tests/v5/limongi18_50k_ti_off.npz src/potluck/tests/v11/calibration
python -m src.potluck.diagnostics.reweighting src/potluck/tests/v5/limongi18_50k_ti_off.npz --outdir src/potluck/tests/v11 --verbose
```

All numerical bounds are in `src/potluck/config.py`.
