# The number-of-PISN prior and the `f_Z^P` distribution

This note answers a narrow question: how much of the shape of the `f_Z^P`
distribution is set by the prior on the number of PISN events.

`f_Z^P` is the PISN share of the metal mass,

\[
f_Z^P = \frac{\sum_{X \in \mathrm{metals}} M_X^{\mathrm{PISN}}}{M_{Z,\mathrm{tot}}}.
\]

With no PISN event the numerator is exactly zero, so **every mock with
`n_PISN = 0` sits on the spike at `f_Z^P = 0`**. The height of that spike is
therefore a direct reflection of the `n_PISN` prior, filtered by the acceptance
cuts. The rest of the distribution is set by the PISN masses, the Pop III and
Pop II contributions, and the dilution mass.

## Test

`src/potluck/diagnostics/npisn_prior.py` generates the model twice with the
same seed and the same configuration, changing only `PriorConfig.n_pisn_probs`:

- default: `(0.5, 0.3, 0.15, 0.05)` for `n_PISN = 0, 1, 2, 3`;
- uniform: `(0.25, 0.25, 0.25, 0.25)`.

The run below uses 20,000 accepted mocks and seed 42.

```bash
python -m src.potluck.diagnostics.npisn_prior --n 20000 --seed 42 --outdir src/potluck/tests/v8
```

## Result

| prior | accepted `n_PISN` = 0 / 1 / 2 / 3 | zero fraction | positive fraction |
|---|---|---|---|
| default | 7,370 / 7,518 / 3,838 / 1,274 | 36.9% | 18.0% |
| uniform | 3,285 / 5,369 / 5,563 / 5,783 | 16.4% | 27.2% |

The zero fraction follows the prior closely: 50% of draws have no PISN, and the
cuts accept 66% of them, giving 36.9% of the accepted sample. Under a uniform
prior the same acceptance rate gives 16.4%. The positive fraction rises from
18.0% to 27.2% purely because the mixture contains more PISN-bearing mocks.

Conditioning on the number of events removes the effect:

| `n_PISN` | default median [16-84%], positive | uniform median [16-84%], positive |
|---|---|---|
| 1 | 0.155 [0.02, 0.66], 23.7% | 0.157 [0.02, 0.68], 24.4% |
| 2 | 0.269 [0.04, 0.83], 34.2% | 0.269 [0.04, 0.82], 34.4% |
| 3 | 0.359 [0.05, 0.90], 40.1% | 0.323 [0.05, 0.85], 38.4% |

Per-group Kolmogorov-Smirnov distances are `D = 0.017`, `0.013`, `0.043` for
`n = 1, 2, 3`. They are consistent with finite-sample noise; the only marginal
case is `n = 3`, which has the smallest samples and the widest `f_Z^P`.

## What this means

1. The `f_Z^P` histogram is a **mixture**, not a direct prior. Its zero spike is
   the `n_PISN = 0` population; its tail is the population with events.
2. The acceptance cuts already reshape the prior. Under the uniform prior, the
   accepted fractions are 16 / 27 / 28 / 29%, not 25% each, because mocks with
   PISN pass the metallicity and support cuts more often.
3. The physics of a single history is prior-independent: once `n_PISN` is
   fixed, the conditional `f_Z^P` distribution is the same for both priors.
   Changing the prior cannot be used to "produce" more PISN-dominated mocks
   without also raising the event rate that the model assumes.

The test does not tune any bound: only `n_pisn_probs` changes, and no cut is
relaxed.
