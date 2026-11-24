# Potluck: a plain-language walkthrough

This is the story of one mock star, from the first random draw to the final
machine-learning score. The formal formulas are in `potluck_final.md`; this file
is the intuition.

---

## The one-paragraph version

Potluck builds a gas cloud, throws a few supernovae into it, mixes the ejecta,
dilutes the mixture with a cloud mass, and reads off the abundance pattern a
star would have if it formed from that gas. It repeats this thousands of times.
Some clouds have a PISN, some do not. The code records the pattern, the PISN
fraction and a “would we notice the PISN?” label. A classifier is then trained
on those labelled patterns. When you give it a real star, it tells you how
PISN-like the pattern is, how far the star is from the simulated clouds, and
what PISN fraction the chemistry would imply.

---

## The picture

A cloud is a bucket. Each source pours element masses into it:

- a PISN,
- a few Pop III core-collapse supernovae,
- a burst of Pop II stars.

The bucket is then diluted to a total gas mass `M_dil`. The observed star forms
from that gas and freezes in the mixture.

```
PISN ejecta ┐
Pop III     ├──►  mix  ──►  dilute by M_dil  ──►  [X/Fe], [Fe/H]
Pop II      ┘
```

![Two-clock timeline](../src/potluck/tests/v19/Limongi18/explanation_plots/fig2_timeline.png)

---

## Step 1: draw the enrichment history

We draw, at random:

- how many PISN events happen: 0, 1, 2 or 3, with probabilities
  0.5, 0.3, 0.15, 0.05;
- the mass of each PISN: between 140 and 270 M☉;
- the number of Pop III events: a Poisson draw with a rate drawn log-uniformly
  between 0.1 and 1000;
- a mass and explosion energy for each Pop III event, from the Ishigaki grid;
- `Δ`, the time between Pop III and Pop II formation: 1–100 Myr;
- `t`, the time between Pop II formation and the observed star: 3–40 Myr;
- `M_dil`, the cloud mass: log-uniform 2e5–1e9 M☉ in v20;
- `M_*II`, the mass of the Pop II burst: log-uniform 1e3–1e7 M☉.

There are two caps: the Pop II burst cannot exceed 10% of the cloud, and the
total stellar mass cannot exceed 10% of the cloud either. These keep the model
from putting a huge burst into a tiny cloud.

---

## Step 2: decide what exploded in time

The timeline has three moments:

- Pop III forms at `T = 0`;
- Pop II forms at `T = Δ`;
- the observed star forms at `T = Δ + t`.

A source is in the pre-enrichment gas if it exploded before Pop II formed:

```
lifetime ≤ Δ
```

A source is in the final cloud if it exploded before the observed star formed:

```
lifetime ≤ Δ + t
```

So a long-lived Pop III star can explode after Pop II has formed. It then
enriches the final cloud without raising the metallicity of the gas the Pop II
burst formed from.

Each source has a lifetime:

- Pop III and PISN: the Schaerer (2002) lifetime for that mass;
- Pop II: the Raiteri et al. (1996) lifetime at the cloud metallicity.

For Pop II, the code inverts the lifetime to get the turnoff mass: the lowest
mass that has had time to explode by `t`.

---

## Step 3: compute the ejecta

For each event, the yield table gives a vector of element masses:

```
M_X = mass of element X ejected by that event
```

- PISN: interpolate the table at the event mass.
- Pop III: use the tabulated row for that `(mass, energy)`.
- Pop II: integrate the IMF above the turnoff mass, or draw a stochastic
  compound-Poisson sample. The expected value is the IMF integral.

Isotopes are folded into elements first. Everything is in solar masses.

---

## Step 4: mix them in the bucket

Put one row per source and one column per element:

```
                 C      N      O    ...   Fe   ...
PISN            ...    ...    ...          ...
Pop III         ...    ...    ...          ...
Pop II          ...    ...    ...          ...
SNIa             0      0      0     ...    0
```

Add the rows to get the total ejecta of each element:

```
M_X = M_X(PISN) + M_X(Pop III) + M_X(Pop II)
```

Also keep the pre-enrichment part, which only includes PISN and Pop III events
that exploded before `Δ`:

```
M_X,pre = M_X,pre(PISN) + M_X,pre(Pop III)
```

The total metal mass is the sum over all elements heavier than helium:

```
M_Z,tot = sum of M_X over metals
M_Z,pre = sum of M_X,pre over metals
```

The matrix is kept, not just the totals. Later we need the PISN row to compute
the PISN fraction and to ask what the pattern would look like without the PISN.

---

## Step 5: dilute the mixture

The cloud mass is `M_dil`. The metallicity of the gas the Pop II burst forms
from is

```
Z* = M_Z,pre / M_dil
```

After Pop II adds its own metals, the final metallicity is

```
Z_final = M_Z,tot / M_dil
```

The observed iron abundance is

```
[Fe/H] = log10( M_Fe / (M_dil * solar_Fe) )
```

Two things are happening at once. `M_Fe` is the iron mass from all sources.
`M_dil` is the gas mass that dilutes it. `[Fe/H]` only sees the ratio.

This is the central degeneracy: a large iron mass in a large cloud and a small
iron mass in a small cloud can give the same `[Fe/H]`. The star cannot tell you
which one happened.

---

## Step 6: apply the gates

Not every draw is useful. The gates reject the bad ones:

- **R1**: at least one PISN or Pop III event. Otherwise nothing exploded.
- **R2**: the pre-enrichment metal mass is positive, `Z*` is above a critical
  floor (1e-6), and the cloud can host the Pop II burst under the 10% cap.
- **R3**: iron and at least three feature elements are above the validity floor,
  so their logarithms are defined.
- **R4** (v20 stage 1): the final `[Fe/H]` is between −5 and −1. For the Li+
  application, R4 is the observed Li+ box, −3.8 to −1.7.

The table floor is not a gate. If `Z*` is below the table's lowest node, the
sample is kept and flagged, and the yield lookup uses a clamp or a blend.

---

## Step 7: build the features

For every element, compute its log abundance relative to solar:

```
a_X = log10( M_X / solar_mass_fraction_X )
```

The feature is the ratio to iron:

```
[X/Fe] = a_X − a_Fe
```

So a feature vector is “how much X do I have compared with iron, compared with
the Sun?” The reference is iron. There are 26 ratio features.

There are also four Fe-independent pairs:

```
[Na/Mg], [Co/Ni], [Mn/Cr], [Zn/Mg]
```

They are useful because they do not depend on the iron normalisation.

---

## Step 8: build the labels

Three PISN fractions are stored:

```
f_ZP  = PISN metal mass / total metal mass
f_FeP = PISN iron mass / total iron mass
f_XP  = PISN mass of X / total mass of X
```

`f_ZP` is the main physical label. It is the analogue of Salvadori's `f_PISN`.
`f_FeP` matters because the features are iron-normalised. `f_XP` says which
elements carry the PISN signal.

Two binary labels come from `f_ZP`:

```
y_0.5 = 1 if f_ZP >= 0.5
y_0.8 = 1 if f_ZP >= 0.8
```

These are the “PISN-dominated” labels. They are useful for comparison with the
literature, but they are not directly observable.

---

## Step 9: build the detectability label

This is the more honest label. Remove the PISN from the matrix, keep everything
else, and recompute the pattern:

```
M_X(minus P) = M_X − M_X(PISN)
x_X(minus P) = [X/Fe] from that
```

Now compare the real pattern with the no-PISN pattern:

```
shift_X = x_X − x_X(minus P)
D^2 = sum_X (shift_X / error_X)^2
y = 1 if D > k
```

With `k = 3`, `y = 1` means “removing the PISN moves the pattern by more than
three sigma.” That is something a survey can actually test. A subdominant PISN
that still shifts Zn or Co by more than the error is a positive, even if `f_ZP`
is below 0.5.

Here `x_X` is the observed pattern. When an observation model is supplied, the
code uses the noisy measured ratios and their errors. That makes `y` a
realised detection label: “in this noise draw, is the no-PISN hypothesis
excluded?” Without an observation model, it uses the noiseless pattern. The
v20 label uses the noisy Li+ observation.

Edge cases:

- fewer than four usable ratios: `y = 0` (not enough information);
- removing the PISN leaves zero of some element: `D = infinity`, `y = 1`;
- no PISN at all: `y = 0`.

---

## Step 10: make it look observed

The noiseless pattern is not what a survey sees. The observation layer copies a
real Li+ star:

1. Pick a Li+ record whose `[Fe/H]` is within 0.2 dex of the mock.
2. Copy its element availability mask, per-element errors, `log g`, and S/N.
3. Add Gaussian noise to each element's log abundance, then form the ratios.
4. Apply the Li+ non-detection thresholds for Si, Mn, Co and Zn.
5. Apply the carbon upper-limit rule.
6. Reweight the mocks to the Li+ `[Fe/H]` histogram, separately for giants and
   dwarfs.

The result is a mock that looks like a real Li+ measurement, including missing
elements and upper limits.

---

## Step 11: repeat until you have enough

The generator draws a batch, applies the gates, computes features, labels and
the observation layer, and keeps the accepted rows. It repeats until it reaches
the target sample size or a maximum number of draws.

Each stored row has:

- the feature vector and mask,
- the observed feature vector, flags and errors,
- `f_ZP`, `f_FeP`, `y_0.5`, `y`,
- the event counts, masses, `M_dil`, `Z*`, `M_*II`, `t`, `Δ`,
- the full ejecta matrix.

---

## Step 12: train the machine

Two models are trained:

- a classifier for the chosen label (`y` or `y_0.5`),
- a regressor for `f_ZP`.

The classifier gives a score. The score is calibrated with isotonic regression
on a held-out set, so that a predicted 0.3 means “about 30% of stars like this
are positive.” Then the probability is shifted to the target prevalence:

```
odds_post = odds_pre × [π/(1−π)] / [π_train/(1−π_train)]
```

If the training mix is 40% positive and the real sky is 0.25% positive, a raw
0.8 becomes about 0.015. The ranking does not change; the probability scale
does.

The regressor gives a continuous `expected_f_Z`. It is not a probability and it
is not shifted.

---

## Step 13: apply to a real star

For a real star, you build the same feature vector from its catalogue
abundances. Then:

- the classifier gives a score and a posterior,
- the regressor gives an expected `f_Z`,
- the support gate gives a distance to the nearest mock cloud.

The support distance is

```
d^2 = min over mocks of sum_X ((real_X − mock_X) / real_error_X)^2
```

If `d` is larger than the 99th percentile of the mock-to-mock distances, the
star is “out of model.” That means the simulated clouds are not close to the
star at the catalogue's precision. It does not mean the star is weird. It often
means a yield zero-point is off.

The useful region is in-model and high posterior. A high posterior on an
out-of-model star is not trustworthy.

---

## Step 14: validate the model

The checks we have run:

- **Salvadori Eq. 13**: put potluck on Salvadori's one-PISN, one-Pop-II slice
  and compare. The bookkeeping matches to about 1e-15 dex.
- **Direct yields**: compute `[X/Fe]` at fixed `Z*` and `t` for every table. The
  spread is the yield systematic. Co and Zn are the worst, 1–2.8 dex.
- **Prior equality**: use one `M_dil` prior for every table, so the accepted
  sample is the same. In v20, acceptance is about 0.51 and zero-PISN about 0.37
  for every policy.
- **M_dil sweep**: change the `M_dil` prior and see what moves. The accepted
  sample and the classifier are stable; the absolute `Z*` and the low-Z
  fraction shift.
- **RF transfer**: train on one table, test on another, and leave one table out.
  With equal gates, transfer is 0.90–0.99. The v19 0.62 was selection, not
  yields.
- **Li+ application**: apply the Li+ masks, errors, weights and support. The raw
  support gate flags most stars because of per-element offsets; allowing one
  offset per element brings 91–95% back in model.

---

## A single mock, end to end

Let us put numbers on one hypothetical sample.

1. Draw one PISN of 200 M☉, five Pop III events, `Δ = 20 Myr`, `t = 10 Myr`,
   `M_dil = 1e7 M☉`, `M_*II = 1e4 M☉`.
2. The PISN lifetime is about 2.2 Myr, so it is in both the pre-enrichment and
   the final cloud. Some Pop III events are younger than 20 Myr; one may explode
   between 20 and 30 Myr, so it reaches the final cloud but not the pre-gas.
3. The PISN ejects about 90 M☉ of metals. The Pop III events add a few M☉. The
   pre-enrichment metal mass is about 100 M☉.
4. `Z* = 100 / 1e7 = 1e-5`. This is below the Limongi18 floor, so the yield
   lookup clamps at 3.2e-5 and the sample is flagged.
5. The Pop II burst adds iron and other elements. The final `[Fe/H]` is around
   −2.5.
6. `f_ZP` is the PISN metal mass divided by the total metal mass. If the Pop II
   burst is small, `f_ZP` is large; if it is large, `f_ZP` is small.
7. Remove the PISN. If the pattern shifts by more than three sigma, `y = 1`.
8. The observation layer copies a Li+ record, adds errors, and marks Zn missing.
9. The classifier gives a raw score, say 0.8. Calibration keeps it near 0.8.
   The prior shift to 0.25% turns it into about 0.015.
10. The support distance is, say, 2.8 and the threshold is 3.2, so the star is
    in model. Its posterior is 0.015, so it is not a candidate at a 0.5
    threshold, but it is a valid star the model can describe.

---

## What the model can and cannot tell you

It can tell you:

- whether a star's pattern looks more PISN-like than the alternatives;
- which elements drive that decision (Co, Zn, Na, Ni, Ca, Cr, Ti);
- how the ranking transfers across yield tables;
- how far a real star is from the simulated clouds;
- what `f_ZP` the chemistry would imply, with a spread.

It cannot tell you:

- the cloud mass of an individual star (it is latent);
- the exact PISN fraction from `[Fe/H]` alone (it is degenerate);
- a precise posterior without a prevalence assumption;
- that a star is a PISN descendant just because the pattern is unusual;
- anything outside the yield tables and the one-cloud, instant-mixing model.

---

## Glossary

| term | meaning |
|---|---|
| `M_dil` | effective gas mass that dilutes the ejecta |
| `Z*` | metallicity of the gas the Pop II burst forms from |
| `Z_final` | final metallicity after all sources |
| `[X/Fe]` | log abundance ratio relative to solar |
| `[Fe/H]` | iron abundance relative to solar |
| `f_ZP` | fraction of the metal mass made by the PISN |
| `y_0.5` | `1` if `f_ZP >= 0.5` |
| `y` | `1` if removing the PISN shifts the pattern by more than `k` sigma |
| ejecta matrix | table of source × element masses |
| pre-enrichment | gas that has been polluted before Pop II forms |
| support distance | nearest-mock distance in units of the star's errors |
| prior shift | conversion from the training class balance to the real prevalence |
