# 1. The big picture

## 1.1 The question

Pair-instability supernovae (PISN) are stars of roughly $140$–$270\,M_\odot$ that
explode completely and leave no remnant. They leave a distinctive chemical
fingerprint in the gas around them. The question potluck is built around is:

> Given the abundance pattern of a metal-poor star, can we tell whether a PISN
> contributed to the gas that star formed from?

We never observe the explosion itself. We observe only the pattern frozen into a
long-lived star. Several different enrichment histories can produce similar
patterns, so this is an inference problem, not a lookup.

## 1.2 Notation used throughout

| symbol | meaning |
| --- | --- |
| $X$ | a chemical element |
| $\mathcal{Z}$ | the metals, i.e. all $X \neq \mathrm{H}, \mathrm{He}$ |
| $M_X$ | mass of element $X$ in the cloud, in $M_\odot$ |
| $M_X^{(s)}$ | mass of $X$ contributed by source $s$ |
| $M_{\mathrm{dil}}$ | dilution mass of the cloud |
| $\kappa_X$ | solar mass fraction of $X$ |

The two abundance quantities we keep coming back to are the log abundance and the
ratio against a reference element:

$$
a_X = \log_{10}\!\left(\frac{M_X}{\kappa_X}\right),
\qquad
\left[\frac{X}{Y}\right] = a_X - a_Y .
$$

Throughout, the reference element is iron, so the features are
$[X/\mathrm{Fe}]$. The quantity $[\mathrm{Fe}/\mathrm{H}]$ is a derived diagnostic
and is never used as a feature.

## 1.3 The bucket picture

The whole model fits in one sentence:

> A gas cloud is a bucket. Each source pours element masses into it. The bucket is
> diluted to a total mass $M_{\mathrm{dil}}$, and the observed star freezes in the
> mixture.

Three source classes are implemented:

- **PISN** — pair-instability supernova, $140$–$270\,M_\odot$ progenitors.
- **Pop III CCSN** — core-collapse supernovae from the first stars.
- **Pop II CCSN** — a single burst of ordinary, slightly enriched stars.

The ejecta are stored as a matrix with one row per source and one column per
element:

$$
\mathbf{M} \in \mathbb{R}_{\ge 0}^{4 \times N_X},
\qquad
s \in \{\mathrm{PISN},\ \mathrm{Pop\,III},\ \mathrm{Pop\,II},\ \mathrm{SN\,Ia}\}.
$$

The SN Ia row exists but is always zero; there is no SN Ia source yet.

Storing the full matrix, rather than only the column sums, is the single most
important design choice. It lets us read off both the PISN share of each element
and the counterfactual "what would this cloud look like without the PISN". Neither
can be recovered from the totals alone.

## 1.4 Why a simulator instead of a forward fit

There are two ways to attack the question.

**Forward fit** (Salvadori et al. 2019). Choose a small set of free parameters,
compute the pattern analytically, and fit it to a real star. Interpretable, but the
parameter space is tiny.

**Simulator + classifier** (potluck). Generate thousands of mock clouds with known
answers, then train a model to recover the answer from the pattern.

The decisive advantage of the second approach is that it has a **negative class**.
In potluck, a substantial fraction of accepted mocks contain no PISN at all. In the
v20 runs that fraction is

$$
f(n_P = 0) \approx 0.368 ,
$$

measured directly from the stored datasets. Salvadori's grid only covers PISN
fractions $f_{\mathrm{PISN}} \ge 0.5$, so a model trained on his grid literally
cannot learn what "no PISN" looks like.

The cost of the approach is that everything downstream is only as honest as the
simulator. We are learning the simulator's opinion, not the universe's.

## 1.5 The two labels — and why they are different

Potluck does not have one answer. It has two, and they answer different questions.
This distinction is the most commonly confused part of the model.

### Label A — a property of the cloud

The PISN share of the metal mass:

$$
f_Z^P = \frac{\sum_{X \in \mathcal{Z}} M_X^{(\mathrm{PISN})}}{\sum_{X \in \mathcal{Z}} M_X},
\qquad
y_{0.5} = \mathbf{1}\!\left[f_Z^P \ge 0.5\right].
$$

This is pure bookkeeping. It is easy to predict, but it is not something an
observer can ever verify: nobody can measure the metal mass of a cloud that no
longer exists.

Measured positive rates in v20 (stored datasets):

| dataset | $y_{0.5}$ rate | $y_{0.8}$ rate |
| --- | ---: | ---: |
| `Limongi18_clamp` | 0.2565 | 0.1447 |
| `WW95_native` | 0.2306 | 0.1209 |
| `Nomoto13_CK20_blend` | 0.2098 | 0.1035 |

### Label B — a property of the observation

Delete the PISN row from the matrix and keep everything else fixed:

$$
M_X^{(-P)} = M_X - M_X^{(\mathrm{PISN})},
\qquad
x_X^{(-P)} = \left[\frac{X}{\mathrm{Fe}}\right]^{(-P)} .
$$

The shift between the observed pattern and the no-PISN pattern is

$$
\delta_X = x_X - x_X^{(-P)},
\qquad
D^2 = \delta^{\mathsf T} \mathbf{C}^{-1} \delta,
\qquad
y = \mathbf{1}[D > 3].
$$

The covariance follows from putting noise on elements rather than on ratios, so
that the shared iron error correlates every ratio:

$$
\mathbf{C} = \operatorname{diag}\!\left(\sigma_X^2\right)
+ \sigma_{\mathrm{Fe}}^2 \mathbf{1}\mathbf{1}^{\mathsf T},
\qquad
\sigma_X = 0.2,\ \ \sigma_{\mathrm{Fe}} = 0.1 .
$$

This label is about the observable: *would we notice if the PISN were not there?*

Measured positive rate in v20: $y \approx 0.594$ for `Limongi18_clamp`.

### They disagree, and that is the interesting part

- A dominant PISN whose pattern ordinary CCSN can also produce:
  $y_{0.5} = 1$ but $y = 0$.
- A minor PISN that pushes Zn far outside the errors:
  $y_{0.5} = 0$ but $y = 1$.

In `Limongi18_clamp` the confusion is extreme and one-sided:

| | $y = 1$ | $y = 0$ |
| --- | ---: | ---: |
| $y_{0.5} = 1$ | 5,241 | 4 |
| $y_{0.5} = 0$ | **6,909** | 8,296 |

So 6,909 mocks are detectable but not PISN-dominated, and only 4 are the reverse.
The detectability label is broader and much harder.

## 1.6 The current state, stated plainly

Most production code trains on **label A**. The stored v20 result tables
(AUC, transfer matrices, operating points) are all label A, because every training
site in `diagnostics/v20_mega.py` reads `y_f_05`.

The detectability label (label B) exists, is harder, and is computed with a
positive rate near 0.59. It is the more operationally honest target, and switching
to it is an open decision rather than a result already claimed.

## 1.7 What to remember from this section

1. A cloud is a bucket; sources pour in element masses; dilution sets the scale.
2. The full ejecta matrix is kept, not just the totals.
3. The model exists to give the problem a **negative class**.
4. Two labels: $y_{0.5}$ (did the PISN dominate the metals?) and
   $y = \mathbf{1}[D>3]$ (would we notice?).
5. The headline results are label $y_{0.5}$; the honest operational target is
   $y$.
