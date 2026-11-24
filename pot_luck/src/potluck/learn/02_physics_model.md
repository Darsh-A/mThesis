# 2. The physics model

This section follows `events.py`, `ejecta.py`, `chronology.py` and
`yields/*.py`. Everything here happens *before* any observation or machine
learning.

## 2.1 What one sample is

A single sample is one gas cloud with one enrichment history. Drawing it means
drawing:

$$
\begin{aligned}
n_P &\sim \operatorname{Cat}(0.5,\ 0.3,\ 0.15,\ 0.05),
&& n_P \in \{0,1,2,3\},\\
m_{P,i} &\sim \mathcal{U}\big[\max(140, m_{\min}),\ \min(270, m_{\max})\big],
&& i = 1 \dots n_P,\\
\lambda_{\rm III} &\sim \operatorname{LogU}(0.1,\ 1000),
&& n_{\rm III} \sim \operatorname{Poisson}(\lambda_{\rm III}),\\
\Delta &\sim \operatorname{LogU}(1,\ 100)\ \mathrm{Myr},
&& t \sim \operatorname{LogU}(3,\ 40)\ \mathrm{Myr},\\
M_\star^{\rm II} &\sim \operatorname{LogU}(10^3,\ 10^7)\ M_\odot,
&& M_\star^{\rm III} + M_\star^{\rm II} \le 0.1\,M_{\rm dil}.
\end{aligned}
$$

The dilution mass is drawn from a log-uniform distribution on
$[M_{{\rm dil},\rm lo},\, M_{{\rm dil},\rm hi}]$ with an optional tilt
$p(\ln M) \propto M^{-\beta}$; the default slope is $\beta = 0$.

One documentation trap: `config.py` defaults to
$[10^5,\,5\times10^8]\,M_\odot$, but the v20 runs recorded in
`v20/manifest.json` used $[2\times10^5,\,10^9]\,M_\odot$. Quote the manifest value
when quoting v20 numbers.

The lower dilution bound is itself raised when the cloud would otherwise be too
small to host the stars already drawn:

$$
M_{{\rm dil},\rm lo}
= \max\!\left(M_{{\rm dil},\min},\
\frac{M_\star^{\rm III} + 10^3}{0.1}\right).
$$

## 2.2 Source 1 — the PISN

PISN yields are tabulated at discrete progenitor masses, so an event of any mass
in between is obtained by interpolation:

$$
M_X^{(\mathrm{P})}(m_P) = \mathcal{I}_X[m_P].
$$

The interpolation rule is per element and deliberately not uniform:

$$
\mathcal{I}_X[m] =
\begin{cases}
\exp\!\big[\operatorname{lin}_{\ln m}\!\big(\ln M_X\big)\big],
& M_X > 0\ \forall m\ \text{and}\ \dfrac{\max M_X}{\min M_X} > 10^3,\\[8pt]
\operatorname{lin}_m\!\big(M_X\big),
& \text{otherwise}.
\end{cases}
$$

The first branch is for elements whose yield turns on steeply, such as Zn and Cu.
Interpolating those linearly in mass would invent intermediate values that never
occur; interpolating in the log tracks the onset. Endpoints outside the grid are
clamped.

Each PISN event contributes one such vector, and events are gated independently
(§2.5).

## 2.3 Source 2 — Pop III core-collapse

The Pop III table (Ishigaki et al. 2018) is **not** interpolated. It is a discrete
set of rows, each one a complete event:

$$
\big(m_i,\ E_i,\ M_{X,i}^{(\rm III)}\big)_{i=1}^{N_{\rm row}}.
$$

Rows are grouped by $(m, E)$, because that pair fixes the progenitor lifetime. One
event means:

1. pick a group, with probabilities $\pi_g$ (uniform over groups, proportional to
   row count, or user weights),
2. pick a row uniformly inside that group,
3. use that row's yield vector verbatim.

This matters: the model is sampling individual stars, not an IMF average. Pop III
stars are few and each one contributes a large, discrete amount.

## 2.4 Source 3 — the Pop II burst

This is the only source that is not a set of discrete events. It is a burst of
total stellar mass $M_\star^{\rm II}$, and the question is how much of each element
the burst has ejected by time $t$ after it formed.

**The IMF.** Larson, normalised per unit stellar mass formed:

$$
\phi(m) = A\left(\frac{m}{0.35}\right)^{-2.35},
\qquad
\int_{0.1}^{100} m\,\phi(m)\,\mathrm{d}m = 1 .
$$

The normalisation $A$ is what makes one unit of stellar mass form produce yields
in units of that same mass.

**The turnoff.** At time $t$, only stars massive enough to have died contribute.
The turnoff mass solves the lifetime relation:

$$
\tau_{\rm II}(m_{\rm TO}, Z_\star) = t .
$$

So the IMF-averaged yield is

$$
Y_X^{\rm II}(t, Z_\star)
= \int_{m_{\rm TO}(t,Z_\star)}^{m_{\max}} m_X(m, Z_\star)\,\phi(m)\,\mathrm{d}m .
$$

Light stars have not died yet and contribute nothing. As $t$ grows, $m_{\rm TO}$
falls and more of the IMF contributes.

**The stochastic sampling.** For generation the integral is replaced by a
compound Poisson sum over mass bins $k$:

$$
\nu_k = \int_{\text{bin } k} \phi(m)\,\mathrm{d}m,
\qquad
n_k \sim \operatorname{Poisson}\!\left(M_\star^{\rm II}\,\nu_k\right),
\qquad
M_X^{\rm II} = \sum_k n_k\, y_{X,k}.
$$

Here $\nu_k$ is the number of stars per unit stellar mass in the bin, and
$y_{X,k}$ is the IMF-weighted mean yield inside the bin. The bin yields are then
rescaled by a factor $f_X \in [0.5, 2]$ so that the mean is exact:

$$
\mathbb{E}\!\left[M_X^{\rm II}\right] = M_\star^{\rm II}\,Y_X^{\rm II}(t, Z_\star).
$$

So the expectation is the deterministic integral, and the scatter around it is
physical. The number of CCSN actually drawn is stored per sample in `popii_n`.

## 2.5 The two clocks

This is the part that separates potluck from a single-clock model.

Three epochs:

$$
T = 0 \quad\text{Pop III forms},
\qquad
T = \Delta \quad\text{Pop II forms},
\qquad
T = \Delta + t \quad\text{the observed star forms}.
$$

Two gates, applied to every PISN and Pop III event:

$$
\tau_i \le \Delta \quad\Longrightarrow\quad \text{pre-enrichment},
\qquad
\tau_i \le \Delta + t \quad\Longrightarrow\quad \text{final mix}.
$$

**Lifetimes.** Pop III and PISN events use the zero-metallicity Schaerer (2002)
table, interpolated in $\log m$:

| $m\,[M_\odot]$ | 500 | 400 | 300 | 200 | 120 | 80 | 60 | 40 | 25 | 15 | 9 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\tau\,[\mathrm{Myr}]$ | 1.899 | 1.974 | 2.047 | 2.204 | 2.521 | 3.012 | 3.464 | 3.864 | 6.459 | 10.40 | 20.22 | 61.90 |

Pop II uses Raiteri et al. (1996), with $Z = \max(Z_\star,\,7\times10^{-5})$:

$$
\log_{10}\frac{\tau}{\mathrm{yr}}
= a_0 + a_1\log_{10}m + a_2\big(\log_{10}m\big)^2,
$$

$$
\begin{aligned}
a_0 &= 10.13 + 0.07547\,\ell - 0.008084\,\ell^2,\\
a_1 &= -4.424 - 0.7939\,\ell - 0.1187\,\ell^2,\\
a_2 &= 1.262 + 0.3385\,\ell + 0.05417\,\ell^2,
\end{aligned}
\qquad \ell \equiv \log_{10} Z .
$$

### Why two clocks matter

A $9\,M_\odot$ Pop III star lives about $20\ \mathrm{Myr}$. If $\Delta = 5\
\mathrm{Myr}$ it has **not** exploded when Pop II forms, so it does not raise
$Z_\star$. But if $\Delta + t = 25\ \mathrm{Myr}$ it **has** exploded by the time
the observed star forms, so it dumps metals into the final cloud anyway.

That is a real physical situation a one-clock model cannot express, because its
only pre-enricher is a PISN that dies within a few Myr.

The consequence to remember:

> $Z_\star$ (what Pop II formed from) and $Z_{\rm final}$ (what the observed star
> sees) are different numbers, and $Z_\star$ can sit below the yield table's floor
> while $Z_{\rm final}$ is perfectly reasonable.

## 2.6 Dilution and the central degeneracy

$$
Z_\star = \frac{M_{Z,\rm pre}}{M_{\rm dil}},
\qquad
Z_{\rm final} = Z_\star\,\frac{M_{Z,\rm tot}}{M_{Z,\rm pre}}
= \frac{M_{Z,\rm tot}}{M_{\rm dil}},
$$

$$
\left[\frac{\mathrm{Fe}}{\mathrm{H}}\right]
= \log_{10}\!\left(\frac{M_{\mathrm{Fe}}}{M_{\rm dil}\,\kappa_{\mathrm{Fe}}}\right).
$$

Because the totals include the pre-enrichment sources, $M_{Z,\rm tot} \ge
M_{Z,\rm pre}$ and therefore $Z_{\rm final} \ge Z_\star$ automatically.

**The degeneracy.** $[\mathrm{Fe}/\mathrm{H}]$ depends only on the ratio of iron
mass to gas mass:

$$
M_{\mathrm{Fe}} \uparrow,\ M_{\rm dil} \uparrow
\quad\text{and}\quad
M_{\mathrm{Fe}} \downarrow,\ M_{\rm dil} \downarrow
$$

give the same $[\mathrm{Fe}/\mathrm{H}]$. The star cannot tell which happened.
$M_{\rm dil}$ is latent and must be marginalised over its prior.

This is also why the observable features are **ratios**: $M_{\rm dil}$ cancels in
$[X/\mathrm{Fe}]$, so the pattern is (to first order) dilution-independent.

### The alternative sampling mode

There is a second configuration, `z_star_range`, which flips the logic: draw
$Z_\star$ directly and derive

$$
M_{\rm dil} = \frac{M_{Z,\rm pre}}{Z_\star}.
$$

The advantage is that the Pop II table is always evaluated inside its valid
metallicity range. v19 used this. v20 went back to drawing $M_{\rm dil}$ so that
all yield tables share one prior and the comparison between them is fair.

## 2.7 The selection cuts

Four gates, applied in order:

$$
\begin{aligned}
R_1:\quad & n_P + n_{\rm III} > 0 &&\text{something exploded},\\
R_2:\quad & M_{Z,\rm pre} > 0,\ Z_\star \ge 10^{-6},\
\text{cloud can host the burst} &&\text{dilution admissible},\\
R_3:\quad & \frac{M_{\rm Fe}}{M_{Z,\rm tot}} \ge 10^{-10}
\ \wedge\ \#\!\left\{X:\ \frac{M_X}{M_{Z,\rm tot}} \ge 10^{-10}\right\} \ge 3
&&\text{logs defined},\\
R_4:\quad & [\mathrm{Fe}/\mathrm{H}]_{\rm obs} \in [-3.8,\,-1.7]
&&\text{observed box}.
\end{aligned}
$$

$R_4$ is applied after the observation layer, because it uses the *measured*
iron abundance.

### What actually rejects

From `v20/manifest.json`, `Limongi18_clamp`, out of 40,000 draws:

| gate | rejected | share |
| --- | ---: | ---: |
| $R_1$ | 3,940 | 9.9 % |
| $R_2$ | **14,188** | **35.5 %** |
| $R_3$ | 1 | ~0 % |
| $R_4$ | 1,421 | 3.6 % |
| accepted | 20,450 | 51.1 % |

Two things follow, and both matter.

1. **$R_2$ dominates.** A gate that removes a third of the draws is not a detail.
   The distribution you actually sample is the prior multiplied by an indicator,
   $p_{\rm acc}(\theta) \propto p(\theta)\,\mathbf{1}[\text{accept}]$, so the
   effective prior is **not** the prior you wrote down.
2. **$R_3$ almost never fires.** Structural zeros are mostly a theoretical
   concern; by the time a sample clears $R_2$ it already has metals.

All six yield-table policies select the same sample: acceptance 0.508–0.513 and
zero-PISN fraction 0.365–0.370. That equality is what makes the cross-table
comparison meaningful.

## 2.8 The low-metallicity floor — the biggest physics caveat

Each Pop II yield table stops at a different minimum non-zero metallicity. Below
that floor the lookup has to extrapolate (clamp to the lowest node, or blend
towards $Z = 0$), and the sample is flagged `clamped`.

Measured from the stored v20 datasets:

| policy | lowest node $Z$ | fraction of mocks below it |
| --- | ---: | ---: |
| `WW95_native` | $1.345\times10^{-6}$ | 0.038 |
| `Limongi18_clamp` | $3.236\times10^{-5}$ | 0.520 |
| `Nomoto13_*` / `*_CK20_*` | $10^{-3}$ | 0.967–0.970 |

For the Nomoto and CK20 tables, **97 % of accepted mocks sit below the table's
lowest computed metallicity**. The yield used for those samples is therefore an
extrapolation, not a calculation.

The `clamped` flag in the dataset is exactly the condition

$$
Z_\star < Z_{\rm grid,\,0},
$$

and I verified that its sample mean equals the "below floor" fraction to machine
precision (0.5197 for `Limongi18_clamp`).

Practical consequence: the only table for which this is a marginal issue is WW95
(3.8 %), and WW95 is also the worst at cross-table transfer (§5). The metallicity
range where the physics is actually computed, not extrapolated, is narrow.

## 2.9 What to remember from this section

1. One sample = one cloud = counts, masses, two delays, one dilution mass.
2. PISN interpolates (log for steep elements); Pop III uses rows verbatim;
   Pop II is IMF-integrated then compound-Poisson sampled.
3. Two clocks: $\tau \le \Delta$ pre-enriches, $\tau \le \Delta + t$ reaches the
   final mix. So $Z_\star \ne Z_{\rm final}$, by construction.
4. $M_{\rm dil}$ is latent; $[X/\mathrm{Fe}]$ is the observable because the
   dilution cancels.
5. $R_2$ rejects a third of all draws — the effective prior is not the written
   prior.
6. 97 % of Nomoto/CK20 mocks use extrapolated yields. That is the dominant
   physics uncertainty, not the ML.
