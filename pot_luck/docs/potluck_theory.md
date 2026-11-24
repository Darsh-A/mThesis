# Potluck: theory

A gas cloud is enriched by three kinds of stars, then a low-mass star forms from it.
Potluck simulates many such clouds and asks how much of the metal came from
pair-instability supernovae (PISN).

```
T = 0                 T = Δ                    T = Δ + t
Pop III forms    →    Pop II forms        →    star forms
```

Everything below follows the order the code runs it.

---

## 1. Notation

| symbol | meaning | code |
| --- | --- | --- |
| $X$ | element | — |
| $M_X$ | total mass of $X$ in the cloud | `M_tot` |
| $M_X^{(\mathrm{P})}$ | mass of $X$ from PISN | `matrix[:,PISN]` |
| $M_{\rm gas}$ | gas mass that receives the metals | `mdil` |
| $\Delta,\ t$ | the two delays | `delta_myr`, `t_myr` |
| $\tau$ | progenitor lifetime | — |
| $Z_{\rm pre}$ | metallicity of the gas Pop II forms from | `z_star` |
| $f_P$ | PISN share of the metals | `f_zp` |
| $y_{0.5},\ y_D$ | the two labels | `y_f_05`, `y` |

$f_P$ is written $f_Z^{\mathrm{P}}$ in the other documents; it is the same quantity.

---

## 2. Draw the sources

The number of PISN, the number of Pop III stars, the two delays and the gas mass are
all drawn from priors:

$$
\begin{aligned}
n_{\rm P} &\sim \mathrm{Cat}(0.5,\,0.3,\,0.15,\,0.05), && n_{\rm P}\in\{0,1,2,3\},\\
m_i &\sim \mathcal{U}\big[\max(140,m_{\min}),\ \min(270,m_{\max})\big], && i=1\dots n_{\rm P},\\
\lambda &\sim \mathrm{LogUniform}(0.1,\,1000), \qquad n_{\rm III}\sim\mathrm{Poisson}(\lambda),\\
\Delta &\sim \mathrm{LogUniform}(1,\,100)\ \mathrm{Myr}, \qquad t\sim\mathrm{LogUniform}(3,\,40)\ \mathrm{Myr},\\
M_{\rm gas} &\sim \mathrm{LogUniform}\big(2\times10^{5},\,10^{9}\big)\ M_\odot,\\
M_\star &\sim \mathrm{LogUniform}\big(10^{3},\,10^{7}\big)\ M_\odot .
\end{aligned}
$$

$M_\star$ is the stellar mass of the Pop II burst. It is capped so the burst fits in
the cloud:

$$
M_{\star,\rm III}+M_\star \le 0.1\,M_{\rm gas}.
$$

---

## 3. Who has exploded?

An event enters the cloud only if its progenitor died before the relevant epoch:

$$
\tau \le \Delta \quad\Longrightarrow\quad \text{pre-enrichment},
\qquad
\tau \le \Delta + t \quad\Longrightarrow\quad \text{final mixture}.
$$

**Pop III and PISN** use zero-metallicity lifetimes from Schaerer (2002),
interpolated in $\log m$:

| $m\,[M_\odot]$ | 500 | 300 | 200 | 120 | 80 | 60 | 40 | 25 | 15 | 9 | 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\tau\,[\mathrm{Myr}]$ | 1.90 | 2.05 | 2.20 | 2.52 | 3.01 | 3.46 | 3.86 | 6.46 | 10.4 | 20.2 | 61.9 |

**Pop II** uses Raiteri et al. (1996), with $Z=\max(Z_{\rm pre},\,7\times10^{-5})$:

$$
\log_{10}\frac{\tau}{\mathrm{yr}} = a_0 + a_1\log_{10}m + a_2\big(\log_{10}m\big)^2,
$$

$$
\begin{aligned}
a_0 &= 10.13 + 0.07547\,\ell - 0.008084\,\ell^2,\\
a_1 &= -4.424 - 0.7939\,\ell - 0.1187\,\ell^2,\\
a_2 &= 1.262 + 0.3385\,\ell + 0.05417\,\ell^2,
\end{aligned}
\qquad \ell \equiv \log_{10}Z .
$$

---

## 4. Ejecta

### PISN

Yields are tabulated at discrete masses. Each element is interpolated geometrically
when it spans more than three decades, and linearly otherwise:

$$
M_X^{(\mathrm{P})}(m) =
\begin{cases}
\exp\big[\mathrm{lin}_{\ln m}\big(\ln M_X^{(\mathrm{P})}\big)\big], & M_X^{(\mathrm{P})}>0 \ \forall m \ \text{and}\ \max/\min > 10^3,\\[4pt]
\mathrm{lin}_m\big(M_X^{(\mathrm{P})}\big), & \text{otherwise}.
\end{cases}
$$

The first branch is for elements with steep onsets such as Zn and Cu.

### Pop III

Each event is one row of the Ishigaki et al. (2018) grid, used unchanged. Rows are
grouped by progenitor mass and energy; the group fixes $\tau$, and the row inside the
group is drawn uniformly.

### Pop II

The IMF is Larson, normalised per unit stellar mass formed:

$$
\Phi(m) = A\left(\frac{m}{0.35}\right)^{-2.35},
\qquad
\int_{0.1}^{100} m\,\Phi(m)\,\mathrm{d}m = 1 .
$$

Only stars above the turn-off mass have died by time $t$, so the yield per unit
stellar mass is

$$
Y_X(t,Z_{\rm pre}) = \int_{m_{\rm TO}}^{m_{\max}} m_X(m,Z_{\rm pre})\,\Phi(m)\,\mathrm{d}m,
\qquad
\tau(m_{\rm TO}) = t .
$$

Simulation replaces the integral with a sum over mass bins, so the number of exploding
stars is a Poisson draw:

$$
\nu_k = \int_{\text{bin }k\ \cap\ [m_{\rm TO},\,m_{\max}]} \Phi(m)\,\mathrm{d}m,
\qquad
N_k \sim \mathrm{Poisson}\big(M_\star\,\nu_k\big),
\qquad
M_X^{(\mathrm{II})} = \sum_k N_k\, y_{k,X}.
$$

Here $y_{k,X}$ is the IMF-weighted mean yield inside bin $k$, and $y_{k,X}$ is rescaled
so that $\mathbb{E}[M_X^{(\mathrm{II})}] = M_\star Y_X$ exactly.

---

## 5. Mix

Sum the sources, with each event gated by the clocks of §3:

$$
\begin{aligned}
M_X^{(\mathrm{P})} &= \sum_i m_{i,X}\,\mathbf{1}\!\left[\tau_i \le \Delta+t\right],
&
M_{X,\rm pre}^{(\mathrm{P})} &= \sum_i m_{i,X}\,\mathbf{1}\!\left[\tau_i \le \Delta\right],\\[4pt]
M_X^{(\mathrm{III})} &= \sum_j m_{j,X}\,\mathbf{1}\!\left[\tau_j \le \Delta+t\right],
&
M_{X,\rm pre}^{(\mathrm{III})} &= \sum_j m_{j,X}\,\mathbf{1}\!\left[\tau_j \le \Delta\right],\\[4pt]
M_X &= M_X^{(\mathrm{P})} + M_X^{(\mathrm{III})} + M_X^{(\mathrm{II})},
&
M_{X,\rm pre} &= M_{X,\rm pre}^{(\mathrm{P})} + M_{X,\rm pre}^{(\mathrm{III})}.
\end{aligned}
$$

There is no SNIa and no AGB.

Metallicity of the pre-enriched gas, and of the final mixture:

$$
Z_{\rm pre} = \frac{M_{Z,\rm pre}}{M_{\rm gas}},
\qquad
Z_{\rm final} = \frac{M_Z}{M_{\rm gas}},
\qquad
M_Z \equiv \sum_{X \in \text{metals}} M_X .
$$

The cloud is accepted only if something enriched it, Pop II can form, and the pattern
is usable:

$$
n_{\rm P}+n_{\rm III} > 0,
\qquad
Z_{\rm pre} \ge 10^{-6},
\qquad
\frac{M_X}{M_Z} \ge 10^{-10}\ \text{for Fe and at least 3 elements}.
$$

---

## 6. Abundances

Solar values are Asplund et al. (2009):

$$
\kappa_X = 10^{A_X-12}\,\frac{m_X}{m_{\rm H}}\,X_{\rm H,\odot},
\qquad
A_X = \log_{10}\!\left(\frac{N_X}{N_{\rm H}}\right)+12 .
$$

Abundance and ratio:

$$
a_X = \log_{10}\!\left(\frac{M_X}{\kappa_X}\right),
\qquad
\left[\frac{X}{\mathrm{Fe}}\right] = a_X - a_{\mathrm{Fe}} .
$$

The observed iron abundance also depends on the gas mass:

$$
\left[\frac{\mathrm{Fe}}{\mathrm{H}}\right] = \log_{10}\!\left(\frac{M_{\mathrm{Fe}}}{M_{\rm gas}\,\kappa_{\mathrm{Fe}}}\right).
$$

Features are the ratios $[X/\mathrm{Fe}]$ plus four pairs without iron,
$[\mathrm{Na}/\mathrm{Mg}]$, $[\mathrm{Co}/\mathrm{Ni}]$, $[\mathrm{Mn}/\mathrm{Cr}]$,
$[\mathrm{Zn}/\mathrm{Mg}]$.

---

## 7. The PISN share and the labels

The fraction of the metals made by PISN:

$$
f_P = \frac{\sum_{X \in \text{metals}} M_X^{(\mathrm{P})}}{M_Z} \in [0,1].
$$

Two per-element diagnostics are also stored,
$f_{\mathrm{Fe}} = M_{\mathrm{Fe}}^{(\mathrm{P})}/M_{\mathrm{Fe}}$ and
$f_X = M_X^{(\mathrm{P})}/M_X$.

**Label 1: majority.** A deterministic function of $f_P$:

$$
y_{0.5} = \mathbf{1}\!\left[f_P \ge 0.5\right].
$$

**Label 2: detectable.** Remove the PISN and ask whether the pattern moves by more
than the errors:

$$
x_X^{(-P)} = \left[\frac{X}{\mathrm{Fe}}\right]^{(-P)},
\qquad
M_X^{(-P)} = M_X - M_X^{(\mathrm{P})},
$$

$$
\delta_X = x_X - x_X^{(-P)},
\qquad
D^2 = \delta^{\mathsf T}\mathbf{C}^{-1}\delta,
\qquad
y_D = \mathbf{1}\!\left[D>3\right].
$$

Noise enters on elements, not ratios, so the shared iron error correlates every ratio:

$$
\mathbf{C} = \mathrm{diag}\big(\sigma_X^2\big) + \sigma_{\mathrm{Fe}}^2\,\mathbf{1}\mathbf{1}^{\mathsf T},
\qquad
\sigma_X = 0.2,\ \ \sigma_{\mathrm{Fe}} = 0.1 .
$$

With an observation model, $\mathbf{C} = \mathrm{diag}(\sigma_{X,\rm obs}^2)$ instead.

The labels differ. $y_{0.5}=1$ means the PISN made most of the metals; $y_D=1$ means a
survey could tell it was there. A dominant PISN with a common pattern gives
$y_{0.5}=1$, $y_D=0$. A minor PISN that moves Zn beyond the errors gives the reverse.

Edge cases: fewer than four ratios gives $y_D=0$; removing the PISN leaves an element
at zero gives $y_D=1$; no PISN gives both zero.

---

## 8. Make it look observed

A mock copies the errors, missing elements and limits of a real Li+ star. Noise is
added to elements, $a_X \to a_X+\varepsilon_X$ with
$\varepsilon_X \sim \mathcal{N}(0,\sigma_X^2)$, so that

$$
\mathrm{Cov}\big(\hat x_X,\hat x_Y\big) = \delta_{XY}\sigma_X^2 + \sigma_{\mathrm{Fe}}^2 .
$$

Giants ($\log g < 3.5$) get extra carbon scatter of 0.3 dex, added in quadrature.
Carbon and the Si, Mn, Co, Zn non-detections are censored against thresholds measured
from the catalogue itself. Finally the mocks are reweighted to the catalogue's
iron-abundance distribution,

$$
w = \frac{p_{\rm Li}\big([\mathrm{Fe}/\mathrm{H}]\big)}{p_{\rm mock}\big([\mathrm{Fe}/\mathrm{H}]\big)},
\qquad
\frac{1}{20} \le w \le 20 .
$$

---

## 9. Classifier

Inputs are the observed ratios, their flags, their errors, and a giant/dwarf flag.

A tree model is fitted to $y_{0.5}$, isotonic regression calibrates it, and the
probability is shifted from the training mix to the sky:

$$
\frac{p'}{1-p'} = \frac{p}{1-p}\cdot
\frac{\pi/(1-\pi)}{\pi_{\rm train}/(1-\pi_{\rm train})},
\qquad
\pi = 2.5\times10^{-3}.
$$

The shift is monotonic, so the ranking is unchanged. A calibrated 0.8 becomes about
0.015, because PISN-dominated stars are about one in 400.

The classifier output and the fraction are different things:

$$
p' = P\big(f_P \ge 0.5 \mid \text{data}\big)
= \int_{0.5}^{1} p(f_P\mid\text{data})\,\mathrm{d}f_P,
\qquad
\hat f_P = \mathbb{E}\big[f_P\mid\text{data}\big].
$$

$p'$ is the tail mass above the cut, $\hat f_P$ is the mean. A second tree model
estimates $\hat f_P$ and is not prior-shifted, because it is not a class probability.

A real star is also checked against the mocks. If it sits further than the 99th
percentile of the mock-to-mock distribution in the core elements, it is out of model:

$$
d^2 = \min_i \sum_{X \in \{\mathrm{Na},\mathrm{Mg},\mathrm{Ca},\mathrm{Ti},\mathrm{Cr},\mathrm{Ni}\}}
\left(\frac{x_X - x_{i,X}}{\sigma_X}\right)^2 .
$$

The reweighting of §8 is summarised by the Kish effective sample size,
$N_{\rm eff} = (\sum w)^2/\sum w^2$.

---

## 10. Worked example

One PISN contributes $40\,M_\odot$ of metals and the Pop II burst adds
$60\,M_\odot$, in a cloud of $M_{\rm gas}=10^{6}\,M_\odot$. Then

$$
f_P = \frac{40}{40+60} = 0.40 \quad\Longrightarrow\quad y_{0.5}=0,
\qquad
Z_{\rm final} = \frac{100}{10^{6}} = 10^{-4}.
$$

The star is a near-threshold negative: not PISN-dominated, but a mixture the
classifier will find harder than a clean zero.

---

## 11. Defaults

| quantity | value |
| --- | --- |
| $Z_\odot$ | $0.01345$ |
| $X_{\rm H,\odot}$ | $0.7381$ |
| IMF | $(m/0.35)^{-2.35}$, $0.1$–$100\,M_\odot$ |
| feature elements | 27, reference Fe |
| validity floor | $10^{-10}$ of $M_Z$ |
| minimum features | 3 |
| $M_{\rm gas}$ | $2\times10^{5}$–$10^{9}\,M_\odot$ |
| Pop II grid | 4 $Z$ nodes (Limongi18), 24 (others); 25 in $t$ |
| detection noise | $\sigma_X=0.2$, $\sigma_{\rm Fe}=0.1$, $k=3$ |
| target prevalence | $2.5\times10^{-3}$ |

Yields: PISN from Heger & Woosley (2002), Pop III from Ishigaki et al. (2018), Pop II
from Limongi & Chieffi (2018) or Nomoto et al. (2013), with WW95 as a transfer test.
Below a table's lowest nonzero metallicity the yield is clamped to that node or blended
towards $Z=0$, and the sample is flagged.

Isotopes are folded into elements before use, with radioactive species routed to their
stable daughter at $t_{\rm decay}=1$ Myr, $f=1-2^{-t_{\rm decay}/t_{1/2}}$. WW95 iron
is scaled by 0.5 on input.

Metallicity interpolation differs by axis. Limongi18 is linear in $[\mathrm{Fe}/\mathrm{H}]$
over $-3,-2,-1,0$, and its low-$Z$ models are $\alpha$-enhanced, so the map is not
solar-scaled:

$$
[\mathrm{Fe}/\mathrm{H}] =
\begin{cases}
\log_{10}\!\big(Z/3.236\times10^{-2}\big), & Z \le 3.236\times10^{-3},\\[3pt]
\dfrac{\log_{10}Z - \log_{10}3.236\times10^{-3}}{\log_{10}Z_\odot - \log_{10}3.236\times10^{-3}} - 1, & 3.236\times10^{-3}<Z<Z_\odot,\\[8pt]
0, & Z \ge Z_\odot .
\end{cases}
$$

Nomoto13 and CK20 interpolate in $\log Z$ between positive nodes, and linearly in $Z$
across the $Z=0$ segment. Pop III has no metallicity axis.

---

## 12. Salvadori limit

Set $n_{\rm III}=0$, one PISN, and a deterministic burst with

$$
M_\star = \beta\,\frac{M_Z^{(\mathrm{P})}}{Y_Z^{\rm II}},
\qquad
\beta = \frac{1-f_{\rm PISN}}{f_{\rm PISN}} .
$$

The gas mass cancels, leaving Salvadori et al. (2019) Eq. 13:

$$
\left[\frac{X}{\mathrm{Fe}}\right] =
\log_{10}
\frac{Y_X^{\mathrm{P}} + \beta\,(Y_Z^{\mathrm{P}}/Y_Z^{\rm II})\,Y_X^{\rm II}}
{Y_{\mathrm{Fe}}^{\mathrm{P}} + \beta\,(Y_Z^{\mathrm{P}}/Y_Z^{\rm II})\,Y_{\mathrm{Fe}}^{\rm II}}
- \log_{10}\frac{\kappa_X}{\kappa_{\mathrm{Fe}}},
$$

with $f_P \equiv f_{\rm PISN}$ in this slice. The identity is reproduced to machine
precision, which validates the bookkeeping, not the yield tables.
