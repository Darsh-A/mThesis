# Potluck: algorithm reference

Generator of metal-poor stellar abundance patterns from a mixed enrichment history.
Everything below is the algorithm implemented in `src/potluck`.

---

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | chemical element |
| $\mathcal{F}$ | feature elements (27), reference $R=\mathrm{Fe}$ |
| $\mathcal{Z}$ | metals ($X\neq$ H, He) |
| $M_X^{(s)}$ | ejected mass of $X$ by source $s$, in $M_\odot$ |
| $M_\odot$ | one solar mass |
| $M_{\rm dil}$ | dilution mass of the cloud |
| $t$ | time between Pop II formation and the observed star |
| $\Delta$ | time between Pop III formation and Pop II formation |
| $\kappa_X$ | solar abundance of $X$ (defined in §1) |

**Idea.** A gas cloud is a bucket. Each source pours element masses into it. The bucket is then diluted to a total mass $M_{\rm dil}$, and the observed star freezes in the mixture.

---

## 1. Abundances

Solar abundance (Asplund et al. 2009):

$$
A_X=\log_{10}\!\left(\frac{N_X}{N_H}\right)+12
$$

Solar mass fraction (used everywhere below):

$$
\kappa_X \equiv X_{\odot,X}
=10^{\,A_X-12}\,\frac{m_X}{m_H}\,X_{H,\odot},
\qquad X_{H,\odot}=0.7381 .
$$

Elemental log abundance and abundance ratio:

$$
a_X=\log_{10}\!\left(\frac{M_X}{\kappa_X}\right),
\qquad
\left[\frac{X}{Y}\right]=a_X-a_Y .
$$

`[X/Fe]` is the feature. `[Fe/H]` is a derived diagnostic, never a feature.

---

## 2. Yield tables: one event → one ejecta vector

Isotopes are first folded into elements:

$$
M_X=\sum_{i\in X} m_i .
$$

All sources use the same 32-element vector in the fixed order of `constants.ELEMENTS`.

### 2.1 PISN (HW2002 by default, or Takahashi)

Ejecta are tabulated at discrete progenitor masses. One event of mass $m_P$ is drawn by interpolating the table:

$$
M_X^{(P)}(m_P)=\mathcal{I}_X[m_P],
\qquad
m_P\in[m_{\min},m_{\max}].
$$

Per element, interpolation is

$$
\mathcal{I}_X[m]=
\begin{cases}
\exp\!\big[\text{lin-interp}(\ln m)\big], & M_X>0\ \forall m\ \text{and}\ \max M_X/\min M_X>10^3,\\[2pt]
\text{lin-interp}(m), & \text{otherwise}.
\end{cases}
$$

**Idea.** Smooth elements are interpolated linearly; threshold elements that jump orders of magnitude (Zn, Cu, …) are interpolated geometrically. Endpoints are clamped outside the grid.

### 2.2 Pop III CCSN (Ishigaki et al. 2018)

The table is a discrete set of rows, each one event:

$$
\big(m_{i},\ E_{i},\ M_{X,i}^{(III)}\big)_{i=1}^{N_{\rm row}} .
$$

Rows are grouped by $(m,E)$, giving $G$ groups with a shared lifetime. One event = one tabulated row, used verbatim.

### 2.3 Pop II CCSN (Limongi18 / Nomoto13 / WW95)

A single burst of total stellar mass $M_{\star II}$. The Larson IMF,

$$
\phi(m)=A\left(\frac{m}{0.35}\right)^{-2.35},
\qquad
A^{-1}=\int_{0.1}^{100} m\,\phi(m)\,dm,
$$

is mass-normalized: $\int m\phi\,dm=1$.

The IMF-averaged yield above the turnoff mass $m_{\rm TO}$ is

$$
Y_X^{II}(t,Z)=\int_{m_{\rm TO}(t,Z)}^{m_{\max}} m_X(m,Z)\,\phi(m)\,dm .
$$

**Stochastic sampling.** The integral is replaced by a compound-Poisson sum over mass bins $k$:

$$
\nu_k=\int_{\text{bin }k}\phi(m)\,dm,
\qquad
n_k\sim\mathrm{Poisson}\!\left(M_{\star II}\,\nu_k\right),
\qquad
M_X^{II}=\sum_k n_k\,y_{X,k}.
$$

Here $y_{X,k}$ is the IMF-weighted mean yield in bin $k$, rescaled by a factor in $[0.5,2]$ so that $\mathbb{E}[M_X^{II}]=M_{\star II}Y_X^{II}$ exactly. Setting `popii_stochastic=False` returns $M_{\star II}Y_X^{II}$ directly.

The $(Z,t)$ grid is log–log linear interpolation, clamped to the tabulated range.

---

## 3. Prior on the enrichment history

One sample draws:

$$
n_P\sim\mathrm{Cat}(0.5,\,0.3,\,0.15,\,0.05),
\qquad n_P\in\{0,1,2,3\},
$$

$$
m_{P,i}\sim U\!\left[\max(140,m_{\min}),\ \min(270,m_{\max})\right],
$$

$$
\lambda_{III}\sim\mathrm{LogU}(0.1,1000),
\qquad
n_{III}\sim\mathrm{Poisson}(\lambda_{III}),
$$

$$
\text{event }j\in\text{group }g\sim\mathrm{Cat}(\pi_g),
\qquad
\pi_g\propto
\begin{cases}
1,&\text{uniform over groups},\\
N_{\rm row}(g),&\text{row-sampling},\\
w_g,&\text{user weights}.
\end{cases}
$$

Each Pop III event then picks one row uniformly inside its group.

$$
M_{\star III}=\sum_j m_{III,j},
$$

$$
\Delta\sim\mathrm{LogU}(1,100)\ \mathrm{Myr},
\qquad
t\sim\mathrm{LogU}(3,40)\ \mathrm{Myr},
$$

$$
M_{\rm dil}\sim p(\ln M)\propto M^{-\beta}
\quad\text{on }[M_{\rm dil,lo},10^9]\,M_\odot,
\qquad \beta=0\ \text{default (log-uniform)},
$$

$$
M_{\star II}\sim\mathrm{LogU}(10^3,\ 10^7)\ M_\odot .
$$

Two caps tie the stellar budget to the cloud:

$$
M_{\rm dil,lo}=\max\!\left(2\times10^5,\ \frac{M_{\star III}+10^3}{0.1}\right),
$$

$$
M_{\star II}\le\min\!\left(10^7,\ 0.1\,M_{\rm dil}-M_{\star III}\right).
$$

The joint prior is the product of all the above densities (stored as `log_prior`).

---

## 4. Clocks and the two enrichment gates

Three epochs: Pop III at $T=0$, Pop II at $T=\Delta$, observed star at $T=\Delta+t$.

### Lifetimes

Pop III (Schaerer 2002, log–log interpolation):

$$
\tau_{III}(m)=\mathrm{interp}\big[\log m;\ \log \tau\big],
$$

with table $(m,M_\odot)=(500,400,300,200,120,80,60,40,25,15,9,5)$ and $\tau/\mathrm{Myr}=(1.899,1.974,2.047,2.204,2.521,3.012,3.464,3.864,6.459,10.40,20.22,61.90)$.

PISN events use the same $\tau_{III}(m_P)$ as their clock.

Pop II (Raiteri et al. 1996), with $z=\max(Z,7\times10^{-5})$:

$$
\log_{10}\tau_{II}(m,Z)=a_0+a_1\log_{10}m+a_2(\log_{10}m)^2\ \ [\mathrm{yr}],
$$

$$
\begin{aligned}
a_0&=10.13+0.07547\log Z-0.008084(\log Z)^2,\\
a_1&=-4.424-0.7939\log Z-0.1187(\log Z)^2,\\
a_2&=1.262+0.3385\log Z+0.05417(\log Z)^2 .
\end{aligned}
$$

The turnoff mass solves the lifetime inverse:

$$
\tau_{II}(m_{\rm TO},Z)=t
\quad\Longrightarrow\quad
m_{\rm TO}(t,Z),
$$

clamped to $[0.1,m_{\max}]$.

### Gates

An event pre-enriches the Pop II gas only if it exploded before Pop II formed, and reaches the observed star only if it exploded before the star formed:

$$
\text{pre-enrichment:}\quad \tau_i\le\Delta,
\qquad
\text{final mix:}\quad \tau_i\le\Delta+t .
$$

**Idea.** $\Delta$ sets which Pop III events are already in the bucket when Pop II forms; $\Delta+t$ sets which ones are in the bucket when the observed star forms. A long-lived Pop III CCSN can therefore enrich the final mix without raising $Z_\star$.

---

## 5. Ejecta matrix and totals

One row per source, one column per element:

$$
\mathbf{M}\in\mathbb{R}_{\ge0}^{4\times N_X},
\qquad
s\in\{\mathrm{PISN},\mathrm{PopIII},\mathrm{PopII},\mathrm{SNIa}\}.
$$

$$
M_X^{(P)}=\sum_i M_{X,i}^{(P)}\,\mathbf{1}\!\left[\tau_{III}(m_{P,i})\le\Delta+t\right],
$$

$$
M_{X,\rm pre}^{(P)}=\sum_i M_{X,i}^{(P)}\,\mathbf{1}\!\left[\tau_{III}(m_{P,i})\le\Delta\right],
$$

$$
M_X^{(III)}=\sum_j M_{X,j}^{(III)}\,\mathbf{1}\!\left[\tau_{III}(m_{III,j})\le\Delta+t\right],
$$

$$
M_{X,\rm pre}^{(III)}=\sum_j M_{X,j}^{(III)}\,\mathbf{1}\!\left[\tau_{III}(m_{III,j})\le\Delta\right],
$$

$$
M_X^{(II)}=\text{compound-Poisson draw at }(Z_\star,t,M_{\star II}),
\qquad
M_X^{(Ia)}=0 .
$$

Totals:

$$
M_X=\sum_s M_X^{(s)}=\left(\mathbf{1}^{\!\top}\mathbf{M}\right)_X,
$$

$$
M_{X,\rm pre}=M_{X,\rm pre}^{(P)}+M_{X,\rm pre}^{(III)},
$$

$$
M_Z^{\rm tot}=\sum_{X\in\mathcal{Z}}M_X,
\qquad
M_Z^{\rm pre}=\sum_{X\in\mathcal{Z}}M_{X,\rm pre}.
$$

**Idea.** Keep the matrix, not just the total: the PISN share of every element, and the counterfactual without PISN, are read off rows of $\mathbf{M}$.

---

## 6. Metallicity and dilution

$$
Z_\star=\frac{M_Z^{\rm pre}}{M_{\rm dil}},
\qquad
Z_{\rm final}=Z_\star\,\frac{M_Z^{\rm tot}}{M_Z^{\rm pre}}
= \frac{M_Z^{\rm tot}}{M_{\rm dil}} .
$$

Since $M_Z^{\rm tot}\ge M_Z^{\rm pre}$, $Z_{\rm final}\ge Z_\star$ automatically.

$$
\left[\mathrm{Fe/H}\right]
=\log_{10}\!\left(\frac{M_{\rm Fe}}{M_{\rm dil}\,\kappa_{\rm Fe}}\right).
$$

**Idea.** $Z_\star$ is the metallicity of the gas the Pop II burst forms from. The observed metallicity includes what Pop II adds later. [Fe/H] is derived, not drawn.

---

## 7. Selection cuts

Applied in this order, with separate rejection counts:

$$
\begin{aligned}
R_1:\quad & n_P+n_{III}>0 &&\text{(something must explode)}\\
R_2:\quad & R_1\ \wedge\ M_Z^{\rm pre}>0\ \wedge\ Z_{\rm cr}\le Z_\star\le Z_{\max} &&\text{(dilution admissible)}\\
R_3:\quad & R_2\ \wedge\ \frac{M_{\rm Fe}}{M_Z^{\rm tot}}\ge\epsilon_{\rm Fe}
\ \wedge\ \#\!\left\{X\in\mathcal{F}:\frac{M_X}{M_Z^{\rm tot}}\ge\epsilon_X\right\}\ge n_{\rm feat}^{\min} &&\text{(logs defined)}\\
R_4:\quad & R_3\ \wedge\ \left[\mathrm{Fe/H}\right]_{\rm obs}\in[-3.8,-1.7] &&\text{(observed box)}
\end{aligned}
$$

Defaults: $Z_{\rm cr}=10^{-6}$, $Z_{\max}=\infty$, $\epsilon_X=\epsilon_{\rm Fe}=10^{-10}$, $n_{\rm feat}^{\min}=3$.

$R_4$ is applied after the observation layer, because it uses the measured [Fe/H]. Whether an element counts as valid is relative to $M_Z^{\rm tot}$, so structural zeros become `NaN` rather than rejecting the sample.

---

## 8. Features and labels

### Features

$$
x_X=\left[\frac{X}{\mathrm{Fe}}\right]=a_X-a_{\rm Fe},
\qquad X\in\mathcal{F}\setminus\{\mathrm{Fe}\}.
$$

Masked invalid if either $X$ or Fe is below the validity floor.

Fe-independent pair features:

$$
\left[\frac{\mathrm{Na}}{\mathrm{Mg}}\right],\
\left[\frac{\mathrm{Co}}{\mathrm{Ni}}\right],\
\left[\frac{\mathrm{Mn}}{\mathrm{Cr}}\right],\
\left[\frac{\mathrm{Zn}}{\mathrm{Mg}}\right].
$$

### Labels

$$
f_Z^P=\frac{\sum_{X\in\mathcal{Z}}M_X^{(P)}}{M_Z^{\rm tot}},
\qquad
f_{\rm Fe}^P=\frac{M_{\rm Fe}^{(P)}}{M_{\rm Fe}},
\qquad
f_X^P=\frac{M_X^{(P)}}{M_X}.
$$

$$
y_{0.5}=\mathbf{1}\!\left[f_Z^P\ge0.5\right],
\qquad
y_{0.8}=\mathbf{1}\!\left[f_Z^P\ge0.8\right].
$$

$f_Z^P$ is the regression target ($\approx$ Salvadori's $f_{\rm PISN}$); $f_{\rm Fe}^P$ is a diagnostic because features are Fe-normalized.

---

## 9. Counterfactual detectability label

Remove the PISN row and keep everything else fixed:

$$
M_X^{(-P)}=M_X-M_X^{(P)},
\qquad
x_X^{(-P)}=\left[\frac{X}{\mathrm{Fe}}\right]^{(-P)}.
$$

Observed shift and Mahalanobis distance:

$$
\delta_X=x_X-x_X^{(-P)},
\qquad
D^2=\delta^{\mathsf T}\mathbf{C}^{-1}\delta,
\qquad
y=\mathbf{1}[D>k].
$$

When an observation model is supplied, $x_X$ is the **noisy observed** ratio from `observation.ratios`; the diagonal covariance uses the catalogue errors. This is realised detection. Without an observation model, the noiseless branch uses `features.ratios`. The v20 label uses the observed branch with the Li+ 11.

A fixed $k=3$ is a distance cut, not a p-value. With $n$ ratios, $D^2$ follows approximately $\chi^2_n$ under the no-PISN hypothesis, so the same $k$ has a different false-positive rate for different $n$. A p-value threshold is the next calibration step.

Noiseless covariance (shared reference):

$$
\mathbf{C}=\mathrm{diag}\!\left(\sigma_X^2\right)+\sigma_{\rm Fe}^2\,\mathbf{1}\mathbf{1}^{\mathsf T},
\qquad
\sigma_X=0.2,\ \sigma_{\rm Fe}=0.1,\ k=3 .
$$

Observed covariance (ratio errors from the catalogue record):

$$
\mathbf{C}=\mathrm{diag}\!\left(\sigma_{X,\rm obs}^2\right).
$$

Edge cases:

$$
\begin{cases}
n_{\rm used}<4: & D=0,\ y=0 \quad\text{(insufficient)},\\
\exists X:\ M_X^{(-P)}\le0: & D=\infty,\ y=1 \quad\text{(PISN-only element)},\\
\text{no PISN}: & D=y=0 .
\end{cases}
$$

**Idea.** $y$ answers “would I notice if the PISN were not there?”, using only detected ratios and their real errors. A large $f_Z^P$ whose pattern other sources can mimic is still $y=0$.

---

## 10. Observation model

Steps 1–9 are noiseless. This layer makes a mock look like a real Li+ star.

### 10.1 Noise on elements, not ratios

$$
\hat a_X=a_X+\epsilon_X,
\qquad
\epsilon_X\sim\mathcal{N}(0,\sigma_X^2),
\qquad
\hat x_X=\hat a_X-\hat a_{\rm Fe}.
$$

Ratio covariance inherits the shared reference:

$$
\mathrm{Cov}(\hat x_X,\hat x_Y)=\delta_{XY}\sigma_X^2+\sigma_{\rm Fe}^2 .
$$

Here the per-element errors are taken from a real catalogue record, not fixed constants.

### 10.2 Record sampling

Pick a Li+ record $r$ with

$$
\left|[\mathrm{Fe/H}]_r-[\mathrm{Fe/H}]_{\rm true}\right|\le0.2\ \mathrm{dex},
$$

then

$$
[\mathrm{Fe/H}]_{\rm obs}=[\mathrm{Fe/H}]_{\rm true}
+\mathcal{N}(0,e_{{\rm FeH},r}).
$$

The record supplies the availability mask, $\sigma_X$, $\log g$, and S/N.

### 10.3 Availability, censoring, limits

*Availability* is copied from the record (missing → `NaN`, no information).
*Censoring* depends on the absolute abundance $[\mathrm{X/H}]=[\mathrm{X/Fe}]+[\mathrm{Fe/H}]$.

Non-detection thresholds for Si, Mn, Co, Zn are a monotone lower envelope in S/N:

$$
\tau_X(\mathrm{S/N})=\text{piecewise-linear, non-increasing},
$$

built from the 0th quantile of $[\mathrm{X/H}]$ in 6 log-S/N bins of the catalogue. If

$$
[\mathrm{X/H}]_{\rm true}<\tau_X,
$$

the element is set missing.

Carbon upper limits use the fitted line

$$
\tau_C=\alpha\,\sigma_C+\beta,
$$

fitted to the catalogue's C limits in $[\mathrm{C/H}]$ vs $\sigma_C$. If $[\mathrm{C/H}]_{\rm obs}<\tau_C$,

$$
\hat x_C=\tau_C-[\mathrm{Fe/H}]_{\rm obs},
\qquad \text{flag}=\text{LIMIT}.
$$

Giant stars ($\log g<3.5$) get extra C scatter:

$$
\sigma_C^2\leftarrow\sigma_C^2+0.3^2
\qquad(\text{or C dropped}).
$$

### 10.4 Final observed ratio

$$
\hat x_X=x_X+\mathcal{N}(0,\sigma_X),
\qquad
\text{missing}\Rightarrow\hat x_X=\mathrm{NaN},
$$

with flag $\in\{\text{missing},\text{detected},\text{limit}\}$.

Fe-independent pairs are formed from elements with

$$
\sigma_{X/Y}^2=\sigma_X^2+\sigma_Y^2 .
$$

### 10.5 Importance weights

Mocks are reweighted to the catalogue [Fe/H] histogram, separately for giants and dwarfs:

$$
w_i=\frac{\hat p_{\rm cat}([\mathrm{Fe/H}]_i)}{\hat p_{\rm mock}([\mathrm{Fe/H}]_i)},
\qquad
w_i\leftarrow\mathrm{clip}\!\left(w_i,\frac{1}{20},20\right),
\qquad
\langle w\rangle=1 .
$$

---

## 11. Rejection sampling and dataset assembly

```
for each batch:
    events  = draw_prior()                    # §3
    ejecta  = build_matrix(events)            # §4–5
    keep    = R1 & R2 & R3                    # §7
    obs     = sample_observation(ejecta)      # §10
    keep   &= R4                              # observed [Fe/H] box
    labels  = detect(ejecta, obs)             # §9
    store rows
until n_accepted == n_target or n_drawn == 20*n_target
```

Stored per sample: features, masks, observation columns, events, ejecta totals, labels, prior density, and the full matrix. Metadata records the config hash, git commit, cut definitions, and the rejection ledger.

---

## 12. Post-generation calibration and scoring

This is not part of the generator but is the intended downstream use.

### Design matrix

$$
\mathbf{z}_i=\Big(
\hat x_X,\ \text{flag}_X,\ \log_{10}\sigma_X,\ \mathbf{1}[\text{giant}],\
[\mathrm{X/Y}],\ \sigma_{X/Y}
\Big)_i .
$$

Columns that are constant or all-missing are dropped (this is how excluded elements like Ti are removed without changing the stored layout).

### Classifier and calibrator

Fit $p(y_{0.5}=1\mid\mathbf{z})$ (gradient-boosted trees), then isotonic regression on a held-out split. Shift to a target prevalence $\pi$ (default $2.5\times10^{-3}$):

$$
\frac{p'}{1-p'}=\frac{p}{1-p}\,
\frac{\pi/(1-\pi)}{\pi_{\rm train}/(1-\pi_{\rm train})}.
$$

### Support gate

For a real star $r$, with core elements Na, Mg, Ca, Ti, Cr, Ni (at least 4 available),

$$
d_r^2=\min_i\sum_{X\in\rm core}
\left(\frac{x_{r,X}-x_{i,X}}{\sigma_{r,X}}\right)^2,
$$

and the star is “out of model” if $d_r$ exceeds the 99th percentile of the mock-vs-mock distribution.

### Regression

A second tree model predicts $f_Z^P$ from $\mathbf{z}$.

---

## 13. Default configuration (summary)

| Quantity | Default |
| --- | --- |
| PISN counts | $(0.5,0.3,0.15,0.05)$ for $n_P=0,1,2,3$ |
| PISN mass | $140$–$270\,M_\odot$ |
| $\lambda_{III}$ | $\mathrm{LogU}(0.1,1000)$ |
| $\Delta$ | $\mathrm{LogU}(1,100)$ Myr |
| $t$ | $\mathrm{LogU}(3,40)$ Myr |
| $M_{\rm dil}$ | $2\times10^5$–$10^9\,M_\odot$, log-uniform |
| $M_{\star II}$ | $10^3$–$10^7\,M_\odot$ |
| Stellar cap | $M_{\star III}+M_{\star II}\le0.1\,M_{\rm dil}$ |
| $Z_{\rm cr}$ | $10^{-6}$ |
| Feature elements | 27, reference Fe |
| Validity floor | $10^{-10}$ relative to $M_Z^{\rm tot}$ |
| Min valid features | 3 |
| Detection noise | $\sigma_X=0.2$, $\sigma_{\rm Fe}=0.1$, $k=3$ |
| Observed [Fe/H] box | $[-3.8,-1.7]$ |
| Target catalogue | Li+ (12 elements) |
| Support prevalence | $2.5\times10^{-3}$ |
