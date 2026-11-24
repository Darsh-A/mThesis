# POTLUK: a mixing model for PISN signatures in metal-poor stars

*Working draft. Algorithm and formulas as currently implemented.*

## 1. The idea in plain words

Picture one gas cloud as a bucket. Over a few tens of Myr, three kinds of stars explode and pour their metals into it: a few very large pair-instability supernovae (PISN), many small Pop III core-collapse supernovae (CCSN), and a Pop II burst of ordinary CCSN. A low-mass star that forms from the cloud afterwards inherits the sum. We simulate many such clouds, degrade the abundances the way a real catalogue (Li et al. 2022) would, and train a classifier to recognise when the PISN supplied a large share of the metals. Salvadori et al. (2019) is the special case of exactly one PISN plus Pop II.

## 2. The clock

```
T = 0                  T = Δ                    T = Δ + t
Pop III forms    →     Pop II forms        →    observed star forms

τ ≤ Δ      : the event exploded before Pop II formed → it pre-enriches the gas (sets Z*)
τ ≤ Δ + t  : the event exploded before the observed star formed → it is in the final mix
```

Two delays matter: Δ (Pop III to Pop II) and t (Pop II to the observed star). A progenitor's lifetime τ decides which clock it beats.

## 3. Symbols

| Symbol | Meaning |
| --- | --- |
| `n_P`, `m_P,i` | number and masses of PISN |
| `n_III`, `λ_III` | number and Poisson rate of Pop III CCSN |
| `M*_III`, `M*_II` | stellar mass of Pop III CCSN and of the Pop II burst |
| `M_dil` | gas mass that receives the metals |
| `Δ`, `t` | the two delays |
| `τ_i` | lifetime of event i's progenitor |
| `m_i,X` | mass of element X ejected by event i |
| `M_X^(s)` | total mass of X from source s (P, III, II) |
| `Z*` | metal mass fraction of the gas Pop II forms from |
| `f_Z^P` | PISN share of all metals |
| `y_f` | label: 1 if `f_Z^P ≥ f0` |

## 4. The algorithm

### Step 1. Cloud and clock

pick how long Pop II waits after Pop III, how long the observed star waits after Pop II, and how big the cloud is.

$$
\begin{aligned}
\Delta &\sim \mathrm{LogUniform}(1,\,100)\ \mathrm{Myr} && \text{Pop III to Pop II delay} \\
t &\sim \mathrm{LogUniform}(3,\,40)\ \mathrm{Myr} && \text{Pop II to observed star} \\
M_{\rm dil} &\sim \mathrm{LogUniform}(2\times 10^{5},\,10^{9})\ M_\odot && \text{gas that receives the metals}
\end{aligned}
$$

### Step 2. Draw the sources

a few PISN, a Poisson number of Pop III CCSN, and a Pop II burst small enough to fit in the cloud.

$$
\begin{aligned}
n_{\rm P} &\in \{0,1,2,3\}, \quad p = (0.5,\,0.3,\,0.15,\,0.05) \\
m_{{\rm P},i} &\sim \mathrm{Uniform}(140,\,270)\ M_\odot, \quad i = 1,\dots,n_{\rm P} \\
\log_{10}\lambda_{\rm III} &\sim \mathrm{Uniform}(-1,\,3), \quad n_{\rm III} \sim \mathrm{Poisson}(\lambda_{\rm III}) \\
g &\sim \mathrm{Uniform}\{1,\dots,10\}, \quad \text{row} \sim \mathrm{Uniform}(\text{rows of group } g) \\
M_{\star,\rm III} &= \text{total progenitor mass of the Pop III CCSN} \\
M_{\star,\rm II} &\sim \mathrm{LogUniform}\big(10^{3},\ \min(10^{7},\ 0.1\,M_{\rm dil} - M_{\star,\rm III})\big) \\
&\Rightarrow\ M_{\star,\rm III} + M_{\star,\rm II} \le 0.1\,M_{\rm dil}
\end{aligned}
$$

### Step 3. Who has exploded? (two clocks)

every star explodes after its lifetime. Early explosions enrich the gas Pop II forms from; later ones still reach the observed star.

$$\begin{aligned}
\text{pre-enriching:} \quad & \tau_i \le \Delta \\
\text{in the final mix:} \quad & \tau_i \le \Delta + t
\end{aligned}$$

Lifetimes: Pop III and PISN progenitors use the zero-metallicity lifetimes of Schaerer (2002); Pop II progenitors use Raiteri et al. (1996) at `Z*`.

All events of one Pop III group share a mass, so the gate passes all of them or none.

### Step 4. Pre-enrichment

the metals released before Pop II forms, divided by the cloud mass, give the metallicity of the gas Pop II forms from.

$$
\begin{aligned}
M_{Z,\rm pre} &= \sum_{i:\,\tau_i \le \Delta}\ \sum_{X \in \text{metals}} m_{i,X} \\
Z_\star &= \frac{M_{Z,\rm pre}}{M_{\rm dil}}
\end{aligned}
$$

Yields m\_i,X: PISN from Heger & Woosley (2002) (Takahashi et al. 2018 as an alternative); Pop III CCSN from Ishigaki et al. (2018).

### Step 5. Pop II ejecta

Pop II stars explode in order of lifetime, so at time t only stars above the turn-off mass have exploded. How many of each mass is a Poisson draw.

$$\begin{aligned}
&\tau_{\rm II}(m_{\rm TO},\,Z_\star) = t \\
&\nu_k = \int_{\text{bin } k\,\cap\,[m_{\rm TO},\,m_{\max}]} \Phi(m)\,dm \\
&n_k \sim \mathrm{Poisson}\big(M_{\star,\rm II}\,\nu_k\big) \\
&M_X^{(\rm II)} = \sum_k n_k\, y_{k,X}(Z_\star)
\end{aligned}$$

Here ν\_k is the number of exploded stars per unit stellar mass formed in bin k, and y\_k,X is the yield of X from a star of bin k.

Φ(m) is the Pop II IMF, normalised so that ∫ m Φ dm = 1. Pop II yields: Nomoto, Kobayashi & Tominaga (2013) by default, Limongi & Chieffi (2018) for comparison. Between Z = 0 and Z = 1e-3 the Nomoto yields are interpolated linearly in Z, above that in log Z.

### Step 6. Final mixture

add the masses. Logs come only at the end.

$$\begin{aligned}
M_X^{(\rm P)} &= \sum_{\text{PISN } i:\ \tau_i \le \Delta + t} m_{i,X} \\
M_X^{(\rm III)} &= \sum_{\text{Pop III } i:\ \tau_i \le \Delta + t} m_{i,X} \\
M_X &= M_X^{(\rm P)} + M_X^{(\rm III)} + M_X^{(\rm II)} \\
M_{Z,\rm tot} &= \sum_{X \in \text{metals}} M_X \\
Z_{\rm final} &= \frac{M_{Z,\rm tot}}{M_{\rm dil}} \\
[\mathrm{Fe/H}] &= \log_{10}\frac{M_{\rm Fe}}{M_{\rm dil}\,X_{\odot,\rm Fe}}
\end{aligned}$$

### Step 7. Cuts

drop clouds that are physically impossible or that the catalogue could not have selected.

$$\begin{aligned}
\text{R1:} \quad & n_{\rm P} + n_{\rm III} = 0 && \text{reject: nothing to enrich the cloud} \\
\text{R2:} \quad & M_{Z,\rm pre} = 0 \ \text{ or } \ Z_\star < z_{\rm cr} = 10^{-6} && \text{reject: Pop II cannot form} \\
\text{R3:} \quad & \text{Fe, or fewer than 3 feature elements, with } M_X / M_{Z,\rm tot} < 10^{-10} && \text{reject} \\
\text{R4:} \quad & [\mathrm{Fe/H}]_{\rm obs} \notin [-3.8,\,-1.7] && \text{reject: Li+ selection, applied after Step 9}
\end{aligned}$$

### Step 8. Abundances, features, labels

turn masses into abundance ratios, and record how much of the metals the PISN made.

$$\begin{aligned}
a_X &= \log_{10}\frac{M_X}{X_{\odot,X}}, \qquad [X/\mathrm{Fe}] = a_X - a_{\rm Fe} \\
f_Z^{\rm P} &= \frac{\sum_{X \in \text{metals}} M_X^{(\rm P)}}{M_{Z,\rm tot}} \\
y_f &= \mathbb{1}\left[f_Z^{\rm P} \ge f_0\right], \qquad f_0 \in \{0.5,\ 0.8\}
\end{aligned}$$

Features: \[X/Fe\] for C, Na, Mg, Si, Ca, Ti, Cr, Mn, Co, Ni and Zn, plus \[Na/Mg\], \[Co/Ni\], \[Mn/Cr\] and \[Zn/Mg\]. X⊙,X is the solar mass fraction (Asplund et al. 2009), and f\_Z^P is the PISN share of the metals.

### Step 9. Observation model (Li et al. 2022)

give each mock star the errors, missing elements and selection of a real catalogue star.

$$\begin{aligned}
& \left|[\mathrm{Fe/H}]_r - [\mathrm{Fe/H}]\right| \le 0.2\ \mathrm{dex} && \text{pick a Li+ record } r \\
& [X/\mathrm{Fe}]_{\rm obs} = [X/\mathrm{Fe}] + \epsilon_X, \quad \epsilon_X \sim \mathcal{N}(0,\,\sigma_{X,r}^2) && \text{diagonal errors from } r \\
& [\mathrm{Fe/H}]_{\rm obs} = [\mathrm{Fe/H}] + \mathcal{N}(0,\,e_{{\rm FeH},r}^2) \\
& w = \frac{p_{\text{Li+}}([\mathrm{Fe/H}]_{\rm obs})}{p_{\rm mock}([\mathrm{Fe/H}]_{\rm obs})} && \text{separately for giants and dwarfs}
\end{aligned}$$

Missing elements are copied from record r. C is reported as an upper limit when \[C/H\]\_obs falls below a fitted threshold, and giants get σ\_C inflated by 0.3 dex. Si, Mn, Co and Zn are set missing when \[X/H\]\_obs lies below a lower-envelope threshold versus log SNR.

### Step 10. Inference (in progress)

train on the mocks, then correct for the fact that real PISN stars are rare.

$$\begin{aligned}
\text{classifier:} \quad & p(y_f = 1 \mid x_{\rm obs}) \\
\text{prevalence:} \quad & \mathrm{odds}_{\rm post} = \mathrm{odds}_{\rm train} \times \frac{\pi/(1-\pi)}{\pi_{\rm train}/(1-\pi_{\rm train})} \\
& \pi = 2.5\times 10^{-3}\ (f_0 = 0.5), \qquad \pi = 10^{-3}\ (f_0 = 0.8) \\
\text{regressor:} \quad & f_Z^{\rm P} \mid x_{\rm obs}
\end{aligned}$$

The classifier is gradient-boosted trees with isotonic calibration. The regressor for f\_Z^P is trained on positives and carries conformal intervals. Metrics are the false-positive rate at fixed true-positive rate and precision at π. A nearest-mock distance on Na, Mg, Ca, Ti, Cr and Ni flags stars the model cannot represent.

## 5. Priors at a glance

| Parameter | Prior | Basis |
| --- | --- | --- |
| `n_P` | 0, 1, 2, 3 with 0.5 / 0.3 / 0.15 / 0.05 | assumption |
| `m_P` | Uniform 140 to 270 M☉ | yield-table range |
| `λ_III` | LogUniform 0.1 to 1000 | spans PISN share of metals from 0 to 1 |
| Pop III group weights | uniform over 10 groups | assumption (sensitivity tested) |
| `Δ` | LogUniform 1 to 100 Myr | Pop III lifetimes |
| `t` | LogUniform 3 to 40 Myr | first SNII explodes to all SNII exploded |
| `M_dil` | LogUniform 2e5 to 1e9 M☉ | Salvadori et al. dilution derivation |
| `M*_II` | LogUniform 1e3 up to the cap | stellar mass at most 0.1 of `M_dil` |
| `z_cr` | 1e-6 | critical metallicity, Pop II table floor |
| \[Fe/H\] window | -3.8 to -1.7 | Li+ selection |

## 6. The Salvadori limit

With `n_III = 0`, `n_P = 1`, and a deterministic Pop II of mass `M*_II = β M_Z^P / Y_Z^II`, where `β = (1 − f_PISN)/f_PISN`, the model reduces to his eq. 13:

$$[X/\mathrm{Fe}] = \log_{10}\frac{Y_X^{\rm P} + \beta\,\dfrac{Y_Z^{\rm P}}{Y_Z^{\rm II}}\,Y_X^{\rm II}}{Y_{\rm Fe}^{\rm P} + \beta\,\dfrac{Y_Z^{\rm P}}{Y_Z^{\rm II}}\,Y_{\rm Fe}^{\rm II}} - \log_{10}\frac{X_{\odot,X}}{X_{\odot,\rm Fe}}$$

This is reproduced numerically to 2e-16.

## 7. Status and known limits

- Steps 1 to 9 are implemented. Training, calibration and validation (Step 10) are in progress.
- One cloud, instant perfect mixing, one Pop II burst, as in Salvadori et al. No SNIa or AGB, which is why t stops at 40 Myr.
- Yield tables dominate the systematics. Against Li+, mock \[Ti/Fe\] is about 0.6 to 0.7 dex too low for every table tried, and \[Na/Fe\] about 0.5 dex.
- Low-metallicity coverage: Nomoto13 has no positive node below Z = 1e-3 and Limongi18 none below 3.2e-5, so most mock Z\* values need interpolation or clamping.
- The priors on `n_P`, `n_III` and `M_dil` are independent, not tied to one IMF.
- Non-detections for elements other than C are approximated with a threshold.
- No real-star validation yet; all performance numbers so far are internal to the simulator.
