# 3. The observation layer

Everything in §2 is noiseless. This section turns a clean mock cloud into something
that looks like a real measurement. It lives in `observation.py` and is the layer
that makes the model applicable to a real catalogue at all.

## 3.1 Why a layer is needed

A real metal-poor star does not hand you a clean abundance pattern. It hands you:

- an error bar on each element,
- some elements measured, some absent,
- some elements bounded only from above,
- an $[\mathrm{Fe}/\mathrm{H}]$, a $\log g$, and a signal-to-noise ratio.

If the classifier is trained on clean patterns and then applied to real data, it
has learned a problem that does not exist. So potluck rebuilds each mock *as if it
had been observed by the survey*.

## 3.2 Borrow a record, do not invent one

For each mock, the code selects a real catalogue record whose iron abundance is
close to the mock's true value:

$$
\big|[\mathrm{Fe}/\mathrm{H}]_r - [\mathrm{Fe}/\mathrm{H}]_{\rm true}\big| \le 0.2\ \mathrm{dex}.
$$

It then copies from that record:

- the per-element errors $\sigma_{X,r}$,
- the availability mask,
- the $\log g$ (hence giant or dwarf),
- the $\mathrm{S/N}$.

Borrowing the whole error vector matters. If each $\sigma_X$ were drawn
independently, the correlation between elements that comes from S/N — a noisy
spectrum is noisy in most lines at once — would be destroyed. Sampling the vector
preserves the heteroscedasticity that real data have: Fe and Mg are usually well
measured, Zn is usually worse.

Finally the observed iron abundance is perturbed with the record's own error:

$$
[\mathrm{Fe}/\mathrm{H}]_{\rm obs}
= [\mathrm{Fe}/\mathrm{H}]_{\rm true} + \mathcal{N}\!\left(0, e_{{\rm FeH},r}\right).
$$

## 3.3 Noise goes on elements, not ratios

This is a correctness point, not a convenience.

If independent noise were added to each ratio $[X/\mathrm{Fe}]$, the covariance
would be diagonal. But Fe appears in every ratio, so an error in Fe shifts **all**
of them together. The consistent prescription is to perturb the log abundances
first:

$$
\hat a_X = a_X + \epsilon_X,
\qquad
\epsilon_X \sim \mathcal{N}\!\left(0, \sigma_X^2\right),
\qquad
\hat x_X = \hat a_X - \hat a_{\mathrm{Fe}} .
$$

The induced ratio covariance then follows automatically:

$$
\operatorname{Cov}\!\left(\hat x_X, \hat x_Y\right)
= \delta_{XY}\,\sigma_X^2 + \sigma_{\mathrm{Fe}}^2
\qquad\Longleftrightarrow\qquad
\mathbf{C} = \operatorname{diag}\!\left(\sigma_X^2\right)
+ \sigma_{\mathrm{Fe}}^2\,\mathbf{1}\mathbf{1}^{\mathsf T}.
$$

A diagonal-plus-rank-one structure is exactly what the detection statistic $D$
assumes. Perturbing ratios directly would make the mock data and the statistic
disagree about the noise.

> **Implementation note.** The noiseless detection branch builds the full
> $\operatorname{diag} + \sigma_{\mathrm{Fe}}^2\mathbf{1}\mathbf{1}^{\mathsf T}$
> covariance, while the observed branch uses
> $\operatorname{diag}(\sigma_X^2)$ only (`detect.py`, lines 130–134). The two
> labels therefore do not use the same error model. Worth knowing before
> comparing them.

## 3.4 Three states, encoded as flags

| flag | name | meaning |
| ---: | --- | --- |
| 0 | missing | no information |
| 1 | detected | a real measurement |
| 2 | limit | an upper limit |

The distinction is not cosmetic. An upper limit still carries information: it says
the abundance is *below* something. The flag is fed to the model as an explicit
column, so a tree can split on "this star has no Zn" or "this star's carbon is only
an upper limit" without needing the numeric value.

## 3.5 Availability is not censoring

Two mechanisms are easy to conflate and must not be:

**Availability** is independent of the abundance. An element is absent because the
spectrum or the wavelength coverage did not include it. It is copied from the
sampled record.

**Censoring** depends on the abundance. A line is undetectable when the element is
too weak. The relevant quantity is the *absolute* abundance, not the ratio:

$$
\left[\frac{X}{\mathrm{H}}\right]
= \left[\frac{X}{\mathrm{Fe}}\right] + \left[\frac{\mathrm{Fe}}{\mathrm{H}}\right].
$$

This is the one place where $[\mathrm{Fe}/\mathrm{H}]$ is needed even though it is
not a feature.

## 3.6 Carbon: real limits and giants

Carbon is the only element with genuine upper limits in the catalogue. The code
fits a line to those limits,

$$
\tau_C = \alpha\,\sigma_C + \beta,
$$

fitted in $[\mathrm{C}/\mathrm{H}]$ against $\sigma_C$. If the observed value falls
below the threshold, the reported value becomes the limit itself and the element is
flagged:

$$
[\mathrm{C}/\mathrm{H}]_{\rm obs} < \tau_C
\quad\Longrightarrow\quad
\hat x_C = \tau_C - [\mathrm{Fe}/\mathrm{H}]_{\rm obs},
\qquad \text{flag} = \texttt{LIMIT}.
$$

Giant stars ($\log g < 3.5$) get extra carbon scatter added in quadrature,

$$
\sigma_C^2 \leftarrow \sigma_C^2 + (0.3\ \mathrm{dex})^2,
$$

because evolved stars mix carbon up from the interior and the yield models do not
describe that.

## 3.7 The other non-detections are inferred

No published limits exist for Si, Mn, Co and Zn. The code therefore builds an
empirical detection boundary: take the catalogue's *detected* stars, bin them in
$\log(\mathrm{S/N})$, and take a low quantile of $[\mathrm{X}/\mathrm{H}]$ in each
bin. The result is forced to be non-increasing:

$$
\tau_X(\mathrm{S/N}) \ \text{piecewise linear, non-increasing}.
$$

A mock whose true absolute abundance falls below that envelope is set missing:

$$
\left[\frac{X}{\mathrm{H}}\right]_{\rm true} < \tau_X
\quad\Longrightarrow\quad \text{missing}.
$$

This is a reasonable proxy, but be clear about what it is: **a threshold inferred
from detections, not from non-detections.** The catalogue does not state where its
detection boundary lies, so this is a construction, not a measurement.

## 3.8 Reweighting to the catalogue

Mocks are reweighted so that their iron-abundance distribution matches the
catalogue's, separately for giants and dwarfs:

$$
w_i = \frac{\hat p_{\rm cat}\!\left([\mathrm{Fe}/\mathrm{H}]_i\right)}
{\hat p_{\rm mock}\!\left([\mathrm{Fe}/\mathrm{H}]_i\right)},
\qquad
w_i \leftarrow \operatorname{clip}\!\left(w_i,\ \frac{1}{20},\ 20\right),
\qquad
\langle w \rangle = 1 .
$$

The cost is measured by the Kish effective sample size:

$$
N_{\rm eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

Measured in `v20/application/Limongi18_clamp`: $N_{\rm eff} = 7{,}986$ out of
10,000, with weights spanning 0.06–3.24. So the reweighting costs roughly a fifth
of the sample.

> **Important scope note.** This reweighting exists only in the *application*
> datasets. The stage-1 v20 datasets used for AUC, transfer and operating points
> have all weights equal to one ($N_{\rm eff} = 20{,}450 = N$) and span
> $[\mathrm{Fe}/\mathrm{H}] \in [-4.99, -1.0]$. Do not mix numbers from the two
> sets.

## 3.9 The catalogue, measured

The target is the Li+ catalogue: **385 stars**, 12 elements
(C, Na, Mg, Si, Ca, Ti, Cr, Mn, Fe, Co, Ni, Zn), giving 11 ratios against Fe.

Coverage is very uneven. Measured directly from `load_li_records()`:

| element | detected | upper limits | missing | detected fraction |
| --- | ---: | ---: | ---: | ---: |
| Mg | 385 | 0 | 0 | 1.000 |
| Ca | 384 | 0 | 1 | 0.997 |
| Ti | 380 | 0 | 5 | 0.987 |
| Cr | 369 | 0 | 16 | 0.958 |
| Na | 364 | 0 | 21 | 0.945 |
| Ni | 288 | 0 | 97 | 0.748 |
| C | 281 | 70 | 34 | 0.730 |
| Co | 265 | 0 | 120 | 0.688 |
| Mn | 178 | 0 | 207 | 0.462 |
| Si | 177 | 0 | 208 | 0.460 |
| Zn | 169 | 0 | 216 | **0.439** |

The same unevenness shows up in the mocks, and is made worse by censoring. Finite
values per ratio column in `v20/application/Limongi18_clamp` (10,000 mocks):

| element | finite in mocks | fraction |
| --- | ---: | ---: |
| Mg | 10,000 | 1.000 |
| Ca | 9,976 | 0.998 |
| Ti | 9,825 | 0.983 |
| Cr | 9,530 | 0.953 |
| Na | 9,460 | 0.946 |
| C | 9,050 | 0.905 |
| Ni | 7,521 | 0.752 |
| Co | 5,563 | 0.556 |
| Mn | 4,497 | 0.450 |
| Si | 4,329 | 0.433 |
| Zn | 2,818 | **0.282** |

**Zn is the element that carries the PISN signal (§5), and it is available in only
44 % of the catalogue and 28 % of the mocks.** That is the most important practical
limitation of the whole Li+ application, and it does not appear in the older prose
docs.

Overall flag budget in the application set: 27,431 missing, 80,930 detected,
1,639 upper limits. Giants are 54.5 % of the mocks.

## 3.10 What the layer buys and what it costs

**Buys.** The mock distribution now resembles the catalogue in the ways that
matter — errors, coverage, limits, iron abundance. Without this, applying the
classifier to real stars would be an extrapolation.

**Costs.**

1. It is fitted to *this* catalogue. The carbon line, the Si/Mn/Co/Zn thresholds
   and the error vectors are all Li+-specific. A different survey needs a
   different layer.
2. The mocks inherit the catalogue's observational biases, and the classifier then
   carries that prior into its predictions.
3. It removes information. The median number of detected ratios is 6 out of 11,
   and 2 % of PISN-present mocks fail the $\ge 4$ ratio requirement and are
   labelled negative by construction.

## 3.11 What to remember from this section

1. Mocks borrow a real record, so errors and missingness are realistic.
2. Noise is applied to **elements**; ratios inherit the shared Fe error.
3. Missing, detected and limit are distinct states, and the flag is a feature.
4. Availability is independent of abundance; censoring is not.
5. Only carbon has real limits; the rest of the detection boundary is inferred.
6. Zn — the key element — is present in 44 % of the catalogue and 28 % of mocks.
