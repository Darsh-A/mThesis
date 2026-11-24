# 6. Likely questions, and honest answers

For the workshop with the Li+ authors. Each item is a question they may ask, the
answer that survives scrutiny, and the reason it survives.

## 6.1 "We don't measure Zn or Co. Does this apply to us?"

This one has a two-part answer, and the second part is the important one.

**In the Li+-free analysis**, where every element is present: alpha elements alone
give AUC $0.863$, adding Zn takes it to $0.966$, dropping Zn from the full set costs
about $0.015$, dropping Co about $0.001$.

**In the deployed Li+ model**, where availability is realistic, Zn is present in only
28 % of mocks. Retrained leave-one-out gives $\Delta\mathrm{AUC} = +0.0001$ for Zn
and $+0.0025$ even restricted to the rows where Zn is measured — against $0.0067$
for Mg and $0.0051$ for Ca.

So the honest line is: *the Zn result describes an idealised survey where all lines
are available. In your catalogue, with your coverage, Zn contributes very little
because you only have it for about half the stars. The discrimination is spread
across Mg, Ca, Cr, Ti and Si.*

Do not lead with "Zn is the key element." It invites the follow-up that Zn is
measured in 44 % of the catalogue, and the answer then has to walk backwards.

## 6.2 "Why is your Ti off by 0.7 dex?"

Answer: **we do not know yet.** Then give the evidence that constrains it:

- Offset is $+0.62$ to $+0.73$ dex across the four v20 tables, and $+0.41$ to
  $+0.73$ across all eight datasets including v19.
- The bootstrap interval is about $\pm 0.01$ dex, so it is not noise.
- There is no $[\mathrm{Fe}/\mathrm{H}]$ trend ($\rho \approx 0.06$), so it is not
  a metallicity-dependent modelling failure.
- It **does not move** when the $Z_\star$ sampling scheme changes, unlike Na, Mg
  and Ca, whose offsets flip sign between v19 and v20.

The last point is the important one: a flat offset that survives every sampling
scheme and every table is a zero-point problem, not a selection artefact. The
candidates are the yield tables, the solar scale, or NLTE.

Do **not** say Li+ Ti is wrong. Say: *our model cannot reproduce the Li+ Ti
zero-point, and we have localised it to a global offset rather than a per-star
anomaly.*

## 6.3 "You report zero candidates. Is the model useless?"

No. Zero candidates is a statement about **prevalence**, not performance.

With

$$
\mathrm{precision}
= \frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi + \mathrm{FPR}\,(1-\pi)}
$$

and $\pi = 2.5\times10^{-3}$, even a small FPR competes with TPR for the numerator.
At FPR $= 0.1\%$ the yield is about 10 true and 10 false per 10,000 stars. That is
a usable candidate list; it just never produces a posterior above 0.5.

The line: *we deliberately do not report a posterior implying certainty we cannot
have. We report a ranking and an operating point.*

Then volunteer the good news: false positives are 6–41× concentrated in genuine
near-threshold mixtures, so the candidates worth chasing are mostly real ambiguous
objects.

## 6.4 "Your label is $f_Z^P \ge 0.5$, which we can never measure."

This is the weakest point. Raise it before they do.

The production classifier — `calibration.py` and every training site in
`v20_mega.py` — predicts

$$
y_{0.5} = \mathbf{1}\!\left[f_Z^P \ge 0.5\right],
$$

which is a property of the cloud, not of the observation. The operationally
meaningful target is instead

$$
y = \mathbf{1}[D > 3],
$$

the detectability label, which has a positive rate near 0.59 and is a harder
problem.

The line: *the classifier is trained on the physical mixture label, which is the
right target for comparison with theory. The detectability label is implemented and
is harder; switching to it is the next step, not a result we have claimed.*

## 6.5 "Isn't this just learning one simulator's yield tables?"

Partly, and the transfer test is the answer.

Leave-one-table-out transfer is 0.94–0.99 for five of six tables, with WW95 at
0.90 from every direction. The discrimination is therefore not tied to one table's
choices.

The stronger answer: **the yield tables are the dominant uncertainty, not the ML.**
Co varies by up to 2.8 dex across tables at $Z_\star = 10^{-5}$, and the low-Z
floor is extrapolated for 97 % of the Nomoto/CK20 samples. Those are the things
that change the answer.

## 6.6 "What about stars outside your model?"

The support gate exists, but describe it accurately.

Raw in-model fractions are 0.065–0.340 across the six application datasets.
Allowing one zero-point offset per core element brings every dataset to
0.909–0.948. So the gate is measuring yield offsets, chiefly Ti, not unusual stars.

The line: *presented raw, the support gate rejects most of your catalogue. That is a
statement about our yields, not about your stars.*

## 6.7 "What about NLTE, 3D, and C/N mixing in giants?"

Partially handled:

- **Giants.** Extra carbon scatter of 0.3 dex in quadrature, or carbon dropped
  entirely, because evolved stars mix carbon up from the interior.
- **Nitrogen.** Not modelled. Li+ has no N column, so it does not enter.
- **NLTE and 3D.** Not modelled at all. This is a candidate explanation for the Ti
  offset, and saying so is honest.

## 6.8 "Why is there no SN Ia?"

There is a row in the matrix and it is zero. At
$[\mathrm{Fe}/\mathrm{H}] \approx -2$ to $-3$, SN Ia can matter for the iron peak,
so this is a real omission rather than a negligible one.

The line: *SN Ia is a placeholder row, currently zero. At these metallicities it is
a genuine omission and it is on the list.*

## 6.9 "Does your $M_{\rm dil}$ prior drive everything?"

The sweep answers it. Three priors, identical gates:

| prior | acceptance | zero-PISN | median $Z_\star$ | within AUC |
| --- | ---: | ---: | ---: | ---: |
| $10^5$–$5\times10^8$ | 0.533 | 0.373 | $3.7\times10^{-5}$ | 0.970 |
| $2\times10^5$–$10^9$ | 0.512 | 0.368 | $2.8\times10^{-5}$ | 0.973 |
| $10^6$–$10^9$ | 0.499 | 0.360 | $1.6\times10^{-5}$ | 0.974 |

The classifier is stable: AUC moves by 0.004. The absolute metallicity is not:
median $Z_\star$ moves by 0.35 dex, and the below-floor fraction by 0.05.

The line: *rankings are robust to the dilution prior; absolute metallicity is not.*

## 6.10 "Where does the 0.25 % prevalence come from?"

From chemical evolution models, not from the catalogue:

$$
\pi = 2.5\times10^{-3}
$$

following de Bennassuti et al. (2017) and Salvadori et al. (2019). It should always
be quoted with a sensitivity band, for example
$\pi \in \{10^{-3},\,2.5\times10^{-3},\,10^{-2}\}$.

Changing $\pi$ moves every posterior and moves no ranking, because the prior shift
is a monotone odds rescaling.

## 6.11 "What would make this publishable?"

The blockers are not the core model. They are:

1. **Label choice.** Decide whether the headline is $y_{0.5}$ or $y = \mathbf{1}[D>3]$
   and say so explicitly.
2. **Low-Z treatment.** Report the below-floor fractions, and either restrict the
   claims to mocks above the floor or replace the clamp/blend with an explicit
   extrapolation model.
3. **Yield-systematic treatment.** Present Co and Zn as a systematic band, not a
   point prediction.
4. **Real-star controls.** Add simulation-based calibration, or at least a
   held-out catalogue test.
5. **Framing.** Present it as simulation-based inference with explicit assumptions,
   not as a validated physical model.

## 6.12 What to remember from this section

1. Zn is required, Co is optional; alpha-only is not enough.
2. The Ti offset is the one offset that survives every scheme — treat it as real.
3. Zero candidates is prevalence, not failure; quote the operating point instead.
4. Own the label mismatch before the audience finds it.
5. Yield tables, not the ML, are the dominant uncertainty.
6. The support gate is a yield diagnostic.
