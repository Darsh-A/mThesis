# Overview

## The problem

A very metal-poor star records the gas from which it formed. If one or more
pair-instability supernovae (PISN) enriched that gas, the star's abundance pattern
should carry a signature: a large odd-even effect, high alpha elements, and very
little Fe, Cu, or Zn compared with a normal core-collapse mixture.

Salvadori et al. model this as a forward problem. They fix one PISN, one Pop II
burst, and a dilution factor, compute the resulting abundances, and fit individual
stars on a grid. Their grid always has a PISN, so it cannot represent the alternative
hypothesis "no PISN".

This package is an inverse-problem simulator. It draws a mixture, asks what a star
would observe, and produces labelled data. A classifier or regressor trained on that
data answers the inverse question: given an abundance pattern, what mixture produced
it, and could a PISN be detected at all?

## The three sources

| source | role | current table |
|---|---|---|
| PISN | few huge events; the signal | Heger & Woosley 2002, 140–270 M☉ |
| Pop III CCSN | many small events; the main confuser | Ishigaki et al. 2018, 10 mass/energy groups |
| Pop II CCSN | normal enrichment after the cloud is polluted | Limongi & Chieffi 2018, IMF-integrated |
| SNIa | late Fe-rich enrichment | not implemented (zero row) |

All tables are folded onto the same 32 elements. That makes the ejecta matrix
well defined and lets the model compare sources element by element.

## The two clocks

There are three times:

- `T = 0`: the Pop III population forms.
- `T = Δ`: the Pop II burst forms from the already polluted cloud.
- `T = Δ + t`: the observed low-mass star forms.

A Pop III star of mass `m` has lifetime `τ_III(m)`. It contributes to the gas that
Pop II forms from only if `τ_III(m) ≤ Δ`. It contributes to the final cloud if
`τ_III(m) ≤ Δ + t`.

This matters because a 15 M☉ Pop III star lives about 10 Myr, comparable to `Δ`.
A one-clock model that assumes every Pop III event exploded before Pop II would
wrongly let those stars pre-enrich the cloud.

The Pop II burst has its own clock. At time `t` after the burst, only stars with
lifetime below `t` have exploded. The turnoff mass `m_TO(t, Z*)` is integrated
against the IMF to get the Pop II ejecta.

## The pipeline

```
draw counts, masses, Δ, t, M_dil
        │
        ├─ R1: no PISN and no Pop III? reject
        │
        ├─ gate Pop III on Δ and Δ+t
        ├─ build M_Zpre, Z* = M_Zpre / M_dil
        │
        ├─ R2: no pre-enrichment, or Z* below the table floor? reject
        ├─ R4: final [Fe/H] outside the target box? reject
        │
        ├─ add Pop II ejecta at (Z*, t)
        ├─ add PISN ejecta that arrived before the star
        │
        ├─ R3: Fe and enough features above the relative floor? reject
        │
        ├─ abundances a_X, ratios [X/Fe], labels f_Z^P, f_Fe^P
        ├─ counterfactual remove PISN -> D -> y
        └─ store sample + ledger
```

## What is different from Salvadori

- A negative class exists: gas enriched by Pop III CCSN and Pop II only.
- Confusers live in the same mixture, not in a separate test.
- The mixture is parametrised by amounts, not by a fixed `f_PISN`.
- The clocks are separate, so late Pop III events are allowed.
- The output is a dataset with labels and nuisance parameters, not a grid fit.
- Selection cuts are recorded, so the training prior can be measured.

## What is still shared with Salvadori

- Instant, complete mixing in one cloud.
- One Pop II burst.
- No source-specific metal loss.
- The same yield tables are used where possible.

These shared assumptions are the main structural limitations.
