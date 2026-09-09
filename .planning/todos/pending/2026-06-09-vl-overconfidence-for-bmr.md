---
created: 2026-06-09T23:55
title: VL Laplace overconfidence suppresses absolute BMR pruning
area: model-selection / inference
priority: medium
files:
  - src/pyro_dcm/inference/variational_laplace.py
  - src/pyro_dcm/model_selection/bmr.py
  - cluster/scripts/lc_vl_bmr_selection.py
  - .planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md
---

## Problem

Discovered closing SYNTH-03 (job 56270544): the VL (Laplace/ReML) posterior on the latent
circuit is **over-confident** at high SNR — posterior std/prior std ~0.001. Bayesian Model
Reduction then never prunes anything: every single-connection prune ΔF is strongly negative
(true chain ~ -1.1M, genuinely-absent edges -34 to -77k), so exhaustive best-model
`bmr_circuit_selection` keeps the FULL model (full_model_rank=1/4096).

The *relative* prune-cost ranking still recovers the true structure perfectly (essential
edges = true chain, 3/3 seeds), which is how SYNTH-03 was closed. But the **absolute** "prune
if ΔF > 0" rule is miscalibrated by the overconfident covariance. This affects **any absolute
BMR usage** (Phase 23), not just the latent circuit.

## Solution (sketch / options)

1. **Temper the VL posterior covariance** before BMR: ReML can underestimate parameter
   covariance when the observation precision is large. Inflate `sigma_post` by a calibrated
   factor, or cross-check against a posterior-predictive / held-out evidence.
2. **Use BMR relative ranking + a separation-gap criterion** as the standard structure-
   selection API (what SYNTH-03 now does) rather than the absolute ΔF>0 threshold. Consider
   adding a `rank_connections()` helper to `model_selection/bmr.py`.
3. Sanity-check whether the overconfidence is partly a *synthetic-data* artifact (clean,
   high-SNR, model exactly correct) that will be less severe on real M/EEG latents.
4. Validate against the Phase 23 brute-force ELBO comparison (which passed) to quantify how
   much the absolute ΔF is off.

Note: this does NOT affect the Phase 20-05 closure (structure recovered via ranking) or the
SYNTH-01/02 recovery numbers — it is a calibration caveat for absolute BMR evidence.

## Audit disposition (2026-06-10) — DEFER to v0.7.0 (Phase C)

Does **not** block v0.6.0: relative-ranking BMR is the standardized API and it recovers structure
(SYNTH-03 3/3). The absolute-ΔF calibration fix (posterior tempering + a `rank_connections()`
helper) is a **VL refinement** → maps to **v0.7.0 Phase C (VL + BMR model comparison)** in
`.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`. Carry forward as-is.


---

## ⚠️ 2026-09-09 MEASURED: two of this todo's premises do not hold

Job 55242781, the SYNTH-03 setup verbatim (latent circuit N=4, seeds 42-44), so
the numbers are directly comparable with the run that opened this todo.

### 1. Shrinkage is ~0.12, not ~0.001

`posterior sd / prior sd` over the off-diagonal A parameters:

| seed | min | median | max |
|---|---|---|---|
| 42 | 2.62e-02 | 1.22e-01 | 2.50e-01 |
| 43 | 2.46e-02 | 1.24e-01 | 2.64e-01 |
| 44 | 2.36e-02 | 1.15e-01 | 2.73e-01 |

Two orders of magnitude above the ~0.001 recorded here. A posterior at 2-27% of
prior width is ordinary for informative data, not pathological overconfidence.
The recorded figure is either from a different configuration or was never a
`sd` ratio; either way **"the VL posterior is over-confident" is not supported
by this measurement** and should not be repeated without re-deriving it.

### 2. Tempering the covariance does essentially nothing

Inflating the posterior covariance by `k`:

| k | chain dF (max), seed 42 | absent dF (min) | absent prunable |
|---|---|---|---|
| 1 | -1085008.24 | -77293.43 | 0/9 |
| 1024 | -1085011.70 | -77296.90 | 0/9 |

A **1024x** inflation moves dF by ~3.5 nats against magnitudes of 1e5-1e6, and
in the WRONG direction. Solution option 1 in this todo ("temper the VL posterior
covariance before BMR") is therefore not a viable fix, and the sketch should be
struck rather than left as a candidate.

### What the magnitudes point at instead

dF is ~-1e6 for true-chain edges and ~-1e5 for genuinely absent ones -- both
enormously negative, which is why the absolute rule never fires for anything.
The likely driver is not the posterior at all but the **reduced-prior variance**:
`reduced_cov[idx,idx] = 1e-8` does not encode "this edge is absent", it encodes
"this edge is exactly zero with near-infinite precision". Any nonzero posterior
mean is then penalised by roughly `mean^2 / 1e-8`, which swamps every other
term -- including for absent edges, whose posterior means are small but not
zero.

Under test in job 55242955: sweep the reduced-prior variance (1e-8 .. 0.5) and
check whether any value separates TRUE_ABSENT (prunable) from TRUE_CHAIN
(essential). **Hypothesis, not a conclusion** -- the magnitudes are consistent
with it but it has not yet been confirmed.

Whatever the outcome, the two measurements above stand and this todo's stated
diagnosis needs rewriting.


---

## ✅ 2026-09-09 DIAGNOSED: it is the reduced-prior variance, not the posterior

Job 55242955 confirms the hypothesis. Sweeping the reduced-prior variance with
the posterior untouched (prior variance = 0.0625):

| red_var | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| 1e-8 (current) | 0/9 prunable | 0/9 | 0/9 |
| 1e-6 | 3/9 | 2/9 | 3/9 |
| 1e-4 | 4/9 | 5/9 | 3/9 |
| **1e-3** | **9/9, separates** | **9/9, separates** | 6/9 |
| 1e-2 | 9/9 but chain also prunable | same | same |
| 1e-1, 5e-1 | 0/9 | 0/9 | 0/9 |

At `red_var = 1e-3` (~1/60 of the prior variance) absolute BMR separates all
nine genuinely-absent edges from the true chain on seeds 42 and 43: every absent
edge has `dF > 0` and every chain edge `dF < 0`.

**So absolute BMR is recoverable and this todo's diagnosis was wrong.** The
suppression is not posterior overconfidence -- it is that
`reduced_cov[idx,idx] = 1e-8` encodes "this edge is exactly zero with
near-infinite precision" rather than "this edge is absent". Any nonzero
posterior mean is then penalised by roughly `mean^2 / red_var`, which at 1e-8
produces `dF ~ -1e6` for everything, true edges and absent edges alike.

The non-monotonicity at the top end is the expected mirror: as `red_var`
approaches the prior variance the reduced model approaches the full model and
`dF -> 0`; at 0.5 the "reduced" prior is WIDER than the full prior, which is not
a prune at all.

### Seed 44 does not separate, and that looks real

Its hardest absent edge has `|posterior mean| = 0.0959` against a true-chain
range of 0.1287..0.1384 -- genuinely close, unlike seeds 42/43 where absent tops
out at ~0.039. A finer sweep between 1e-3 and 1e-2 might separate it, but this
may equally be a real identifiability limit on that seed. Do not assume a
threshold exists for it.

### Do NOT tune red_var to the answer

Choosing `red_var` by maximising separation on seeds where the truth is known is
circular and would manufacture the result. It is a modelling statement -- "an
edge counts as absent when its effect is below this scale" -- and needs a
principled justification, e.g. a fixed fraction of the prior variance tied to a
smallest-effect-of-interest, fixed before looking at recovery. What this
experiment establishes is that a workable regime EXISTS, not which value to ship.

### Rewrite required

- Strike solution option 1 (temper the posterior covariance): measured
  ineffective, a 1024x inflation moves `dF` by ~3.5 nats against 1e5-1e6.
- Strike the "posterior std/prior std ~0.001" premise: measured 0.024..0.273,
  median ~0.12.
- Option 2 (relative ranking + separation gap) remains sound and shipped
  (`rank_connections`), but is no longer the ONLY defensible route.
- New work: choose and justify a `red_var` convention, then re-validate absolute
  pruning across more seeds.
