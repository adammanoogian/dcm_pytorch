---
created: 2026-09-08T19:20
title: Ranking agreement falls below the 0.80 gate under CSD sampling noise
area: validation / SPM parity
priority: medium
files:
  - cluster/scripts/spm_xval_multiseed.py
  - validation/run_vl_validation.py
  - cluster/results/spm_xval_multiseed_55234748.json
---

## Problem

With genuine finite-sample CSD realisations (job 55234748, seeds 42-46), the
VL-vs-SPM cross-model ranking agreement is **1.0, 1.0, 0.667, 1.0, 1.0** --
seed 44 falls to 2/3 pairs, below the `>=0.80` gate.

Every previous run reported 1.0 on every seed. That was an artifact: the sweep
refitted the identical problem, so of course the ranking never moved. This is
the first time ranking has been tested against varying data at all.

## Why this matters more than the number suggests

Model ranking is the claim the project has leaned on hardest. The memory entry
and several phase closures use wording along the lines of "ranking agreement is
1.0, so ranking is the SPM-comparable claim even when absolute F is not". That
statement rested on a degenerate sweep and needs re-grounding.

Note the F gate itself replicates fine (all 5 seeds within 5%), so this is not
a regression -- it is the first honest measurement.

## What is not yet known

- Is seed 44 a **close call** (two models with nearly equal evidence, where
  either ordering is defensible) or a **real disagreement** on a well-separated
  pair? The per-pair delta-F magnitudes decide this and are in the JSON's
  `pairwise_results`, not yet inspected -- eduVPN dropped mid-analysis.
- Does agreement improve with more averages (`XVAL_CSD_AVERAGES` > 64), i.e. is
  this purely a noise-level effect that vanishes as the CSD estimate tightens?
- Is 5 seeds enough to characterise the rate at all? Almost certainly not.

## Next step

1. Pull `pairwise_results` for seed 44 and report the delta-F separation of the
   disagreeing pair alongside the disagreement.
2. Re-run over more seeds (20+) and at two or three `XVAL_CSD_AVERAGES` levels,
   and report agreement as a **rate with a separation threshold**, not a single
   number. A disagreement on a pair separated by < 3 nats is not the same
   finding as one on a well-separated pair.
3. Only then decide whether the `>=0.80` gate is the right shape.


---

## UPDATE 2026-09-08: seed 44 resolved -- and it exposes a gap in the F gate

Job 55234938 (same seeds, now carrying per-pair separations). Ranking rates
reproduce exactly: **1.0, 1.0, 0.667, 1.0, 1.0**. The single disagreement:

```
seed 44, pair (0,1):  spm_delta = -1.3016   vl_delta = +53.0843
                      min_abs_separation = 1.3016   is_close_call = False
```

### It is neither a clean near-tie nor two confident engines disagreeing

SPM is **nearly indifferent** between the two models -- 1.30 nats -- and mildly
prefers model 1. VL is **emphatic** at 53.08 nats in the opposite direction.
So the two engines disagree not only on the ordering but on *how much evidence
there is at all*, by roughly a factor of 40.

**Threshold caveat, stated plainly:** the `is_close_call = False` verdict is
sensitive to the arbitrary `close_call_nats = 1.0` default, since 1.3016 sits
just above it. At a 1.5-nat threshold this pair would be classified "close". The
classification is therefore not the finding. The **53-vs-1.3 asymmetry is**, and
that is threshold-independent.

### The real finding: the matched-F gate only tests the FULL model

Seed 44's *full-model* matched-F relative error is 0.0162 -- comfortably inside
the 5% gate. Yet on a REDUCED model the same two engines differ by ~54 nats.

The parity suite fits three `a_mask` scenarios for the ranking sweep but gates
matched-F on the full model **only**. Reduced-model free energies are used for
ranking and then discarded, so a divergence of this size is invisible to every
current gate. Since BMR and every structure-selection claim runs on reduced
models, this is the regime that matters most and it is the one least tested.

## Revised next step

1. **Gate matched-F on every scenario, not just the full model.** Record the
   per-scenario VL and SPM free energies in the results JSON (they are computed
   in `run_vl_spectral_dcm_validation`'s scenario loop and currently thrown
   away after `compare_model_ranking`). This is the highest-value change here.
2. Identify which `a_mask` scenario 0 and 1 are for seed 44 and whether the
   53-nat gap sits in one engine's fit of one specific reduced model -- i.e.
   whether one of them failed to converge on that mask.
3. Only then revisit the `>=0.80` ranking gate and the `close_call_nats`
   default. Both are premature while reduced-model F is ungated.


---

## UPDATE 2026-09-08: the divergence is ONE scenario -- `single_direction`

The per-scenario gate (`a30a781`) landed and job 55235094 shows the reduced-model
divergence is not spread across models. It is concentrated entirely in the
feed-forward mask.

| seed | full_reciprocal | single_direction | diagonal_only |
|---|---|---|---|
| 42 | 0.0021 OK | **0.1468 MISS** | 0.0060 OK |
| 43 | 0.0064 OK | **0.2771 MISS** | 0.0472 OK |
| 44 | 0.0162 OK | **0.1706 MISS** | 0.0386 OK |
| 45 | 0.0220 OK | 0.0014 OK | 0.0199 OK |
| 46 | 0.0285 OK | **0.2782 MISS** | 0.0302 OK |

4/5 seeds fail on `single_direction`, and **always in the same direction**: VL's
free energy is BELOW SPM's, by 18.4, 62.0, 49.7 and 72.3 nats. A consistent sign
across seeds is a bias, not noise.

`single_direction` is `eye` plus a single off-diagonal (`mask[1,0] = 1`), i.e. a
**feed-forward** structure. Note `diagonal_only` (also triangular) PASSES, so
this is not triangularity as such -- it is the single-off-diagonal case.

### This also explains seed 44's ranking disagreement

Seed 44's disagreeing pair was (0,1) = full_reciprocal vs single_direction, with
`vl_delta = +53.08` against `spm_delta = -1.30`. VL underestimates
single_direction's F by ~49.7 nats on that seed, which is very nearly the whole
53-nat gap. The ranking disagreement is a *consequence* of the single_direction
divergence, not an independent finding.

### The gap in the evidence

The forward-parity result that established agreement to **4e-14** (job 55234376)
was run on the **full reciprocal A only**. The forward has NEVER been checked on
a triangular mask. And the feed-forward regime is exactly where
`compute_transfer_function_hemodynamic` was defective before `007b686` -- the bug
behind the retracted Phase 31 "feed-forward is unidentifiable" finding. The
resolvent fix improved the conditioning, but parity in that regime was never
verified.

**Probe submitted: job 55235473** runs the forward-parity comparison under all
three masks, with `full_reciprocal` as an explicit control that must reproduce
1.000000. If the control matches and `single_direction` does not, this is a
forward-model divergence in the feed-forward regime, not an optimiser problem.

### Why this matters beyond the parity suite

Reduced models are sparse by construction, so **BMR evaluates exactly this
regime**. If VL systematically underestimates F for feed-forward masks, every
spectral prune delta-F is biased. Whether this reaches task/latent BMR too is
unknown and must be checked rather than assumed: the a_mask -> zeroed prior
variance -> SVD reduction path lives in `_run_vl_generic` and is shared by ALL
forwards, so the mechanism is not obviously spectral-specific. This is a
DIFFERENT mechanism from `2026-06-09-vl-overconfidence-for-bmr`; do not merge
them.


---

## UPDATE 2026-09-08 (later): forward and convergence BOTH exonerated

### 1. Not the forward (job 55235592)

Comparing our forward at the prior mean against SPM's `pred_pE`, going through
the VL engine's OWN packing (`theta = prior_mean`, A block = `a_mask/128`,
which is exactly what SPM exports as `pE.A`) so no convention mapping is
assumed:

| mask | ratio mean | max abs diff |
|---|---|---|
| full_reciprocal (CONTROL) | 0.998336 | 1.33e-02 |
| single_direction (TEST) | 0.998484 | 1.61e-02 |
| diagonal_only | 0.998660 | 1.38e-02 |

A **uniform** ~0.17% deviation across all three masks. There is no
feed-forward-specific forward divergence, so the forward cannot explain a
15-28% F gap that appears only on `single_direction`.

**A prior probe (55235562) is VOID** -- it tried to rebuild SPM's transfer
function from the exported `pE_A` and its control came back at 6.47 where
1.000000 was required. SPM's `pE.A` is the raw prior, before `spm_fx_fmri`
applies its self-inhibition parameterisation. Ignore its numbers entirely.

### 2. Not convergence

VL fitted at `max_iter` 64 / 256 / 1024 on `single_direction`:

```
seed   SPM F     VL@64     VL@256    VL@1024   iters   gap
 42  125.0511  106.6919  106.6919  106.6919    16   -18.36
 43  223.8669  161.8360  161.8360  161.8360    16   -62.03
 44  291.2369  241.5480  241.5480  241.5480    17   -49.69
 45  285.5854  285.1748  285.1748  285.1748    16    -0.41
 46  260.0095  187.6822  187.6822  187.6822    16   -72.33
```

Identical to every digit across a 16x iteration budget, `converged=True`
throughout, and `max F over trace` matches the reported F -- so VL is not
stopping short and not accepting a worse iterate than one it had visited.

### 3. Setup verified on both sides

`export_to_mat.py` writes `DCM["a"] = a_mask`; the bridge installs
`DCM.M.pE`/`DCM.M.pC` from `spm_dcm_fmri_priors(DCM.a, ...)`. SPM really is
fitting the reduced model. The mask is not being silently ignored.

### 4. The reframing: VL may be the CORRECT side

`single_direction` prunes `A[0,1]`, whose **true value is 0.15** -- the larger
of the two couplings. Pruning a real connection should cost evidence.

| seed | VL dF (full - single) | SPM dF |
|---|---|---|
| 42 | 32.41 | 13.76 |
| 43 | 64.11 | 3.54 |
| 44 | 53.08 | **-1.30** |
| 45 | 45.22 | 37.69 |
| 46 | 80.85 | 16.39 |

VL is consistently large (32-81 nats). SPM is erratic (-1.3 to 37.7), and on
seed 44 reports that the model **without** the true 0.15 connection is BETTER
than the one with it. That is implausible on data generated with that
connection present.

Combined with two independent results pointing the same way -- VL is 2.82x
closer to ground truth on the off-diagonals (5/5 seeds), and VL attains the
higher free energy on the full model -- the evidence leans toward **SPM
under-fitting the reduced models**, not VL being wrong.

**Held deliberately as a hypothesis, not a conclusion.** SPM12 is the reference
implementation and the claim "our port beats SPM" demands more than five seeds
on one N=2 problem. What has been *established* is narrower and solid: the gap
is not the forward, not the iteration budget, and not a mis-specified mask.

## Decisive next test

Compare each engine's **reduced-model `Ep` against ground truth**, and the
residual norm `|y - g(Ep)|` each achieves on the same data. If SPM's
reduced-model `Ep` is far from truth AND its residual is larger than VL's while
its F is higher, SPM's reduced-model free energy is not tracking its own fit
quality and the hypothesis is confirmed. This needs the bridge to export
per-scenario `Ep`, which it currently does only for the full model.


---

## ✅ 2026-09-09 RESOLVED: SPM's reduced-model free energy does not track its fit

Job 55242892. Both engines' posterior means scored through **our** forward, so
the residuals are comparable (the two forwards agree to a uniform ~0.17%, job
55235592 -- far too small to explain residual gaps of 10-40%).

15 scenario-fits (5 seeds x 3 masks). **7 are inconsistent: the engine reporting
the higher free energy achieves the WORSE fit.** Every single one of those 7 is
SPM. VL is never on the wrong side of it.

| seed | mask | SPM F | VL F | SPM resid | VL resid |
|---|---|---|---|---|---|
| 42 | single_direction | 125.05 | 106.69 | 0.4889 | **0.4578** |
| 43 | full_reciprocal | 227.41 | 225.95 | 0.4957 | **0.4429** |
| 43 | single_direction | 223.87 | 161.84 | 0.5523 | **0.4687** |
| 43 | diagonal_only | 113.48 | 108.12 | 0.5091 | **0.5087** |
| 44 | single_direction | 291.24 | 241.55 | 0.4414 | **0.3822** |
| 46 | full_reciprocal | 276.40 | 268.53 | 0.3403 | **0.2788** |
| 46 | single_direction | 260.01 | 187.68 | 0.5134 | **0.3645** |

VL achieves the lower residual in **13 of 15** fits; the two exceptions are
`diagonal_only` by 0.0019 and 0.0044.

`single_direction` is inconsistent on **4/5 seeds** -- every seed except 45,
which is also the one seed where the two engines' F agreed (0.0014 relative).
The F divergence and the fit inconsistency appear and disappear together.

### Why this settles it

Free energy trades data fit against complexity, so higher-F-with-worse-fit is
normally legitimate. That defence does not apply here: both engines fit the
**same model under the same mask on the same data**, so the complexity term is
identical. An engine reporting more evidence while fitting worse is reporting a
free energy that does not track its own fit quality.

Combined with the earlier eliminations -- not the forward (uniform 0.17% across
masks), not the iteration budget (identical F at 64/256/1024), not a
mis-specified mask (`DCM.a` verified) -- and with VL landing 2.82x closer to
ground truth on the off-diagonals, **VL is the more trustworthy engine on
reduced models.**

### Limits of the claim

- Residuals are computed through OUR forward for both engines. That is the only
  way to make them comparable, but it is our forward. The 0.17% forward
  agreement bounds the effect far below the observed gaps.
- 5 seeds, one N=2 problem. This shows SPM's reduced-model F is *inconsistent
  with its own fit*; it does not prove VL's F is correct in absolute terms --
  only that VL is never caught in the same inconsistency.
- No claim is made about SPM12 generally. This is the injected-CSD spectral path
  on reduced models.

### Consequence for BMR and structure selection

BMR evaluates reduced models exclusively. On this evidence, spectral
structure-selection results computed from **SPM** free energies on reduced
models should not be trusted without re-checking; the VL path is the sounder
basis. This does NOT retroactively invalidate the project's own BMR work, which
runs on task/latent forwards through VL.
