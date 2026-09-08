---
created: 2026-09-08T18:10
title: spm_xval_multiseed reports fake replication -- the spectral simulator ignores its seed
area: validation
priority: done
files:
  - cluster/scripts/spm_xval_multiseed.py
  - src/pyro_dcm/simulators/spectral_simulator.py
  - validation/run_vl_validation.py
---

## Problem

`simulate_spectral_dcm` is **fully deterministic**. It computes the analytic CSD
from `A` and the noise *parameters*; its `seed` argument only calls
`torch.manual_seed`, which nothing downstream consumes. Verified 2026-09-08:

```
seed 43: max|csd - csd_42| = 0.000e+00
seed 44: max|csd - csd_42| = 0.000e+00
seed 45: max|csd - csd_42| = 0.000e+00
seed 46: max|csd - csd_42| = 0.000e+00
```

`_build_reciprocal_asymmetric_A` is likewise seed-independent.

So `cluster/scripts/spm_xval_multiseed.py` fits the **identical problem N times**.
Job 55234529 (seeds 42-46) returned five bit-identical rows -- `vl_F` 517.880604,
`spm_F` 515.979543, offset 1.901061 every time -- and reported
`f_offset_std = 0.0`, `f_offset_cv = 0.0`, `f_offset_is_constant = True`.

That reads as strong replication evidence. It is a sweep of size one.

## Why it matters

`f_offset_is_constant` was the harness's answer to "is the VL-vs-SPM gap a
constant offset?" -- a question whose answer shaped the whole free-energy
investigation. The harness could never have answered it.

## Done so far

Added a loud degeneracy guard to `spm_xval_multiseed.py`: if every seed yields
an identical offset, it sets `f_offset_is_constant = None`, records
`seeds_are_degenerate` / `degeneracy_warning`, and prints a banner. This makes
the failure impossible to mistake for a result. It does NOT fix the sweep.

## Open: how should the problem actually vary?

A scientific choice, deliberately not made unilaterally:

1. **Vary `A` per seed** (`make_stable_A_spectral(n, seed=seed)`). Tests engine
   agreement across connectivity structures, but abandons the prior-matched
   reciprocal-asymmetric problem the gates were designed around.
2. **Add sampling noise to the CSD** -- draw a finite-sample CSD (Wishart around
   the analytic one) so each seed is a different realisation of the same truth.
   Closest to what "seed" normally means, and closest to real data, since
   `spm_dcm_csd_Q` assumes a *sample* spectral density.
3. **Both**, reported separately.

Option 2 is the most faithful to what the gates claim to measure. Confirm before
implementing -- it changes what the parity gates test.


---

## ✅ CLOSED 2026-09-08: sampling noise implemented, sweep is non-degenerate

Implemented option 2 (sampling noise on the CSD), per user decision.
`sample_csd_realisation` (`3027ab4`) draws a complex-Wishart finite-sample CSD --
the model `spm_dcm_csd_Q` already assumes (REF-111, Camba-Mendez & Kapetanios
2005). `spm_xval_multiseed` now defaults to `XVAL_CSD_AVERAGES=64`.

Job 55234748, seeds 42-46:

```
n_distinct_offsets     = 5
seeds_are_degenerate   = False
f_offset_is_constant   = False
f_offset_mean          =  0.5552
f_offset_std           =  5.2032
```

| seed | VL F | SPM F | offset | matched-F rel err | ranking |
|---|---|---|---|---|---|
| 42 | 139.1035 | 138.8089 | +0.2946 | 0.0021 | 1.000 |
| 43 | 225.9482 | 227.4108 | -1.4626 | 0.0064 | 1.000 |
| 44 | 294.6323 | 289.9352 | +4.6971 | 0.0162 | **0.667** |
| 45 | 330.3965 | 323.2787 | +7.1178 | 0.0220 | 1.000 |
| 46 | 268.5280 | 276.3992 | -7.8712 | 0.0285 | 1.000 |

### Three real findings the degenerate sweep was hiding

1. **The matched-F gate replicates.** All 5 seeds inside the strict 5% gate,
   worst 0.0285. This is the first genuine replication of that result.

2. **The offset is NOT a constant -- and that is the better outcome.** It is
   sign-varying (-7.87 to +7.12) around a mean of 0.555, i.e. realisation noise
   with no systematic bias, rather than a structural offset. The long-standing
   "constant offset" hypothesis is retired on real evidence for the first time;
   previously it could neither be confirmed nor refuted because the harness
   never varied anything.

3. **Ranking agreement is NOT always 1.0.** Seed 44 gives 0.667 (2/3 pairs),
   below the >=0.80 gate. Ranking had looked perfect in every prior run purely
   because the same problem was refitted. With a noisy CSD two models of similar
   evidence can be ordered differently by two optimisers landing in slightly
   different optima. **This is a new open question**, tracked in
   `2026-09-08-ranking-agreement-under-noise.md`.

Note F drops from ~518 (analytic CSD) to 139-330 under sampling noise, as
expected: a finite-sample CSD is harder to fit.
