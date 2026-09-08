---
created: 2026-09-08T18:10
title: spm_xval_multiseed reports fake replication -- the spectral simulator ignores its seed
area: validation
priority: medium
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
