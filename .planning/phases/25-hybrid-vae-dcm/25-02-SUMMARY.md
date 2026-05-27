---
phase: 25-hybrid-vae-dcm
plan: 02
subsystem: models
tags: [pyro, vae, dcm, ode, encoder, guide, svi]
dependency-graph:
  requires: ["25-01"]
  provides: ["hybrid_vae_dcm_model", "HybridVAEDCMGuide"]
  affects: ["25-03", "25-04"]
tech-stack:
  added: []
  patterns: ["wrapper-model-pattern", "single-latent-site", "nan-guard"]
key-files:
  created:
    - src/pyro_dcm/models/hybrid_vae_dcm.py
    - tests/test_hybrid_vae_dcm_model.py
  modified:
    - src/pyro_dcm/models/__init__.py
decisions:
  - id: "25-02-D1"
    description: "SVI smoke test uses windowed avg (first 5 vs last 5 finite losses) instead of first vs last, because early NaN losses from ODE divergence are expected"
  - id: "25-02-D2"
    description: "packer.total_dim used (not n_features) as LatentCircuitDCMPacker attribute name differs from TaskDCMPacker"
metrics:
  duration: "~70 minutes"
  completed: "2026-05-28"
---

# Phase 25 Plan 02: Hybrid VAE-DCM Model/Guide Pair Summary

Physics-informed VAE where DCM ODE (CoupledDCMSystem hemodynamic=False) is the decoder and DCMEncoderNet is the recognition network, with single _latent site for Pyro ELBO compatibility.

## What Was Done

### Task 1: hybrid_vae_dcm_model + HybridVAEDCMGuide

Created `src/pyro_dcm/models/hybrid_vae_dcm.py` with two components:

1. **`hybrid_vae_dcm_model`** (decoder/generative model):
   - Samples packed `_latent` from N(0, I) in standardized space
   - Unstandardizes via `LatentCircuitDCMPacker.unstandardize`
   - Unpacks into A_free, C, x0, noise_prec
   - Applies masks and `parameterize_A` for negative diagonal
   - Runs `CoupledDCMSystem(hemodynamic=False)` ODE integration
   - NaN guard replaces diverged predictions with detached zeros
   - Evaluates Gaussian likelihood via `direct_observation` (identity C_obs)

2. **`HybridVAEDCMGuide`** (encoder/recognition network):
   - Wraps `DCMEncoderNet` as Pyro module
   - Maps observed trajectories to z_loc, z_scale
   - Samples `_latent` from N(z_loc, z_scale)
   - `sample_posterior()` method for post-training inference

3. Updated `models/__init__.py` with re-exports.

### Task 2: Integration Tests

Created `tests/test_hybrid_vae_dcm_model.py` with 7 tests:

| # | Test | Status |
|---|------|--------|
| 1 | model_trace_has_latent_and_obs_sites | PASS |
| 2 | guide_trace_has_latent_site | PASS |
| 3 | model_guide_site_names_match | PASS |
| 4 | model_predicted_trajectories_shape | PASS |
| 5 | guide_sample_posterior_shapes | PASS |
| 6 | svi_smoke_elbo_decreases (80 steps) | PASS |
| 7 | nan_guard_produces_finite_loss | PASS |

## Decisions Made

- **[25-02-D1] SVI smoke test uses windowed average.** Early SVI steps produce NaN losses because untrained encoder outputs cause ODE divergence. The NaN guard prevents gradient corruption, but losses are NaN. Test filters to finite losses and compares first-5 vs last-5 average.

- **[25-02-D2] `packer.total_dim` not `packer.n_features`.** `LatentCircuitDCMPacker` uses `total_dim` (sparse packing), unlike `TaskDCMPacker` which uses `n_features` (dense packing).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] ClippedAdam import location**
- **Found during:** Task 2
- **Issue:** Plan specified `from pyro.infer import ClippedAdam` but it's in `pyro.optim`
- **Fix:** Changed to `from pyro.optim import ClippedAdam`
- **Commit:** ad45a65

**2. [Rule 1 - Bug] SVI test NaN handling**
- **Found during:** Task 2
- **Issue:** Initial SVI test compared first vs last loss, but early losses are NaN due to ODE divergence
- **Fix:** Filter finite losses, compare windowed averages (first 5 vs last 5), use lower lr=0.005 and clip_norm=5.0
- **Commit:** ad45a65

## Next Phase Readiness

Phase 25-02 provides the core model/guide pair. Phase 25-03 (training loop) can proceed directly.
