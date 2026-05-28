---
phase: 25-hybrid-vae-dcm
plan: 03
subsystem: models
tags: [pyro, vae, dcm, training, kl-annealing, synthetic-data, svi]
dependency-graph:
  requires: ["25-02"]
  provides: ["generate_synthetic_vae_dataset", "train_hybrid_vae_dcm", "train_hybrid_vae_dcm.py"]
  affects: ["25-04"]
tech-stack:
  added: []
  patterns: ["kl-annealing-beta-warmup", "poutine-scale-model", "nan-tolerant-epoch-averaging"]
key-files:
  created:
    - scripts/train_hybrid_vae_dcm.py
    - tests/test_hybrid_vae_dcm_training.py
  modified:
    - src/pyro_dcm/models/hybrid_vae_dcm.py
    - src/pyro_dcm/models/__init__.py
decisions:
  - id: "25-03-D1"
    description: "Beta clamped to >= 1e-6 (not 0.0) because pyro.poutine.scale raises ValueError on scale=0"
  - id: "25-03-D2"
    description: "SVI instance recreated each epoch with current scaled_model rather than modifying scale mid-loop; simpler and avoids Pyro messenger state issues"
metrics:
  duration: "~17 minutes"
  completed: "2026-05-28"
---

# Phase 25 Plan 03: Training Infrastructure Summary

Synthetic dataset generator + SVI training loop with KL annealing (beta warmup 0->1) for hybrid VAE-DCM, with CLI training script and 5 integration tests.

## What Was Done

### Task 1: Synthetic dataset generator + training loop

Added two functions to `src/pyro_dcm/models/hybrid_vae_dcm.py`:

1. **`generate_synthetic_vae_dataset`**:
   - Generates `n_samples` diverse DCM configurations
   - Each sample: random stable A via `make_stable_latent_circuit_A`, random C ~ N(0, 0.5), random x0 ~ N(0, 0.1), noise_prec ~ U(5, 50)
   - Simulates trajectories via `simulate_latent_circuit` with block stimulus (on at t=1-2s)
   - Adds Gaussian noise with std = 1/sqrt(noise_prec)
   - Handles diverged simulations (retry with stronger self-inhibition)
   - Returns list of dicts with all tensors in float64

2. **`train_hybrid_vae_dcm`**:
   - KL annealing: beta ramps from ~0 to 1 over `warmup_epochs` via `pyro.poutine.scale(model_fn, scale=beta)`
   - Uses `ClippedAdam` optimizer with configurable lr and clip_norm
   - Shuffles training data each epoch
   - NaN-tolerant: filters diverged SVI steps from epoch average
   - Returns dict with per-epoch losses and beta schedule

3. Updated `models/__init__.py` with new exports.

### Task 2: Training script + integration tests

1. **`scripts/train_hybrid_vae_dcm.py`**:
   - CLI args: n_samples, n_epochs, warmup_epochs, n_regions, n_inputs, duration, dt, lr, seed, output_dir
   - Lazy imports (fast --help)
   - Generates train + test data, creates packer, fits standardization
   - Trains encoder-decoder pair via `train_hybrid_vae_dcm`
   - Evaluates amortized inference on held-out test set (sign recovery)
   - Saves: training curves plot, encoder state_dict, packer stats, recovery summary

2. **`tests/test_hybrid_vae_dcm_training.py`** (5 tests):

| # | Test | Status |
|---|------|--------|
| 1 | test_generate_synthetic_dataset_shapes | PASS |
| 2 | test_generate_synthetic_dataset_all_stable | PASS |
| 3 | test_training_loop_smoke (20 samples, 10 epochs) | PASS |
| 4 | test_kl_annealing_schedule | PASS |
| 5 | test_amortized_inference_recovers_sign_pattern | SLOW (cluster) |

## Decisions Made

- **[25-03-D1] Beta clamped to >= 1e-6.** `pyro.poutine.scale(scale=0.0)` raises `ValueError: Expected scale > 0`. At epoch 0, beta = epoch/warmup = 0.0, so we clamp to 1e-6 (effectively zero but avoids the error).

- **[25-03-D2] SVI recreated each epoch with current scaled model.** Rather than mutating the scale on a single SVI instance, we create `poutine.scale(model_fn, scale=beta)` each epoch and pass it to a fresh SVI. This is simpler and avoids Pyro messenger state issues.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Beta=0 crashes poutine.scale**
- **Found during:** Task 1
- **Issue:** `poutine.scale(scale=0.0)` raises ValueError
- **Fix:** Clamp beta to >= 1e-6
- **Commit:** 306cbf4

## Verification Results

- `pytest tests/test_hybrid_vae_dcm_training.py -v -m "not slow"` -- 4/4 pass (93s)
- `python scripts/train_hybrid_vae_dcm.py --n_samples 20 --n_epochs 10 --duration 2.0 --n_regions 3` -- runs end-to-end (~60s)
- `pytest tests/test_hybrid_vae_dcm_primitives.py tests/test_hybrid_vae_dcm_model.py -v` -- 19/19 pass (no regressions)
- `ruff check src/pyro_dcm/models/hybrid_vae_dcm.py scripts/train_hybrid_vae_dcm.py` -- all checks passed

## Next Phase Readiness

Phase 25-03 provides the training infrastructure. Phase 25-04 (cluster-scale training + acceptance metrics) can proceed. The slow test (`test_amortized_inference_recovers_sign_pattern`) should be run on M3 to validate at larger scale.
