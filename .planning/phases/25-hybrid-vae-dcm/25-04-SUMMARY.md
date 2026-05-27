---
phase: 25-hybrid-vae-dcm
plan: 04
subsystem: models
tags: [pyro, vae, dcm, training, cluster, amortized-inference, recovery]
dependency-graph:
  requires: ["25-02"]
  provides: ["generate_synthetic_vae_dataset", "train_hybrid_vae_dcm", "sbatch_hybrid_vae_dcm", "recovery_tests"]
  affects: []
tech-stack:
  added: []
  patterns: ["kl-annealing", "poutine-scale-beta", "encoder-checkpoint", "recovery-report-json"]
key-files:
  created:
    - scripts/train_hybrid_vae_dcm.py
    - cluster/sbatch_hybrid_vae_dcm.sh
    - tests/test_hybrid_vae_dcm_recovery.py
  modified:
    - src/pyro_dcm/models/hybrid_vae_dcm.py
    - src/pyro_dcm/models/__init__.py
decisions:
  - id: "25-04-D1"
    description: "KL annealing uses poutine.scale with mutable beta container (list[float]) read by scaled_model closure. SVI is created once with the wrapper, beta updated per epoch."
  - id: "25-04-D2"
    description: "Beta floor is 1e-3 (not 0.0) at epoch 0 to avoid degenerate ELBO when scale=0 zeros out all log-probs."
  - id: "25-04-D3"
    description: "KL divergence estimated analytically from encoder z_loc/z_scale vs standard normal, not from Trace_ELBO decomposition, for speed and simplicity."
metrics:
  duration: "~11 minutes"
  completed: "2026-05-28"
---

# Phase 25 Plan 04: Hybrid VAE-DCM Cluster Training and Recovery Validation Summary

Full training pipeline with synthetic data generation, KL-annealed SVI training, M3 cluster sbatch, encoder checkpointing, amortized recovery report, and validation test suite with skip-if-no-checkpoint guards.

## What Was Done

### Task 1: Training pipeline + cluster sbatch + recovery tests

**1. Added training infrastructure to `hybrid_vae_dcm.py`:**

- `generate_synthetic_vae_dataset(n_samples, n_regions, n_inputs, duration, dt, seed)`:
  Generates diverse DCM parameter configurations (random stable A via
  `make_stable_latent_circuit_A`, random C ~ N(0, 0.5), random x0 ~ N(0, 0.1),
  random noise_prec ~ U(5, 50)), simulates trajectories via `simulate_latent_circuit`,
  and adds Gaussian observation noise. Block stimulus design (ON at 20-40% of duration).

- `train_hybrid_vae_dcm(model_fn, guide, train_data, n_epochs, warmup_epochs, ...)`:
  Training loop with KL annealing via `pyro.poutine.scale`. Uses a mutable
  `beta_container` list so SVI can be created once with a `scaled_model` closure.
  Beta linearly increases from 1e-3 to 1.0 over warmup epochs. ClippedAdam optimizer
  with configurable gradient clipping. Handles NaN/Inf losses gracefully (skips them).

**2. Created `scripts/train_hybrid_vae_dcm.py`:**

Self-contained CLI training script with lazy imports (argparse first for fast --help):
- Generates train + test synthetic datasets
- Fits packer standardization on training parameters
- Creates DCMEncoderNet + HybridVAEDCMGuide
- Runs `train_hybrid_vae_dcm` with configurable epochs/warmup/lr
- Saves: encoder_checkpoint.pt (state_dict + packer stats + training config),
  training_loss.png (matplotlib curve), recovery_report.json
- Recovery report includes per-example metrics (A_free RMSE, sign recovery,
  C RMSE, x0 RMSE), aggregated stats, inference timing, KL divergence
- Prints formatted summary table to stdout

**3. Created `cluster/sbatch_hybrid_vae_dcm.sh`:**

M3 cluster submission script following project conventions:
- Sources `cluster/lib/cluster_env.sh` for standardized environment setup
- 4 CPUs, 16GB RAM, 4-hour wall time, comp partition
- Configurable via `--export` env vars (N_SAMPLES, N_EPOCHS, etc.)
- Default: 1000 samples, 200 epochs, 40 warmup, N=4, M=1, duration=5s
- Uses `pip install --no-deps -e .` (no resolver overhead)
- Prints recovery report JSON on completion

**4. Created `tests/test_hybrid_vae_dcm_recovery.py`:**

5 recovery validation tests with `@pytest.mark.slow` marker:
- `test_load_trained_encoder`: Loads checkpoint, verifies encoder outputs valid shapes
- `test_recovery_a_rmse_below_threshold`: A_free RMSE < 0.3
- `test_recovery_sign_accuracy_above_chance`: Sign recovery > 0.6
- `test_amortized_inference_timing`: Mean inference < 1.0 second
- `test_kl_not_collapsed`: KL > 0.1 (no posterior collapse)

All tests skip gracefully when checkpoint/report files are missing (pre-training).

## Verification

- Smoke test: `python scripts/train_hybrid_vae_dcm.py --n_samples 30 --n_epochs 10 --duration 2.0 --output_dir results/hybrid_vae_dcm_smoke --n_test 5`
  - Completed in 75 seconds on laptop
  - A_free RMSE = 0.06 (well under 0.3 threshold)
  - Inference time = 1.2 ms/example (well under 1s threshold)
  - KL = 16.3 (not collapsed)
  - Sign recovery = 0.46 (expected for 10 epochs; full training will improve)
- All 19 existing hybrid VAE-DCM tests pass (primitives + model)
- All 5 recovery tests skip gracefully (no checkpoint yet)
- `ruff check` clean on all modified files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan 25-03 prerequisites not yet built**

- **Found during:** Task 1 (start)
- **Issue:** Plan 25-04 depends on 25-03 artifacts (`generate_synthetic_vae_dataset`,
  `train_hybrid_vae_dcm`, `scripts/train_hybrid_vae_dcm.py`) which did not exist.
- **Fix:** Built the 25-03 training functions directly into `hybrid_vae_dcm.py`
  as part of the 25-04 task, since the plan specifies them as prerequisites.
- **Files modified:** `src/pyro_dcm/models/hybrid_vae_dcm.py`

**2. [Rule 1 - Bug] Beta=0.0 causes degenerate ELBO at epoch 0**

- **Found during:** Task 1 verification
- **Issue:** When beta=0.0, `poutine.scale(scale=0.0)` zeros out all log-probs,
  causing all SVI steps to return NaN/Inf (0/30 valid samples in epoch 1).
- **Fix:** Set beta floor to `max(1e-3, epoch / warmup_epochs)` so even epoch 0
  has a small positive scale.
- **Files modified:** `src/pyro_dcm/models/hybrid_vae_dcm.py`

## Checkpoint Status

Task 2 is a `checkpoint:human-verify` gate. The automated work (Task 1) is complete.
The checkpoint requires:
1. Submitting the cluster job via `sbatch cluster/sbatch_hybrid_vae_dcm.sh`
2. Waiting for completion (~2-3 hours)
3. Checking recovery_report.json for threshold compliance
4. Running `pytest tests/test_hybrid_vae_dcm_recovery.py -v`

## Next Steps

1. Submit cluster training job (M3)
2. Verify recovery metrics meet thresholds after training
3. Run recovery tests with trained checkpoint
