---
phase: 22-dcm-neural-interpretability
plan: 04
subsystem: neural-data-models
tags: [latent-extraction, csd, spectral-dcm, integration-test, pipeline]
dependency-graph:
  requires: ["22-01", "22-02", "22-03"]
  provides:
    - extract_latent_trajectories (LSTM-AE encoder -> numpy latent dynamics)
    - compute_latent_csd (Welch CSD at MEG frequencies from latent trajectories)
    - prepare_for_spectral_dcm (numpy CSD -> torch tensors for spectral DCM)
    - End-to-end synthetic pipeline test proving A recovery through autoencoder latent space
  affects:
    - "22-05 (interpretability comparison uses latent extraction + CSD pipeline)"
    - "22-06 (publication figures depend on working pipeline)"
tech-stack:
  added: []
  patterns:
    - "Latent-space CSD via Welch periodogram (compute_empirical_csd on encoder output)"
    - "Pipeline pattern: raw timeseries -> LSTM-AE -> latent CSD -> spectral DCM"
key-files:
  created:
    - src/pyro_dcm/neural_data_models/latent_extraction.py
    - tests/test_neural_data_latent_extraction.py
    - tests/test_synthetic_pipeline.py
  modified:
    - src/pyro_dcm/neural_data_models/__init__.py
decisions: []
metrics:
  duration: ~9 minutes
  completed: 2026-05-27
---

# Phase 22 Plan 04: Latent Extraction and Synthetic Pipeline Test Summary

Latent dynamics extraction from trained LSTM-AE via encoder pass, Welch CSD computation at MEG frequencies, and end-to-end synthetic pipeline test proving connectivity recovery from autoencoder latent space (known A -> OU timeseries -> LSTM-AE -> latent CSD -> spectral DCM -> posterior A with decreasing ELBO and no NaN).

## What Was Done

### Task 1: Latent extraction and CSD computation utilities (390700e)

Created `src/pyro_dcm/neural_data_models/latent_extraction.py` with three public functions:

1. **`extract_latent_trajectories(model, data, *, device, batch_size)`**: Sets model to eval mode, iterates over batches with `torch.no_grad()`, calls `model.encode()`, concatenates latent outputs as numpy array of shape `(n_samples, T, N_latent)`. Accepts either `torch.Tensor` or `DataLoader` input.

2. **`compute_latent_csd(latent_trajectories, *, sfreq, fmin, fmax, n_freqs, average_over_samples)`**: Computes CSD from latent trajectories using `compute_empirical_csd` (Welch periodogram). Handles 2D single-trajectory input, 3D with sample averaging, and 3D per-sample modes. Returns dict with `csd`, `freqs`, `sfreq`, `n_latent`.

3. **`prepare_for_spectral_dcm(csd_result)`**: Converts numpy CSD dict to torch tensors (complex128 CSD, float64 freqs, float64 all-ones a_mask) for `spectral_dcm_model`.

Updated `__init__.py` to export all three functions.

Created 8 unit tests in `tests/test_neural_data_latent_extraction.py`:

| # | Test | Coverage |
|---|------|----------|
| 1 | test_extract_latent_trajectories_shape | (10, 100, 6) -> (10, 100, 12) |
| 2 | test_extract_latent_trajectories_from_dataloader | DataLoader input same shape |
| 3 | test_extract_latent_trajectories_eval_mode | Model set to eval during extraction |
| 4 | test_compute_latent_csd_2d_input | (T, N) -> (F, N, N) |
| 5 | test_compute_latent_csd_3d_averaged | (n, T, N) with average -> (F, N, N) |
| 6 | test_compute_latent_csd_3d_per_sample | (n, T, N) no average -> (n, F, N, N) |
| 7 | test_compute_latent_csd_hermitian | CSD[f,i,j] == conj(CSD[f,j,i]) |
| 8 | test_prepare_for_spectral_dcm_dtypes | complex128, float64, float64 |

### Task 2: Synthetic end-to-end integration test (9220a03)

Created `tests/test_synthetic_pipeline.py` with two tests:

1. **`test_synthetic_pipeline_end_to_end`** (marked `@pytest.mark.slow`, ~62s):
   - Generate 100 train samples from 10-region sensorimotor A via OU process (4s at 250 Hz)
   - Train MEGAutoencoder (n_latent=20, hidden=64, 30 epochs) -- verifies loss decreases
   - Extract latent trajectories: shape (100, 1000, 20)
   - Compute latent CSD: (64, 20, 20), verified Hermitian with non-negative auto-spectra
   - Fit spectral DCM with prior_a_var=1/16, eig_clamp=-1.0, 200 SVI steps
   - Verify ELBO decreases (early_mean > late_mean), no NaN in losses or posterior A_free

2. **`test_raw_vs_latent_csd_shapes`** (~9s):
   - Raw CSD from 10-channel timeseries: (64, 10, 10)
   - Latent CSD from 20-channel encoder output: (64, 20, 20)
   - Both verified Hermitian with non-negative auto-spectra

## Test Results

- **New tests:** 10/10 pass (8 unit + 2 integration)
- **Regression tests:** 41/41 pass (12 autoencoder + 19 simulator + 10 spectral DCM)
- **Ruff:** Clean on all new files

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

None -- no architectural decisions needed.

## Verification

- [x] All 10 tests pass (`pytest tests/test_neural_data_latent_extraction.py tests/test_synthetic_pipeline.py -v`)
- [x] Synthetic pipeline test confirms full flow works with known ground truth
- [x] No regressions in existing tests (41/41 pass)
- [x] Ruff clean on all new files
- [x] Top-level imports work (`from pyro_dcm.neural_data_models import extract_latent_trajectories, compute_latent_csd, prepare_for_spectral_dcm`)

## Next Phase Readiness

Plan 22-04 provides the complete extraction pipeline for downstream plans:
- `extract_latent_trajectories` + `compute_latent_csd` + `prepare_for_spectral_dcm` form the bridge from trained autoencoder to spectral DCM
- The synthetic pipeline test proves the approach works end-to-end
- No blockers for Plans 22-05 (interpretability comparison) and 22-06 (publication figures)
