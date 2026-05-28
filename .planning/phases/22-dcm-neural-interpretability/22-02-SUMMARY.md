---
phase: 22-dcm-neural-interpretability
plan: 02
subsystem: neural-data-models
tags: [lstm, autoencoder, meg, spectral-loss, overcomplete]
dependency-graph:
  requires: []
  provides:
    - MEGAutoencoder nn.Module with overcomplete latent space
    - AutoencoderTrainer with spectral consistency loss
    - 12 unit tests for shape/gradient/training/checkpoint/spectral
  affects:
    - 22-03 (MEG simulator feeds training data)
    - 22-04 (spectral DCM fits latent dynamics from autoencoder)
    - 22-05 (interpretability comparison uses trained autoencoder)
tech-stack:
  added: []
  patterns:
    - LSTM autoencoder with overcomplete latent (2x input dim)
    - Differentiable spectral loss via torch.fft.rfft log-power PSD
    - Trainer pattern with early stopping, validation, checkpointing
key-files:
  created:
    - src/pyro_dcm/neural_data_models/__init__.py
    - src/pyro_dcm/neural_data_models/lstm_autoencoder.py
    - src/pyro_dcm/neural_data_models/trainer.py
    - tests/test_lstm_autoencoder.py
  modified: []
decisions: []
metrics:
  duration: ~14 minutes
  completed: 2026-05-27
---

# Phase 22 Plan 02: LSTM Autoencoder for MEG ROI Timeseries Summary

LSTM autoencoder with overcomplete latent space (n_latent = 2 * n_roi) and differentiable spectral consistency loss via torch.fft.rfft log-power PSD comparison.

## What Was Done

### Task 1: MEGAutoencoder nn.Module (ecf4827)

Created `src/pyro_dcm/neural_data_models/lstm_autoencoder.py` containing `MEGAutoencoder(nn.Module)`:

- **Encoder:** LSTM -> Linear projection to latent space at every timestep (not just final hidden state)
- **Decoder:** Linear projection -> LSTM back to ROI space
- **Default overcomplete representation:** `n_latent = 2 * n_roi` when `None` is passed
- **Shape contract:** `(batch, T, N_roi)` -> encode -> `(batch, T, N_latent)` -> decode -> `(batch, T, N_roi)`
- **forward()** returns `(reconstruction, latent)` tuple

Created `src/pyro_dcm/neural_data_models/__init__.py` exporting `MEGAutoencoder`.

### Task 2: AutoencoderTrainer + Tests (96cc548)

Created `src/pyro_dcm/neural_data_models/trainer.py` containing `AutoencoderTrainer`:

- **MSE reconstruction loss** as primary criterion
- **Optional spectral consistency loss** via `torch.fft.rfft` (fully differentiable):
  - Computes PSD as `|rfft(x)|^2` per channel
  - Compares log-power spectra: `loss = MSE(log(PSD_x + eps), log(PSD_recon + eps))`
  - Activated by `spectral_weight > 0` (default 0.0 = disabled)
- **Training loop** with per-epoch loss tracking
- **Validation** with per-epoch val loss computation
- **Early stopping** based on validation loss patience
- **Checkpointing** via `torch.save`/`torch.load` (model + optimizer state)
- **evaluate()** for computing mean MSE on any DataLoader

Created 12 unit tests in `tests/test_lstm_autoencoder.py`:

| # | Test | Coverage |
|---|------|----------|
| 1 | test_output_shapes | Forward pass shapes correct |
| 2 | test_default_n_latent | n_latent = 2 * n_roi |
| 3 | test_custom_n_latent | Explicit n_latent overrides |
| 4 | test_encode_decode_shapes | Encode/decode shape complementarity |
| 5 | test_gradient_flow | All parameters receive non-zero gradients |
| 6 | test_batch_invariance | Sample-i output identical in batch=1 vs batch=4 |
| 7 | test_train_reduces_loss | 20 epochs on synthetic data reduces loss |
| 8 | test_checkpoint_roundtrip | Save/load produces identical state_dict |
| 9 | test_early_stopping | patience parameter triggers early exit |
| 10 | test_evaluate | evaluate() returns positive float MSE |
| 11 | test_spectral_loss_nonzero | spectral_weight=1.0 changes total loss |
| 12 | test_spectral_loss_zero_default | spectral_weight=0.0 skips spectral path |

Updated `__init__.py` to export both `MEGAutoencoder` and `AutoencoderTrainer`.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

None -- no architectural decisions needed.

## Verification Results

- `pytest tests/test_lstm_autoencoder.py -v`: 12/12 passed
- `ruff check src/pyro_dcm/neural_data_models/`: All checks passed
- `python -c "from pyro_dcm.neural_data_models import MEGAutoencoder, AutoencoderTrainer"`: OK

## Next Phase Readiness

The autoencoder module is ready for:
- Phase 22-03: Training on MEG simulator data or real Cam-CAN data
- Phase 22-04: Extracting latent dynamics for spectral DCM fitting
- No blockers for downstream plans
