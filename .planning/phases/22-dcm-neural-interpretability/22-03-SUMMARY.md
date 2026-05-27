---
phase: 22-dcm-neural-interpretability
plan: 03
subsystem: simulators
tags: [meg-simulator, ornstein-uhlenbeck, sensorimotor-network, timeseries-generation]
depends_on:
  requires: []
  provides:
    - simulate_meg_timeseries (OU SDE timeseries from known A)
    - make_sensorimotor_A (10-region bilateral sensorimotor connectivity)
    - generate_meg_dataset (train/val splits with ground-truth A)
  affects:
    - 22-04 (LSTM autoencoder training uses MEG datasets)
    - 22-05 (spectral DCM fitting on autoencoder latent dynamics)
tech-stack:
  added: []
  patterns:
    - Euler-Maruyama SDE integration for time-domain spectral DCM
    - Structured A matrix factory for sensorimotor network simulation
key-files:
  created:
    - src/pyro_dcm/simulators/meg_simulator.py
    - tests/test_meg_simulator.py
  modified:
    - src/pyro_dcm/simulators/__init__.py
decisions:
  - id: 22-03-D1
    summary: "OU process uses Euler-Maruyama with dt=1/sfreq, not adaptive ODE solver"
    rationale: "Simple, fast, matches plan specification; spectral DCM linear model doesn't need adaptive stepping"
  - id: 22-03-D2
    summary: "CSD consistency test uses 100 samples x 20s at 50 Hz with correlation > 0.5 threshold"
    rationale: "Finite-sample OU noise requires many realizations and lenient threshold; 50 Hz keeps test fast"
metrics:
  duration: ~13 minutes
  completed: 2026-05-27
---

# Phase 22 Plan 03: MEG Timeseries Simulator Summary

**One-liner:** Ornstein-Uhlenbeck SDE simulator generating (n_samples, T, N) MEG-like timeseries from known A matrix, with 10-region sensorimotor network factory and CSD consistency validation against analytical spectral DCM.

## What Was Done

### Task 1: MEG timeseries simulator and sensorimotor network factory

Created `src/pyro_dcm/simulators/meg_simulator.py` with three public functions:

1. **`simulate_meg_timeseries(A, *, sfreq, duration, n_samples, sigma, seed)`**: Generates multivariate timeseries via Euler-Maruyama integration of the OU SDE `dx = Ax dt + sigma dW`. Validates A stability (all eigenvalues with negative real parts). Returns dict with `timeseries` tensor shape `(n_samples, T, N)` and metadata.

2. **`make_sensorimotor_A(*, self_connection, intra_strength, bilateral_strength, feedforward_strength, seed)`**: Creates a 10x10 structured A matrix with: M1-S1 bidirectional coupling, PMC/SMA->M1 feedforward, A1->M1 auditory-motor, bilateral homotopic connections, and small random noise on other entries. Guarantees stability via eigenvalue-checked rescaling.

3. **`generate_meg_dataset(A, *, n_roi, sfreq, duration, n_train, n_val, sigma, seed)`**: Convenience wrapper producing train/val splits with ground-truth A. Uses `make_sensorimotor_A` for 10-region default or `make_stable_A_spectral` for other sizes.

Updated `src/pyro_dcm/simulators/__init__.py` to export all three functions.

### Task 2: Comprehensive tests with CSD consistency check

Created `tests/test_meg_simulator.py` with 19 tests across 10 categories:

- **(a) Shapes**: 4 tests verifying (n_samples, T, N) for various configs including sensorimotor
- **(b) Reproducibility**: 2 tests for seed determinism and seed variation
- **(c) Stability**: 1 test confirming no NaN/Inf in 10s simulation
- **(d) Variance bounds**: 1 test that variance is positive and bounded
- **(e) Sensorimotor shape**: 1 test for (10, 10) float64
- **(f) Sensorimotor stability**: 1 test across 4 seeds
- **(g) Sensorimotor structure**: 4 tests for diagonal negativity, M1-S1 connections, bilateral symmetry, ROI names
- **(h) CSD consistency**: 1 test generating 100 OU trials at 50 Hz x 20s, computing average empirical CSD, and verifying auto-spectra correlation > 0.5 with analytical spectral DCM prediction
- **(i) Dataset shapes**: 2 tests for train/val shapes with default and custom n_roi
- **(j) Default A**: 2 tests verifying sensorimotor ROI names are set when A=None/n_roi=10

All 19 tests pass in ~4.4 seconds. Ruff clean. No regressions in existing spectral simulator tests (26/26 pass).

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 22-03-D1 | Euler-Maruyama with dt=1/sfreq for OU integration | Simple, fast, matches linear stochastic model; adaptive solver unnecessary |
| 22-03-D2 | CSD consistency: 100 samples, 20s, 50 Hz, r > 0.5 | Finite-sample noise requires averaging; 50 Hz keeps test under 5s |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- [x] All 19 tests pass (`pytest tests/test_meg_simulator.py -v`)
- [x] `ruff check` clean on all new files
- [x] Top-level imports work (`from pyro_dcm.simulators import simulate_meg_timeseries, make_sensorimotor_A, generate_meg_dataset`)
- [x] No regressions in existing simulator tests (26/26 pass)

## Next Phase Readiness

Plan 22-04 (LSTM autoencoder training pipeline) can proceed. It will use `generate_meg_dataset` to create training data with known ground-truth A for the autoencoder.
