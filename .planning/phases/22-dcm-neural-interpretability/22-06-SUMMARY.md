---
phase: 22-dcm-neural-interpretability
plan: 06
subsystem: interpretability-validation
tags: [validation, comparison, raw-csd, latent-csd, spectral-dcm, perturbation, cluster]
dependency-graph:
  requires: ["22-01", "22-02", "22-03", "22-04", "22-05"]
  provides:
    - Full validation orchestrator comparing raw CSD DCM vs latent CSD DCM perturbation sensitivity
    - Automated sbatch generation for M3 cluster submission
  affects:
    - "Phase 22 scientific conclusions (raw vs latent DCM comparison results)"
tech-stack:
  added: []
  patterns:
    - "Dual-path validation: same ground truth A, same perturbation conditions, different observation models"
    - "Z-score comparison: per-condition detection sensitivity across raw (N=10) and latent (N=20) DCM"
key-files:
  created:
    - scripts/22_run_full_validation.py
  modified: []
decisions: []
metrics:
  duration: ~5 minutes
  completed: 2026-05-27
---

# Phase 22 Plan 06: Full Validation Orchestrator Summary

Dual-path validation orchestrator comparing raw CSD spectral DCM (N=10, no autoencoder) vs latent CSD spectral DCM (N=20, through LSTM-AE) across 9 perturbation conditions, with automated M3 cluster sbatch generation and comparison z-score output.

## What Was Done

### Task 1: Create full validation orchestrator (998046a)

Created `scripts/22_run_full_validation.py` implementing two-path perturbation comparison:

**Path A (raw CSD baseline):**
- Generates timeseries from baseline A (10-region sensorimotor network)
- Computes CSD directly from raw timeseries using `compute_empirical_csd` (Welch periodogram)
- Fits spectral DCM to raw CSD (N=10, fully connected)
- For each of 9 perturbation conditions: generates perturbed timeseries, computes raw CSD, fits DCM, measures delta_A

**Path B (latent CSD via autoencoder):**
- Same baseline A and perturbation conditions
- Trains LSTM autoencoder on baseline data (N_latent = 2 * N_roi = 20)
- Extracts latent trajectories, computes latent CSD, fits spectral DCM (N=20)
- For each perturbation condition: passes perturbed data through SAME autoencoder, fits DCM

**Comparison:**
- Matches conditions by name across both paths
- Computes z-score for each condition in each path
- Prints formatted comparison table to log
- Saves `raw_vs_latent_comparison.npz` with z-scores, detection flags, delta_A arrays, and all metadata

**CLI arguments:**
- `--output-dir`, `--n-train`, `--n-eval`, `--n-latent-multiplier`, `--hidden-size`
- `--ae-epochs`, `--svi-steps`, `--n-restarts`, `--seed`
- `--submit-cluster`: generates sbatch script (24h, 32G, comp partition) and prints submission command

### Task 2: Human verification (PENDING)

Task 2 is a `checkpoint:human-verify` gate -- awaiting cluster results to verify:
- Perturbation detection sensitivity for both paths
- Whether latent DCM detects same/different/better/worse perturbations than raw DCM
- Scientific conclusions about DCM as an interpretability tool

## Test Results

- `python scripts/22_run_full_validation.py --help`: prints usage with all CLI arguments
- `ruff check scripts/22_run_full_validation.py`: all checks passed
- `--submit-cluster` flag: generates valid sbatch script with forward-slash paths

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Windows path separators in generated sbatch script**
- **Found during:** Task 1 verification
- **Issue:** `Path` objects on Windows produce backslash separators, which break bash scripts on the M3 Linux cluster
- **Fix:** Added `.replace("\\", "/")` to the output_dir path in the sbatch template
- **Files modified:** scripts/22_run_full_validation.py
- **Commit:** 998046a

## Decisions Made

None -- no architectural decisions needed. The script reuses existing infrastructure (`compute_empirical_csd`, `prepare_for_spectral_dcm`, `_fit_spectral_dcm` pattern from 22-05).

## Verification

- [x] `python scripts/22_run_full_validation.py --help` prints usage
- [x] `ruff check scripts/22_run_full_validation.py` clean
- [x] `--submit-cluster` generates valid sbatch script
- [x] Script orchestrates both raw and latent paths without duplicating perturbation experiment logic
- [ ] Full validation runs on M3 without error (PENDING -- Task 2)
- [ ] Comparison results are interpretable (PENDING -- Task 2)

## Next Phase Readiness

Plan 22-06 Task 1 delivers the validation orchestrator. Task 2 (human verification of cluster results) is pending -- this is the final gate for Phase 22 scientific conclusions about DCM as an interpretability tool for learned neural representations.
