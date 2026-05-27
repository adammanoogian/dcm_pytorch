---
phase: 22-dcm-neural-interpretability
plan: 01
subsystem: spectral-dcm
tags: [meg, spectral-dcm, eigenvalue-clamp, frequency-grid, backward-compat]
dependency-graph:
  requires: []
  provides:
    - "default_frequency_grid_meg (1-45 Hz for MEG electrophysiology)"
    - "Parameterized eig_clamp in compute_transfer_function and spectral_dcm_forward"
    - "prior_a_var and eig_clamp keyword args in spectral_dcm_model"
  affects:
    - "22-04 (fit spectral DCM to MEG CSD -- uses eig_clamp and prior_a_var)"
    - "22-05 (compare raw vs latent DCM -- same infrastructure)"
tech-stack:
  added: []
  patterns:
    - "Keyword-only parameters for backward-compatible extension"
    - "Nyquist validation in frequency grid factory"
key-files:
  created:
    - tests/test_meg_spectral_dcm.py
  modified:
    - src/pyro_dcm/forward_models/spectral_transfer.py
    - src/pyro_dcm/forward_models/__init__.py
    - src/pyro_dcm/models/spectral_dcm_model.py
decisions:
  - id: "22-01-D1"
    summary: "eig_clamp=None disables clamping entirely; -1.0 is recommended for MEG"
  - id: "22-01-D2"
    summary: "prior_a_var uses variance (not std) to match SPM12 convention"
metrics:
  duration: "~12 min"
  completed: "2026-05-27"
---

# Phase 22 Plan 01: Adapt Spectral DCM for MEG Electrophysiology Summary

Parameterized eigenvalue clamp and prior variance in spectral DCM, added 1-45 Hz MEG frequency grid with Nyquist validation

## What Was Done

### Task 1: Parameterize eigenvalue clamp and add MEG frequency grid
**Commit:** 2b5fc93

- Added `default_frequency_grid_meg(sfreq=250.0, n_freqs=64)` returning `torch.linspace(1.0, 45.0, n_freqs)` with Nyquist validation
- Added keyword-only `eig_clamp` parameter to `compute_transfer_function` (default `-1/32`, `None` disables)
- Propagated `eig_clamp` through `spectral_dcm_forward`
- Exported `default_frequency_grid_meg` from `forward_models/__init__.py`
- All 27 existing spectral tests pass unchanged

### Task 2: Add prior_a_var to spectral_dcm_model and write MEG tests
**Commit:** c5b7dac

- Added keyword-only `prior_a_var` (default `1/64`) and `eig_clamp` (default `-1/32`) to `spectral_dcm_model`
- Replaced hardcoded `(1.0/64.0)**0.5` with `prior_a_var**0.5`
- Passed `eig_clamp` through to `spectral_dcm_forward`
- Created `tests/test_meg_spectral_dcm.py` with 10 tests across 3 test classes
- Removed unused `F` variable and fixed import sorting (pre-existing ruff lints)

## Test Results

- **New tests:** 10/10 pass (`tests/test_meg_spectral_dcm.py`)
- **Existing tests:** 27/27 pass (`tests/test_spectral_transfer.py` + `tests/test_spectral_dcm_model.py`)
- **Ruff:** Clean on all modified files

## Decisions Made

1. **[22-01-D1] eig_clamp=None disables clamping entirely.** For MEG, `eig_clamp=-1.0` is recommended (matches neural timescales). `None` relies entirely on `parameterize_A` for stability.

2. **[22-01-D2] prior_a_var uses variance (not std).** Matches SPM12 convention where prior precision is specified as variance. The standard deviation is computed as `prior_a_var**0.5`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_none_disables eigenvalue direction**
- **Found during:** Task 2
- **Issue:** Test used eigenvalue -2.0 (more negative than -1/32) expecting clamp to change it, but `torch.clamp(max=-1/32)` only pushes values *up* toward -1/32, not down.
- **Fix:** Changed test to use eigenvalue -0.01 (less negative than -1/32=-0.03125), where clamping actually pulls it to -0.03125.
- **Files modified:** tests/test_meg_spectral_dcm.py
- **Commit:** c5b7dac

**2. [Rule 1 - Bug] Removed unused F variable (pre-existing ruff F841)**
- **Found during:** Task 2
- **Issue:** `F = freqs.shape[0]` was assigned but never used in spectral_dcm_model.
- **Fix:** Removed the assignment.
- **Files modified:** src/pyro_dcm/models/spectral_dcm_model.py
- **Commit:** c5b7dac

**3. [Rule 1 - Bug] Fixed import sorting (pre-existing ruff I001)**
- **Found during:** Task 2
- **Issue:** Imports in spectral_dcm_model.py and forward_models/__init__.py were not sorted per ruff isort rules.
- **Fix:** Ran `ruff check --fix` to auto-sort.
- **Files modified:** src/pyro_dcm/models/spectral_dcm_model.py, src/pyro_dcm/forward_models/__init__.py
- **Commits:** 2b5fc93, c5b7dac

## Backward Compatibility

All new parameters are keyword-only with defaults that reproduce the exact pre-change behavior:

| Parameter | Default | fMRI behavior |
|-----------|---------|---------------|
| `eig_clamp` | `-1/32` | Identical to hardcoded clamp |
| `prior_a_var` | `1/64` | Identical to hardcoded `(1/64)**0.5` std |

No existing caller needs modification. The plan asked for 8 tests; 10 were written (3 for frequency grid, 4 for eig_clamp, 3 for model parameters).

## Next Phase Readiness

Plan 22-01 provides the foundation for fitting spectral DCM to MEG data:
- `default_frequency_grid_meg()` is ready for use in pipeline scripts
- `eig_clamp=-1.0` and `prior_a_var=1/16` are the recommended MEG settings
- No blockers for downstream plans 22-04 and 22-05
