---
phase: 02-spectral-dcm-forward-model
plan: 01
subsystem: forward-model
tags: [spectral-dcm, transfer-function, eigendecomposition, noise-spectrum, CSD, complex128]

# Dependency graph
requires:
  - phase: 01-neural-hemodynamic-forward-model
    provides: "Neural state equation, A matrix parameterization, project structure"
provides:
  - "Eigendecomposition-based spectral transfer function H(w) per [REF-010] Eq. 3"
  - "Predicted CSD assembly S(w) = H @ Gu @ H^H + Gn per [REF-010] Eq. 4"
  - "Neuronal noise CSD (1/f power-law, diagonal) per [REF-010] Eq. 5-6"
  - "Observation noise CSD (global + regional) per [REF-010] Eq. 7"
  - "Default SPM12 frequency grid and noise priors"
  - "spectral_dcm_forward convenience wrapper for full pipeline"
affects:
  - 02-03 (integration plan combines 02-01 and 02-02 exports)
  - phase-03 (spectral simulator uses transfer function and noise models)
  - phase-04 (Pyro generative model calls spectral_dcm_forward)
  - phase-06 (SPM validation compares predicted CSD output)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Eigendecomposition-based modal transfer (avoid direct matrix inversion per frequency)"
    - "Eigenvalue stabilization: clamp real parts to max(-1/32) for fMRI"
    - "torch.complex128 throughout spectral computations"
    - "torch.diag_embed for efficient diagonal CSD construction"
    - "torch.einsum for modal summation in transfer function"

key-files:
  created:
    - src/pyro_dcm/forward_models/spectral_transfer.py
    - src/pyro_dcm/forward_models/spectral_noise.py
    - tests/test_spectral_transfer.py
    - tests/test_spectral_noise.py
  modified: []

key-decisions:
  - "Eigenvalue stabilization at -1/32 (SPM convention for fMRI frequencies)"
  - "C_in = C_out = identity for standard spDCM (hemodynamic Jacobian deferred to Phase 2/3)"
  - "SPM scaling constant C = 1/256 applied to all noise spectra"
  - "Observation noise exponent divided by 2 (flatter spectrum than neuronal)"
  - "Global observation noise divided by 8.0 (matching spm_csd_fmri_mtf.m)"

patterns-established:
  - "Modal transfer function: eigdecomp -> stabilize -> project -> einsum assembly"
  - "Noise parameterization: (2, N) log-space params [amplitude, exponent] per group"
  - "Frequency grid: linspace(1/128, 1/(2*TR), 32) as SPM default"

# Metrics
duration: 17min
completed: 2026-03-26
---

# Phase 2 Plan 1: Spectral Transfer Function and Noise Models Summary

**Eigendecomposition-based spectral transfer function H(w) with SPM noise models (4N+2 params), validated against direct matrix inversion within 1e-10 tolerance and verified Hermitian/PSD predicted CSD**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-26T07:01:49Z
- **Completed:** 2026-03-26T07:18:55Z
- **Tasks:** 3/3 completed
- **Files created:** 4
- **Tests added:** 27

## Accomplishments

- Implemented eigendecomposition-based transfer function matching SPM12 spm_dcm_mtf.m with eigenvalue stabilization
- Implemented neuronal (1/f power-law) and observation (global + regional) noise spectral models matching SPM12 spm_csd_fmri_mtf.m
- Predicted CSD assembly verified as Hermitian and positive-semidefinite at all frequencies
- Full pipeline differentiable via autograd (finite gradients w.r.t. A matrix)
- 27 comprehensive unit tests covering shape, correctness, stability, and mathematical properties

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Transfer function + predicted CSD | `8638fe0` | `spectral_transfer.py` |
| 1-fix | Remove accidentally staged 02-02 file | `d703417` | (corrective) |
| 2 | Noise spectral models | `ec9596e` | `spectral_noise.py` |
| 3 | Unit tests | `f6b74b4` | `test_spectral_transfer.py`, `test_spectral_noise.py` |

## Key Files

- `src/pyro_dcm/forward_models/spectral_transfer.py` -- Transfer function H(w) via eigendecomposition, predicted CSD assembly, frequency grid, convenience wrapper
- `src/pyro_dcm/forward_models/spectral_noise.py` -- Neuronal noise CSD (diagonal 1/f), observation noise CSD (global + regional), default priors
- `tests/test_spectral_transfer.py` -- 14 tests: grid, transfer function, CSD properties, pipeline, autograd
- `tests/test_spectral_noise.py` -- 13 tests: noise shapes, 1/f behavior, amplitude scaling, priors

## Decisions Made

- **Eigenvalue stabilization at -1/32**: SPM convention for fMRI frequencies. Eigenvalues with real parts > -1/32 are clamped before transfer function computation. This prevents numerical blow-up at low frequencies while preserving the spectral shape.
- **Identity C_in/C_out for standard spDCM**: The full hemodynamic Jacobian is not used in this plan. Standard spDCM uses identity matrices for input/output projection. The hemodynamic linearization at steady state will be addressed in Phase 2/3 integration.
- **SPM noise constants**: C = 1/256 scaling, observation exponent /2, global observation /8.0 -- all match spm_csd_fmri_mtf.m exactly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Accidentally staged file from parallel plan 02-02**
- **Found during:** Task 1 commit
- **Issue:** `tests/test_csd_computation.py` was pre-staged in git index from parallel 02-02 execution
- **Fix:** Created corrective commit removing the file from tracking (kept on disk for 02-02)
- **Commit:** `d703417`

---

**Total deviations:** 1 auto-fixed (blocking git staging issue)
**Impact on plan:** No impact on code quality. Corrective commit restores clean separation between parallel plans.

## Issues Encountered

- PyTorch DLL load failure on Windows Python 3.13 -- resolved by using conda environment (dcm_psilocybin_clean) with Python 3.11 and torch 2.11.0+cpu

## Verification Results

1. Both modules import without error
2. All 27 new tests pass
3. Full test suite (94 tests) passes with no regressions
4. No NaN or Inf in any test output
5. All files contain `from __future__ import annotations`
6. All mathematical functions have docstrings citing [REF-010] Eq. N
7. Transfer function validated against direct matrix inversion within 1e-10

## Next Phase Readiness

- spectral_transfer.py and spectral_noise.py ready for export via `forward_models/__init__.py` (deferred to 02-03 integration plan)
- spectral_dcm_forward ready for use by spectral simulator (Phase 3) and Pyro generative model (Phase 4)
- MAR smoothing step documented as omitted (Phase 6 validation may need it as optional post-processing)

---
*Phase: 02-spectral-dcm-forward-model*
*Plan: 01*
*Completed: 2026-03-26*
