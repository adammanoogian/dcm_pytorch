---
phase: 02-spectral-dcm-forward-model
plan: 02
subsystem: forward-models
tags: [csd, welch, scipy, spectral, cross-spectral-density, fmri]

# Dependency graph
requires:
  - phase: 01-neural-hemodynamic-forward-model
    provides: BOLD time series output (T, N) from hemodynamic forward model
provides:
  - Empirical CSD computation from BOLD time series via Welch method
  - Frequency interpolation to SPM-compatible target grids
  - Torch wrapper for Pyro model integration
  - Default Welch parameters for fMRI data
affects:
  - 02-spectral-dcm-forward-model (plan 03 integration)
  - 04-pyro-generative-models (spectral DCM observed data preparation)

# Tech tracking
tech-stack:
  added: [scipy.signal.csd]
  patterns: [Welch periodogram CSD, Hermitian enforcement via upper-triangle, numpy-to-torch wrapper]

key-files:
  created:
    - src/pyro_dcm/forward_models/csd_computation.py
    - tests/test_csd_computation.py
  modified: []

key-decisions:
  - "Welch CSD (scipy.signal.csd) over MAR-based CSD for empirical computation — simpler, well-validated, SPM matching deferred to predicted CSD"
  - "np.interp for frequency interpolation (real/imag separately) — simple linear interpolation sufficient for smooth CSD"
  - "Hermitian enforcement via upper-triangle computation + conjugation — exact symmetry guaranteed"

patterns-established:
  - "CSD modules use numpy/scipy (not torch) for data preparation — torch only at model boundary"
  - "bold_to_csd_torch wrapper pattern: numpy computation + torch.as_tensor conversion"

# Metrics
duration: 14min
completed: 2026-03-26
---

# Phase 2 Plan 02: Empirical CSD Computation Summary

**Welch-based empirical CSD from BOLD time series with scipy.signal.csd, Hermitian enforcement, frequency interpolation, and torch wrapper for Phase 4 Pyro integration**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-26T07:02:14Z
- **Completed:** 2026-03-26T07:16:42Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Empirical CSD computation using Welch periodogram with configurable segment length and target frequency grid interpolation
- Hermitian symmetry enforced by construction (upper triangle + conjugate)
- Torch wrapper (bold_to_csd_torch) providing complex128 tensor interface for Phase 4 Pyro models
- 12 unit tests validating shape, dtype, symmetry, spectral properties, edge cases, and cross-framework consistency

## Task Commits

Each task was committed atomically:

1. **Task 1: Empirical CSD computation** - `73ff6b7` (feat)
2. **Task 2: Unit tests for CSD computation** - `22b46a7` (test)

## Files Created/Modified
- `src/pyro_dcm/forward_models/csd_computation.py` - Welch-based empirical CSD with compute_empirical_csd, bold_to_csd_torch, default_welch_params
- `tests/test_csd_computation.py` - 12 unit tests covering shape, dtype, Hermitian symmetry, positive auto-spectra, white noise flatness, sinusoidal peak detection, frequency interpolation, short series handling, torch roundtrip, and default parameters

## Decisions Made
- **Welch CSD approach**: Used scipy.signal.csd with 'density' scaling, matching one-sided PSD convention. The CONTEXT.md decision to use standard signal processing (not SPM's MAR-based approach) was followed.
- **Frequency interpolation**: Linear interpolation (np.interp) on real and imaginary parts separately. Simple and effective for smooth CSD curves.
- **nperseg default**: min(256, T) handles both standard and short fMRI runs gracefully.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Project not installed in editable mode initially; required `pip install -e . --no-deps` in conda base environment (torch DLL issue on Windows with Python 3.13, worked with conda Python 3.12)
- Parallel plan 02-01 accidentally committed then reverted test_csd_computation.py; resolved cleanly since the file was untracked when I committed it

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CSD computation module ready for integration in plan 02-03
- bold_to_csd_torch provides the observed data interface for Phase 4 spectral DCM Pyro model
- All 12 tests pass, validating spectral properties required by downstream modules

---
*Phase: 02-spectral-dcm-forward-model*
*Completed: 2026-03-26*
