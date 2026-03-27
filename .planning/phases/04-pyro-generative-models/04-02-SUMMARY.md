---
phase: 04-pyro-generative-models
plan: 02
subsystem: generative-models
tags: [pyro, dcm, spectral, csd, svi, connectivity]

# Dependency graph
requires:
  - phase: 02-spectral-dcm
    provides: "spectral_dcm_forward, compute_transfer_function, neuronal_noise_csd, observation_noise_csd"
  - phase: 01-neural-hemodynamic
    provides: "parameterize_A"
provides:
  - "spectral_dcm_model Pyro generative function for spectral DCM"
affects: [05-inference-pipeline, 06-validation, 07-amortized-guide]

key-files:
  created:
    - "src/pyro_dcm/models/spectral_dcm_model.py"
    - "tests/test_spectral_dcm_model.py"
  modified: []

key-decisions:
  - "Complex CSD decomposed to stacked real/imag for Pyro Gaussian likelihood"
  - "SPM noise priors: a(2,N), b(2,1), c(2,N) all sampled as Normal(0, 1/64)"

duration: ~10min
completed: 2026-03-27
---

# Phase 04 Plan 02: Spectral DCM Pyro Model Summary

**Pyro generative model for spectral DCM: samples A_free and noise params (a,b,c), computes predicted CSD via eigendecomposition transfer function, evaluates Gaussian likelihood on real/imag decomposed CSD**

## Performance

- **Duration:** ~10 min
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Spectral DCM Pyro model sampling A_free ~ N(0, 1/64) and noise params a/b/c ~ N(0, 1/64)
- Complex CSD decomposed to stacked real/imaginary vector for Pyro Gaussian observation
- Unit tests covering model trace structure, shapes, noise sampling, CSD decomposition, SVI smoke test

## Task Commits

1. **Task 1: Spectral DCM Pyro generative model** - `03b12b0` (feat)
2. **Task 2: Unit tests for spectral DCM Pyro model** - `ad97431` (test)

## Files Created/Modified
- `src/pyro_dcm/models/spectral_dcm_model.py` - Pyro generative model: spectral_dcm_model function
- `tests/test_spectral_dcm_model.py` - Unit tests for spectral DCM Pyro model

## Deviations from Plan
None.

## Next Phase Readiness
- Spectral DCM Pyro model ready for inference pipeline (Phase 5)
- Complex CSD handling pattern established for downstream use

---
*Phase: 04-pyro-generative-models*
*Completed: 2026-03-27*
