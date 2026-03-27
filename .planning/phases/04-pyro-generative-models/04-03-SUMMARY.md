---
phase: 04-pyro-generative-models
plan: 03
subsystem: generative-models
tags: [pyro, dcm, rdcm, svi, autoguide, regression, frequency-domain, connectivity]

# Dependency graph
requires:
  - phase: 04-pyro-generative-models
    plan: 01
    provides: "task_dcm_model Pyro generative function, models/__init__.py"
  - phase: 04-pyro-generative-models
    plan: 02
    provides: "spectral_dcm_model Pyro generative function, decompose_csd_for_likelihood"
  - phase: 03-rdcm-forward
    provides: "create_regressors, generate_bold, get_hrf, rigid_inversion, sparse_inversion"
provides:
  - "rdcm_model Pyro generative function for regression DCM"
  - "create_guide factory (AutoNormal with configurable init_scale)"
  - "run_svi runner (ClippedAdam + LR decay + NaN detection)"
  - "extract_posterior_params helper"
  - "Complete package exports: all 3 models + inference utilities from pyro_dcm.models and pyro_dcm"
affects: [05-inference-pipeline, 06-validation, 07-amortized-guide]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Region-wise Pyro model with Python loop over regions (variable-size theta per region)"
    - "AutoNormal guide factory with init_scale=0.01 for ODE stability"
    - "SVI runner with ClippedAdam, exponential LR decay, and NaN ELBO detection"

key-files:
  created:
    - "src/pyro_dcm/models/rdcm_model.py"
    - "src/pyro_dcm/models/guides.py"
    - "tests/test_rdcm_model.py"
    - "tests/test_svi_integration.py"
  modified:
    - "src/pyro_dcm/models/__init__.py"
    - "src/pyro_dcm/__init__.py"

key-decisions:
  - "Per-region Python loop (not plate) for rDCM because each region has different D_r"
  - "N(0,1) prior on theta (broader than analytic VB) since Pyro model serves ELBO comparison"
  - "Gamma(2,1) prior on noise precision matching analytic VB convention"
  - "math.isnan for NaN detection (not torch.isnan) for scalar loss values"
  - "Mock-based NaN detection test (Pyro validates distribution params before NaN can propagate)"

patterns-established:
  - "Guide factory pattern: create_guide(model, init_scale) -> AutoNormal"
  - "SVI runner pattern: run_svi(model, guide, args) -> {losses, final_loss, num_steps}"
  - "Posterior extraction pattern: extract_posterior_params(guide, args) -> {median, params}"
  - "Integration test pattern: simulator -> create_regressors -> model -> guide -> run_svi -> extract_posterior_params"

# Metrics
duration: 13min
completed: 2026-03-27
---

# Phase 04 Plan 03: rDCM Pyro Model + Guide Factory + SVI Runner Summary

**rDCM Pyro model with per-region variable-size theta sampling, AutoNormal guide factory (init_scale=0.01), SVI runner (ClippedAdam + LR decay + NaN detection), and integration tests proving all 3 DCM variants work end-to-end with SVI**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-27T09:01:28Z
- **Completed:** 2026-03-27T09:14:40Z
- **Tasks:** 2
- **Files created:** 4
- **Files modified:** 2

## Accomplishments
- rDCM Pyro model wrapping frequency-domain regression likelihood with per-region theta vectors sized by active connections in a_mask/c_mask
- Guide factory (create_guide) and SVI runner (run_svi) providing shared inference infrastructure for all 3 DCM variants
- All 3 models verified end-to-end: no NaN in SVI, loss decreases over 100 steps, posteriors extractable
- 15 new tests (6 rDCM unit + 9 integration), all 232 tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: rDCM Pyro model + guide factory + SVI runner + package exports** - `d8b244b` (feat)
2. **Task 2: Unit tests for rDCM model + integration tests for all three models** - `d82490c` (test)

## Files Created/Modified
- `src/pyro_dcm/models/rdcm_model.py` - Pyro model: per-region theta sampling, Gamma noise prior, Gaussian likelihood
- `src/pyro_dcm/models/guides.py` - create_guide (AutoNormal factory), run_svi (SVI runner), extract_posterior_params
- `src/pyro_dcm/models/__init__.py` - Updated exports: all 3 models + guide + runner + extraction
- `src/pyro_dcm/__init__.py` - Top-level exports: models + inference utilities
- `tests/test_rdcm_model.py` - 6 unit tests: region sites, theta shapes, variable masks, log_prob, SVI
- `tests/test_svi_integration.py` - 9 integration tests: guide factory, SVI runner, end-to-end all models

## Decisions Made
- **Per-region Python loop for rDCM**: Each region can have a different number of active connections (D_r), making vectorization with pyro.plate impossible. A Python loop over regions with individual pyro.sample sites is the correct approach.
- **N(0,1) prior on theta**: Intentionally broader than the analytic VB priors (which use SPM-convention precision scaling). The Pyro model serves ELBO comparison and amortization, not primary inference.
- **Mock-based NaN detection test**: Pyro validates distribution parameters and raises ValueError before NaN can propagate to ELBO computation. Used unittest.mock.patch on SVI.step to reliably test the NaN detection logic.
- **math.isnan over torch.isnan**: The loss value from SVI.step is a Python float, not a tensor. Using math.isnan is cleaner and avoids unnecessary tensor creation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed init_scale test approach for AutoNormal**
- **Found during:** Task 2 (integration tests)
- **Issue:** AutoNormal does not expose `init_scale` as a direct attribute. The test `guide.init_scale == 0.05` raised AttributeError.
- **Fix:** Changed test to verify that different init_scale values produce different guide scale parameters after one SVI step. This functionally validates the init_scale parameter is used.
- **Files modified:** tests/test_svi_integration.py
- **Committed in:** d82490c

**2. [Rule 1 - Bug] Fixed NaN detection test to use mock approach**
- **Found during:** Task 2 (integration tests)
- **Issue:** Pyro validates distribution parameters and raises ValueError when scale=0.0 or loc=NaN, preventing NaN from reaching the ELBO computation. Cannot reliably trigger NaN ELBO through a real model.
- **Fix:** Used unittest.mock.patch on SVI.step to return NaN on the second call, verifying the NaN detection logic in run_svi works correctly.
- **Files modified:** tests/test_svi_integration.py
- **Committed in:** d82490c

---

**Total deviations:** 2 auto-fixed (2 bugs in test assertions)
**Impact on plan:** Essential fixes for test correctness. No scope creep. All tests validate the intended behavior.

## Issues Encountered
None beyond the deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 3 DCM Pyro models complete: task_dcm_model, spectral_dcm_model, rdcm_model
- Shared inference infrastructure ready: create_guide, run_svi, extract_posterior_params
- Package exports complete: all models importable from pyro_dcm.models and pyro_dcm
- 232 tests passing, ready for Phase 5 (inference pipeline with parameter recovery)
- Phase 4 complete: all 3 plans (04-01, 04-02, 04-03) delivered

---
*Phase: 04-pyro-generative-models*
*Completed: 2026-03-27*
