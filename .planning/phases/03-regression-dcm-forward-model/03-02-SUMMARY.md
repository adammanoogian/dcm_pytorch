---
phase: 03-regression-dcm-forward-model
plan: 02
subsystem: forward-models
tags: [rdcm, variational-bayes, analytic-posterior, free-energy, ard-sparsity, binary-indicators]
dependency_graph:
  requires:
    - phase: 03-01
      provides: rdcm-forward-pipeline (HRF, BOLD generation, design matrix, create_regressors)
  provides:
    - rdcm-analytic-posterior (rigid and sparse VB inversion)
    - rdcm-free-energy (5-component rigid, 7-component sparse)
    - rdcm-prior-specification (rigid and sparse, matching Julia get_priors.jl)
    - rdcm-standalone-likelihood (compute_rdcm_likelihood)
  affects: [03-03-rdcm-simulator, 04-pyro-generative-models, 05-parameter-recovery, 06-validation]
tech_stack:
  added: []
  patterns: [region-wise-vb-inversion, analytic-gaussian-gamma-posterior, ard-binary-indicators, random-sweep-z-update, multi-rerun-best-selection]
key_files:
  created:
    - src/pyro_dcm/forward_models/rdcm_posterior.py
    - tests/test_rdcm_posterior.py
  modified: []
key_decisions:
  - "torch.as_tensor for scalar-to-tensor conversion avoiding copy warnings"
  - "Confound prior precision = 1.0 (weak prior) for confound columns"
  - "l0 clamped at 1e16 max to handle inf precision from absent connections safely"
patterns_established:
  - "Region-wise VB loop: precompute W=X^T@X, V=X^T@Y, iterate Sigma/mu/beta/F"
  - "Sparse z-update: random permutation sweep with sigmoid gating"
  - "Multi-rerun selection: n_reruns independent runs, select max free energy"
  - "Hard thresholding: |mu| < 1e-5 -> mu=0, z=0"
metrics:
  duration: "~12 minutes"
  completed: "2026-03-26"
---

# Phase 3 Plan 02: rDCM Analytic Posterior Summary

**Region-wise analytic VB posterior (rigid + sparse) with 5/7-component free energy, ARD binary indicators with random sweep z-update, and multi-rerun best-selection following Julia RegressionDynamicCausalModeling.jl.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-26T09:46:47Z
- **Completed:** 2026-03-26T09:58:59Z
- **Tasks:** 3/3
- **Files modified:** 2

## Accomplishments

- Implemented complete rDCM analytic posterior inference matching Julia implementation
- Rigid VB inversion converges in < 50 iterations and recovers known parameters
- Sparse VB inversion correctly prunes absent connections (z -> 0) and retains present ones (z -> 1)
- Free energy computation with 5 components (rigid) and 7 components (sparse) verified against manual calculations
- Prior specification exactly matches Julia get_priors.jl: A diagonal mean = -0.5, precision scaling 8/nr (off-diag), 8*nr (diag), Gamma(2,1)
- Standalone likelihood function exposed for testing and downstream use
- All 33 unit tests pass with zero warnings

## Task Commits

Each task was committed atomically:

1. **Task 1: Prior specification, rigid VB inversion, and standalone likelihood** - `e1be682` (feat)
2. **Task 2: Sparse VB inversion with binary indicators** - `8935d99` (feat)
3. **Task 3: Unit tests for analytic posterior, likelihood, and free energy** - `bae070c` (test)

## Files Created/Modified

- `src/pyro_dcm/forward_models/rdcm_posterior.py` -- Complete rDCM analytic posterior: get_priors_rigid, get_priors_sparse, compute_rdcm_likelihood, compute_free_energy_rigid, compute_free_energy_sparse, rigid_inversion, sparse_inversion
- `tests/test_rdcm_posterior.py` -- 33 unit tests covering priors (9), likelihood (4), free energy rigid (4), rigid inversion (7), sparse inversion (6), free energy sparse (2), integration (1)

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Confound prior precision = 1.0 | Weak prior on confound regressors; they are auxiliary, not connectivity parameters |
| l0 clamped at 1e16 max | Absent connections in rigid masking produce inf precision; clamping prevents numerical issues in matrix inversion |
| torch.as_tensor for g_i in sparse | Avoids UserWarning about tensor copy construction when g_i is already a tensor |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed torch.tensor copy warning in sparse z-update**
- **Found during:** Task 3 (running tests)
- **Issue:** `torch.tensor(g_i, dtype=dtype)` produces UserWarning when g_i is already a tensor
- **Fix:** Changed to `torch.as_tensor(g_i, dtype=dtype)` which avoids unnecessary copy
- **Files modified:** src/pyro_dcm/forward_models/rdcm_posterior.py
- **Verification:** All tests pass with -W error (warnings as errors)
- **Committed in:** bae070c (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Minor fix for clean test execution. No scope creep.

## Issues Encountered

None -- plan executed smoothly.

## User Setup Required

None -- no external service configuration required.

## Test Results

```
33 passed in 2.45s
```

All success criteria verified:
- Rigid VB converges in < 50 iterations for well-conditioned problems
- Posterior mean recovers known parameters within 3 posterior standard deviations
- Posterior covariance matrices are positive definite (all eigenvalues > 0)
- Free energy (rigid) has 5 correct components matching [REF-020] Eq. 15
- Free energy (sparse) adds 2 entropy terms for z indicators (7 total)
- Sparse inversion prunes true-zero connections (z < 0.5) and retains true-present (z > 0.5)
- Multiple reruns selects best free energy solution
- Priors: A diagonal mean = -0.5, precision scaling = nr/8 (off-diag), 8*nr (diag), Gamma(2,1)
- Standalone likelihood matches closed-form analytic Gaussian log-likelihood within 1e-8

## Next Phase Readiness

Plan 03-03 (rDCM simulator) can proceed. The posterior module provides:
- `rigid_inversion(X, Y, a_mask, c_mask)` for fixed-architecture inversion
- `sparse_inversion(X, Y, a_mask, c_mask)` for ARD-based sparsity learning
- `compute_rdcm_likelihood(Y_r, X_r, mu_r, tau_r)` for standalone log-likelihood

The `create_regressors` output from 03-01 is directly consumable by both inversion functions. The simulator in 03-03 will combine `generate_bold` + `create_regressors` + `rigid_inversion`/`sparse_inversion` into an end-to-end pipeline.

---
*Completed: 2026-03-26*
