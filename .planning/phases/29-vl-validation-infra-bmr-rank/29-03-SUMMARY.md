---
phase: 29-vl-validation-infra-bmr-rank
plan: 03
subsystem: testing
tags: [variational-laplace, csd-precision, c-order, numerical-robustness, regression-test, task-dcm]

# Dependency graph
requires:
  - phase: 28-variational-laplace-engine
    provides: "ForwardModel protocol + compute_csd_precision (spm_dcm_csd_Q) + TaskDCMForward.build_precision"
provides:
  - "C-order CSD round-trip regression test locking the (j fastest, i, w) index contract (guards commit 64e326f against regression)"
  - "TaskDCMForward.build_precision intractability guard with expected-vs-actual matrix size and dt>=0.1 floor documentation"
  - "task precision guard regression test (tractable identity + oversized ValueError)"
  - "registered `vl` pytest marker for Variational Laplace validation tests"
affects: [30-recovery-sweep, 32-spm-cross-validation, vl-validation-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-loud intractability guard with expected-vs-actual error message (project rule)"
    - "Pure-tensor-algebra regression test guarding a fixed index-layout bug class (no fit, <1s)"

key-files:
  created:
    - tests/test_csd_corder_roundtrip.py
    - tests/test_task_precision_guard.py
  modified:
    - src/pyro_dcm/inference/forward_models.py
    - pyproject.toml

key-decisions:
  - "_TASK_PRECISION_MAX_DIM = 5000 cap on the dense (T*N, T*N) task-DCM precision; enforces dt>=0.1 floor"
  - "Registered `vl` pytest marker (was unregistered) to support the plan's `-m vl` selection"

patterns-established:
  - "Index-contract regression test: assert flat[idx]==tensor[w,i,j] for the C-order (j,i,w) map on an asymmetric complex input"
  - "Intractability guard: ValueError message reports actual size, resulting dense shape, expected cap, and the mitigating parameter floor"

# Metrics
duration: 16 min
completed: 2026-06-10
---

# Phase 29 Plan 03: VL Numerical-Robustness Guards Summary

**C-order CSD round-trip regression test (locks commit 64e326f) plus a fail-loud task-DCM precision intractability guard with expected-vs-actual matrix size and a documented dt>=0.1 floor.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-06-10T16:27:11Z
- **Completed:** 2026-06-10T16:43:04Z
- **Tasks:** 3
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- **VLREC-05**: a C-order CSD round-trip regression test that asserts the
  `j = idx % N; i = (idx // N) % N; w = idx // (N*N)` map recovers every element
  of an asymmetric complex `(F, N, N)` CSD, and that `compute_csd_precision`
  preserves the asymmetric block layout (no transpose/symmetrize). This locks
  the contract fixed in commit 64e326f before any SPM cross-validation (Phase 32).
- **VLROBUST-02**: `TaskDCMForward.build_precision` now fails loud when
  `ny = T*N > 5000`, with an error message reporting the actual size, the
  resulting dense `(ny, ny)` shape, the expected cap, and the `dt >= 0.1` floor.
  The tractable path (`ny <= 5000`) is unchanged.
- Both guards covered by fast (<1s execution) regression tests; the `vl` pytest
  marker was registered so `-m vl` selection works.

## Task Commits

Each task was committed atomically:

1. **Task 1: C-order CSD round-trip regression test** - `bc30477` (test)
2. **Task 2: task DCM precision intractability guard** - `d12eb84` (feat)
3. **Task 3: task precision guard regression test** - `09c9375` (test)

**Plan metadata:** (docs commit follows this summary)

## Files Created/Modified

- `tests/test_csd_corder_roundtrip.py` - C-order index round-trip + precision
  block-structure regression test (guards commit 64e326f / pitfall S4).
- `tests/test_task_precision_guard.py` - tractable-identity + oversized-ValueError
  regression test for the precision guard.
- `src/pyro_dcm/inference/forward_models.py` - added `_TASK_PRECISION_MAX_DIM`
  constant and the intractability guard + docstring in
  `TaskDCMForward.build_precision`.
- `pyproject.toml` - registered the `vl` pytest marker.

## Decisions Made

- **[29-03-D1] `_TASK_PRECISION_MAX_DIM = 5000` cap on the dense task-DCM
  precision.** A dense `(T*N, T*N)` float64 matrix is intractable at fine dt
  (dt=0.01, 100s, N=4 -> 4e4 -> ~13 GB). The cap enforces the `dt >= 0.1` floor
  and the error message names the mitigation. Matches the plan-specified value.
- **[29-03-D2] Registered the `vl` pytest marker** (it was absent from
  `pyproject.toml`), a prerequisite for the plan's `pytest -m vl` verification.
  Folded into the Task 1 commit since it enables that test's selection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Registered the missing `vl` pytest marker**
- **Found during:** Task 1 (C-order round-trip test)
- **Issue:** The plan marks all tests `@pytest.mark.vl` and verifies with
  `pytest -m vl`, but `vl` was not in `pyproject.toml`'s `markers` list (only
  slow/spm/tapas/mne/foundation/latent were registered).
- **Fix:** Added `"vl: marks ... Variational Laplace validation ..."` to
  `[tool.pytest.ini_options].markers`.
- **Files modified:** `pyproject.toml`
- **Verification:** `pytest -m vl` selects the new tests; ruff clean.
- **Committed in:** `bc30477` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Necessary to satisfy the plan's own `-m vl` verification
commands. No scope creep.

## Issues Encountered

- **Pre-existing failures in `tests/test_vl_forward_model_protocol.py` (5 of the
  task-DCM cases).** Confirmed unrelated to this plan: they fail at test setup
  with `make_block_stimulus()`/`simulate_task_dcm()` signature mismatches
  (`missing 1 required positional argument: 'rest_duration'`, `unexpected keyword
  argument 'dt_sim'`) and reproduce on the pre-plan commit `a064e69` before any
  of my edits. They never reach `build_precision`. Left untouched (out of scope;
  test-helper signature drift, not a precision-guard regression).
- **Pre-existing ruff D102 debt (23 missing-docstring violations) in
  `forward_models.py`** on unrelated methods. My `build_precision` edit *added*
  a docstring (net -1 D102). New test files are fully ruff-clean. mypy shows only
  pre-existing protocol-level `object`/`dict` typing issues unrelated to the
  added constant/guard. Left untouched (project-wide lint debt, out of scope).

## Next Phase Readiness

- The two numerical-robustness guards required before any recovery sweep
  (Phase 30) or SPM cross-validation (Phase 32) are now in place and locked by
  regression tests.
- No blockers introduced. Pre-existing `test_vl_forward_model_protocol.py`
  task-DCM setup-helper signature drift is a separate, latent issue worth a
  cleanup pass but does not affect the VL engine or these guards.

---
*Phase: 29-vl-validation-infra-bmr-rank*
*Completed: 2026-06-10*
