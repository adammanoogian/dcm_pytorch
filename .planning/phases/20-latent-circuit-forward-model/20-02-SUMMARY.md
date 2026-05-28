---
phase: 20-latent-circuit-forward-model
plan: 02
subsystem: inference
tags: [pyro, svi, multi-start, elbo, guide, variational-inference]

# Dependency graph
requires:
  - phase: 04-pyro-generative-models
    provides: run_svi, create_guide baseline infrastructure in guides.py
provides:
  - run_svi with n_restarts and guide_factory for multi-start SVI
  - backward-compatible single-restart path (n_restarts=1 default)
  - NaN resilience in multi-restart SVI (inf penalty, continue)
  - param store save/restore to best restart state
  - 6-test suite covering all multi-start contracts
affects:
  - 20-03-PLAN.md (latent circuit forward model uses run_svi with n_restarts)
  - 20-04-PLAN.md (prior recalibration uses multi-start fitting)
  - 20-05-PLAN.md (recovery validation uses multi-start)
  - Any phase that calls run_svi with n_restarts>1

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Multi-start SVI with best-ELBO selection via guide_factory callable
    - Param store save/restore using pyro.get_param_store().get_state/set_state
    - Internal _run_single_svi helper for shared loop logic (DRY)

key-files:
  created:
    - tests/test_multi_start_svi.py
  modified:
    - src/pyro_dcm/models/guides.py

key-decisions:
  - "n_restarts=1 path is bit-exact with pre-Phase-20 single-run path -- no structural changes to existing code"
  - "guide_factory is required when n_restarts>1 (ValueError on None) -- prevents silent reuse of a guide already in trained state"
  - "NaN restarts get final_loss=inf and are skipped; RuntimeError only if ALL restarts NaN"
  - "Param store restored via get_state/set_state after all restarts -- avoids re-running best restart"
  - "Return dict for n_restarts>1 extends (not replaces) single-restart keys: all_restarts, n_restarts, best_restart_idx"

patterns-established:
  - "guide_factory pattern: partial(create_guide, model, guide_type='auto_normal') as zero-arg callable"
  - "Multi-start logging: pyro_dcm.svi logger at INFO level per restart"

# Metrics
duration: 10min
completed: 2026-05-24
---

# Phase 20 Plan 02: Multi-start SVI Summary

**Extended run_svi with n_restarts + guide_factory multi-start support, backward-compatible param store restoration, and 6-test suite covering all contracts**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-24T20:50:46Z
- **Completed:** 2026-05-24T21:00:47Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended `run_svi()` with `n_restarts: int = 1` and `guide_factory: Callable | None = None` parameters
- Preserved exact backward-compatible single-restart path: no change to dict structure for n_restarts=1
- Multi-restart path: outer loop with fresh guide per restart, NaN resilience (inf penalty), param store save/restore to best ELBO
- Logging via `logging.getLogger("pyro_dcm.svi")` at INFO per restart and for best selection
- 6 tests covering: backward compat, best selection, guide_factory requirement, fresh init, NaN resilience, param store restoration
- All 6 new tests pass (10s); all 30 guide factory tests pass; all 11 SVI integration tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add n_restarts and guide_factory to run_svi** - `11d1090` (feat)
2. **Task 2: Multi-start SVI test suite** - `60beab7` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/pyro_dcm/models/guides.py` - Extended run_svi with multi-start support; added _build_elbo and _run_single_svi helpers; added `import logging` and module-level `_log`
- `tests/test_multi_start_svi.py` - 6 tests for multi-start SVI contracts (new file)

## Decisions Made

- `n_restarts=1` single-run path delegates entirely to `_run_single_svi` with `nan_is_error=True`, preserving pre-Phase-20 RuntimeError behavior on NaN.
- `_run_single_svi` internal helper extracts the shared SVI loop for DRY code between single and multi-restart paths.
- `guide_factory=None` with `n_restarts>1` raises `ValueError` immediately with a descriptive message. This prevents silent misuse where a pre-trained guide would be reused across restarts.
- Param store restoration uses `get_state()` saved after each restart's completion, then `set_state()` on the best -- avoids the performance cost of re-running the best restart.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `run_svi` is ready for Phase 20-03 latent circuit forward model which will call `run_svi` with `n_restarts=10` as minimum (addressing pitfall LC11).
- `guide_factory` pattern documented in docstring and smoke-test example for downstream plan authors.
- No blockers.

---
*Phase: 20-latent-circuit-forward-model*
*Completed: 2026-05-24*
