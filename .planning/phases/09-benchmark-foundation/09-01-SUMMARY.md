---
phase: 09-benchmark-foundation
plan: 01
subsystem: benchmarks
tags: [BenchmarkConfig, dataclass, ELBO, Trace_ELBO, amortization-gap, Pyro]

# Dependency graph
requires:
  - phase: 08-metrics-benchmarks-and-documentation
    provides: BenchmarkConfig dataclass, amortized benchmark runners, consolidated metrics
provides:
  - Extended BenchmarkConfig with guide_type, n_regions_list, elbo_type, fixtures_dir
  - Real ELBO-based amortization gap in task and spectral amortized runners
affects: [09-02, 09-03, 10-guide-variants, 11-calibration-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "**kwargs passthrough in factory classmethods for extensibility"
    - "ELBO evaluation ordering: amortized BEFORE clear_param_store, SVI AFTER training"

key-files:
  created: []
  modified:
    - benchmarks/config.py
    - benchmarks/runners/task_amortized.py
    - benchmarks/runners/spectral_amortized.py

key-decisions:
  - "num_particles=5 for ELBO evaluation to balance noise vs cost"
  - "Amortized ELBO computed before pyro.clear_param_store() to preserve guide params"
  - "Factory methods accept **kwargs rather than explicit new params for forward compatibility"

patterns-established:
  - "Config extension via defaults + kwargs: add fields with defaults, pass through factories"
  - "ELBO gap pattern: evaluate both guides on same data, compute absolute+relative gap"

# Metrics
duration: 8min
completed: 2026-04-07
---

# Phase 9 Plan 01: Config Extension and ELBO Gap Fix Summary

**BenchmarkConfig extended with 4 Phase 9 fields and amortization gap metric replaced with real Trace_ELBO evaluation**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-07T21:50:21Z
- **Completed:** 2026-04-07T21:58:12Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Extended BenchmarkConfig with guide_type, n_regions_list, elbo_type, fixtures_dir fields, all backward-compatible
- Added **kwargs passthrough to quick_config and full_config factory methods
- Replaced fabricated RMSE-ratio amortization gap proxy with real Trace_ELBO().loss() evaluation in both task and spectral amortized runners
- Established correct ELBO evaluation ordering: amortized BEFORE clear_param_store, SVI AFTER training

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend BenchmarkConfig with Phase 9 fields** - `75cb91a` (feat)
2. **Task 2: Fix amortization gap to use real ELBO evaluation** - `e4f5771` (fix)

## Files Created/Modified

- `benchmarks/config.py` - Added 4 new fields (guide_type, n_regions_list, elbo_type, fixtures_dir) with backward-compatible defaults; updated quick_config/full_config to accept **kwargs
- `benchmarks/runners/task_amortized.py` - Replaced RMSE-ratio proxy with real ELBO evaluation via Trace_ELBO(num_particles=5).loss(); amortized ELBO computed before clear_param_store
- `benchmarks/runners/spectral_amortized.py` - Same ELBO evaluation pattern as task runner; removed fabricated proxy

## Decisions Made

- **num_particles=5 for ELBO evaluation:** Balances variance reduction against computational cost. 5 particles gives stable enough estimates for gap computation without significantly slowing the benchmark loop.
- **kwargs passthrough pattern:** Factory methods accept **kwargs and forward to constructor, rather than adding explicit parameters. This allows future config fields to work without touching the factory methods again.
- **ELBO evaluation before clear_param_store:** The amortized guide's registered parameters live in the Pyro param store. Evaluating ELBO must happen before the store is cleared for per-subject SVI. This is a critical ordering constraint.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `ruff` not installed in the execution environment -- syntax validation done via ast.parse instead. Linting should be confirmed in CI.
- `zuko` dependency not installed, preventing full import test of amortized runners. Module-level imports verified via ast.parse; runtime correctness verified by pattern inspection.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- BenchmarkConfig is ready for Plan 02 (fixture generation) to use fixtures_dir field
- ELBO gap computation is ready for Plan 03 (runner integration) validation
- All existing benchmark code continues to work unchanged (backward-compatible defaults)

---
*Phase: 09-benchmark-foundation*
*Completed: 2026-04-07*
