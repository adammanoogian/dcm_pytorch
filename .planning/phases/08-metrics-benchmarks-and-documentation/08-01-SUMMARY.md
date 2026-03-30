---
phase: 08-metrics-benchmarks-and-documentation
plan: 01
subsystem: testing
tags: [benchmarks, metrics, rmse, coverage, correlation, argparse, cli]

# Dependency graph
requires:
  - phase: 05-parameter-recovery-and-model-comparison
    provides: "Recovery metrics (RMSE, coverage, correlation) used in test files"
  - phase: 07-amortized-neural-inference-guides
    provides: "Amortization gap metric and benchmark patterns"
provides:
  - "benchmarks/metrics.py with 5 consolidated metric functions"
  - "benchmarks/config.py with BenchmarkConfig dataclass"
  - "benchmarks/run_all_benchmarks.py CLI entry point"
  - "benchmarks/runners/ registry with 7 placeholder entries"
  - "[benchmark] optional-dependencies in pyproject.toml"
affects: [08-02, 08-03, 08-04]

# Tech tracking
tech-stack:
  added: ["tabulate (optional)", "matplotlib (optional)"]
  patterns: ["RUNNER_REGISTRY dispatch pattern", "BenchmarkConfig quick/full classmethods"]

key-files:
  created:
    - "benchmarks/__init__.py"
    - "benchmarks/metrics.py"
    - "benchmarks/config.py"
    - "benchmarks/runners/__init__.py"
    - "benchmarks/run_all_benchmarks.py"
    - "benchmarks/results/.gitkeep"
    - "tests/test_benchmark_metrics.py"
  modified:
    - "pyproject.toml"

key-decisions:
  - "sys.path.insert for CLI script to find benchmarks package"

patterns-established:
  - "RUNNER_REGISTRY: dict mapping (variant, method) tuples to runner callables"
  - "BenchmarkConfig: dataclass with quick_config/full_config classmethods for CI vs paper-quality"
  - "CLI dispatches to registry, gracefully skips NotImplementedError runners"

# Metrics
duration: 8min
completed: 2026-03-30
---

# Phase 8 Plan 01: Benchmark Infrastructure Summary

**Consolidated metrics module (RMSE, coverage, correlation, amortization gap), BenchmarkConfig dataclass with quick/full classmethods, CLI entry point with argparse dispatch via RUNNER_REGISTRY**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-30T10:20:18Z
- **Completed:** 2026-03-30T10:28:16Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Consolidated 5 duplicate metric functions from 3+ test files into single authoritative benchmarks/metrics.py
- Created BenchmarkConfig dataclass with quick_config (CI) and full_config (paper) classmethods
- Built CLI entry point (run_all_benchmarks.py) with --variant, --method, --quick flags and JSON output
- Established RUNNER_REGISTRY with 7 entries (task/spectral/rdcm_rigid/rdcm_sparse/spm) for Plan 08-03

## Task Commits

Each task was committed atomically:

1. **Task 1: Create benchmarks/metrics.py and benchmarks/config.py** - `b331a7e` (feat)
2. **Task 2: Create CLI entry point and update pyproject.toml** - `a732dc8` (feat)
3. **Task 3: Unit tests for metrics module** - `bbcacd1` (test)

## Files Created/Modified

- `benchmarks/__init__.py` - Package init with docstring
- `benchmarks/metrics.py` - 5 consolidated metric functions (compute_rmse, pearson_corr, compute_coverage_from_ci, compute_coverage_from_samples, compute_amortization_gap)
- `benchmarks/config.py` - BenchmarkConfig dataclass with quick_config/full_config classmethods
- `benchmarks/runners/__init__.py` - RUNNER_REGISTRY with 7 placeholder entries
- `benchmarks/run_all_benchmarks.py` - CLI entry point with argparse, JSON output, metadata collection
- `benchmarks/results/.gitkeep` - Track results directory
- `tests/test_benchmark_metrics.py` - 9 unit tests for metric functions
- `pyproject.toml` - Added [benchmark] optional-dependencies (matplotlib, tabulate)

## Decisions Made

- Added sys.path.insert in run_all_benchmarks.py to resolve benchmarks package imports when run as standalone script

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added sys.path manipulation for CLI script imports**
- **Found during:** Task 2 (CLI entry point)
- **Issue:** Running `python benchmarks/run_all_benchmarks.py` directly failed with ModuleNotFoundError because benchmarks/ is not an installed package
- **Fix:** Added `sys.path.insert(0, project_root)` using Path(__file__).parent.parent
- **Files modified:** benchmarks/run_all_benchmarks.py
- **Verification:** `python benchmarks/run_all_benchmarks.py --help` works correctly
- **Committed in:** a732dc8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Standard pattern for CLI scripts in non-installed packages. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- benchmarks/ package scaffolding complete, ready for Plan 08-02 (benchmark runners)
- RUNNER_REGISTRY entries are placeholders; runners will be implemented in Plan 08-03
- Metrics are tested and ready for use by runner implementations

---
*Phase: 08-metrics-benchmarks-and-documentation*
*Completed: 2026-03-30*
