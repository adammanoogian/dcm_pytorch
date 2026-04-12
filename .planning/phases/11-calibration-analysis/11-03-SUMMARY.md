---
phase: 11-calibration-analysis
plan: 03
subsystem: benchmarks
tags: [violin-plots, timing-profiler, pareto-frontier, poutine-trace, matplotlib, calibration]
dependency_graph:
  requires:
    - phase: 11-01
      provides: multi-level coverage metrics, calibration sweep orchestrator
    - phase: 11-02
      provides: calibration curve plots, comparison tables, scaling study figures, GUIDE_COLORS
  provides:
    - plot_posterior_violins for per-A_ij violin overlay across guide types (CAL-04)
    - profile_svi_step for poutine.trace-based timing decomposition (CAL-05)
    - plot_pareto_frontier for wall-time vs RMSE with Pareto front overlay (CAL-05)
    - plot_timing_breakdown for stacked bar chart of timing components (CAL-05)
    - Updated calibration_analysis.py with --violin and --timing CLI flags
  affects: [08-UAT]
tech_stack:
  added: []
  patterns: [poutine-trace-profiling, pareto-front-computation, violin-grid-layout]
key_files:
  created:
    - benchmarks/timing_profiler.py
  modified:
    - benchmarks/plotting.py
    - benchmarks/calibration_analysis.py
decisions:
  - id: spectral-only-profiling
    choice: "profile_all_guides supports variant='spectral' only"
    rationale: "Fastest variant for profiling; task DCM requires fixture IO; extensible later"
  - id: skip-auto-delta-violins
    choice: "Exclude auto_delta from violin plots"
    rationale: "Point estimate with no distribution to visualize; misleading violin"
patterns_established:
  - "poutine.trace for separate model/guide timing: isolates forward, guide, and gradient costs"
  - "Pareto front via sorted-sweep: O(n log n), correct non-dominated point identification"
  - "Violin NxN grid layout: one subplot per A[i,j] element for dense parameter comparison"
metrics:
  duration: ~9min
  completed: 2026-04-12
---

# Phase 11 Plan 03: Supplementary Calibration Analysis Summary

**Per-A_ij posterior violin plots, poutine.trace timing profiler with stacked breakdown, and Pareto frontier visualization for wall-time vs RMSE trade-offs**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-12T19:45:24Z
- **Completed:** 2026-04-12T19:53:59Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `benchmarks/timing_profiler.py` with `profile_svi_step` (decomposes wall-clock into forward/guide/gradient via `poutine.trace`) and `profile_all_guides` (trains + profiles all 6 SVI guide types on a representative dataset)
- Added `plot_posterior_violins` generating NxN grid of per-A_ij violin overlays across guide types with ground truth marked as red dashed line
- Added `plot_pareto_frontier` showing median wall-time (log scale) vs median RMSE with IQR error bars and Pareto front overlay connecting non-dominated points
- Added `plot_timing_breakdown` producing stacked horizontal bar chart of forward model / guide evaluation / gradient percentage per guide type
- Wired all new functions into `calibration_analysis.py` with `--violin`, `--timing`, `--variant`, `--fixtures-dir`, and `--seed` CLI flags

## Task Commits

Each task was committed atomically:

1. **Task 1: Add timing profiler and violin/Pareto plot functions** - `d3b4371` (feat)
2. **Task 2: Wire violin/timing/Pareto into calibration_analysis.py** - `91ca225` (feat)

## Files Created/Modified

- `benchmarks/timing_profiler.py` - New module: profile_svi_step, profile_all_guides, _build_spectral_model_args
- `benchmarks/plotting.py` - Added plot_posterior_violins, plot_pareto_frontier, plot_timing_breakdown; added torch and typing.Any imports
- `benchmarks/calibration_analysis.py` - Added generate_violin_plots, generate_timing_figures; added Pareto frontier to default mode; extended CLI with --violin, --timing, --variant, --fixtures-dir, --seed flags

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| profile_all_guides supports spectral variant only | Fastest variant for profiling; task DCM requires fixture IO; pattern is extensible |
| Exclude auto_delta from violin plots | Point estimate has no distribution to visualize; would be misleading as a violin |
| _build_spectral_model_args shared helper | Avoids duplicating fixture loading logic between profiler and violin generator |

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 11 (Calibration Analysis) is now complete. All three plans delivered:
- 11-01: Multi-level coverage metrics + calibration sweep orchestrator
- 11-02: Calibration curve plots + comparison tables + scaling studies
- 11-03: Posterior violin plots + timing profiler + Pareto frontier

The single entry point `python benchmarks/calibration_analysis.py` generates all figures. Ready for Phase 08 UAT or Phase 12 wrap-up.

---
*Phase: 11-calibration-analysis*
*Completed: 2026-04-12*
