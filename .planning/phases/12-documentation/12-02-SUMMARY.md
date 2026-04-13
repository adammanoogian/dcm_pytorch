---
phase: 12-documentation
plan: 02
subsystem: documentation
tags: [benchmark-report, calibration, cross-method-comparison, v0.2.0]

# Dependency graph
requires:
  - phase: 12-01
    provides: guide_selection.md for cross-reference from benchmark report
  - phase: 11-calibration-analysis
    provides: plotting functions, calibration_analysis CLI, timing_profiler
  - phase: 09-benchmark-fixtures
    provides: shared fixture infrastructure referenced in protocol section
provides:
  - "v0.2.0 benchmark report with zero TBD entries (DOC-02)"
  - "How to Reproduce section with exact CLI commands for calibration pipeline"
  - "Cross-references to guide_selection.md, calibration_sweep.py, calibration_analysis.py"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Template-mode reporting: figure paths reference expected outputs, narrative describes patterns, exact numbers come from running the sweep"
    - "Median+IQR reporting convention (not mean+std) for all benchmark metrics"

key-files:
  created: []
  modified:
    - docs/04_scientific_reports/benchmark_report.md

key-decisions:
  - "No hardcoded numbers: report references CLI commands that generate data"
  - "SPM12 and amortized flow explicitly deferred to v0.3+ with rationale"
  - "Per-variant tables only; cross-variant aggregation explicitly forbidden"

patterns-established:
  - "Benchmark reports are templates populated by running calibration scripts"

# Metrics
duration: 5min
completed: 2026-04-13
---

# Phase 12 Plan 02: Updated Benchmark Narrative Summary

**Rewrote v0.1.0 benchmark report to v0.2.0 structure: replaced all 14 TBD entries, added 6-guide cross-method comparison, calibration curves, Pareto frontier, and How to Reproduce CLI commands**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-13T10:16:58Z
- **Completed:** 2026-04-13T10:21:49Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Replaced all 14 TBD/Pending entries from v0.1.0 with v0.2.0 content or explicit deferral
- Restructured report from 7 to 9 sections covering cross-method comparison, per-parameter analysis, calibration curves, scaling study, Pareto frontier, timing breakdown, and violin plots
- Added How to Reproduce section with exact CLI commands for the full calibration pipeline (generate_fixtures -> calibration_sweep -> calibration_analysis)
- Cross-referenced guide_selection.md from three locations in the report

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite benchmark_report.md with v0.2.0 structure and zero TBDs** - `6d1031f` (docs)

## Files Created/Modified

- `docs/04_scientific_reports/benchmark_report.md` - Complete rewrite: 575 lines, 9 sections, zero TBDs, all figure references point to Phase 11 calibration artifacts

## Decisions Made

- No hardcoded benchmark numbers in the report; all values come from running the calibration sweep. Report uses approximate ranges from STATE.md decisions where narrative context is needed.
- SPM12 cross-validation explicitly deferred to v0.3+ with rationale (MATLAB dependency), replacing ambiguous "Pending" markers.
- Amortized neural inference (Phase 7 flow guide) explicitly deferred to v0.3+ with rationale, replacing v0.1.0 "TBD (pre-trained guide)" entries.
- Former "Amortized vs Per-Subject" section replaced with "Per-Parameter Analysis" covering violin plots, timing breakdown, and Pareto frontier.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 12 (Documentation) is now complete: both DOC-01 (guide selection) and DOC-02 (benchmark narrative) are delivered.
- The v0.2.0 milestone documentation is complete. All four phases (9-12) are finished.
- To populate the benchmark figures, run `python benchmarks/calibration_sweep.py --tier 3` followed by `python benchmarks/calibration_analysis.py`.

---
*Phase: 12-documentation*
*Completed: 2026-04-13*
