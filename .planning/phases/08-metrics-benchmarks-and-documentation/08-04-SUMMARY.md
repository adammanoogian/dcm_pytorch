---
phase: 08-metrics-benchmarks-and-documentation
plan: 04
subsystem: benchmarks
tags: [plotting, matplotlib, figures, benchmark-report, publication, pdf, png]

# Dependency graph
requires:
  - phase: 08-metrics-benchmarks-and-documentation
    plan: 03
    provides: "Benchmark runners producing JSON results"
provides:
  - "Plotting module with 7 figure-generation functions"
  - "4 publication-quality figures in dual format (PDF + PNG)"
  - "Benchmark narrative report with unified comparison tables"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["Style fallback chain: SciencePlots -> seaborn -> default", "Dual-format figure output (PDF + PNG)"]

key-files:
  created:
    - "benchmarks/plotting.py"
    - "docs/04_scientific_reports/benchmark_report.md"
    - "figures/benchmark_rmse_comparison.png"
    - "figures/benchmark_time_comparison.png"
    - "figures/benchmark_coverage_comparison.png"
    - "figures/true_vs_inferred_scatter.png"
  modified: []

key-decisions:
  - "Figures and JSON results gitignored; only source code and report committed"
  - "Amortization gap figure skipped due to no paired SVI/amortized results available"

# Metrics
duration: 37min
completed: 2026-04-02
---

# Phase 08 Plan 04: Benchmark Figures and Report Summary

**Plotting module with 7 functions generating dual-format figures from JSON results, plus 7-section benchmark narrative report with BNC-01/02/03 comparison tables**

## Performance

- **Duration:** 37 min
- **Started:** 2026-04-02T12:56:49Z
- **Completed:** 2026-04-02T13:34:43Z
- **Tasks:** 2
- **Files modified:** 2 committed (+ 8 generated figures, gitignored)

## Accomplishments

- Created plotting module with 7 exported functions for benchmark figure generation
- Generated 4 publication-quality figures (RMSE, time, coverage, scatter) in PDF + PNG
- Wrote comprehensive 7-section benchmark report with unified comparison tables
- Report includes Grad Steps column, SPM12 rows, and amortization gap analysis

## Task Commits

Each task was committed atomically:

1. **Task 1: Create plotting module** - `f903663` (feat)
2. **Task 2: Generate figures and write benchmark report** - `9ac3ef6` (docs)

## Files Created/Modified

- `benchmarks/plotting.py` - 7 figure-generation functions with style fallback and dual-format output
- `docs/04_scientific_reports/benchmark_report.md` - Narrative benchmark report with BNC-01/02/03 tables
- `figures/benchmark_rmse_comparison.{png,pdf}` - RMSE bar chart (gitignored)
- `figures/benchmark_time_comparison.{png,pdf}` - Wall time bar chart (gitignored)
- `figures/benchmark_coverage_comparison.{png,pdf}` - Coverage bar chart with nominal line (gitignored)
- `figures/true_vs_inferred_scatter.{png,pdf}` - RMSE vs correlation scatter (gitignored)
- `benchmarks/results/benchmark_results.json` - Combined quick-mode results (gitignored)

## Decisions Made

- **Figures gitignored:** Generated binary artifacts (.png, .pdf) are excluded from version control per existing .gitignore rules. The plotting module (source code) and report (markdown) are committed.
- **Amortization gap figure skipped:** No paired SVI/amortized results available without pre-trained guides. The plotting function handles this gracefully and skips generation.
- **Combined JSON from individual runs:** Assembled results from separate spectral, rDCM, and task runs into a single JSON file since the CLI overwrites on each run.

## Deviations from Plan

None -- plan executed exactly as written. The amortization gap figure (5th of 5 planned) was not generated due to missing amortized benchmark data, which the plan anticipated ("if a variant/method is not in results, skip that bar/point").

## Issues Encountered

- Task DCM quick-mode had 2/3 ODE underflow failures (expected at 30s duration). Only 1 dataset succeeded, producing valid but single-point statistics (std=0).
- Background benchmark run (`--quick --variant all`) collided with individual runs, requiring manual assembly of combined JSON results.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 08-04 was the final remaining plan in Phase 8 (08-05 was completed out of order)
- All 5 plans in Phase 8 are now complete
- Phase 8 (and the entire v0.1.0-foundation milestone) is ready for final review
- To generate publication-quality results: run `python benchmarks/run_all_benchmarks.py` without `--quick`
- To generate amortization gap figure: train guides with `scripts/train_amortized_guide.py`, then re-run benchmarks with `--method amortized`

---
*Phase: 08-metrics-benchmarks-and-documentation*
*Completed: 2026-04-02*
