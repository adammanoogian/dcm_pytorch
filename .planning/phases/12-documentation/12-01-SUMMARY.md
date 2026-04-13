---
phase: 12-documentation
plan: 01
subsystem: documentation
tags: [guide-selection, decision-tree, mermaid, elbo, calibration]

dependency_graph:
  requires: [phase-10, phase-11]
  provides: [guide-selection-doc, quickstart-crossref]
  affects: [12-02]

tech_stack:
  added: []
  patterns: [mermaid-flowchart, text-fallback-tree]

key_files:
  created:
    - docs/02_pipeline_guide/guide_selection.md
  modified:
    - docs/02_pipeline_guide/quickstart.md

decisions:
  - id: mermaid-plus-text-fallback
    choice: "Mermaid flowchart with ASCII text fallback"
    rationale: "GitHub renders Mermaid natively; text fallback covers local editors and PDF"
  - id: ranges-not-exact-numbers
    choice: "Reference approximate calibration ranges, not hardcoded numbers"
    rationale: "Exact values change when sweep is re-run; ranges convey the right information"

metrics:
  duration: "4m 18s"
  completed: 2026-04-13
---

# Phase 12 Plan 01: Guide Selection Recommendation Summary

**One-liner:** Decision tree and comparison guide for selecting among 6 SVI guide types and 2 rDCM modes, with coverage ceiling and memory limit warnings.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create guide_selection.md with decision tree and recommendation content | 3d5ad18 | docs/02_pipeline_guide/guide_selection.md |
| 2 | Add cross-reference from quickstart.md to guide_selection.md | 1ca67b0 | docs/02_pipeline_guide/quickstart.md |

## What Was Built

1. **Guide Selection Document** (307 lines) at `docs/02_pipeline_guide/guide_selection.md`:
   - Mermaid flowchart (`graph TD`) branching on DCM variant, network size, compute budget
   - ASCII text fallback tree for non-GitHub rendering contexts
   - 8-row method comparison table covering all inference methods with registry keys
   - Per-method prose sections (2-3 sentences each)
   - ELBO objective guidance table (3 ELBO types with compatibility matrix)
   - 5 dedicated warning blocks in blockquote format
   - Reproducibility section with exact CLI commands for calibration sweep

2. **Quickstart Cross-Reference**: New first bullet in Next Steps section of
   `docs/02_pipeline_guide/quickstart.md` linking to `guide_selection.md`.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Mermaid + ASCII text fallback | GitHub renders Mermaid natively; text covers other contexts |
| Approximate ranges, not exact numbers | Values change per sweep run; ranges convey the right guidance |
| Variant-first branching in decision tree | Prevents Simpson's paradox from cross-variant comparison |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

All 7 verification criteria passed:

1. `guide_selection.md` exists with all required sections
2. Mermaid flowchart uses `graph TD` (GitHub-compatible)
3. All 6 GUIDE_REGISTRY keys appear (auto_delta, auto_normal, auto_lowrank_mvn, auto_mvn, auto_iaf, auto_laplace)
4. All 3 ELBO_REGISTRY keys appear (trace_elbo, tracemeanfield_elbo, renyi_elbo)
5. Warnings section has 5 dedicated warning blocks
6. quickstart.md references guide_selection.md in Next Steps
7. No hardcoded exact benchmark numbers (uses ranges and relative comparisons)

All 6 must-have truths verified.

## Next Phase Readiness

Plan 12-02 (updated benchmark narrative) can proceed. The guide_selection.md
document is available for cross-referencing from the benchmark report.
