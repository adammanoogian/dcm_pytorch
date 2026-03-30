---
phase: 08-metrics-benchmarks-and-documentation
plan: 02
subsystem: docs
tags: [documentation, quickstart, methods, latex, bibtex, equations]

# Dependency graph
requires:
  - phase: 01-forward-models-and-ode-integration
    provides: Neural state equation, Balloon-Windkessel, BOLD signal implementations
  - phase: 02-spectral-dcm-forward-model
    provides: Transfer function, CSD computation, noise models
  - phase: 03-regression-dcm-forward-model
    provides: rDCM forward pipeline, analytic posterior
  - phase: 04-pyro-generative-models-and-svi
    provides: Pyro models, guides, SVI runner
  - phase: 07-amortized-neural-inference-guides
    provides: AmortizedFlowGuide, summary networks
provides:
  - Quickstart tutorial (simulate -> infer -> inspect -> compare in ~40 lines)
  - Paper-ready methods section (Markdown + LaTeX)
  - BibTeX bibliography with 11 entries
  - Equations quick-reference for all 3 DCM variants
affects: [08-05-package-polish]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level imports in documentation (from pyro_dcm.simulators.task_simulator)"
    - "Dual-format methods (Markdown primary, LaTeX secondary)"

key-files:
  created:
    - docs/02_pipeline_guide/quickstart.md
    - docs/03_methods_reference/methods.md
    - docs/03_methods_reference/methods.tex
    - docs/03_methods_reference/equations.md
    - docs/03_methods_reference/references.bib
  modified: []

key-decisions:
  - "Module-level imports in quickstart (not top-level pyro_dcm.*) for robustness"
  - "LaTeX methods.tex as includeable fragment (no documentclass/preamble)"
  - "SPM12 code defaults in parameter tables (not paper values)"

patterns-established:
  - "Documentation references use [REF-XXX] Eq. N matching code docstrings exactly"
  - "Equations reference organized by DCM variant with source file cross-references"

# Metrics
duration: 10min
completed: 2026-03-30
---

# Phase 8 Plan 02: Documentation Summary

**Quickstart tutorial, paper-ready methods section (Markdown + LaTeX), equations reference, and BibTeX bibliography for all 3 DCM variants**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-30T10:20:24Z
- **Completed:** 2026-03-30T10:30:14Z
- **Tasks:** 2
- **Files created:** 5

## Accomplishments

- Quickstart tutorial walks through simulate -> infer -> inspect -> compare -> spectral DCM in ~40 lines of user code with 7 module-level imports
- Paper-ready methods section with 5 sections covering generative model (neural state, hemodynamic, BOLD, spectral, rDCM), inference (SVI, amortized, analytic VB), model comparison, implementation, and benchmarks
- LaTeX methods fragment with labeled equations, parameter tables, and BibTeX citations ready for journal submission
- Equations quick-reference covering 40+ equations across all 3 DCM variants with source file cross-references
- BibTeX bibliography with 11 entries for all cited papers

## Task Commits

Each task was committed atomically:

1. **Task 1: Quickstart tutorial + equations reference** - `52523f9` (docs)
2. **Task 2: Methods section (Markdown + LaTeX + BibTeX)** - `7237beb` (docs)

## Files Created

- `docs/02_pipeline_guide/quickstart.md` - End-to-end tutorial: simulate, SVI, posteriors, model comparison, spectral DCM
- `docs/03_methods_reference/equations.md` - Quick-reference tables for all implemented equations with REF citations
- `docs/03_methods_reference/methods.md` - Paper-ready methods section in Markdown (5 sections, 39 REF citations)
- `docs/03_methods_reference/methods.tex` - LaTeX fragment with labeled equations for journal inclusion
- `docs/03_methods_reference/references.bib` - BibTeX bibliography (11 entries: REF-001, 002, 010, 020, 021, 040, 041, 042, 043, 060, Baldy 2025)

## Decisions Made

- **Module-level imports in quickstart**: Used `from pyro_dcm.simulators.task_simulator import ...` instead of top-level `from pyro_dcm import ...` to ensure the quickstart works regardless of __init__.py export state
- **LaTeX as includeable fragment**: methods.tex has no \documentclass or preamble, designed for \input{methods.tex} in a paper template
- **SPM12 code defaults**: Parameter tables use SPM12 spm_fx_fmri.m values (kappa=0.64, gamma=0.32, etc.), not Stephan 2007 paper values, matching the actual implementation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Documentation artifacts complete for Plans 08-03 through 08-05
- Methods section references benchmark report (placeholder for Plan 08-04 results)
- Equations reference provides cross-reference for any future equation additions

---
*Phase: 08-metrics-benchmarks-and-documentation*
*Completed: 2026-03-30*
