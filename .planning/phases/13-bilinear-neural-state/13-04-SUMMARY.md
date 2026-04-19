---
phase: 13-bilinear-neural-state
plan: 04
subsystem: docs
tags: [documentation, drift-correction, bilinear, linear, CLAUDE.md, PROJECT.md]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    provides: "Plan 13-01 Task 1 handles the source-code half of BILIN-07 (neural_state.py module + class docstrings). This plan covers the two non-source doc sites only."
provides:
  - "CLAUDE.md directory-tree now accurately reflects src/pyro_dcm/models/ (task_dcm_model.py, spectral_dcm_model.py, rdcm_model.py, guides.py, amortized_wrappers.py)"
  - "PROJECT.md v0.1.0 Validated list correctly labels the dx/dt = Ax + Cu form as **Linear** (not Bilinear)"
  - "Unblocks the v0.3.0 release phase from having to land this cleanup under ship pressure"
affects: [v0.3.0-release, phase-15-pyro-model-review, phase-16-benchmark-review]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Doc-rename scope split across plans: source-code edits colocated with feature work (13-01), non-source drift correction in a parallel docs-only plan (13-04)"

key-files:
  created:
    - .planning/phases/13-bilinear-neural-state/13-04-SUMMARY.md
  modified:
    - CLAUDE.md
    - .planning/PROJECT.md

key-decisions:
  - "Annotated task_dcm_model.py in CLAUDE.md tree with '[v0.3.0: + bilinear B path]' so future readers of the tree see the bilinear entrypoint without reading the code"
  - "Did NOT add a new v0.3.0 Validated line for the true bilinear form to PROJECT.md — that belongs to the v0.3.0 release phase after Phase 16 acceptance passes"
  - "Did NOT touch PROJECT.md line 58 (v0.3.0 Target features bullet). That legitimate use of 'bilinear' describes the correct B-matrix form with compute_effective_A — it is the forward target, not drift"

patterns-established:
  - "Parallel Wave 1 docs-only plan: intentionally narrow file scope (CLAUDE.md + PROJECT.md only) so no file-conflict with 13-01/13-02/13-03 source-code plans"
  - "Plan scope is enforced by line-level surgical Edit calls, not wholesale rewrites — prevents scope creep during wave-parallel execution"

# Metrics
duration: ~6min
completed: 2026-04-17
---

# Phase 13 Plan 04: Doc-Rename for Non-Source Drift Sites Summary

**Two surgical documentation edits correcting stale `generative_models/` directory-tree and the mislabeled `A + Cu` Validated line (linear, not bilinear), closing the non-source half of BILIN-07.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-17T20:18:00Z (approx)
- **Completed:** 2026-04-17T20:24:36Z
- **Tasks:** 3 (2 doc edits + 1 commit)
- **Files modified:** 2 (`CLAUDE.md`, `.planning/PROJECT.md`)

## Accomplishments

- `CLAUDE.md` directory-tree block at line ~101 replaced: stale `generative_models/` with `task_dcm.py` / `spectral_dcm.py` / `regression_dcm.py` is now the accurate `models/` layout listing the five real files (`task_dcm_model.py`, `spectral_dcm_model.py`, `rdcm_model.py`, `guides.py`, `amortized_wrappers.py`). `task_dcm_model.py` is annotated `[v0.3.0: + bilinear B path]` so future readers see the bilinear entrypoint without grep.
- `.planning/PROJECT.md` line 23 rewritten: `- Bilinear neural state equation (dx/dt = Ax + Cu) with explicit A matrix — v0.1.0` → `- **Linear** neural state equation (dx/dt = Ax + Cu) with explicit A matrix — v0.1.0`. The `A + Cu` form is linear; the true bilinear form is the v0.3.0 Phase 13 deliverable and will be added to Validated when Phase 16 ships.
- Zero source/test changes — `git show --stat f77560d` shows only two `.md` files, confirming no overlap with Plans 13-01/13-02/13-03 (Wave 1 parallelism preserved).

## Task Commits

Each task was committed atomically as part of the single combined plan commit (no per-task commit for docs-only surgical edits):

1. **Task 1: Rewrite CLAUDE.md directory-tree generative_models/ block** — bundled into `f77560d`
2. **Task 2: Rewrite PROJECT.md line 23 Bilinear → **Linear**** — bundled into `f77560d`
3. **Task 3: Commit plan 13-04 work** — `f77560d` (`docs(13-04): correct stale doc drift (CLAUDE.md tree + PROJECT.md linear vs bilinear)`)

**Plan metadata commit:** (this SUMMARY.md) — next commit

## Files Created/Modified

- `CLAUDE.md` — Directory-tree block (lines ~101-108): removed `generative_models/` (3 files), added `models/` (5 files with `task_dcm_model.py` annotated for v0.3.0 bilinear path).
- `.planning/PROJECT.md` — Line 23: `- Bilinear neural state equation (dx/dt = Ax + Cu) ...` → `- **Linear** neural state equation (dx/dt = Ax + Cu) ...`.
- `.planning/phases/13-bilinear-neural-state/13-04-SUMMARY.md` — This summary.

## Decisions Made

- **Plan scope enforced strictly at line level.** PROJECT.md line 58 (`- Bilinear neural state equation with compute_effective_A(A, B_list, u_mod)`) is a legitimate v0.3.0 Target-features bullet that describes the correct Friston 2003 bilinear form with B-matrix modulators. It is NOT drift and was intentionally NOT touched.
- **No new v0.3.0 Validated line added to PROJECT.md.** That belongs to the v0.3.0 release phase (post-Phase-16 acceptance), not Phase 13 work.
- **`models/` entries include `guides.py` and `amortized_wrappers.py`.** Verified against `ls src/pyro_dcm/models/` before the edit — CLAUDE.md tree now matches filesystem exactly.
- **No opportunistic full-repo bilinear-terminology audit.** Per CONTEXT.md, that is explicitly out of scope; Phases 15 and 16 will each sweep their own file scopes.

## Deviations from Plan

None — plan executed exactly as written.

All four must_have truths from the plan frontmatter verified:
- `rg "generative_models/" CLAUDE.md` → 0 hits ✓
- `rg "task_dcm_model.py|rdcm_model.py|amortized_wrappers.py" CLAUDE.md` → 3 hits ✓
- `rg "\*\*Linear\*\* neural state equation \(dx/dt = Ax \+ Cu\)" .planning/PROJECT.md` → 1 hit (line 23) ✓
- `rg "Bilinear neural state equation.*dx/dt = Ax \+ Cu" .planning/PROJECT.md` → 0 hits ✓

## Issues Encountered

None. Git emitted a benign `LF will be replaced by CRLF` warning on staging (Windows line-ending autoconversion for text files) — expected and harmless.

## User Setup Required

None — no external service configuration required for docs-only edits.

## Next Phase Readiness

- BILIN-07 non-source half closed. When Plan 13-01's source-code half lands (module docstring + `NeuralStateEquation` class docstring in `src/pyro_dcm/forward_models/neural_state.py`), BILIN-07 is fully complete and Phase 13 success criterion 5 is green.
- No blockers introduced. Plans 13-02 (stability monitor) and 13-03 (worst-case 3σ test) remain independent of this work.
- The v0.3.0 release phase no longer carries the pre-existing documentation drift debt as a blocker.

---
*Phase: 13-bilinear-neural-state*
*Completed: 2026-04-17*
