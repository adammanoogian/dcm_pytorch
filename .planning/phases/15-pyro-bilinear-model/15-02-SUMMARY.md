---
phase: 15-pyro-bilinear-model
plan: 02
subsystem: models
tags: [pytorch, pyro, bilinear-dcm, autoguide, auto-discovery, model-06]

# Dependency graph
requires:
  - phase: 15-pyro-bilinear-model
    plan: 01
    provides: "bilinear task_dcm_model with f'B_free_{j}' sample sites + deterministic B site + NaN-safe BOLD guard"
provides:
  - "tests/test_guide_factory.py::TestBilinearDiscovery: 6 parametrized tests across AutoNormal, AutoLowRankMultivariateNormal, AutoIAFNormal"
  - "Module-scoped fixture task_bilinear_guide_data (3-region, J=1, make_epoch_stimulus + PiecewiseConstantInput) for downstream bilinear guide-factory tests"
  - "_guide_kwargs_for(guide_type) helper: variant-specific create_guide kwargs that auto-scale hidden_dim for auto_iaf bilinear latent"
  - "Executable verification of MODEL-06 passive claim: create_guide auto-discovers dynamic bilinear sites without factory changes"
affects: [16-recovery-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "autoguide-discovery-verification: force AutoGuide._setup_prototype via one guide() call + assert dynamic site names present in guide.prototype_trace.nodes"
    - "test-local bilinear fixture duplication (over cross-test-file import): mirrors test_task_dcm_model.py::task_bilinear_data structure without fragile sibling imports"

key-files:
  created: []
  modified:
    - "tests/test_guide_factory.py (imports + module-scoped fixture + _guide_kwargs_for helper + TestBilinearDiscovery class = 320 new lines appended)"

key-decisions:
  - "Test-local fixture over cross-test-file import (per plan truth #3)"
  - "hidden_dim=64 injected via _guide_kwargs_for for auto_iaf (Rule 3 blocker fix: AutoRegressiveNN requires min(hidden_dims) >= input_dim; create_guide default [20] < bilinear latent 22)"
  - "auto_mvn intentionally excluded from MODEL-06 scope (research R3 note; full-rank covariance wasteful for bilinear J > 1)"
  - "Zero source changes confirmed: src/pyro_dcm/models/guides.py untouched by this plan (MODEL-06 is passive)"

patterns-established:
  - "autoguide-discovery-verification-pattern: for any new dynamic pyro.sample site (f'{name}_{j}' loop), a passive gate test forces _setup_prototype + asserts site membership in guide.prototype_trace.nodes across all supported AutoGuide variants"

# Metrics
duration: 15min
completed: 2026-04-18
---

# Phase 15 Plan 15-02: Guide Factory Bilinear Auto-Discovery Summary

**Executable MODEL-06 gate: create_guide auto-discovers the Plan-15-01 f'B_free_{j}' sites across AutoNormal, AutoLowRankMultivariateNormal, and AutoIAFNormal without any factory changes; 6 new parametrized tests in TestBilinearDiscovery (3 variants * prototype-trace + SVI-smoke) all green in 25.55s.**

## Performance

- **Duration:** 15 min 21 s
- **Started:** 2026-04-18T12:38:39Z
- **Completed:** 2026-04-18T12:54:00Z
- **Tasks:** 1/1
- **Files modified:** 1 (tests/test_guide_factory.py)

## Accomplishments

- Verified MODEL-06 passive claim empirically: `AutoGuide._setup_prototype` (lazy-called on first `guide()`) correctly discovers dynamic bilinear sites `B_free_0` identically to static sites `A_free`, `C` across all three supported AutoGuide families (mean-field, low-rank MVN, IAF flow).
- Added SVI-smoke gate: 20 ClippedAdam + Trace_ELBO steps per variant with L2 `init_scale=0.005`, all losses finite, param store non-empty after training. For `auto_normal` the param-store explicitly contains names referencing `B_free_0`; for `auto_lowrank_mvn` / `auto_iaf` the bilinear dims fold into the AutoContinuous `_latent` vector (prototype-trace continues to reference `B_free_0` structurally).
- Established `_guide_kwargs_for(guide_type)` test-side helper to size `hidden_dim=64` for `auto_iaf` (clears `AutoRegressiveNN` `min(hidden_dims) >= input_dim` constraint; bilinear `task_dcm_model` latent dim = 22 exceeds `create_guide` default `hidden_dim=[20]`). Confirmed `init_scale=0.005` is portable across all three variants (silently dropped for `auto_iaf` per `guides.py:171-172` `_INIT_SCALE_GUIDES` guard — no `TypeError`).
- Zero source-code changes (`src/pyro_dcm/models/guides.py` untouched). Full test_guide_factory.py: 30/30 green (24 pre-existing + 6 new). Phase 13 + 14 + 15-01 regression subset: 113/113 green in 6:32.

## Task Commits

1. **Task 1: Append TestBilinearDiscovery class + helpers + fixture to tests/test_guide_factory.py** — `9b796c0` (test)

**Plan metadata:** `e1d986b` (docs: complete guide-factory bilinear auto-discovery plan)

## Test Inventory

| Class | Test | Variant | Runtime | Status |
|-------|------|---------|---------|--------|
| TestBilinearDiscovery | test_b_free_sites_in_prototype_trace | auto_normal | ~1.0s | PASS |
| TestBilinearDiscovery | test_b_free_sites_in_prototype_trace | auto_lowrank_mvn | ~1.0s | PASS |
| TestBilinearDiscovery | test_b_free_sites_in_prototype_trace | auto_iaf | ~2.5s | PASS |
| TestBilinearDiscovery | test_b_free_sites_in_param_store_after_svi | auto_normal | ~4.5s | PASS |
| TestBilinearDiscovery | test_b_free_sites_in_param_store_after_svi | auto_lowrank_mvn | ~7.5s | PASS |
| TestBilinearDiscovery | test_b_free_sites_in_param_store_after_svi | auto_iaf | ~9.0s | PASS |

**Cumulative runtime:** 25.55 s (budget <90 s; comfortably 71% under). Per-variant SVI smoke runtime <10 s; no variant exceeded the 45 s half-budget warning threshold, so no Phase 16 budget re-tuning recommended.

**Pre-existing tests (unchanged):** 24/24 green — parametrized instantiation (6), default back-compat (2), init_scale asymmetry (3), blocklist (2), invalid guide (1), kwargs passthrough (2), SVI step smoke (6), registry sanity (2).

## Files Created/Modified

- `tests/test_guide_factory.py` — imports widened (task_dcm_model, make_block_stimulus, make_epoch_stimulus, make_random_stable_A, simulate_task_dcm, PiecewiseConstantInput); new `task_bilinear_guide_data` module-scoped fixture (3-region, J=1); new module-level `_BILINEAR_GUIDE_VARIANTS` list and `_guide_kwargs_for` helper; new `TestBilinearDiscovery` class with autouse `_silence_stability_logger` caplog fixture and 6 parametrized tests. 320 new lines appended. Zero lines removed or modified from pre-existing content.

## Decisions Made

**Locked decisions inherited from 15-01 applied:**
- **L1 (full (N,N) .to_event(2) B-site shape):** confirmed active by 15-01; discovery test relies on `B_free_0` being a stochastic site with event-dim-2 shape, which is exactly what `AutoGuide._setup_prototype` iterates over. Test would have failed if 15-01 had used a flat-vector alternative.
- **L2 (init_scale=0.005 for bilinear SVI):** passed explicitly to `create_guide` for all three variants. For `auto_normal` and `auto_lowrank_mvn` this flows through to the ctor (`_INIT_SCALE_GUIDES` guard); for `auto_iaf` it is silently dropped, confirming the plan's single-parametrize portability claim.
- **L3 (pyro.deterministic('B', ...) guarded to bilinear branch):** not directly asserted by 15-02 tests but implicitly relied on by 15-01's trace structure (pre-existing `test_linear_reduction_when_b_masks_none` in `test_task_dcm_model.py` continues to assert exact set equality on linear-mode sites).

**This plan's decisions:**
- None — plan executed exactly as written, with one auto-fixed blocker (see Deviations below) that did NOT invalidate any locked-decision truth.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Added `hidden_dim=64` to `create_guide` call for `auto_iaf` variant**

- **Found during:** Task 1 verification (first pytest run of TestBilinearDiscovery)
- **Issue:** `AutoIAFNormal` wraps `pyro.nn.AutoRegressiveNN`, which raises `ValueError: Hidden dimension must not be less than input dimension.` at `auto_reg_nn.py:206` when `min(hidden_dims) < input_dim`. The bilinear `task_dcm_model` has latent dim 22 (A_free=9 + C=3 + noise_prec=1 + B_free_0=9), which exceeds `create_guide`'s default `hidden_dim=[20]` (set at `guides.py:181`, passed through to `AutoIAFNormal` ctor at line 189). This caused both `auto_iaf` parametrizations to fail before reaching the MODEL-06 discovery assertion. The plan's truth-7 claim that `create_guide(..., init_scale=0.005)` is portable across all three variants is CORRECT (init_scale is indeed silently dropped for `auto_iaf`), but the `hidden_dim=[20]` default was a separate, orthogonal constructor-arg failure that the plan did not anticipate.
- **Fix:** Added a small `_guide_kwargs_for(guide_type: str) -> dict` helper module-function (~10 lines) that returns `{"init_scale": 0.005}` for all variants AND adds `"hidden_dim": 64` when `guide_type == "auto_iaf"`. The helper is called as `create_guide(task_dcm_model, guide_type=guide_type, **_guide_kwargs_for(guide_type))` in both test methods. This keeps the single-parametrize-list pattern intact (matches plan truth-6) and is a test-side-only adjustment (matches plan truth-5). `hidden_dim=64` comfortably clears the floor of 22 with a 2.9x margin — no tuning sensitivity expected for Phase 16 (3-8 regions; worst-case latent ~20 + J*N*N).
- **Files modified:** `tests/test_guide_factory.py` (helper def + 2 call-site replacements inside `TestBilinearDiscovery`; also one sentence appended to the `_BILINEAR_GUIDE_VARIANTS` module-docstring explaining the `hidden_dim` sizing rationale).
- **Verification:** After the fix, all 6 TestBilinearDiscovery tests pass in 25.55 s (`auto_iaf` variants are now the slowest at ~9 s SVI smoke but still well under per-variant 30 s plan budget). `create_guide` source remains untouched (grep confirms `git diff --stat src/` empty for this plan's commit).
- **Committed in:** `9b796c0` (part of the single Task-1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 3 blocker).
**Impact on plan:** Auto-fix is a TEST-SIDE-ONLY adjustment (no source change; no factory change). MODEL-06 truth-5 ("Zero edits to src/pyro_dcm/models/guides.py") remains satisfied. The hidden_dim sizing is orthogonal to the init_scale portability truth-6 — both constraints are real and both are now handled test-side. Flagged for potential follow-up (see Next Phase Readiness).

## Issues Encountered

- **Parallel-execution branch coordination:** Plan 15-03 committed to the same branch during this plan's execution window (`6c68b10 feat(15-03): refuse bilinear sites in TaskDCMPacker + amortized_task_dcm_model` landed between `git log` check and `git commit`). Verified no file-level conflict with the guarded scope of this plan: 15-03 touched `src/pyro_dcm/guides/parameter_packing.py`, `src/pyro_dcm/models/amortized_wrappers.py`, `tests/test_amortized_task_dcm.py`, `tests/test_parameter_packing.py`, `tests/test_posterior_extraction.py` (and, per prompt, `guides.py` docstring-only — though that change was not present in the final branch state visible to this plan; likely folded into `6c68b10` or deferred). This plan touched only `tests/test_guide_factory.py` as instructed; staged individually via `git add tests/test_guide_factory.py` (no `git add .` or `-A`).
- **Pre-existing ruff lint in `tests/test_guide_factory.py`:** I001 on lines 9-32 (import block organization) and D403 on line 245 (`test_kwargs_passthrough_lowrank` docstring `rank` not capitalized) both confirmed pre-existing via `git stash` round-trip before my edits were applied. Per Phase 14 precedent (14-01, 14-02, 15-01), pre-existing lint is not touched by additive plans. Candidate for a future chore commit.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **MODEL-06 closed.** Phase 15 now has 5/7 requirements closed (MODEL-01, MODEL-02, MODEL-03 both halves, MODEL-04 from Plan 15-01; MODEL-06 from this plan). MODEL-05 (extract_posterior_params bilinear key contract) and MODEL-07 (TaskDCMPacker explicit refusal) are in Plan 15-03's scope.
- **Phase 16 recovery benchmark is unblocked for guide-factory concerns.** RECOV-01..08 can now use `create_guide(task_dcm_model, guide_type="auto_normal", init_scale=0.005)` for the primary recovery path with MODEL-06 confidence that bilinear sites auto-discover; can use `auto_lowrank_mvn` as secondary posterior-comparison path; `auto_iaf` remains available for DCM.V1 validation with `hidden_dim=64` explicit override (downstream callers must mirror this; see follow-up).
- **Known follow-ups (both out of scope for 15-02):**
  1. `auto_mvn` docstring update in `src/pyro_dcm/models/guides.py` noting bilinear J > 1 cost scaling (research Section 3 R3 note) — deferred per plan truth-7; track for v0.3.1 polish or Phase 16 chore.
  2. Consider whether `create_guide` should auto-scale `hidden_dim` default when `guide_type=="auto_iaf"` based on model latent dim (heuristic: `max(20, 2 * estimated_latent)`). Currently callers must pass `hidden_dim` explicitly for any model with >20 latent dims. This plan's test-side `_guide_kwargs_for` is the pattern to emulate downstream. Explicit `create_guide(..., bilinear_mode=True)` auto-switch is REJECTED per 15-RESEARCH.md Section 14 Q1.
- **Parallel-plan coordination verified:** Plan 15-03 has already committed (`6c68b10`); its unstaged tests (test_amortized_task_dcm.py, test_parameter_packing.py, test_posterior_extraction.py) remain in the working tree for 15-03's metadata commit to sweep. This plan's metadata commit stages only `PLAN.md` + `SUMMARY.md` + `STATE.md`, leaving 15-03's tree untouched.

---
*Phase: 15-pyro-bilinear-model*
*Completed: 2026-04-18*
