---
phase: 17
name: Circuit Visualization Module
milestone: v0.4.0
status: passed
score: 15/15
verified: 2026-04-24
verified_against_commit: ef2a8d8552996729307c1f3e08db09dfcb39f03b
---

# Phase 17: Circuit Visualization Module -- Verification Report
**Phase Goal:** Implement `src/pyro_dcm/utils/circuit_viz.py` -- a `CircuitViz` class with `from_model_config`, `from_posterior`, `to_dict`, `export`, and `load` methods producing `dcm_circuit_explorer/v1` JSON from Pyro-DCM model configs and/or SVI posteriors, verified by structural unit tests and a Pyro smoke integration test.
**Verified:** 2026-04-24
**Status:** PASSED
**Re-verification:** No -- initial verification
**Commit verified against:** ef2a8d8 (docs(17-01): finalize CircuitViz SUMMARY duration field)

## 1. Summary

Phase 17 delivers a pure-Python `dcm_circuit_explorer/v1` JSON serializer as its sole output. All 5 ROADMAP success criteria and all 10 VIZ-NN requirements are satisfied. The new file `src/pyro_dcm/utils/circuit_viz.py` (659 LOC) implements `CircuitVizConfig` dataclass, `CircuitViz` factory class, and `flatten_posterior_for_viz` helper. The test file `tests/test_circuit_viz.py` (506 LOC) has 17 tests (15 fast A-series + regression, 2 slow B-series integration smokes) -- all 17 pass. Zero upstream files were modified. The executor deviation in B-01 (using `functools.partial` instead of `run_svi(model_kwargs=...)`) is accepted as a correct fix matching the established bilinear posterior-extraction pattern. No fitting-metric assertions (RMSE, coverage, shrinkage, ELBO convergence thresholds) appear anywhere in the test file.
## 2. Must-Have Coverage

| ID | Must-Have | Evidence | Status |
|----|-----------|----------|--------|
| ROADMAP-1 | `from_model_config` produces `_status=planned` and `fitted_params is None` | `circuit_viz.py:454-468`; `test_from_posterior_flips_status` lines 113-115 | VERIFIED |
| ROADMAP-2 | `from_posterior` produces `_status=fitted` with populated `fitted_params`; traces through `flatten_posterior_for_viz` | `circuit_viz.py:505-509`; `test_smoke_planned_to_fitted` lines 429-433 | VERIFIED |
| ROADMAP-3 | Round-trip: `CircuitViz.load(cfg.export(path)).to_dict() == cfg.to_dict()` on 3-region fixture; uses `tmp_path`; not skipped | `test_export_roundtrip` lines 202-210; `tmp_path` fixture used | VERIFIED |
| ROADMAP-4 | Schema tolerance: bare bilinear with empty phenotypes/hypotheses/drugs; no headless browser deps | `test_empty_optional_collections` lines 213-225; no selenium/pyppeteer/playwright in codebase | VERIFIED |
| ROADMAP-5 | Zero upstream edits | `git diff --name-only 9979b7e..HEAD` shows only STATE.md, SUMMARY.md, utils/__init__.py, circuit_viz.py, test_circuit_viz.py | VERIFIED |
| VIZ-01 | `CircuitVizConfig` dataclass with 13 + `extras` field; `to_dict()` and `export()` methods | `circuit_viz.py:150-272`; 14 fields at lines 184-197 | VERIFIED |
| VIZ-02 | `from_model_config` raises `ValueError` on missing/mismatched `region_colors` | `circuit_viz.py:387-398`; both missing and length-mismatch regression tests pass | VERIFIED |
| VIZ-03 | `from_posterior` deepcopy + NaN/Inf validation before deepcopy | `circuit_viz.py:505-509`; no-mutation, NaN, and Inf tests all pass | VERIFIED |
| VIZ-04 | `load` + V1 extras harvesting via `_FIRST_CLASS_KEYS` | `circuit_viz.py:533,549`; `test_roundtrip_heart2adapt` confirms extras populated | VERIFIED |
| VIZ-05 | V7 deterministic `mat_order` (sorted B-keys) | `circuit_viz.py:445`; shuffled {B3,B1,B2} -> [A,B1,B2,B3,C] passes | VERIFIED |
| VIZ-06 | V8 tensor/ndarray/list coercion in `_to_list_of_list` | `circuit_viz.py:73-108`; `test_tensor_input_accepted` -- torch.float64 coerced lossless | VERIFIED |
| VIZ-07 | `flatten_posterior_for_viz` module-level function exported | `circuit_viz.py:553-659`; exported via `__init__.py` lines 3-6; used in B-01 test | VERIFIED |
| VIZ-08 | Round-trip equality via export+load | `test_export_roundtrip` and `test_roundtrip_heart2adapt` both pass | VERIFIED |
| VIZ-09 | `src/pyro_dcm/utils/__init__.py` re-exports the 3 surface names | `__init__.py` lines 3-7 and 20-22 with `# Phase 17` markers | VERIFIED |
| VIZ-10 | Zero upstream edits (same gate as ROADMAP-5) | No changes to task_dcm_model, extract_posterior_params, parameterize_A/B, create_guide, run_svi, connectivity/, guides/, simulators/ | VERIFIED |

**Score:** 15/15 must-haves verified

## 3. Locked Decisions V1-V8

| Decision | Evidence | Status |
|----------|----------|--------|
| V1 `extras: dict = field(default_factory=dict)`; `to_dict` merges extras last (first-class wins); `load` harvests unknown keys | `circuit_viz.py:197` (field def), `228-241` (merge logic), `549` (load harvest) | VERIFIED |
| V2 `ValueError` on missing/mismatched `region_colors`; expected-vs-actual message | `circuit_viz.py:387-398`; messages include expected/got phrasing | VERIFIED |
| V3 No `jsonschema` dependency | grep across circuit_viz.py, test file, pyproject.toml returned 0 results | VERIFIED |
| V4 `from_posterior` takes pre-flattened dict; `flatten_posterior_for_viz` separate, not auto-invoked | `circuit_viz.py:471-509` and `553-659`; helper documented as recommended but not auto-invoked | VERIFIED |
| V5 No headless browser deps | grep for selenium, pyppeteer, playwright in all Phase 17 files returned 0 results | VERIFIED |
| V6 `_validate_no_nan` before deepcopy in `from_posterior` AND before write in `export`; error includes `[row={i}, col={j}]` | `circuit_viz.py:505` and `269`; error at lines 137-146. Minor: spec used combined NaN/Inf wording; implementation uses separate messages -- functionally equivalent | VERIFIED |
| V7 `mat_order = [A] + sorted(B_keys) + ([C] if C present)` | `circuit_viz.py:445` -- exact match to spec | VERIFIED |
| V8 `_to_list_of_list` dispatches Tensor/ndarray/list; numpy guarded by `try/except ImportError` | `circuit_viz.py:37-42` (guard), `97-108` (dispatch) | VERIFIED |
## 4. Directive Check

No fitting-metric gates found in acceptance. Scan of `tests/test_circuit_viz.py` for RMSE, coverage, shrinkage, recovery (as assertion thresholds), convergence returned zero assertion lines. `Trace_ELBO` (line 325) and `SVI` (line 401) appear as Pyro class imports/instantiation for smoke test setup only. The class docstring (line 301) explicitly states the B-class is NOT a recovery test, only a shape-contract smoke. B-01 assertions are structural: `d[_status] == fitted` and shape checks on `fitted_params` key set and dimensions. User directive fully honored.

## 5. Test Results

| Suite | Count | Result | Wall time |
|-------|-------|--------|-----------|
| A-series + regression (fast, -m not slow) | 15 | 15 passed | 1.62s |
| B-series integration smokes (-m slow) | 2 | 2 passed | 12.21s |
| Total Phase 17 | 17 | 17 passed | ~14s |

Individual tests:

- A-01 test_roundtrip_heart2adapt: PASSED (HEART2ADAPT 3-extras byte-equal round-trip: _study, _description, node_info)
- A-02 test_from_posterior_flips_status: PASSED
- A-03 test_from_posterior_no_mutation: PASSED
- A-04 test_to_dict_top_level_keys: PASSED
- A-05 test_mat_order_deterministic: PASSED
- A-06 test_schema_version: PASSED
- A-07 test_vals_are_list_of_list: PASSED
- A-08 test_tensor_input_accepted: PASSED
- A-09 test_export_roundtrip: PASSED (uses tmp_path)
- A-10 test_empty_optional_collections: PASSED
- test_region_colors_missing_raises: PASSED
- test_region_colors_length_mismatch_raises: PASSED
- test_nan_in_posterior_raises: PASSED
- test_inf_in_posterior_raises: PASSED
- test_extras_roundtrip_preserved: PASSED
- B-01 test_smoke_planned_to_fitted: PASSED (20-step bare SVI, structural assertion only)
- B-02 test_extract_posterior_shapes_match_serializer: PASSED (5-step SVI, shape contract only)
## 6. Deviations from Plan (Accepted / Rejected)

### B-01 bare SVI + functools.partial (ACCEPTED)

Plan spec called for `run_svi(..., model_kwargs={b_masks, stim_mod})` followed by `extract_posterior_params(guide, model_args, model=task_dcm_model)`. The executor identified that `Predictive(model=task_dcm_model, ...)` invokes `model(*args)` with no kwarg forwarding, so the bilinear branch never activates during posterior sampling -- `B_free_0` and `B` sites would be absent.

Fix: bare SVI loop (20 steps) + `functools.partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` bound before `extract_posterior_params`. This pattern is already established in `tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction` (Phase 15). No V-decision semantics changed; `flatten_posterior_for_viz` signature and behavior are untouched. ACCEPTED.

### Task 3 REQUIREMENTS.md idempotent no-op (ACCEPTED)

Executor claimed VIZ-01..10 block was already present from planner commit `9979b7e`. Verified: `git log --oneline --all -- .planning/REQUIREMENTS.md` confirms `9979b7e` is the first commit to introduce the VIZ-01..10 block, preceding all executor commits `57c5922`, `b5af070`, `55ca869`, `ef2a8d8`. The executor correctly created no additional commit for Task 3. ACCEPTED.

### HEART2ADAPT extras are 3 keys, not 6 (INFO)

Plan/research docs listed _study, _description, node_info, node_positions, svg_edges, b_overlays as expected extras. Actual `configs/heart2adapt_dcm_config.json` on this branch contains only _study, _description, node_info. The V1 pass-through is schema-agnostic. Not a code deviation; noted for plan-text maintenance only.

## 7. Pre-existing Issues Surfaced

### tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys (WARNING)

Fails with AssertionError: simulation_diverged key unexpectedly present. The `simulation_diverged` key was added to the simulator return dict in Phase 16 commit `f7c2ba9` (fix(phase16): filter corrupt fixtures via seed pool). Confirmed pre-existing: `f7c2ba9` predates all Phase 17 commits; no Phase 17 file touches `tests/test_task_simulator.py` or `task_simulator.py`. This is a Phase 16 residual; the test expected_keys set needs a one-line update to add `simulation_diverged`. Not a Phase 17 blocker.

## 8. Gaps Found

None.

## 9. Human Verification Needed

None. All acceptance gates are structural and were verified programmatically.

---

## VERIFICATION PASSED

All 15/15 must-haves verified. 17/17 Phase 17 tests pass (15 fast + 2 slow integration smokes). Zero upstream edits. Zero fitting-metric gates. Phase 17 goal achieved.

---
_Verified: 2026-04-24_
_Verifier: Claude (gsd-verifier)_
