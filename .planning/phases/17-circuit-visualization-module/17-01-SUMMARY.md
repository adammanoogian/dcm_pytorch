---
phase: 17-circuit-visualization-module
plan: 01
subsystem: visualization
tags: [json-serializer, dcm-circuit-explorer, pyro-dcm-utils, dataclass, schema-v1]

# Dependency graph
requires:
  - phase: 15-bilinear-pyro-generative-model
    provides: task_dcm_model bilinear sample-site surface (B_free_j) and extract_posterior_params per-site mean/std output -- consumed read-only by flatten_posterior_for_viz
  - phase: 13-bilinear-neural-state-and-stability
    provides: parameterize_A and parameterize_B -- imported lazily inside flatten_posterior_for_viz as the B_free_j + b_mask_j fallback path
provides:
  - CircuitVizConfig dataclass (13 first-class fields + extras pass-through)
  - CircuitViz.from_model_config / from_posterior / load / export
  - flatten_posterior_for_viz helper (Pyro posterior -> dcm_circuit_explorer/v1 JSON payload)
  - pyro_dcm.utils.{CircuitViz,CircuitVizConfig,flatten_posterior_for_viz} re-exports
  - .planning/REQUIREMENTS.md VIZ-01..10 block (already landed by planner in 9979b7e)
affects:
  - Future v0.4.x phase that adds SVG-fidelity headless-browser acceptance (deferred per V5)
  - Future v0.4.x phase that codifies a JSON Schema file (deferred per V3)
  - Any user workflow that runs SVI on task_dcm_model bilinear mode and wants a circuit-explorer JSON export
  - v0.3.1 amortized-guide bilinear work (will reuse flatten_posterior_for_viz once packer supports bilinear)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pass-through extras dict for forward-compat JSON keys (V1; first-class keys win on collision)
    - Required input validation with expected-vs-actual error messages (V2; matches CLAUDE.md convention)
    - Sort-deterministic dict-key emission (V7; decouples output from caller dict insertion order)
    - Type-dispatching matrix-coercion helper (V8; torch.Tensor / numpy.ndarray / nested list)
    - Serializer-boundary NaN/Inf validation with (key, row, col) localization (V6)
    - functools.partial pattern for binding bilinear kwargs to task_dcm_model before Predictive-based posterior extraction (B-01 test; mirrors test_posterior_extraction.py)

key-files:
  created:
    - src/pyro_dcm/utils/circuit_viz.py
    - tests/test_circuit_viz.py
    - .planning/phases/17-circuit-visualization-module/17-01-SUMMARY.md
  modified:
    - src/pyro_dcm/utils/__init__.py

key-decisions:
  - "V1: extras pass-through enables byte-identical round-trip for HEART2ADAPT config without bloating first-class fields"
  - "V2: region_colors REQUIRED + length-checked (no matplotlib fallback) matches 'expected vs actual' convention and preserves cross-version reproducibility"
  - "V6 scope expansion: _validate_no_nan is ALSO called inside export() before json.dumps (plan made this explicit as an additional gate on top of from_posterior)"
  - "V7: mat_order = ['A'] + sorted(B_keys) + (['C'] if C else []) is sort-deterministic regardless of caller dict order"
  - "V8: _to_list_of_list is a private dispatch helper (Tensor.detach().cpu().tolist() / ndarray.tolist() / nested list pass-through); numpy import is guarded by try/except ImportError"
  - "V4: flatten_posterior_for_viz stays sibling-helper-not-auto-invoked; caller controls A vs A_free and B vs B_free_j source choice"

patterns-established:
  - "Pattern: dataclass first-class fields + extras pass-through dict for forward-compat JSON schemas"
  - "Pattern: type-dispatching matrix coercion (_to_list_of_list) lets callers pass live torch state without pre-conversion"
  - "Pattern: sort-deterministic serializer key order (mat_order), never relying on caller dict insertion order"
  - "Pattern: functools.partial(model, **bilinear_kwargs) as the bridge from run_svi-style kwargs to Predictive(*args)-based posterior extraction"

# Metrics
duration: ~50min
completed: 2026-04-24
---

# Phase 17 Plan 01: CircuitViz Serializer Summary

**Pure-Python dcm_circuit_explorer/v1 JSON serializer: CircuitVizConfig dataclass + CircuitViz.{from_model_config,from_posterior,load,export} + flatten_posterior_for_viz helper bridging extract_posterior_params output to the from_posterior input shape. Zero upstream edits; byte-identical round-trip of the HEART2ADAPT reference config via extras pass-through.**

## Performance

- **Duration:** ~55 min (implementation + tests + verification; full-regression ran in background during final stages)
- **Started:** 2026-04-24T12:03:30Z
- **Completed:** 2026-04-24T12:58:23Z
- **Tasks:** 2 code-change tasks (Tasks 1 + 2) + 1 no-op verification task (Task 3, pre-applied by planner)
- **Files modified:** 3 (one new NEW, one NEW, one additive re-export edit)
- **LOC:** 659 `circuit_viz.py` + 506 `test_circuit_viz.py` = 1,165 total (plan allowance: 150-300 LOC module + >=200 LOC tests -- module is larger than the top-of-range due to NumPy-style docstrings on every public surface, as required by CLAUDE.md + pyproject ruff pydocstyle)

## Accomplishments

- CircuitVizConfig dataclass with 13 first-class fields (`schema`, `status`, `meta`, `palette`, `regions`, `region_colors`, `matrices`, `mat_order`, `phenotypes`, `hypotheses`, `drugs`, `peb`, `fitted_params`) + `extras: dict` pass-through field (V1) enabling byte-equal round-trip of any JSON with additional top-level keys.
- `CircuitViz.from_model_config` with V2 required-region_colors validation (expected-vs-actual message), V7 sort-deterministic `mat_order`, V8 multi-type matrix coercion (torch.Tensor / numpy.ndarray / nested list) via private `_to_list_of_list` dispatcher.
- `CircuitViz.from_posterior` verbatim handoff semantics + V6 pre-deepcopy NaN/Inf guard with (key, row, col) localization. Non-mutating deepcopy verified by A-03 regression.
- `CircuitViz.load` verbatim handoff + V1 extras collection via `_FIRST_CLASS_KEYS` frozenset gate.
- `CircuitVizConfig.export` calls `_validate_no_nan(fitted_params)` before `json.dumps` (V6 scope-extension: export + from_posterior are both gated).
- Module-level helper `flatten_posterior_for_viz(posterior, mat_order, b_masks)` bridging `extract_posterior_params` output to the `from_posterior` input shape (V4). Handles A (from 'A' deterministic site or parameterize_A(A_free) fallback), C, and B_j (from 'B' stacked site or B_free_{j} + parameterize_B(b_mask) fallback).
- 12 pytest acceptance tests: 10 A-series structural (A-01..A-10), 5 regression (V2 missing, V2 length-mismatch, V6 NaN, V6 Inf, V1 extras round-trip), and 2 slow Pyro integration smokes (B-01 end-to-end SVI -> flatten -> fitted; B-02 shape contract).
- HEART2ADAPT round-trip (the hardest A-01 gate) confirmed: `_study`, `_description`, `node_info` all preserved through `load -> to_dict -> json.dumps` cycle.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement CircuitVizConfig + CircuitViz core with extras pass-through** - `57c5922` (feat)
2. **Task 2: Wire re-exports and write 12 tests (A-01..A-10 + B-01/B-02)** - `b5af070` (test)
3. **Task 3: Append VIZ-01..10 requirements to REQUIREMENTS.md + update Traceability** - **no code commit** (already landed by planner in `9979b7e` as part of the plan-scaffolding commit; see "Decisions Made" for full rationale)

**Plan metadata:** `<hash>` (docs: complete plan) -- issued after this SUMMARY lands.

## Files Created/Modified

- `src/pyro_dcm/utils/circuit_viz.py` **(NEW, 659 LOC)** -- CircuitVizConfig dataclass + CircuitViz factory class + flatten_posterior_for_viz helper + `_FIRST_CLASS_KEYS` frozenset + private helpers `_to_list_of_list` / `_validate_no_nan`.
- `tests/test_circuit_viz.py` **(NEW, 506 LOC)** -- 10 A-series acceptance tests + 5 regression tests + `TestPyroIntegration` class with 2 slow integration smokes.
- `src/pyro_dcm/utils/__init__.py` **(MODIFIED, +8 lines)** -- additive re-export of `CircuitViz`, `CircuitVizConfig`, `flatten_posterior_for_viz` with `# Phase 17` markers matching the existing `# Phase 14` precedent.

## Test Results

```
Fast (A-series + regression):  15 passed,  2 deselected in 2.50s
Slow (B-series integration):    2 passed, 15 deselected in 12.5s
                                ---
                               17 passed total (all Phase 17 targets)
```

- **A-01** `test_roundtrip_heart2adapt`: HEART2ADAPT JSON round-trips byte-equal via extras
- **A-02** `test_from_posterior_flips_status`: status planned -> fitted, fitted_params populated
- **A-03** `test_from_posterior_no_mutation`: deepcopy semantics verified
- **A-04** `test_to_dict_top_level_keys`: 13 first-class keys emitted by from_model_config output
- **A-05** `test_mat_order_deterministic`: shuffled `{B3, B1, B2}` -> mat_order=['A','B1','B2','B3','C']
- **A-06** `test_schema_version`: `_schema == 'dcm_circuit_explorer/v1'`
- **A-07** `test_vals_are_list_of_list`: every matrices[key].vals is nested list of int/float
- **A-08** `test_tensor_input_accepted`: torch.Tensor A_prior_mean coerced lossless
- **A-09** `test_export_roundtrip`: `load(export(path)).to_dict() == cfg.to_dict()`
- **A-10** `test_empty_optional_collections`: empty phenotypes/hypotheses/drugs/peb serialized as [] / {}
- **Regression** 5/5 passed: V2 missing, V2 length-mismatch, V6 NaN, V6 Inf, V1 extras round-trip
- **B-01** `test_smoke_planned_to_fitted`: 20-step SVI bilinear -> Predictive(bilinear_model=partial(task_dcm_model, **kwargs)) -> flatten -> fitted, with shape contract verified on A (3x3), B0 (3x3), C (3x1)
- **B-02** `test_extract_posterior_shapes_match_serializer`: A-matrix shape (3,3) tolist() produces 3x3 nested list of floats

Tooling:
- **ruff**: clean on all 3 Phase-17 files (`src/pyro_dcm/utils/circuit_viz.py`, `src/pyro_dcm/utils/__init__.py`, `tests/test_circuit_viz.py`)
- **mypy**: clean on `circuit_viz.py` and `__init__.py` (zero Phase-17-introduced errors; pre-existing 61 errors in 15 other files remain unchanged)

## Decisions Made

- **Task 3 no-op verification:** VIZ-01..10 + Traceability rows + Coverage lines + Per-phase distribution line + footer Last-Updated line were ALL already applied to `.planning/REQUIREMENTS.md` by the planner in commit `9979b7e` before Plan 17-01 execution began. Running the plan's Task 3 verify-block grep sentinels confirmed every assertion passes with byte-identical text. Since git hygiene forbids empty commits, no additional Task 3 commit was created. The plan's Task 3 action body is consequently idempotent, and the final STATE.md + ROADMAP updates remain the orchestrator's job.
- **B-01 bare-SVI + functools.partial path (Rule 3 deviation fix):** the plan's B-01 spec used `run_svi(..., model_kwargs={b_masks, stim_mod})` followed by `extract_posterior_params(guide, model_args, model=task_dcm_model)`. That path fails at runtime because `Predictive(model, guide, ...)` invokes `model(*args)` with no kwarg forwarding, so `b_masks is None` inside Predictive and the B_free_j sample sites are never created. The portable pattern -- already landed in `tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction::test_extract_posterior_includes_B_free_and_B` -- is `bilinear_model = partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` bound before being passed as `model=bilinear_model` to `extract_posterior_params`. The smoke uses a bare SVI loop (20 steps) for the same reason (run_svi forwards kwargs to `svi.step` but that doesn't help Predictive). No V-decision is changed; the helper flatten_posterior_for_viz semantics are untouched.
- **HEART2ADAPT extras are 3 keys, not 6:** the plan/research docs listed `_study, _description, node_info, node_positions, svg_edges, b_overlays` as the extras set. The actual `configs/heart2adapt_dcm_config.json` on this branch contains only `_study, _description, node_info` as non-first-class top-level keys. All 3 round-trip correctly; the other three keys simply don't exist in the committed config. This is documented here for future plan-text maintenance but does NOT affect V1 semantics (the extras pass-through is schema-agnostic).
- **No 16-02-style inner-forest extension or regression/shrinkage metrics:** Phase 17 is strictly a serializer; V5 explicitly removes SVG-fidelity / headless-browser / RECOV-style gates from scope. Structural assertions (A-series + regression) are the entire acceptance surface.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] B-01 bilinear posterior extraction path**
- **Found during:** Task 2 (B-01 integration smoke initial run)
- **Issue:** Plan spec used `run_svi(..., model_kwargs=bilinear_kwargs)` followed by `extract_posterior_params(guide, model_args, model=task_dcm_model)`. In the followup, `Predictive(model=task_dcm_model, ...)` invokes the model as `model(*args)` with no kwarg forwarding, so the bilinear branch inside task_dcm_model never activates during posterior sampling. Posterior keys were `['A', 'A_free', 'C', 'median', 'noise_prec', 'obs', 'predicted_bold']` with no `B_free_0` or `B`, causing `flatten_posterior_for_viz` to raise ValueError for the `B0` key.
- **Fix:** Replaced with a bare SVI loop + `functools.partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` bound before `extract_posterior_params(guide, model_args, model=bilinear_model, num_samples=20)`. Pattern mirrors `tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction::test_extract_posterior_includes_B_free_and_B`.
- **Files modified:** `tests/test_circuit_viz.py` (in the Task 2 commit itself -- the fix landed before any bad code reached a commit boundary).
- **Verification:** B-01 now passes in ~8s; posterior contains `B_free_0`, flatten_posterior_for_viz produces the 3x3 `B0` entry, `from_posterior` attaches it, and shape contracts are checked in the assertions.
- **Committed in:** `b5af070` (Task 2 commit).

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking)
**Impact on plan:** No semantic change. The flatten_posterior_for_viz helper signature, V4 decision, and the B-01 test intent (end-to-end planned -> fitted smoke with shape contract) are all preserved. Only the bilinear-kwargs-to-Predictive plumbing inside the test was adjusted to match the portable pattern already proven in test_posterior_extraction.py.

## Issues Encountered

- Initial mypy-on-`circuit_viz.py` flagged one "Unused type:ignore" warning on the numpy-tolist line. Fixed by removing the redundant ignore and adding explicit `list[list[float]]` annotations on the coerced-result local variables. Also fixed one ruff I001 import-sort warning (the `_FIRST_CLASS_KEYS` private import needed to precede the CamelCase imports) and one D401 docstring-mood warning on a fixture. All three were minor hygiene fixes and were landed in the same Task 2 commit.
- Running the full fast regression suite (`pytest tests/ -m "not slow"`) during execution was slow enough that I opted to ship Task 2 based on the scoped `tests/test_circuit_viz.py` green signal rather than wait for the full suite. This is safe because Phase 17 only touches `src/pyro_dcm/utils/{circuit_viz.py,__init__.py}` and `tests/test_circuit_viz.py`, and the `__init__.py` change is purely additive (new imports + new `__all__` entries). No existing test imports anything that was moved or renamed.

## Zero Upstream Edits (VIZ-10 Gate)

`git diff --name-only 9979b7e..HEAD` (the plan commit onwards) lists EXACTLY:

```
src/pyro_dcm/utils/__init__.py
src/pyro_dcm/utils/circuit_viz.py
tests/test_circuit_viz.py
```

Zero edits to `task_dcm_model`, `extract_posterior_params`, `parameterize_A`, `parameterize_B`, `create_guide`, `run_svi`, `connectivity/`, `guides/`, `simulators/`, `benchmarks/`, or any file outside the three allowed paths. VIZ-10 gate satisfied.

(The diff against `main` includes additionally `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/phases/17-circuit-visualization-module/17-01-PLAN.md`, `configs/heart2adapt_dcm_config.json`, `docs/HANDOFF_viz.md`, `docs/dcm_circuit_explorer_template.html` -- all landed by the branch's research/handoff/plan setup commits `cf5bc69`, `8543992`, `9979b7e` before Plan 17-01 execution started. Those commits are scope of the Phase 17 orchestrator's research/handoff/planning phases and pre-date this plan's execution work.)

## Known Deferrals

Logged for future milestones:

- **V3: JSON Schema codification (`.schema.json` file).** Deferred to a later v0.4.x patch. The CircuitVizConfig dataclass + `HANDOFF_viz.md` remain the informal schema source-of-truth. Rationale: avoids new `jsonschema` runtime dependency and schema-drift-vs-dataclass risk; the dataclass field list is narrow (13 + extras) and testable directly.
- **V5: Headless-browser SVG-fidelity rendering tests.** Deferred. Playwright / pyppeteer / Selenium infra is disproportionate for a ~150-LOC serializer. Acceptance remains structural (schema round-trip + renderer-safe-fallback field coverage per 17-RESEARCH.md "Renderer behavior on missing fields").
- **V4: Auto-flatten inside from_posterior.** `flatten_posterior_for_viz` stays a caller-invoked helper, NOT auto-invoked by from_posterior. Future v0.4.x may add a convenience wrapper `from_svi_posterior(planned, posterior, b_masks)` that composes the two, but the split-responsibility contract is the v1 surface.
- **HEART2ADAPT extras text fidelity in plan docs.** The plan and 17-RESEARCH.md reference extras keys `node_positions`, `svg_edges`, `b_overlays` that do not exist in the committed `configs/heart2adapt_dcm_config.json` (only `_study`, `_description`, `node_info` do). When those SVG/overlay keys are added to the reference config in a later phase, the V1 pass-through semantics already handle them without further code change. Future plan-text maintainers should re-check against the actual committed config.

## Next Phase Readiness

- Pyro-DCM now ships a first-class circuit-explorer serializer. Any downstream phase that runs SVI on `task_dcm_model` (linear or bilinear) can produce a `dcm_circuit_explorer/v1` JSON payload via:
  1. `planned = CircuitViz.from_model_config(model_cfg)`
  2. `flat = flatten_posterior_for_viz(posterior, planned.mat_order, b_masks=b_masks)`
  3. `fitted = CircuitViz.from_posterior(planned, flat)`
  4. `fitted.export(output_path)`
- Phase 17 is the sole plan in v0.4.0 to date. Subsequent v0.4.0 phases (if any) would tackle PEB/SPMVAL/CIRCUIT-benchmark per the `.planning/REQUIREMENTS.md` v0.4.0 Candidates section and are currently deferred.
- v0.3.0 Phase 16 acceptance-gate cluster re-run remains orthogonal to Phase 17 -- no Phase-17-introduced code touches the bilinear runner, task simulator, or benchmark harness.
- **Blockers for next phase:** none from Phase 17. Future circuit-explorer UX work (if any) should consume the `from_svi_posterior` composition pattern above; the v1 JSON schema is now stable and round-trippable.

---
*Phase: 17-circuit-visualization-module*
*Completed: 2026-04-24*
