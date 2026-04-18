---
phase: 14-stimulus-utilities-and-bilinear-simulator
plan: 02
subsystem: simulators
tags: [pytorch, bilinear-dcm, simulator, dt-invariance, structural-shortcircuit]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    provides: CoupledDCMSystem bilinear path + linear short-circuit gate (coupled_system.py:287-291); compute_effective_A
  - phase: 14-stimulus-utilities-and-bilinear-simulator
    plan: 01
    provides: make_event_stimulus, make_epoch_stimulus (breakpoint-dict); merge_piecewise_inputs helper in pyro_dcm.utils
provides:
  - simulate_task_dcm extended signature with keyword-only B_list, stimulus_mod, n_driving_inputs
  - Structural linear short-circuit at simulator level (CoupledDCMSystem called with no B= kwarg when B_list is None) guaranteeing torch.equal bit-exactness vs pre-Phase-14
  - Bilinear simulator path via merge_piecewise_inputs + CoupledDCMSystem(B=B_stacked, n_driving_inputs=C.shape[1])
  - Return-dict keys 'B_list' and 'stimulus_mod' (None in linear mode; populated in bilinear mode)
  - _normalize_B_list + _normalize_stimulus_to_input_fn module-private helpers
  - dt-invariance regression coverage (rk4 dt=0.01 vs dt=0.005, atol=1e-4) for BOTH linear AND bilinear paths
affects: [15-pyro-generative-model, 16-recovery-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - bilinear-simulator-extension
    - structural-linear-shortcircuit
    - keyword-only-additive-api-extension

key-files:
  created:
    - tests/test_bilinear_simulator.py
  modified:
    - src/pyro_dcm/simulators/task_simulator.py
    - tests/test_task_simulator.py

key-decisions:
  - "L3 applied (from plan frontmatter): n_driving_inputs defaults to C.shape[1] at simulator level when B_list is supplied but n_driving_inputs is not. Simulator has C in scope, so it always infers and passes the value explicitly to CoupledDCMSystem; the CoupledDCMSystem raise branch is never triggered by the simulator path. Inconsistent explicit n_driving_inputs raises ValueError."
  - "L4 applied (from plan frontmatter): SIM-05 dt-invariance fixture uses B off-diagonal magnitude = 0.1 (not 0.3) to keep Phase 13 stability monitor silent."
  - "L5 applied (from plan frontmatter): linear-path dt-invariance regression test added as test_dt_invariance_linear (symmetry with bilinear); guards against silent future breakage of linear rk4 reproducibility."
  - "L6 applied (phase-level wave split): Plan 14-02 runs after Plan 14-01 in Wave 2 so the simulator extension can import the Plan-14-01 stimulus utilities and merge_piecewise_inputs."
  - "C matrix convention in tests: C = [[0.25], [0.0], [0.0]] (task-DCM suite convention) instead of torch.eye(3,1). Discovered during execution: unit-amplitude driving inputs caused dopri5 adaptive step to underflow to 0.0 on make_random_stable_A(seed=42) for Tests 1-3 (dopri5 default solver). Switching to the 0.25 amplitude convention (matching test_task_simulator.py fixtures) eliminates the issue. Tests 4 and 5 use parameterize_A(zeros(N,N)) + rk4 fixed-step so the underflow was never at risk."

patterns-established:
  - "Keyword-only additive API extension: new arguments live after `*` sentinel so every pre-Phase-14 positional caller continues to work unchanged."
  - "Structural linear short-circuit: when bilinear kwargs default to None, the simulator calls CoupledDCMSystem with no B= or n_driving_inputs= kwarg, inheriting Phase 13's gate at coupled_system.py:287-291. Bit-exactness is asserted structurally (torch.equal), not numerically (atol)."
  - "Simulator-owned inference of n_driving_inputs: simulator has C in scope, so it defaults n_driving_inputs = C.shape[1] (L3). CoupledDCMSystem retains its raise-policy for direct callers that don't have C in scope."

# Metrics
requirements-closed: [SIM-03, SIM-04, SIM-05]
duration: 23min
completed: 2026-04-18
---

# Phase 14 Plan 02: Bilinear Simulator Extension Summary

**`simulate_task_dcm` extended with three keyword-only arguments (`B_list`, `stimulus_mod`, `n_driving_inputs`) to support the Friston 2003 bilinear neural state equation, while preserving bit-exact linear behavior via a structural short-circuit (no `B=` kwarg passed to `CoupledDCMSystem` when `B_list is None`). New 5-test `tests/test_bilinear_simulator.py` closes SIM-03 (linear bit-exactness + bilinear distinguishability), SIM-04 (return-dict keys), and SIM-05 (dt-invariance, linear + bilinear).**

## Performance

- **Duration:** ~23 min (wall clock)
- **Started:** 2026-04-18T09:30:08Z
- **Completed:** 2026-04-18T09:53:37Z
- **Tasks:** 3
- **Commits:** 3 (feat + test + test)
- **Files changed:** 3 (1 source modified, 1 test modified, 1 new test file)

## Accomplishments

- **Extended `simulate_task_dcm` signature** with three keyword-only args (`B_list`, `stimulus_mod`, `n_driving_inputs`) after the `*` sentinel. All pre-Phase-14 positional callers work unchanged.
- **Linear short-circuit is structural, not numerical.** When `B_list is None`, the simulator calls `CoupledDCMSystem(A_dev, C_dev, input_fn, hemo_params)` with no `B=` or `n_driving_inputs=` kwarg, inheriting the Phase 13 linear gate at `coupled_system.py:287-291`. SIM-03 bit-exactness is asserted with `torch.equal` (not `torch.testing.assert_close`).
- **Bilinear path wiring:** merges `stimulus` (driving) + `stimulus_mod` (modulatory) via `merge_piecewise_inputs` into a widened `(M + J)`-column input, stacks `B_list` via `_normalize_B_list`, then calls `CoupledDCMSystem(A_dev, C_dev, merged_input_fn, hemo_params, B=B_stacked, n_driving_inputs=C.shape[1])`.
- **Return dict gains two keys** — `'B_list'` and `'stimulus_mod'` — both `None` in linear mode, stacked `(J, N, N)` tensor + `PiecewiseConstantInput` in bilinear mode (SIM-04).
- **Two module-private helpers** factored out from inline code: `_normalize_B_list(B_list, device, dtype)` (accepts None / list / tuple / tensor; empty collapses to None; raises TypeError on invalid types and ValueError on non-3-D tensors) and `_normalize_stimulus_to_input_fn(stim, device, dtype)` (accepts dict or `PiecewiseConstantInput`).
- **Validation gates:** missing `stimulus_mod` in bilinear mode raises `ValueError`; `stimulus_mod.values.shape[1]` mismatch with `B_list.shape[0]` raises `ValueError`; explicit `n_driving_inputs != C.shape[1]` raises `ValueError`. No silent defaults.

## Task Commits

Each task was committed atomically:

1. **Task 1: extend simulate_task_dcm with bilinear mode** — `abeb5d8` (`feat(14-02)`). +198 lines / -12 lines. Adds two helpers, extends signature with `*`-guarded kwargs, branches body on `B_stacked is None`, extends return dict, updates imports, expands docstring with Parameters/Returns/Notes/References sections.
2. **Task 2: update expected_keys for new bilinear return-dict keys** — `88cc1bb` (`test(14-02)`). +1 line. Additive one-line edit to `TestSimulatorOutputStructure::test_simulator_output_keys`.
3. **Task 3: bilinear simulator regression and dt-invariance tests** — `22ee2f7` (`test(14-02)`). +219 lines. New file `tests/test_bilinear_simulator.py` with 5 tests.

## Files Created/Modified

- **`src/pyro_dcm/simulators/task_simulator.py`** (modified):
  - New module-private helper `_normalize_B_list` (normalizes list/tuple/tensor/None inputs to stacked `(J, N, N)` tensor or None, with type- and shape-validation).
  - New module-private helper `_normalize_stimulus_to_input_fn` (accepts dict or `PiecewiseConstantInput`).
  - Extended `simulate_task_dcm` signature with `*, B_list=None, stimulus_mod=None, n_driving_inputs=None`.
  - Body rewritten to branch on `B_stacked is None`. Linear branch inherits Phase 13's literal expression by omitting `B=` kwarg. Bilinear branch wires `merge_piecewise_inputs` + `CoupledDCMSystem(B=..., n_driving_inputs=...)`.
  - Return dict extended with `'B_list'` and `'stimulus_mod'` keys.
  - Import widened to include `merge_piecewise_inputs`.
  - Docstring expanded with three new parameter entries, two new return-key entries, a "Linear short-circuit" note, a "References" section citing Friston 2003.
- **`tests/test_task_simulator.py`** (modified): ONE-LINE additive edit to `TestSimulatorOutputStructure::test_simulator_output_keys` `expected_keys` set adding `"B_list"` and `"stimulus_mod"`. All 17 other tests untouched.
- **`tests/test_bilinear_simulator.py`** (NEW, 5 tests).

## Test Inventory

### New tests (5) in `tests/test_bilinear_simulator.py`

| Test | Covers | Gate |
|------|--------|------|
| `test_bilinear_arg_none_matches_no_kwarg` | SIM-03 primary: structural linear short-circuit | `torch.equal` on `bold_clean`, `bold`, `neural` |
| `test_bilinear_output_distinguishable_from_linear` | SIM-03 secondary: bilinear mode is not a no-op | `max\|diff\| > 0.01` |
| `test_return_dict_has_bilinear_keys` | SIM-04: return-dict keys in both modes | `None` checks + `isinstance(PiecewiseConstantInput)` |
| `test_dt_invariance_bilinear` | SIM-05 primary: rk4 dt=0.01 vs dt=0.005 | `torch.testing.assert_close(atol=1e-4, rtol=0.0)` |
| `test_dt_invariance_linear` | SIM-05 L5 regression symmetry | `torch.testing.assert_close(atol=1e-4, rtol=0.0)` |

All 5 green in 306.11s (the two dt-invariance tests account for the bulk of the runtime — each runs four simulate_task_dcm calls at duration=200s).

### Modified tests (1-line edit)

- `tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys`: `expected_keys` set augmented with `"B_list"` and `"stimulus_mod"` (plus a comment explaining the Phase 14 additions). All 18 tests in the file pass unchanged.

### Regression (all green)

- `tests/test_task_simulator.py`: 18/18 in 36.64s (confirms no pre-existing test broke).
- Phase 14 + Phase 13 + downstream subset (`test_task_simulator + test_bilinear_simulator + test_stimulus_utils + test_ode_integrator + test_coupled_system_bilinear + test_linear_invariance + test_bilinear_utils + test_neural_state + test_stability_monitor`): 98/98 in 253.85s.
- `tests/test_task_dcm_model.py`: 10/10 in 19.67s (direct downstream of simulator; Pyro model unchanged).

### Grep sentinels

| Sentinel | File | Count | Expected |
|----------|------|------:|---------:|
| `def simulate_task_dcm` | task_simulator.py | 1 | 1 |
| `B_list` | task_simulator.py | 31 | ≥10 |
| `stimulus_mod` | task_simulator.py | 14 | ≥6 |
| `merge_piecewise_inputs` | task_simulator.py | 2 | ≥2 |
| `set(result.keys())` | tests/test_task_simulator.py | 1 | 1 |
| `B_list` (dict access) | test_bilinear_simulator.py | 5 | ≥4 |
| `torch.equal` | test_bilinear_simulator.py | 4 | ≥3 |
| `atol=1e-4` | test_bilinear_simulator.py | 4 | ≥2 |

### Linting

- `ruff check src/pyro_dcm/simulators/task_simulator.py tests/test_bilinear_simulator.py`: clean on all new code. The single E501 violation on line 1033 of `task_simulator.py` is the pre-existing `make_random_stable_A` issue flagged in 14-01 SUMMARY and is untouched by this plan.
- `tests/test_task_simulator.py` has 4 pre-existing E501 violations in other test methods (not in the one-line diff made here; verified pre-existing via `git stash` round-trip).
- mypy: not available in local env (consistent with 14-01 SUMMARY).

## Decisions Made

All decisions applied verbatim from the plan frontmatter (no new decisions made). See the `key-decisions` block above for the full list of L3 / L4 / L5 / L6 and the test-fixture C matrix choice.

## Deviations from Plan

**1. [Rule 1 — Bug] Test fixture C matrix changed from `torch.eye(3, 1)` to `torch.tensor([[0.25], [0.0], [0.0]])` in Tests 1-3.**

- **Found during:** Task 3 initial test run (before commit).
- **Issue:** The plan's pseudocode specified `C = torch.eye(3, 1)` (unit amplitude on region 0) for Tests 1-3. With `solver='dopri5'` (the `simulate_task_dcm` default) and `A = make_random_stable_A(n_regions=3, density=0.5, seed=42)`, dopri5's adaptive step size underflowed to 0.0 at integration start (torchdiffeq assertion error `underflow in dt 0.0` in `rk_common.py:284`). The underflow is triggered by the combination of unit-amplitude step-function drive + random-A coupling off-diagonal structure; the default dopri5 tolerance (rtol=1e-5) demands a step small enough that double-precision can no longer represent `t0 + dt > t0`.
- **Fix:** Switched `C` to `torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)` in Tests 1, 2, 3. This matches the `simple_C_3` fixture convention in `tests/test_task_simulator.py` (0.25 amplitude on region 0, zero elsewhere), which has been used successfully across 40+ existing tests with dopri5. Tests 4 and 5 use `parameterize_A(zeros)` + `rk4` fixed-step so they were never at risk.
- **Why this is a testbed bug, not an algorithmic bug:** the simulator itself is correct. The issue is that the original plan's test fixture inadvertently stress-tested the torchdiffeq adaptive-step solver with an amplitude 4× the SPM convention. The fix keeps the tests faithful to their stated purpose (Test 1 bit-exactness, Test 2 distinguishability, Test 3 key presence) while using fixtures consistent with the rest of the task-DCM test suite.
- **Files modified:** `tests/test_bilinear_simulator.py` (three occurrences replaced in Tests 1-3; Tests 4-5 unchanged).
- **Commit:** `22ee2f7` (no separate fix commit; Task 3 commit already includes the corrected fixture).

## Divergence Note: Simulator vs CoupledDCMSystem n_driving_inputs policy

The simulator's `n_driving_inputs` defaulting policy is intentionally more permissive than `CoupledDCMSystem`'s. The researcher 14-RESEARCH.md §4.4 and the plan L3 decision both codify this:

- **`CoupledDCMSystem.__init__`** raises `ValueError` when `B` is non-empty and `n_driving_inputs is None`. This is the right policy for `CoupledDCMSystem` because it does not have `C` in scope (the `C` matrix is passed in separately and its column count must be explicitly declared).
- **`simulate_task_dcm`** defaults `n_driving_inputs = C.shape[1]` when `B_list` is supplied but `n_driving_inputs` is not. This is the right policy for the simulator because it DOES have `C` in scope at the call site and can infer the correct value. Inconsistent explicit values raise `ValueError`.

Because `simulate_task_dcm` always passes an inferred-or-explicit `n_driving_inputs` to `CoupledDCMSystem`, the `CoupledDCMSystem` raise branch is never triggered by the simulator path. Direct users of `CoupledDCMSystem` (Phase 15's Pyro model, anyone building custom integrators) must continue to pass `n_driving_inputs` explicitly when supplying `B`.

## Issues Encountered

- **dopri5 adaptive step underflow** on `C = torch.eye(3, 1)` + `make_random_stable_A(seed=42)`. Resolved by switching to the 0.25-amplitude convention (see Deviation #1 above).
- **Full-suite pytest under the harness runs in background without streaming output** (noted in 14-01 SUMMARY). Regression was verified via targeted subsets covering 108 tests across every file that imports or depends on the modified modules: task_simulator, ode_integrator, task_dcm_model, neural_state, coupled_system, stability_monitor, bilinear_utils, linear_invariance, stimulus_utils. All green.
- **mypy not available in execution environment** (noted in 14-01 SUMMARY). Ruff passed on all new code; type hints follow Python 3.10+ native syntax per CLAUDE.md conventions.

## User Setup Required

None.

## Phase 14 Closure Evidence

Phase 14 success criteria traced to source + test artifacts:

| SC | Criterion | Source artifact | Test artifact |
|----|-----------|-----------------|---------------|
| SIM-01 | Variable-amplitude stick-function stimuli | Plan 14-01: `make_event_stimulus` in task_simulator.py | `tests/test_stimulus_utils.py::TestMakeEventStimulus` (13 tests) |
| SIM-02 | Boxcar modulatory inputs | Plan 14-01: `make_epoch_stimulus` in task_simulator.py | `tests/test_stimulus_utils.py::TestMakeEpochStimulus` (8 tests) |
| SIM-03 | Bilinear simulator extension with linear bit-exactness | Plan 14-02: extended `simulate_task_dcm` signature + structural linear short-circuit | `tests/test_bilinear_simulator.py::test_bilinear_arg_none_matches_no_kwarg` + `::test_bilinear_output_distinguishable_from_linear` |
| SIM-04 | Return dict has 'B_list' and 'stimulus_mod' keys | Plan 14-02: extended return dict | `tests/test_bilinear_simulator.py::test_return_dict_has_bilinear_keys` + `tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys` |
| SIM-05 | dt-invariance at atol=1e-4 under rk4 | Plan 14-02: bilinear simulator path uses existing `integrate_ode` rk4 support | `tests/test_bilinear_simulator.py::test_dt_invariance_bilinear` + `::test_dt_invariance_linear` (L5 symmetry) |

All 5 Phase 14 requirements closed. Plan 14-01 covered SIM-01, SIM-02. Plan 14-02 covers SIM-03, SIM-04, SIM-05.

## Downstream Contracts for Phase 15

Phase 15 (`task_bilinear.py` Pyro generative model; RECOV-01 setup) can rely on the following frozen contracts from Phase 14:

1. **`simulate_task_dcm` call signature.** Three keyword-only args after `*`:
   - `B_list: torch.Tensor | list[torch.Tensor] | None = None`
   - `stimulus_mod: dict[str, torch.Tensor] | PiecewiseConstantInput | None = None`
   - `n_driving_inputs: int | None = None`

2. **Return-dict keys `'B_list'` and `'stimulus_mod'`.** Populated in bilinear mode (stacked `(J, N, N)` tensor + `PiecewiseConstantInput`) and `None` in linear mode. The amortized bilinear recovery benchmark in Phase 16 can introspect these keys to determine ground-truth B for scoring.

3. **B_list normalization contract.** `_normalize_B_list` accepts `list[Tensor]`, `tuple[Tensor, ...]`, or stacked `(J, N, N)` tensor. Empty list / shape `(0, N, N)` → `None` (linear mode). Phase 15 fixture authors can pass Python lists freely.

4. **Bilinear ValueError policy.** Three specific error cases at the simulator boundary: (a) `B_list non-None + stimulus_mod=None`; (b) `stimulus_mod.values.shape[1] != B_list.shape[0]`; (c) `n_driving_inputs != C.shape[1]` when both are explicit. Phase 15 error-handling tests should cite these by message fragment.

5. **`merge_piecewise_inputs` usage.** Phase 15's Pyro model, when it needs to construct the widened input inside a pyro.sample / pyro.plate context, should use `merge_piecewise_inputs(drive, mod)` from `pyro_dcm.utils` directly — the simulator's wiring is reference-only (Phase 15 constructs widened inputs inside the plate).

## Known Follow-ups

- **`LinearInterpolatedInput`** (SIM-06) deferred to v0.3.1 per STATE.md decision D2 (variable-amplitude semantics = per-event piecewise-constant in v0.3.0).
- **Amortized-guide bilinear support** deferred to v0.3.1 per D5. `amortized_wrappers.py` and `TaskDCMPacker` remain linear-only; DCM.V1 acceptance (Phase 16) uses SVI paths only.
- **Pre-existing E501 in `make_random_stable_A`** (task_simulator.py:1033) and in `tests/test_task_simulator.py` (4 occurrences). Unrelated to Plan 14-02 surface; candidates for a future `chore` commit.

## Next Phase Readiness

Phase 15 (`pyro-generative-model`) can now:

- Import the extended `simulate_task_dcm` to generate bilinear ground truth for its Pyro-model recovery fixtures.
- Rely on the breakpoint-dict or `PiecewiseConstantInput` input contract at the `stimulus` and `stimulus_mod` boundaries.
- Use `merge_piecewise_inputs` from `pyro_dcm.utils` to construct widened inputs inside the Pyro plate/sample contexts.
- Emulate the L3 "infer `n_driving_inputs` from `C.shape[1]`" policy in the Pyro model's `input_fn` construction (or pass explicitly).

No blockers. SIM-03, SIM-04, SIM-05 closed. Phase 14 closed 5/5.

---
*Phase: 14-stimulus-utilities-and-bilinear-simulator*
*Completed: 2026-04-18*
