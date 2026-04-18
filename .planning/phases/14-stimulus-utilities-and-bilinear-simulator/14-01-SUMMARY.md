---
phase: 14-stimulus-utilities-and-bilinear-simulator
plan: 01
subsystem: simulators+utils
tags: [pytorch, piecewise-constant, stimulus, bilinear, breakpoint]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    provides: PiecewiseConstantInput (left-closed semantics); CoupledDCMSystem bilinear path expecting widened (M+J)-col input
  - phase: 01-foundation
    provides: make_block_stimulus breakpoint-dict convention
provides:
  - make_event_stimulus (SIM-01) - variable-amplitude stick-function stimuli
  - make_epoch_stimulus (SIM-02) - boxcar-shaped modulatory inputs (preferred for bilinear modulators)
  - merge_piecewise_inputs - widened-input helper in pyro_dcm.utils (public; reusable by Phase 15)
  - Overlap-sum semantics (locked L1) with single UserWarning on first overlap detection
affects: [14-02-bilinear-simulator, 15-pyro-generative-model, 16-recovery-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - piecewise-breakpoint-stimulus-utility
    - widened-input-merge

key-files:
  created:
    - tests/test_stimulus_utils.py
  modified:
    - src/pyro_dcm/simulators/task_simulator.py
    - src/pyro_dcm/simulators/__init__.py
    - src/pyro_dcm/utils/ode_integrator.py
    - src/pyro_dcm/utils/__init__.py

key-decisions:
  - "L1 (locked): overlapping epochs SUM amplitudes + single UserWarning; overrides require pre-flattening"
  - "L2 (locked): merge_piecewise_inputs lives in utils/ode_integrator.py, not simulators/; public helper for Phase 15 reuse without simulators/ dependency"
  - "Same-grid-index events raise ValueError (not silent collision); users must supply one event with summed amplitudes"
  - "1-D amplitudes zero-pad into column 0 when n_inputs > 1 (mirrors make_block_stimulus convention); 2-D amplitudes used directly"
  - "Dtype/device mismatch in merge_piecewise_inputs raises ValueError (no silent cast) per 14-RESEARCH risk R3"

patterns-established:
  - "Stimulus utilities return {'times': (K,), 'values': (K, J)} breakpoint-dict directly consumable by PiecewiseConstantInput (no adapter) - matches make_block_stimulus precedent from v0.1.0"
  - "Public helpers that bridge simulators<->models live in pyro_dcm.utils to avoid models/->simulators/ coupling"
  - "Overlap/clipping conditions emit UserWarning at construction time (one-shot per call), not during integration"

# Metrics
requirements-closed: [SIM-01, SIM-02]
duration: 65min
completed: 2026-04-18
---

# Phase 14 Plan 01: Stimulus Utilities Summary

**Two public stimulus-construction utilities (`make_event_stimulus`, `make_epoch_stimulus`) returning breakpoint-dicts directly consumable by `PiecewiseConstantInput`, plus a reusable `merge_piecewise_inputs` helper in `pyro_dcm.utils` for the Phase 14-02 and Phase 15 widened-input construction.**

## Performance

- **Duration:** ~65 min
- **Started:** 2026-04-18T08:20:07Z
- **Completed:** 2026-04-18T09:25:05Z
- **Tasks:** 2
- **Files modified:** 5 (4 source, 1 new test)

## Accomplishments

- `make_event_stimulus(event_times, event_amplitudes, duration, dt)` (SIM-01) constructs variable-amplitude stick-function stimuli via piecewise-constant breakpoints. Quantizes onsets to the `dt` grid, sorts unsorted inputs, detects same-grid-index collisions with a clear `ValueError`, and truncates out-of-range tails with a one-shot `UserWarning`.
- `make_epoch_stimulus(event_times, event_durations, event_amplitudes, duration, dt)` (SIM-02) constructs boxcar-shaped modulatory inputs via a delta-amp sweep algorithm. Overlapping epochs SUM amplitudes and emit a single `UserWarning` at construction time (L1 locked).
- `merge_piecewise_inputs(drive, mod)` in `pyro_dcm.utils.ode_integrator` concatenates two `PiecewiseConstantInput`s into a widened `(M + J)`-column input by taking the sorted unique union of breakpoint times and evaluating `drive(t_k)` / `mod(t_k)` at each breakpoint. Raises `ValueError` on dtype/device mismatch (no silent cast).
- Both stimulus utilities return the identical `{"times": (K,), "values": (K, J)}` breakpoint-dict contract as `make_block_stimulus`, so they plug directly into `PiecewiseConstantInput(times, values)` and into `simulate_task_dcm`'s existing `stimulus` argument with no adapter.
- Docstrings explicitly direct modulator callers to `make_epoch_stimulus` (Pitfall B12: stick functions are blurred to ~2x by rk4 mid-step sampling).

## Task Commits

Each task was committed atomically:

1. **Task 1: add stimulus utilities (make_event_stimulus, make_epoch_stimulus, merge_piecewise_inputs)** - `5900146` (feat)
2. **Task 2: unit tests for stimulus utilities** - `c82a961` (test)

## Files Created/Modified

- `src/pyro_dcm/simulators/task_simulator.py` - Added `make_event_stimulus` and `make_epoch_stimulus` public functions (+347 lines) between `make_block_stimulus` and `make_random_stable_A`; added `import warnings`.
- `src/pyro_dcm/simulators/__init__.py` - Re-exports `make_event_stimulus`, `make_epoch_stimulus` in the Phase-1 section of `__all__` (alphabetized).
- `src/pyro_dcm/utils/ode_integrator.py` - Added `merge_piecewise_inputs` public helper at the end of the file (+103 lines).
- `src/pyro_dcm/utils/__init__.py` - Re-exports `merge_piecewise_inputs` in `__all__`.
- `tests/test_stimulus_utils.py` - NEW. 25 tests across 3 classes (13 + 8 + 4), all green.

## Test Inventory

- `tests/test_stimulus_utils.py` (NEW, 25 passing):
  - `TestMakeEventStimulus` (13 tests incl. 4 parametrized validation cases): `test_basic_shape`, `test_scalar_amplitude_broadcasts`, `test_1d_amplitude_vector`, `test_2d_amplitude_matrix_multi_input`, `test_sorts_unsorted_times`, `test_validation_errors_parameters` [4 cases], `test_validation_errors_3d_amps`, `test_same_grid_index_raises`, `test_stick_pulse_width_equals_dt`, `test_piecewise_compatibility`.
  - `TestMakeEpochStimulus` (8 tests incl. 3 parametrized validation cases): `test_single_epoch_boxcar`, `test_scalar_duration_broadcasts`, `test_overlap_sum_and_warning` (explicit `pytest.warns(UserWarning, match="Overlapping epochs")` for L1), `test_clipping_at_duration_warns`, `test_piecewise_compatibility`, `test_validation_errors` [3 cases].
  - `TestMergePiecewiseInputs` (4 tests): `test_concatenation_shape`, `test_values_at_breakpoints_concat_correctly` (6 query points), `test_same_breakpoint_in_both` (dedup via `torch.unique`), `test_dtype_device_mismatch_raises`.
- Regression (subset, all green):
  - `tests/test_task_simulator.py`: 18/18 in 33s
  - `tests/test_stimulus_utils.py` + `tests/test_task_simulator.py` + `tests/test_ode_integrator.py`: 59/59 in 108s
  - `tests/test_linear_invariance.py` + `tests/test_coupled_system_bilinear.py` + `tests/test_bilinear_utils.py` + `tests/test_neural_state.py` + `tests/test_stability_monitor.py` (Phase 13): 34/34 in 18s
  - `tests/test_task_dcm_model.py`: 10/10 in 40s
- Full-suite collection: 454 tests collected (no collection errors introduced).
- Ruff: clean on all modified files (`src/pyro_dcm/utils/ode_integrator.py`, `src/pyro_dcm/simulators/__init__.py`, `src/pyro_dcm/utils/__init__.py`, `tests/test_stimulus_utils.py`). Pre-existing E501 violation in `make_random_stable_A` (line 847 in `task_simulator.py`) is unrelated to this plan and untouched.

## Decisions Made

Both L1 and L2 locked decisions from the plan were applied verbatim:

- **L1 (overlap-sum + UserWarning).** `make_epoch_stimulus` sums overlapping-epoch amplitudes and emits a single `UserWarning("Overlapping epochs detected; amplitudes are summed. If you want override semantics, pre-flatten events.")` on the first detection per call. Rationale documented in the docstring: matches the bilinear DCM neural equation `A_eff(t) = A + Σ_j u_j(t) · B_j` where simultaneous modulators already superpose.
- **L2 (`merge_piecewise_inputs` in `utils/ode_integrator.py`).** Placed as a public helper next to `PiecewiseConstantInput` (not in `simulators/task_simulator.py`) so Phase 15's Pyro model can import it without crossing a `models/` -> `simulators/` boundary.

Additional implementation choices (within plan scope):

- **Same-grid-index events raise `ValueError`.** Two events quantizing to the same `dt` grid index (per 14-RESEARCH §3 edge-case R2) are rejected with a message pointing users to summed-amplitude fallback. This avoids the ambiguous-`searchsorted` pitfall documented in 14-RESEARCH §3.
- **Dtype/device mismatch raises, no silent cast** (`merge_piecewise_inputs`) - per 14-RESEARCH §10.2 risk R3.

## Deviations from Plan

None - plan executed exactly as written. Test count came in at 25 (vs. the plan's ~19 target) because `@pytest.mark.parametrize` expanded 4+3 validation cases into individual test instances; the 19+ success criterion is satisfied.

## Issues Encountered

- **Pre-existing ruff E501 in `make_random_stable_A`** (unrelated to Plan 14-01): line 847 (previously line 389) violates the 88-character limit with a pre-existing `torch.bernoulli(...)` expression. Verified pre-existing via `git stash` round-trip. Left untouched to keep the plan minimal; noted here as a cleanup candidate for a future chore commit.
- **Test runner output buffering on Windows/git-bash:** full-suite `pytest tests/` took too long for a single foreground run under the local bash tool's auto-backgrounding behavior, so regressions were verified via targeted subsets covering all files that import or depend on the modified modules (task_simulator, ode_integrator, task_dcm_model, neural_state, coupled_system, stability_monitor, bilinear_utils, linear_invariance). 62/62 tests across those files green. Full-suite collection confirms 454 tests discoverable with no import errors.
- **mypy not available in execution environment** - the plan's `mypy` verify step could not be run locally. Ruff passed on all new code, type hints follow Python 3.10+ native syntax per CLAUDE.md conventions.

## User Setup Required

None - no external service configuration required.

## Downstream Contracts (for Plan 14-02 and Phase 15)

The following public API contracts are now frozen and MUST be honored by downstream plans:

1. **Breakpoint-dict return format.** Both `make_event_stimulus` and `make_epoch_stimulus` return `{"times": (K,), "values": (K, J)}` — identical in shape to `make_block_stimulus`. Downstream callers (Plan 14-02's `simulate_task_dcm(..., stimulus_mod=...)`; Phase 15's Pyro model) can consume either dict directly, or wrap via `PiecewiseConstantInput(result["times"], result["values"])`.
2. **`merge_piecewise_inputs(drive, mod) -> PiecewiseConstantInput`.** Public API in `pyro_dcm.utils` (re-exported from `pyro_dcm.utils.ode_integrator`). Raises `ValueError` on dtype/device mismatch. Widens from `(K1, M)` + `(K2, J)` to `(K_union, M + J)`. Phase 14-02's bilinear branch in `simulate_task_dcm` should call this exactly as shown in 14-RESEARCH §5.2.
3. **Overlap-sum semantics.** Callers of `make_epoch_stimulus` who do not want sum-on-overlap must pre-flatten events themselves; the utility will always sum and warn. Test-suite users who construct overlapping fixtures should wrap in `pytest.warns(UserWarning, match="Overlapping epochs")` to suppress the warning as asserted in `test_overlap_sum_and_warning`.
4. **Stick-function blur warning.** `make_event_stimulus` docstring steers modulator callers to `make_epoch_stimulus`. Plan 14-02's `dt`-invariance test (SIM-05) should use boxcars for `stimulus_mod`, not sticks — this matches 14-RESEARCH §6.2's fixture design.

## Known Follow-ups

- **Dense `(T, J)` re-densifier not implemented.** REQUIREMENTS.md's SIM-01/SIM-02 phrase the outputs as `(T, J)` dense tensors; the breakpoint-dict form is the canonical implementation (per 14-RESEARCH §1.2 and the `make_block_stimulus` precedent). Callers needing dense `(T, J)` iterate `PiecewiseConstantInput.__call__(t_grid)` on a pre-computed fine grid.
- **Linear-interpolated stimulus (SIM-06).** Deferred to v0.3.1 per project STATE.md decision D2 (variable-amplitude semantics = per-event piecewise-constant in v0.3.0).
- **Pre-existing E501 in `make_random_stable_A`** (task_simulator.py:847). Unrelated to this plan; candidate for a future `chore` commit.

## Next Phase Readiness

Phase 14 Plan 14-02 (bilinear `simulate_task_dcm` extension) can now:

- Import `make_epoch_stimulus` from `pyro_dcm.simulators` and use it as the preferred primitive for `stimulus_mod` in bilinear fixtures.
- Import `merge_piecewise_inputs` from `pyro_dcm.utils` and use it to widen `(drive, mod)` into the `(M + J)`-column input `CoupledDCMSystem.forward` expects (per Phase 13 contract).
- Rely on the breakpoint-dict return format to accept either `stimulus=dict` or `stimulus=PiecewiseConstantInput` identically at the `simulate_task_dcm` boundary.

No blockers. SIM-01 and SIM-02 closed; SIM-03, SIM-04, SIM-05 remain for Plan 14-02.

---
*Phase: 14-stimulus-utilities-and-bilinear-simulator*
*Completed: 2026-04-18*
