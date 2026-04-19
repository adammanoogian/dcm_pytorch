---
phase: 14-stimulus-utilities-and-bilinear-simulator
verified: 2026-04-18T10:05:42Z
status: passed
score: 14/14 must-haves verified
---

# Phase 14: Stimulus Utilities and Bilinear Simulator Verification Report

**Phase Goal:** Users can construct variable-amplitude event and epoch stimuli
and run the simulator in bilinear mode to produce context-dependent BOLD ground truth,
while the existing linear simulator output is exactly preserved when bilinear arguments are omitted.

**Verified:** 2026-04-18T10:05:42Z
**Status:** passed
**Re-verification:** No - initial verification

---
## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | make_event_stimulus constructs variable-amplitude stick-function stimuli via piecewise-constant breakpoint representation (SIM-01) | VERIFIED | Implemented at task_simulator.py:490; returns breakpoint dict (K,) + (K, n_inputs); 14 tests in TestMakeEventStimulus pass |
| 2 | make_epoch_stimulus constructs boxcar-shaped modulatory inputs; preferred per Pitfall B12 (SIM-02) | VERIFIED | Implemented at task_simulator.py:707; docstring names it preferred for modulatory inputs; 8 tests in TestMakeEpochStimulus pass |
| 3 | Overlapping epochs SUM amplitudes and emit UserWarning at construction time (L1 locked) | VERIFIED | warnings.warn at task_simulator.py:933; test_overlap_sum_and_warning uses pytest.warns and asserts value==2.0 in overlap window |
| 4 | make_event_stimulus docstring warns stick functions blurred at rk4 mid-steps (Pitfall B12) | VERIFIED | Pitfall B12 note at task_simulator.py:566; also referenced at lines 203, 729, 792 |
| 5 | merge_piecewise_inputs lives in utils/ode_integrator.py and is exported from pyro_dcm.utils (L2 locked) | VERIFIED | Defined at ode_integrator.py:244; re-exported in utils/__init__.py line 7 and __all__ line 14 |
| 6 | Both stimulus utilities return breakpoint dicts directly consumable by PiecewiseConstantInput | VERIFIED | test_piecewise_compatibility passes in both TestMakeEventStimulus and TestMakeEpochStimulus |
| 7 | simulate_task_dcm with B_list=None output identical to pre-Phase-14 linear call (SIM-03) | VERIFIED | test_bilinear_arg_none_matches_no_kwarg uses torch.equal on bold_clean, bold, neural; passes |
| 8 | Linear short-circuit structural: B_list=None calls CoupledDCMSystem without B= or n_driving_inputs= kwarg | VERIFIED | task_simulator.py:306 confirmed by grep: no B= in linear branch |
| 9 | Bilinear mode BOLD numerically distinguishable from linear null on same seed (SIM-03 secondary gate) | VERIFIED | test_bilinear_output_distinguishable_from_linear asserts max(diff) > 0.01; passes |
| 10 | Return dict contains B_list and stimulus_mod keys; both None in linear mode (SIM-04) | VERIFIED | task_simulator.py:420-421; test_return_dict_has_bilinear_keys and augmented expected_keys at test_task_simulator.py:83 confirm |
| 11 | dt-invariance: rk4 at dt=0.01 vs dt=0.005 within atol=1e-4 under bilinear ground truth (SIM-05) | VERIFIED | test_dt_invariance_bilinear uses assert_close(atol=1e-4, rtol=0.0); passes |
| 12 | All existing test_task_simulator.py tests pass unchanged; only additive change is augmented expected_keys | VERIFIED | 18 tests pass; expected_keys at line 83 is the only change to pre-existing file |
| 13 | B_list normalization accepts list, tuple, stacked (J,N,N) tensor; empty collapses to linear mode | VERIFIED | _normalize_B_list at task_simulator.py:37 handles all four input forms |
| 14 | n_driving_inputs defaults to C.shape[1] in bilinear mode (L3); missing stimulus_mod raises ValueError | VERIFIED | task_simulator.py:325-343 (L3 default) and :309-313 (ValueError guard) |

**Score:** 14/14 truths verified

---
### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/pyro_dcm/simulators/task_simulator.py | make_event_stimulus, make_epoch_stimulus, extended simulate_task_dcm | VERIFIED | All present; 1038 lines; no stubs |
| src/pyro_dcm/simulators/__init__.py | Re-exports both functions in __all__ | VERIFIED | Lines 14-15 (import) + lines 23-24 (__all__) |
| src/pyro_dcm/utils/ode_integrator.py | merge_piecewise_inputs public helper | VERIFIED | Line 244; substantive with dtype/device validation |
| src/pyro_dcm/utils/__init__.py | Re-exports merge_piecewise_inputs in __all__ | VERIFIED | Line 7 (import) + line 14 (__all__) |
| tests/test_stimulus_utils.py | 19+ unit tests across 3 test classes | VERIFIED | 25 tests collected; 426 lines |
| tests/test_bilinear_simulator.py | 5 tests covering SIM-03/04/05 | VERIFIED | 5 tests collected; 219 lines |
| tests/test_task_simulator.py | Augmented expected_keys with B_list and stimulus_mod | VERIFIED | Line 83 adds both keys; no other changes |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| make_event_stimulus / make_epoch_stimulus return dicts | PiecewiseConstantInput(times, values) | Identical breakpoint-dict shape contract | WIRED | test_piecewise_compatibility passes in both test classes |
| simulate_task_dcm linear branch (B_list=None) | CoupledDCMSystem(A, C, input_fn, hemo_params) no B= kwarg | Structural short-circuit at task_simulator.py:306 | WIRED | Grep confirmed: line 306 has only four positional args |
| simulate_task_dcm bilinear branch | CoupledDCMSystem with B=B_stacked, n_driving_inputs=n_driv | merge_piecewise_inputs at task_simulator.py:334 | WIRED | task_simulator.py:337-344 passes B= and n_driving_inputs= |
| test_bilinear_arg_none_matches_no_kwarg | torch.equal on bold_clean, bold, neural | SIM-03 structural bit-exact guarantee | WIRED | Three torch.equal assertions at test_bilinear_simulator.py:60-62 pass |

---

### Requirements Coverage

| Requirement | Closed by | Status |
|-------------|-----------|--------|
| SIM-01 | Plan 14-01: make_event_stimulus | SATISFIED |
| SIM-02 | Plan 14-01: make_epoch_stimulus with Pitfall B12 docstring | SATISFIED |
| SIM-03 | Plan 14-02: structural short-circuit + torch.equal test + distinguishability test | SATISFIED |
| SIM-04 | Plan 14-02: return dict B_list/stimulus_mod keys + expected_keys augmentation | SATISFIED |
| SIM-05 | Plan 14-02: test_dt_invariance_bilinear + test_dt_invariance_linear (L5) | SATISFIED |

---
### Anti-Patterns Found

None. All Phase 14 modified files checked for TODO/FIXME/placeholder comments (0), empty implementations (0), and stub patterns (0).

---

### Test Suite Results

| Suite | Tests | Result |
|-------|-------|--------|
| tests/test_stimulus_utils.py | 25 | PASS |
| tests/test_bilinear_simulator.py | 5 | PASS |
| tests/test_task_simulator.py | 18 | PASS |
| Phase 13 regression (5 test files) | 34 | PASS |

Combined Phase 14 run: 48 tests pass in 190s. Phase 13 regression: 34/34 in 8.5s. No cross-phase regression.

Note on plan estimate vs actual: Plan 14-02 stated test_task_simulator.py has 40+ tests; actual collected count is 18. The functional requirement (all existing tests pass unchanged) is met -- the count discrepancy is in the plan estimate only.

---

### Human Verification Required

None. All five ROADMAP success criteria are verifiable from automated test execution and static code analysis. No visual, real-time, or external service checks required.

---

## Summary

Phase 14 goal is fully achieved. All 14 must-haves are verified at all three levels (exists, substantive, wired). The phase delivers:

1. Two new stimulus utilities (make_event_stimulus, make_epoch_stimulus) with complete NumPy-style docstrings, Pitfall B12 warnings, overlap-sum semantics (L1), and 25 unit tests.
2. merge_piecewise_inputs placed in utils/ode_integrator.py per L2 decision, with dtype/device mismatch validation (no silent cast).
3. simulate_task_dcm extended with three keyword-only args; linear short-circuit is structural (line 306, no B= kwarg); bilinear path wired through merge_piecewise_inputs to CoupledDCMSystem.
4. All prior-phase tests continue to pass: Phase 13 34/34 green, no regression.

---

_Verified: 2026-04-18T10:05:42Z_
_Verifier: Claude (gsd-verifier)_
