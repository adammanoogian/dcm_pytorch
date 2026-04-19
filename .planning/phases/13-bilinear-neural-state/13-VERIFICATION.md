---
phase: 13-bilinear-neural-state
verified: 2026-04-17T00:00:00Z
status: passed
score: 18/18 must-haves verified
re_verification:
  previous_status: none
  previous_score: none
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 13: Bilinear Neural State & Stability Monitor -- Verification Report

**Phase Goal:** The neural state equation computes the Friston 2003 bilinear form A_eff(t)*x + C*u with a documented eigenvalue stability monitor, while preserving bit-exact linear behavior when bilinear arguments are omitted.

**Verified:** 2026-04-17
**Status:** passed
**Re-verification:** No (initial verification).

## Goal Achievement

### Observable Truths (aggregated across 4 plan must-haves blocks)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | parameterize_B(B_free, b_mask) returns masked (J,N,N) with zero diagonal by default | VERIFIED | neural_state.py:62-153; test_default_diagonal_is_zero PASS |
| 2 | parameterize_B emits DeprecationWarning on non-zero diagonal mask | VERIFIED | neural_state.py:137-152; test_nonzero_diagonal_triggers_deprecation_warning PASS |
| 3 | compute_effective_A(A, B, u_mod) returns A + einsum(j,jnm->nm, u_mod, B) | VERIFIED | neural_state.py:156-224; test_einsum_correctness PASS |
| 4 | J=0 edge case: compute_effective_A returns A bit-exactly | VERIFIED | neural_state.py:220-221 explicit short-circuit; test_empty_J_returns_A_unchanged PASS (torch.equal) |
| 5 | neural_state.py module docstring no longer calls A+Cu bilinear | VERIFIED | Module header (lines 1-15) labels linear form; grep sentinel returns 0 hits |
| 6 | NeuralStateEquation.derivatives(B=None) returns literal A@x + C@u bit-exactly | VERIFIED | neural_state.py:322-323 literal expression; test_linear_invariance.py PASS at atol=1e-10 on 3 fixtures |
| 7 | derivatives with B.shape[0] == 0 routes through linear short-circuit | VERIFIED | neural_state.py:322 branch; test_empty_J_bit_exact PASS |
| 8 | derivatives with non-trivial B, u_mod routes through compute_effective_A | VERIFIED | neural_state.py:334-335; test_bilinear_changes_output PASS |
| 9 | test_linear_invariance.py passes at rtol=0, atol=1e-10 | VERIFIED | 3 fixtures (2-region hand-crafted, N=3 seed=42, N=5 seed=7) all PASS |
| 10 | CoupledDCMSystem accepts optional B, n_driving_inputs, stability_check_every kwargs | VERIFIED | coupled_system.py:140-143 kwargs; test_b_registered_as_buffer + test_missing_n_driving_inputs_raises PASS |
| 11 | CoupledDCMSystem None defaults preserve exact linear behavior | VERIFIED | coupled_system.py:287-291 literal v0.2.0 branch; test_linear_kwarg_none_matches_no_kwarg_bit_exact PASS (torch.equal) |
| 12 | Existing test_neural_state.py, test_ode_integrator.py, test_task_simulator.py, test_task_dcm_model.py all pass unchanged | VERIFIED | 24 + 28 tests PASS |
| 13 | A_eff eigenvalue monitor logs WARNING to pyro_dcm.stability when max Re > 0 | VERIFIED | coupled_system.py:50 logging.getLogger; coupled_system.py:366 _STABILITY_LOGGER.warning; test_unstable_A_eff_emits_warning PASS |
| 14 | Monitor never raises (D4) | VERIFIED | coupled_system.py:308-372 contains no raise; test_monitor_never_raises PASS |
| 15 | Monitor cadence is subsample (counter % stability_check_every; default 10 ~= every 2.5 rk4 ODE steps) | VERIFIED | coupled_system.py:347-351 counter-modulo; test_stability_check_every_zero_disables PASS |
| 16 | 3-sigma worst-case B: off-diag=3.0, u_mod=1 sustained, 500s rk4 dt=0.1, no NaN | VERIFIED | test_three_sigma_b_sustained_mod_no_nan_500s PASS |
| 17 | NeuralStateEquation class docstring no longer calls A+Cu bilinear | VERIFIED | neural_state.py:228 reads dx/dt = Ax + Cu (linear form; bilinear B-matrix path added in v0.3.0) |
| 18 | CLAUDE.md directory tree + PROJECT.md line 23 corrected | VERIFIED | CLAUDE.md:101-107 lists actual models/ files; grep generative_models/ returns 0 hits; PROJECT.md:23 reads **Linear** neural state equation |

**Score:** 18/18 truths VERIFIED.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/pyro_dcm/forward_models/neural_state.py | parameterize_B, compute_effective_A; NeuralStateEquation.derivatives extended | VERIFIED | 336 lines; functions at lines 62, 156; derivatives kwargs at 262-269; linear short-circuit at 322-323 |
| src/pyro_dcm/forward_models/__init__.py | Re-exports parameterize_B, compute_effective_A | VERIFIED | Lines 11-16 import, lines 52-54 in __all__ |
| src/pyro_dcm/forward_models/coupled_system.py | B, n_driving_inputs, stability_check_every kwargs; _maybe_check_stability; literal linear short-circuit | VERIFIED | 373 lines; kwargs at 140-143; _maybe_check_stability at 308-372; literal short-circuit at 287-291 |
| src/pyro_dcm/__init__.py | NullHandler for pyro_dcm logger | VERIFIED | Line 13 adds NullHandler; runtime-verified via import check |
| CLAUDE.md | Directory tree corrected (models/, not generative_models/) | VERIFIED | Lines 101-107 match filesystem contents of src/pyro_dcm/models/ |
| .planning/PROJECT.md | Line 23 Linear label | VERIFIED | **Linear** neural state equation (dx/dt = Ax + Cu) |
| tests/test_bilinear_utils.py | parameterize_B + compute_effective_A coverage | VERIFIED | 9 tests including deprecation + J=0 edge cases; ALL PASS |
| tests/test_linear_invariance.py | 3 fixtures at atol=1e-10 | VERIFIED | TestLinearInvariance covers 2-region + 2 random seeds; atol=1e-10, rtol=0 |
| tests/test_coupled_system_bilinear.py | Bilinear path + ValueError test | VERIFIED | 5 tests, all PASS |
| tests/test_stability_monitor.py | BILIN-05 caplog + BILIN-06 3-sigma 500s | VERIFIED | 5 tests, all PASS including 500s no-NaN test |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| tests/test_bilinear_utils.py | parameterize_B, compute_effective_A | direct import | WIRED | from pyro_dcm.forward_models.neural_state import at lines 15-19 |
| NeuralStateEquation.derivatives | compute_effective_A | function call | WIRED | neural_state.py:334 A_eff = compute_effective_A(self.A, B, u_mod) |
| CoupledDCMSystem.forward | compute_effective_A | function call | WIRED | coupled_system.py:296 A_eff = compute_effective_A(self.A, self.B, u_mod) |
| CoupledDCMSystem._maybe_check_stability | pyro_dcm.stability logger | module-level logging.getLogger | WIRED | coupled_system.py:50 + coupled_system.py:366 .warning(...) |
| pyro_dcm package | pyro_dcm root logger | NullHandler attachment | WIRED | __init__.py:13; runtime-verified: NullHandler attached = True |
| CLAUDE.md tree | src/pyro_dcm/models/ | documentation alignment | WIRED | Tree entries match filesystem listing exactly |

### Requirements Coverage (BILIN-01..07)

| Req | Description | Status | Supporting Evidence |
|-----|-------------|--------|---------------------|
| BILIN-01 | parameterize_B masked factory with safe-default zero diag + DeprecationWarning | SATISFIED | neural_state.py:62-153 + 5 tests pass |
| BILIN-02 | compute_effective_A composition via torch.einsum(j,jnm->nm, ...) | SATISFIED | neural_state.py:224 einsum call + 3 tests pass |
| BILIN-03 | Bit-exact v0.2.0 linear behavior when B=None (atol=1e-10) | SATISFIED | Literal short-circuit at neural_state.py:323 + coupled_system.py:291; test_linear_invariance.py all PASS at atol=1e-10, rtol=0 |
| BILIN-04 | All existing tests pass unchanged | SATISFIED | 24 test_neural_state + test_ode_integrator + 28 test_task_simulator + test_task_dcm_model ALL PASS |
| BILIN-05 | Eigenvalue monitor logs WARNING, log-only, subsample cadence | SATISFIED | _maybe_check_stability + counter-modulo + 4 caplog tests PASS |
| BILIN-06 | 3-sigma B worst case 500s no NaN | SATISFIED | test_three_sigma_b_sustained_mod_no_nan_500s PASS |
| BILIN-07 | Docs no longer mislabel A+Cu as bilinear (source + CLAUDE.md + PROJECT.md) | SATISFIED | All 3 sentinel greps correct: neural_state module header linear, generative_models/ 0 hits in CLAUDE.md, PROJECT.md:23 Linear |

### Anti-Patterns Found

None. Scan of modified files (neural_state.py, coupled_system.py, __init__.py, new test files) turned up no TODO, FIXME, placeholder, empty returns, or stub handlers. The one warnings.warn call is the documented DeprecationWarning in parameterize_B, not a stub marker.

### Success Criteria (ROADMAP.md Phase 13)

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | test_linear_invariance.py passes at atol=1e-10 | PASS | All 5 tests in TestLinearInvariance + TestBilinearPathSanity PASS |
| 2 | Existing test_neural_state.py + test_ode_integrator.py stay green | PASS | 24 tests PASS (32.36s) |
| 3 | 3-sigma B stability test passes: 500s no NaN | PASS | test_three_sigma_b_sustained_mod_no_nan_500s PASS |
| 4 | A_eff eigenvalue monitor logs WARNING when max Re > 0 | PASS | test_unstable_A_eff_emits_warning PASS; format string verified in test |
| 5 | Class + module docstrings + non-source docs no longer label A+Cu bilinear | PASS | Sentinel greps all correct (see BILIN-07 row above) |

### Test-Run Summary

| Command | Tests | Result | Wall-time |
|---------|-------|--------|-----------|
| pytest tests/test_bilinear_utils.py tests/test_linear_invariance.py tests/test_coupled_system_bilinear.py tests/test_stability_monitor.py -v | 26 | 26 PASS / 0 FAIL | 12.76s |
| pytest tests/test_neural_state.py tests/test_ode_integrator.py -x -q | 24 | 24 PASS / 0 FAIL | 32.36s |
| pytest tests/test_task_simulator.py tests/test_task_dcm_model.py -x -q | 28 | 28 PASS / 0 FAIL | 110.21s |

Total: 78/78 tests pass.

### Sentinel Command Checks

| Command | Expected | Got |
|---------|----------|-----|
| grep -nE for stale bilinear-labeled A+Cu docstring in neural_state.py | 0 hits | 0 hits |
| grep -n generative_models/ CLAUDE.md | 0 hits | 0 hits |
| grep -n "Linear neural state equation" .planning/PROJECT.md | line 23 | line 23 |
| python -c import pyro_dcm (NullHandler attached check) | NullHandler attached = True | NullHandler attached = True |

### Human Verification Required

None. All goal-critical behavior is mechanically verifiable and has been verified.

### Gaps Summary

No gaps. All 18 must-haves, all 7 requirements (BILIN-01..07), and all 5 ROADMAP success criteria are satisfied in the codebase. The phase goal is achieved:

- **Bilinear form:** compute_effective_A implements A + sum_j u_j * B_j via torch.einsum; wired into both NeuralStateEquation.derivatives and CoupledDCMSystem.forward on the non-empty-B branch.
- **Stability monitor:** _maybe_check_stability logs max Re(eig(A_eff)) to pyro_dcm.stability at configurable cadence, never raises, disables cleanly. 3-sigma worst-case 500s integration test gates against Pitfall B1.
- **Bit-exact linear preservation:** Both short-circuit sites execute the literal v0.2.0 expression self.A @ x + self.C @ u; test_linear_invariance.py locks this at atol=1e-10, rtol=0 across 3 fixtures + empty-J + no-kwarg cases.
- **Documentation drift closed:** Source docstrings (neural_state.py module + class), CLAUDE.md tree, PROJECT.md line 23 (all corrected per BILIN-07).

---

*Verified: 2026-04-17*
*Verifier: Claude (gsd-verifier)*
