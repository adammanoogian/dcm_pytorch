---
phase: 03-regression-dcm-forward-model
verified: 2026-03-26T12:44:15Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 3: Regression DCM Forward Model Verification Report

**Phase Goal:** Build the regression DCM frequency-domain likelihood and simulator, including the analytic posterior and ARD sparsity priors.
**Verified:** 2026-03-26T12:44:15Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Frequency-domain likelihood computes correctly for known parameters | VERIFIED | `compute_rdcm_likelihood` matches closed-form Gaussian log-likelihood to 1e-8 (test_known_value, test_perfect_fit pass) |
| 2 | Region-wise factorization p(y_j | A_j, C_j) matches analytic formula from [REF-020] | VERIFIED | `rigid_inversion` iterates VB update Eq. 11-15 per-region; `compute_free_energy_rigid` has 5 additive components validated against manual calculation; `test_five_components` and `test_known_value` pass |
| 3 | Simulator produces synthetic frequency-domain data for given (A, C, noise) | VERIFIED | `simulate_rdcm` chains `generate_bold` -> `create_regressors` -> `rigid_inversion`/`sparse_inversion`; produces `y` (75, 3) BOLD and passes to frequency-domain regression pipeline |
| 4 | Sparse A recovery: VB with Bernoulli ARD on simulated data recovers zero-pattern with F1 > 0.85 | VERIFIED | `test_sparse_recovery_3_region` passes with F1 > 0.85 on 3-region cyclic network; `test_prunes_absent_connections` confirms z < 0.5 for zero entries, z > 0.5 for present entries |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/pyro_dcm/forward_models/rdcm_forward.py` | rDCM forward pipeline: HRF, BOLD generation, design matrix, derivative coefficients | VERIFIED | 564 lines, 8 public functions, no stubs, `from __future__ import annotations`, REF-020 Eq. 4-8 citations |
| `src/pyro_dcm/forward_models/rdcm_posterior.py` | Region-wise analytic posterior (rigid + sparse), standalone likelihood, free energy, prior specs | VERIFIED | 994 lines, 7 public functions, no stubs, REF-020 Eq. 9-15 citations |
| `src/pyro_dcm/simulators/rdcm_simulator.py` | End-to-end rDCM simulator with BOLD generation and VB inversion | VERIFIED | 339 lines, 3 public functions, `simulate_rdcm` calls `generate_bold` + `create_regressors` + VB inversion |
| `src/pyro_dcm/forward_models/__init__.py` | Updated exports including rDCM forward + posterior functions | VERIFIED | 13 Phase 3 exports under "Phase 3: Regression DCM" section |
| `src/pyro_dcm/simulators/__init__.py` | Updated exports including rDCM simulator | VERIFIED | 3 Phase 3 exports: `simulate_rdcm`, `make_stable_A_rdcm`, `make_block_stimulus_rdcm` |
| `tests/test_rdcm_forward.py` | Unit tests for rdcm_forward functions | VERIFIED | 542 lines, 27 tests, all pass |
| `tests/test_rdcm_posterior.py` | Unit tests for VB inversion, free energy, prior computation | VERIFIED | 781 lines, 33 tests, all pass |
| `tests/test_rdcm_simulator.py` | Integration tests for rDCM pipeline and parameter recovery | VERIFIED | 435 lines, 14 tests, all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `rdcm_simulator.py` | `rdcm_forward.py` | `from pyro_dcm.forward_models.rdcm_forward import create_regressors, generate_bold, get_hrf` | WIRED | Line 24-28; `bold_result = generate_bold(...)` at line 298, `X, Y, N_eff = create_regressors(...)` at line 301 |
| `rdcm_simulator.py` | `rdcm_posterior.py` | `from pyro_dcm.forward_models.rdcm_posterior import rigid_inversion, sparse_inversion` | WIRED | Line 29-32; `rigid_inversion(X, Y, ...)` at line 317, `sparse_inversion(X, Y, ...)` at line 322 |
| `forward_models/__init__.py` | `rdcm_forward.py` + `rdcm_posterior.py` | direct import + `__all__` listing | WIRED | 13 Phase 3 functions imported and exported |
| `simulate_rdcm` form → VB result | posterior dict | result.update(inv_result) | WIRED | Line 337 merges VB inversion result with ground truth; all required keys present |
| `compute_rdcm_likelihood` | frequency-domain Gaussian formula [REF-020] Eq. 15 | direct implementation | WIRED | `log_lik = -0.5*N*log(2pi) + 0.5*N*log(tau) - 0.5*tau*(residual@residual)` verified to 1e-8 against manual formula |
| `compute_free_energy_rigid` | `rigid_inversion` | called inside VB loop | WIRED | Lines 474-487 in `rigid_inversion` call `compute_free_energy_rigid` on each iteration |

### Requirements Coverage

| Requirement | Status | Supporting Evidence |
|-------------|--------|---------------------|
| FWD-07: Implement regression DCM analytic likelihood in frequency domain per Frassle et al. (2017) [REF-020] Eq. 4-15 | SATISFIED | `rdcm_forward.py` implements Eq. 4-8 (design matrix construction); `rdcm_posterior.py` implements Eq. 9-15 (priors, posterior, free energy); all equations cited |
| SIM-03: Build data simulator for regression DCM: given A, C, produce synthetic frequency-domain data | SATISFIED | `simulate_rdcm(A, C, u, ...)` generates BOLD, constructs frequency-domain regressors X and Y, and runs VB inversion end-to-end |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No TODO, FIXME, placeholder, empty returns, or stub patterns found in any Phase 3 source file |

**Note:** `reduce_zeros` and `dcm_euler_step` are not exported from `pyro_dcm.forward_models.__init__.py` even though they are public functions in `rdcm_forward.py`. This is a minor incompleteness (the PLAN listed them as providing "create_regressors" etc.), but neither function is needed for downstream consumers and the export tests pass without them. No impact on phase goal.

### Human Verification Required

None — all four success criteria are verifiable programmatically via the test suite.

### Gaps Summary

No gaps found. All four observable truths are verified by the test suite. 194 total tests pass (120 from Phase 1-2 + 74 from Phase 3: 27 + 33 + 14), with zero regressions against the Phase 1-2 baseline.

---

## Test Run Summary

```
Platform: win32 — Python 3.12.10 (miniforge3), pytest 9.0.1
Python env: C:\Users\aman0087\AppData\Local\miniforge3
PyTorch version: 2.9.1+cpu

tests/test_rdcm_forward.py:   27 passed in 73.47s (SUMMARY-reported)
tests/test_rdcm_posterior.py: 33 passed in 2.45s (SUMMARY-reported)
tests/test_rdcm_simulator.py: 14 passed in 17.10s (verified live)

Full suite: 194 passed in 109.09s (verified live, 2026-03-26)
```

All Phase 1-2 test modules pass without regression.

---

_Verified: 2026-03-26T12:44:15Z_
_Verifier: Claude (gsd-verifier)_
