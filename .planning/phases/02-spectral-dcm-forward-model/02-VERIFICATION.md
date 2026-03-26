---
phase: 02-spectral-dcm-forward-model
verified: 2026-03-26T08:15:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 2: Spectral DCM Forward Model Verification Report

**Phase Goal:** Build the spectral DCM observation pipeline -- cross-spectral density computation from time series, transfer function mapping, and a simulator for synthetic CSD data.
**Verified:** 2026-03-26T08:15:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | CSD computation from time series matches scipy.signal.csd within numerical tolerance | VERIFIED | Max diff = 0.00e+00 (machine zero) on diagonal and off-diagonal pairs |
| 2 | Transfer function H(w) = C_out(iwI-A)^-1 C_in correctly predicts CSD peaks at eigenfrequencies | VERIFIED | Max error vs direct inversion = 7.22e-16; peak at 0.0402 Hz vs eigenfrequency 0.0477 Hz |
| 3 | Endogenous (1/f^alpha) and observation noise spectral models integrated per SPM parameterization | VERIFIED | 1/f ratio 32.0 neuronal vs 5.66 observation; C=1/256 scaling exact; off-diagonal global noise confirmed |
| 4 | Simulator produces synthetic CSD given (A, noise params, frequency range) | VERIFIED | Shape (32,4,4) and (64,4,4), dtype complex128, no NaN/Inf, all intermediates returned |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/pyro_dcm/forward_models/spectral_transfer.py` | Transfer function + predicted CSD (FWD-06) | VERIFIED | 207 lines, no stubs, exports 4 functions |
| `src/pyro_dcm/forward_models/spectral_noise.py` | Neuronal + observation noise CSD (FWD-06) | VERIFIED | 183 lines, no stubs, exports 3 functions |
| `src/pyro_dcm/forward_models/csd_computation.py` | Welch-based empirical CSD from BOLD (FWD-05) | VERIFIED | 176 lines, no stubs, exports 3 functions |
| `src/pyro_dcm/simulators/spectral_simulator.py` | Synthetic CSD simulator (SIM-02) | VERIFIED | 249 lines, no stubs, exports 2 functions |
| `src/pyro_dcm/forward_models/__init__.py` | Package exports for all Phase 2 modules | VERIFIED | 13 exports total (5 Phase 1 + 8 Phase 2) |
| `src/pyro_dcm/simulators/__init__.py` | Package exports for spectral simulator | VERIFIED | 5 exports total (3 Phase 1 + 2 Phase 2) |
| `tests/test_spectral_transfer.py` | Unit tests for transfer function and CSD | VERIFIED | 14 tests, all pass |
| `tests/test_spectral_noise.py` | Unit tests for noise models | VERIFIED | 13 tests, all pass |
| `tests/test_csd_computation.py` | Unit tests for empirical CSD | VERIFIED | 12 tests, all pass |
| `tests/test_spectral_simulator.py` | Integration tests for full pipeline | VERIFIED | 26 tests, all pass |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `spectral_simulator.py` | `spectral_transfer.py` | import spectral_dcm_forward | WIRED | Lines 31-34: imports compute_transfer_function and spectral_dcm_forward |
| `spectral_simulator.py` | `spectral_noise.py` | import noise functions | WIRED | Lines 25-29: imports neuronal_noise_csd, observation_noise_csd, default_noise_priors |
| `spectral_transfer.py` | `spectral_noise.py` | import for CSD assembly | WIRED | Lines 18-21: imports both noise CSD functions |
| `forward_models/__init__.py` | all Phase 2 forward modules | re-export via __all__ | WIRED | 8 Phase 2 symbols exported, all importable at package level |
| `simulators/__init__.py` | `spectral_simulator.py` | re-export via __all__ | WIRED | simulate_spectral_dcm and make_stable_A_spectral in __all__ |
| `compute_empirical_csd` | scipy.signal.csd | direct call with scaling=density | WIRED | Lines 119-125: calls scipy.signal.csd for each region pair |
| `bold_to_csd_torch` | `compute_empirical_csd` | numpy conversion then call | WIRED | Lines 171-176: converts torch to numpy via compute_empirical_csd |
| `predicted_csd` | H, Gu, Gn | H @ Gu @ H.conj().T + Gn | WIRED | Lines 151-152: implements [REF-010] Eq. 4 exactly |

### Requirements Coverage

| Requirement | Status | Supporting Artifacts |
| ----------- | ------ | -------------------- |
| FWD-05 | SATISFIED | csd_computation.py (Welch CSD via scipy.signal.csd), test_csd_computation.py (12 tests including sinusoidal peak) |
| FWD-06 | SATISFIED | spectral_transfer.py (H(w) Eq. 3-4), spectral_noise.py (Eq. 5-7 noise), 27 unit tests across both files |
| SIM-02 | SATISFIED | spectral_simulator.py (simulate_spectral_dcm), test_spectral_simulator.py (eigenfrequency + roundtrip) |

### Anti-Patterns Found

None. All Phase 2 source files are clean:

- Zero TODO/FIXME/XXX/HACK/placeholder comments in any source file
- Zero stub return patterns (no return null, return {}, return [])
- All four source files contain `from __future__ import annotations`
- All mathematical functions cite [REF-010] equation numbers in docstrings

### Human Verification Required

None. All four success criteria are fully verifiable from the codebase and automated tests.

## Test Count Verification

| Test File | Tests | Phase |
| --------- | -----:| -----:|
| test_balloon.py | 7 | 1 |
| test_bold_signal.py | 6 | 1 |
| test_neural_state.py | 8 | 1 |
| test_ode_integrator.py | 17 | 1 |
| test_task_simulator.py | 17 | 1 |
| **Phase 1 subtotal** | **55** | |
| test_spectral_transfer.py | 14 | 2 |
| test_spectral_noise.py | 13 | 2 |
| test_csd_computation.py | 12 | 2 |
| test_spectral_simulator.py | 26 | 2 |
| **Phase 2 subtotal** | **65** | |
| **TOTAL** | **120** | |

All 120 tests pass. Zero regressions: all 55 Phase 1 tests continue to pass.

## Implementation Quality Notes

**Eigendecomposition-based transfer function:** Modal decomposition avoids per-frequency matrix inversion, matching SPM12 spm_dcm_mtf.m. Eigenvalue stabilization clamps real parts to max(-1/32) per SPM fMRI convention. Verified against direct inversion within 7.22e-16.

**SPM noise constants faithfully implemented:**
- C = 1/256 global scaling on all noise spectra (computed value matches expected within 1e-10)
- Observation exponent divided by 2, giving flatter spectrum (neuronal ratio 32.0 vs observation 5.66)
- Global observation noise divided by 8.0 matching spm_csd_fmri_mtf.m
- 4N+2 parameters total, all in log-space, per [REF-010]

**Hermitian CSD by construction:** Upper-triangle computation plus conjugation guarantees exact Hermitian symmetry in empirical CSD. Predicted CSD formula H @ Gu @ H^H + Gn is Hermitian by construction.

**Autograd-compatible:** spectral_dcm_forward produces finite gradients w.r.t. A matrix, enabling use in Phase 4 Pyro generative models.

**Eigenfrequency physics test:** Integration test in test_spectral_simulator.py constructs A with known complex eigenvalues (sigma=-0.5, omega=0.3), then verifies the transfer function magnitude peak is within 3 frequency bins of the analytically computed resonance and within 0.02 Hz of the eigenfrequency -- directly validating the physical correctness of the spectral DCM forward model.

---

_Verified: 2026-03-26T08:15:00Z_
_Verifier: Claude (gsd-verifier)_
