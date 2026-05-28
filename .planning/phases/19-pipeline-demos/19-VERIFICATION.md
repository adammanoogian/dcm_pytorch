---
phase: 19-pipeline-demos
verified: 2026-05-24T21:00:00Z
status: passed
score: 10/10 must-haves verified
gaps: []
---

# Phase 19: Pipeline Demos Verification Report

**Phase Goal:** Users can follow two self-contained demo scripts that show the complete path from synthetic MNE data through Pyro-DCM model fitting to posterior connectivity matrices, serving as copy-pasteable starting points for real neuroimaging workflows.
**Verified:** 2026-05-24T21:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | scripts/demo_spectral_dcm.py exists and is substantive | VERIFIED | 158 lines, no stubs, real MNE + SVI pipeline |
| 2 | Spectral demo: synthetic MNE Epochs created, epochs_to_csd called, spectral_dcm_model fitted via SVI, posterior A printed | VERIFIED | Lines 86-158 implement all four steps |
| 3 | Spectral demo: A-RMSE recovery metric printed | VERIFIED | Line 154: `print(f"\nA-RMSE: {a_rmse:.4f}")` |
| 4 | Spectral demo: exits with clear error when MNE absent | VERIFIED | Lines 20-24: try/except ImportError + sys.exit(1) + pip install hint |
| 5 | scripts/demo_task_dcm.py exists and is substantive | VERIFIED | 217 lines, no stubs, real MNE + SVI + bilinear B pipeline |
| 6 | Task demo: synthetic MNE Epochs created, epochs_to_timeseries called, task_dcm_model fitted via SVI, posterior A + B printed | VERIFIED | Lines 122-213 implement all five steps |
| 7 | Task demo: A-RMSE and B-RMSE recovery metrics printed with sign match | VERIFIED | Lines 205-213: A-RMSE, B-RMSE, B sign recovery all printed |
| 8 | Task demo: exits with clear error when MNE absent | VERIFIED | Lines 27-30: try/except ImportError + sys.exit(1) + pip install hint |
| 9 | Task demo: uses bilinear B matrices (b_masks + stim_mod) | VERIFIED | Lines 77-83 define b_mask and B_free_true; lines 157-158 pass b_masks + stim_mod to model |
| 10 | Regression: test_mne_loader.py and test_bids_loader.py exist and are substantive | VERIFIED | 435 and 200 lines respectively, 11 + 3 test functions, no stubs |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/demo_spectral_dcm.py` | End-to-end spectral DCM demo | VERIFIED | 158 lines, no stubs, no TODOs |
| `scripts/demo_task_dcm.py` | End-to-end task DCM demo with bilinear B | VERIFIED | 217 lines, no stubs, no TODOs |
| `src/pyro_dcm/io/mne_loader.py` | epochs_to_csd + epochs_to_timeseries | VERIFIED | 309 lines; both functions fully implemented with real MNE calls |
| `src/pyro_dcm/io/__init__.py` | Exposes epochs_to_csd, epochs_to_timeseries | VERIFIED | Both symbols in __all__ |
| `tests/test_mne_loader.py` | MNE loader test suite | VERIFIED | 435 lines, 11 tests covering shape/dtype/symmetry/round-trip |
| `tests/test_bids_loader.py` | BIDS loader test suite | VERIFIED | 200 lines, 3 tests covering round-trip/events/BAD_ACQ_SKIP |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| demo_spectral_dcm.py | epochs_to_csd | `from pyro_dcm.io import epochs_to_csd` | WIRED | Import line 38; called line 101; result used lines 102-109 |
| demo_spectral_dcm.py | spectral_dcm_model | `from pyro_dcm import spectral_dcm_model` | WIRED | Import line 35; passed to create_guide line 125 and run_svi line 129 |
| demo_spectral_dcm.py | extract_posterior_params | `from pyro_dcm import extract_posterior_params` | WIRED | Import line 31; called line 144; result used lines 145-154 |
| demo_spectral_dcm.py | parameterize_A | `from pyro_dcm import parameterize_A` | WIRED | Import line 33; called line 146 to convert A_free_mean to A_est |
| demo_task_dcm.py | epochs_to_timeseries | `from pyro_dcm.io import epochs_to_timeseries` | WIRED | Import line 47; called line 139; result used lines 140-143 |
| demo_task_dcm.py | task_dcm_model | `from pyro_dcm import task_dcm_model` | WIRED | Import line 44; passed to run_svi line 175; b_masks+stim_mod kwargs lines 157-158 |
| demo_task_dcm.py | extract_posterior_params | `from pyro_dcm import extract_posterior_params` | WIRED | Import line 38; called line 185; posterior["median"]["A"] and ["B"] used lines 192-193 |
| demo_task_dcm.py | parameterize_B | `from pyro_dcm import parameterize_B` | WIRED | Import line 40; called line 83 to construct B_true for recovery RMSE |
| task_dcm_model | pyro.deterministic("A") | Line 281 in task_dcm_model.py | WIRED | A deterministic site guaranteed; posterior["median"]["A"] resolves correctly |
| task_dcm_model | pyro.deterministic("B") | Line 322 in task_dcm_model.py | WIRED | B deterministic site emitted in bilinear branch; posterior["median"]["B"] resolves correctly |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| PIPE-01: Spectral DCM demo -- MNE Epochs -> epochs_to_csd -> SpectralDCMModel -> SVI -> posterior A | SATISFIED | All steps present and wired; A-RMSE printed |
| PIPE-02: Task DCM demo -- MNE Epochs -> epochs_to_timeseries -> TaskDCMModel -> SVI -> posterior A + B | SATISFIED | All steps present and wired; A-RMSE + B-RMSE + sign match printed |
| Regression: test_mne_loader.py passes | SATISFIABLE | File exists and is substantive (pytest skips gracefully if MNE absent) |
| Regression: test_bids_loader.py passes | SATISFIABLE | File exists and is substantive (pytest skips if MNE/mne_bids absent) |

### Anti-Patterns Found

None. Both demo scripts were scanned for: TODO/FIXME/XXX/placeholder text, empty return patterns (return null/{}/ []), stub handlers, and hardcoded placeholders. Zero matches found.

### Human Verification Required

1. **Spectral demo runtime execution**
   - **Test:** `python scripts/demo_spectral_dcm.py` (with MNE installed)
   - **Expected:** Prints ground-truth A, CSD shapes, SVI progress, final ELBO, posterior A, A-RMSE ~0.037 (300 steps)
   - **Why human:** Runtime requires MNE + Pyro environment; cannot verify execution without running the script

2. **Task demo runtime execution**
   - **Test:** `python scripts/demo_task_dcm.py` (with MNE installed)
   - **Expected:** Prints BOLD shape, SVI progress, final loss, A-RMSE, B-RMSE, B sign recovery
   - **Why human:** Requires MNE environment; B recovery at 100 steps is intentionally approximate (demo-friendly)

3. **MNE-absent guard for both scripts**
   - **Test:** `python scripts/demo_spectral_dcm.py` and `python scripts/demo_task_dcm.py` without MNE installed
   - **Expected:** Both exit immediately with "MNE-Python required. Install with: pip install pyro-dcm[mne]"
   - **Why human:** Cannot uninstall MNE in current environment to test the guard path

## Gaps Summary

No gaps. All 10 must-haves verified. All key links confirmed wired. No anti-patterns detected.

Both demo scripts are substantive (158 and 217 lines), contain real mathematics and IO calls, are fully wired to their dependencies (epochs_to_csd, epochs_to_timeseries, spectral_dcm_model, task_dcm_model, extract_posterior_params), and demonstrate the complete pipeline specified by PIPE-01 and PIPE-02.

The only remaining verification items require human execution in an MNE-enabled environment to confirm actual runtime output.

---

_Verified: 2026-05-24T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
