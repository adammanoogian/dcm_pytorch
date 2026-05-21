---
phase: 18-mne-bids-io-test-suite
verified: 2026-05-21T16:59:32Z
status: passed
score: 17/17 must-haves verified
---

# Phase 18: MNE/BIDS IO Test Suite Verification Report

**Phase Goal:** The MNE and BIDS IO loaders have a comprehensive test suite that validates shape contracts, mathematical properties, error handling, and critical scientific pitfalls, runnable via `pytest -m mne` and cleanly skipped when MNE is not installed.
**Verified:** 2026-05-21T16:59:32Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | pytest marker mne is registered in pyproject.toml markers list | VERIFIED | pyproject.toml line 77 contains the marker string; confirmed by `pytest --markers` |
| 2  | epochs_to_csd returns (F, N, N) complex tensor and (F,) frequency vector | VERIFIED | test_epochs_to_csd_shape asserts shape (32,3,3) complex, freqs (32,); PASSES |
| 3  | epochs_to_timeseries returns (T, N) averaged and (n_epochs, T, N) unaveraged | VERIFIED | test_epochs_to_timeseries_shape_averaged (256,3) / _unaveraged (10,256,3) both PASS |
| 4  | raw_to_timeseries returns (T, N) float tensor from synthetic Raw | VERIFIED | test_raw_to_timeseries_shape asserts (2560,3) float64; PASSES |
| 5  | stc_to_roi_timeseries returns (T, N) float tensor with mocked extract_label_time_course | VERIFIED | test_stc_to_roi_timeseries_shape asserts (100,3) float64, roi_names, sfreq=256; PASSES |
| 6  | Channel picks subsetting produces N matching pick count | VERIFIED | test_channel_picks_subsetting asserts N=2 from 5, N=1 from 5; PASSES |
| 7  | Explicit picks excluding bad channels reduces N correctly | VERIFIED | test_bad_channel_exclusion: picks=None gives 3, explicit picks gives 2; PASSES |
| 8  | CSD satisfies Hermitian symmetry csd[f,i,j] == conj(csd[f,j,i]) | VERIFIED | test_csd_hermitian_symmetry: torch.allclose(csd, csd.conj().transpose(-2,-1), atol=1e-10); PASSES |
| 9  | CSD auto-spectra diagonal is real and non-negative | VERIFIED | test_csd_nonnegative_autospectra: real >= -1e-12 and imag allclose 0 for all channels; PASSES |
| 10 | 10 Hz sine injection produces CSD peak at 10 Hz bin +/- 1 bin | VERIFIED | test_csd_sine_injection_roundtrip: abs(peak_idx - target_idx) <= 1; PASSES |
| 11 | _require_mne() raises ImportError with pip install instruction | VERIFIED | test_require_mne_import_error mocks __import__ and asserts ImportError with correct message; PASSES |
| 12 | epochs_to_csd raises ValueError for invalid method | VERIFIED | test_epochs_to_csd_invalid_method: raises ValueError match="method must be"; PASSES |
| 13 | pytest.importorskip(mne) at module level skips file when MNE absent | VERIFIED | test_mne_loader.py line 23: `mne = pytest.importorskip("mne")` |
| 14 | pytestmark = pytest.mark.mne applied to all mne tests | VERIFIED | test_mne_loader.py line 33: `pytestmark = pytest.mark.mne` |
| 15 | load_bids_raw returns valid mne.io.BaseRaw from synthetic BIDS dataset | VERIFIED | test_load_bids_raw: isinstance(BaseRaw), channels present, sfreq=256, n_times>0; PASSES |
| 16 | load_bids_epochs returns valid mne.Epochs from BIDS with events | VERIFIED | test_load_bids_epochs: isinstance(Epochs), len>0, sfreq=256; PASSES |
| 17 | BAD_ACQ_SKIP annotation handled without error | VERIFIED | test_bids_bad_acq_skip_annotation: no crash, len(epochs)>=1; PASSES |

**Score:** 17/17 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Contains mne pytest marker string | VERIFIED | Line 77 has the marker; confirmed via pytest --markers |
| `tests/test_mne_loader.py` | pytestmark, importorskip, 11+ test functions | VERIFIED | 12 test functions, 435 lines, pytestmark line 33, importorskip line 23 |
| `tests/test_bids_loader.py` | pytestmark, importorskip for mne+mne_bids, 3 test functions | VERIFIED | 3 test functions, 200 lines, pytestmark line 24, both importorskip lines 19-20 |
| `src/pyro_dcm/io/mne_loader.py` | 4 loader functions + _require_mne | VERIFIED | 309 lines; all 5 symbols implemented with real math, no stubs |
| `src/pyro_dcm/io/bids_loader.py` | load_bids_raw, load_bids_epochs | VERIFIED | 137 lines; both functions fully implemented |
| `src/pyro_dcm/io/__init__.py` | Re-exports loaders | VERIFIED | Imports all 4 MNE functions; bids under try/except for optional dep |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/test_mne_loader.py` | `src/pyro_dcm/io/mne_loader.py` | `from pyro_dcm.io.mne_loader import` | WIRED | Line 25 imports 5 symbols; all used in test functions |
| `tests/test_bids_loader.py` | `src/pyro_dcm/io/bids_loader.py` | `from pyro_dcm.io.bids_loader import` | WIRED | Line 22 imports load_bids_epochs, load_bids_raw; both used in 3 tests |

### Anti-Patterns Found

None. No TODOs, FIXME, placeholder content, empty returns, or stub patterns found in any test or source file.

### Lint and Format

| Check | Result |
|-------|--------|
| `ruff check tests/test_mne_loader.py tests/test_bids_loader.py` | All checks passed |
| `ruff format --check tests/test_mne_loader.py tests/test_bids_loader.py` | 2 files already formatted |

### Test Execution

All 15 tests pass in 12.32s.

```
tests/test_mne_loader.py::test_epochs_to_csd_shape PASSED
tests/test_mne_loader.py::test_epochs_to_timeseries_shape_averaged PASSED
tests/test_mne_loader.py::test_epochs_to_timeseries_shape_unaveraged PASSED
tests/test_mne_loader.py::test_raw_to_timeseries_shape PASSED
tests/test_mne_loader.py::test_stc_to_roi_timeseries_shape PASSED
tests/test_mne_loader.py::test_channel_picks_subsetting PASSED
tests/test_mne_loader.py::test_bad_channel_exclusion PASSED
tests/test_mne_loader.py::test_csd_hermitian_symmetry PASSED
tests/test_mne_loader.py::test_csd_nonnegative_autospectra PASSED
tests/test_mne_loader.py::test_csd_sine_injection_roundtrip PASSED
tests/test_mne_loader.py::test_require_mne_import_error PASSED
tests/test_mne_loader.py::test_epochs_to_csd_invalid_method PASSED
tests/test_bids_loader.py::test_load_bids_raw PASSED
tests/test_bids_loader.py::test_load_bids_epochs PASSED
tests/test_bids_loader.py::test_bids_bad_acq_skip_annotation PASSED
15 passed, 12 warnings in 12.32s
```

12 warnings are all RuntimeWarning from MNE/mne-bids internals (baseline correction advisory, BrainVision format conversion) -- none are assertion errors.

### Human Verification Required

None. All truths are programmatically verifiable via test assertions.

## Gaps Summary

No gaps. All 17 must-haves verified.

---

_Verified: 2026-05-21T16:59:32Z_
_Verifier: Claude (gsd-verifier)_
