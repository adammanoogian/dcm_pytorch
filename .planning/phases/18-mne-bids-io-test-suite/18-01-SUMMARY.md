# Phase 18 Plan 01: MNE Loader Test Suite Summary

**One-liner:** 12 tests covering all 4 MNE loader functions -- shape validation, CSD mathematical properties (Hermitian, non-negative diagonal, sine peak), channel picks, bad channel behavior, error paths, and module-level skip/marker.

## Plan Metadata

- **Phase:** 18 (MNE/BIDS IO Test Suite)
- **Plan:** 01
- **Type:** execute
- **Duration:** ~10 minutes
- **Completed:** 2026-05-21

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Register mne pytest marker + add MNE IO module | `77b3fe8` | `pyproject.toml`, `src/pyro_dcm/io/__init__.py`, `src/pyro_dcm/io/mne_loader.py` |
| 2 | Create test_mne_loader.py with all MNE loader tests | `4d9a91b` | `tests/test_mne_loader.py` |

## Test Coverage

12 test functions covering 13 requirements (TEST-12 and TEST-13 are implicit module-level behaviors):

| Test ID | Function | What it verifies |
|---------|----------|-----------------|
| TEST-01 | `test_epochs_to_csd_shape` | CSD `(F, N, N)` complex + metadata |
| TEST-02 | `test_epochs_to_timeseries_shape_averaged` | Averaged `(T, N)` float64 |
| TEST-02 | `test_epochs_to_timeseries_shape_unaveraged` | Unaveraged `(n_ep, T, N)` float64 |
| TEST-03 | `test_raw_to_timeseries_shape` | Raw `(T, N)` float64 + sfreq |
| TEST-04 | `test_stc_to_roi_timeseries_shape` | Mocked STC `(T, N)` + ROI names |
| TEST-05 | `test_channel_picks_subsetting` | Explicit picks reduce N dimension |
| TEST-06 | `test_bad_channel_exclusion` | Bads NOT excluded with picks=None |
| TEST-07 | `test_csd_hermitian_symmetry` | `csd == conj(csd.T)` to atol=1e-10 |
| TEST-08 | `test_csd_nonnegative_autospectra` | Diagonal real >= 0, imag ~= 0 |
| TEST-09 | `test_csd_sine_injection_roundtrip` | 10 Hz peak within 1 bin |
| TEST-10 | `test_require_mne_import_error` | ImportError with install hint |
| TEST-11 | `test_epochs_to_csd_invalid_method` | ValueError for bad method arg |
| TEST-12 | (implicit) `pytest.importorskip("mne")` | Module-level skip when MNE absent |
| TEST-13 | (implicit) `pytestmark = pytest.mark.mne` | Marker applied to all tests |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created MNE IO source module on current branch**

- **Found during:** Task 1 (pre-execution setup)
- **Issue:** `src/pyro_dcm/io/mne_loader.py` and `src/pyro_dcm/io/__init__.py` existed on branch `gsd/phase-16.1-recov-04-b-rmse-diagnostic` (commit `90fe82f`) but were not present on the current test-suite branch `gsd/phase-18-mne-io-test-suite`. Tests cannot import the module under test without the source files.
- **Fix:** Created both files from the source code provided in the plan context (byte-identical to commit `90fe82f`). Also added the `mne` optional dependency group to `pyproject.toml` which was missing on this branch.
- **Files created:** `src/pyro_dcm/io/__init__.py`, `src/pyro_dcm/io/mne_loader.py`
- **Commit:** `77b3fe8` (bundled with Task 1 marker registration)

## Verification Results

All 9 verification criteria from the plan passed:

1. `python -m pytest --markers | grep mne` -- mne marker registered
2. `pytest tests/test_mne_loader.py -v` -- 12/12 passed in 6.4s
3. `ruff check tests/test_mne_loader.py` -- all checks passed
4. `ruff format --check tests/test_mne_loader.py` -- already formatted
5. `from __future__ import annotations` at file top (line 14)
6. CSD Hermitian symmetry holds to atol=1e-10
7. CSD auto-spectra are non-negative
8. 10 Hz sine produces CSD peak within 1 bin of 10 Hz
9. Channel picks subsetting correctly reduces N dimension

## Files Created/Modified

### Created
- `src/pyro_dcm/io/__init__.py` -- IO package init with re-exports
- `src/pyro_dcm/io/mne_loader.py` -- 4 loader functions (epochs_to_csd, epochs_to_timeseries, raw_to_timeseries, stc_to_roi_timeseries) + _require_mne guard
- `tests/test_mne_loader.py` -- 12 test functions, 3 fixtures, 434 lines

### Modified
- `pyproject.toml` -- Added `mne` pytest marker + `mne` optional dependency group

## Decisions Made

None -- plan executed as specified with one Rule 3 deviation for source file creation.

## Next Phase Readiness

Plan 18-02 (BIDS loader tests) can proceed. The MNE IO module is now on this branch with full test coverage. No blockers.
