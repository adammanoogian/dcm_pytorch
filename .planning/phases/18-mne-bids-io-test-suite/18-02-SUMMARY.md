---
phase: 18-mne-bids-io-test-suite
plan: 02
subsystem: io-tests
tags: [mne-bids, bids, testing, round-trip]
dependency-graph:
  requires: [18-01]
  provides: [bids-loader-test-suite]
  affects: [19]
tech-stack:
  added: [mne-bids, pybv]
  patterns: [BIDS-round-trip-testing, write_raw_bids-fixture]
key-files:
  created:
    - src/pyro_dcm/io/bids_loader.py
    - tests/test_bids_loader.py
  modified: []
decisions:
  - id: BIDS-FMT-01
    decision: Use BrainVision format with allow_preload=True for synthetic BIDS writes
    reason: RawArray is always preloaded; BrainVision is the default BIDS EEG format
metrics:
  duration: ~13 minutes
  completed: 2026-05-21
---

# Phase 18 Plan 02: BIDS Loader Tests Summary

BIDS loader round-trip test suite: 3 tests verifying write_raw_bids/read_raw_bids cycle for load_bids_raw and load_bids_epochs, including BAD_ACQ_SKIP annotation handling.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 0 | Create bids_loader.py (deviation) | 34a589b | src/pyro_dcm/io/bids_loader.py |
| 1 | BIDS round-trip test suite | 7d30af2 | tests/test_bids_loader.py |

## Test Coverage

| Test ID | Test Name | What It Validates |
|---------|-----------|-------------------|
| BIDS-01 | test_load_bids_raw | write_raw_bids -> load_bids_raw preserves channels, sfreq, data |
| BIDS-02 | test_load_bids_epochs | write_raw_bids -> load_bids_epochs creates valid Epochs from annotations |
| BIDS-03 | test_bids_bad_acq_skip_annotation | BAD_ACQ_SKIP annotation does not crash epoch creation |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created bids_loader.py on current branch**
- **Found during:** Task 0 (pre-planned deviation)
- **Issue:** src/pyro_dcm/io/bids_loader.py did not exist on gsd/phase-18-mne-io-test-suite branch
- **Fix:** Created the file with the exact content from the phase-16.1 branch
- **Files created:** src/pyro_dcm/io/bids_loader.py
- **Commit:** 34a589b

**2. [Rule 1 - Bug] Added allow_preload and format params to write_raw_bids**
- **Found during:** Task 1 test execution
- **Issue:** RawArray is always preloaded; write_raw_bids rejects preloaded data by default
- **Fix:** Added `allow_preload=True` and `format="BrainVision"` to _make_bids_dataset helper
- **Files modified:** tests/test_bids_loader.py
- **Commit:** 7d30af2

**3. [Rule 3 - Blocking] Installed mne-bids and pybv dependencies**
- **Found during:** Task 1 test execution
- **Issue:** mne-bids not installed in environment; pybv required for BrainVision format
- **Fix:** pip install mne-bids pybv (runtime deps, not committed)
- **Files modified:** none (environment only)

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| BIDS-FMT-01 | Use BrainVision format with allow_preload=True | RawArray is preloaded by construction; BrainVision is the standard BIDS EEG format; requires pybv |
| BIDS-EVT-01 | Use event_id=None for epoch tests | Lets MNE auto-detect event codes from BIDS annotations; more robust to annotation round-trip code changes |

## Verification Results

- `pytest tests/test_bids_loader.py -v`: 3/3 passed
- `ruff check tests/test_bids_loader.py`: All checks passed
- `ruff format --check tests/test_bids_loader.py`: Already formatted
- `from __future__ import annotations` present at top
- Both mne and mne_bids importorskip'd at module level
- pytestmark = pytest.mark.mne applied
- Tests skip gracefully when mne or mne_bids absent

## Next Phase Readiness

Phase 18 is now fully complete (plans 18-01 and 18-02). The MNE/BIDS IO test suite provides:
- 12 MNE loader tests (18-01)
- 3 BIDS loader tests (18-02)
- Full skip behavior when optional deps absent
- Marker-based selection via `pytest -m mne`

Ready to proceed to Phase 19 (end-to-end pipeline demos) or merge to main.
