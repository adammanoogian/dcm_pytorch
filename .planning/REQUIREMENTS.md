# Requirements: Pyro-DCM v0.5.0 MNE-Python Integration

**Defined:** 2026-05-21
**Core Value:** The A matrix (effective connectivity) remains an explicit, interpretable object with full posterior uncertainty throughout inference

## v0.5.0 Requirements

Requirements for MNE-Python Integration testing and pipeline demonstrations. The IO
loaders already exist (`src/pyro_dcm/io/`); this milestone validates them via tests,
demonstrates usage via end-to-end pipeline scripts, and encodes critical scientific
pitfalls (CSD conventions, channel picks) as explicit test cases.

### IO Loader Tests

- [ ] **TEST-01**: Shape validation for `epochs_to_csd` — output `(F, N, N)` complex CSD tensor matches expected dimensions from synthetic Epochs
- [ ] **TEST-02**: Shape validation for `epochs_to_timeseries` — output `(T, N)` float tensor matches expected dimensions, both averaged and unaveraged paths
- [ ] **TEST-03**: Shape validation for `raw_to_timeseries` — output `(T, N)` float tensor matches expected dimensions from synthetic Raw
- [ ] **TEST-04**: Shape validation for `stc_to_roi_timeseries` — output `(T, N)` float tensor matches expected dimensions (mocked `extract_label_time_course`)
- [ ] **TEST-05**: Channel picks subsetting — loaders correctly subset channels by name list and by type string; output shape matches picks, not full channel count
- [ ] **TEST-06**: Bad channel annotation handling — output shape excludes channels marked as `info['bads']` when using default picks (critical pitfall P3)
- [ ] **TEST-07**: CSD Hermitian symmetry — `csd[f,i,j] == conj(csd[f,j,i])` for all frequency bins
- [ ] **TEST-08**: CSD non-negative auto-spectra diagonal — `csd[f,i,i].real >= 0` for all frequency bins and channels
- [ ] **TEST-09**: CSD sine-injection round-trip — inject 10 Hz sine into synthetic Epochs, verify CSD peak at 10 Hz bin (within 1 bin tolerance)
- [ ] **TEST-10**: `_require_mne()` raises ImportError with install instructions when MNE not installed
- [ ] **TEST-11**: `epochs_to_csd` raises ValueError for invalid `method` argument
- [ ] **TEST-12**: `pytest.importorskip("mne")` at module level — test file skips entirely when MNE absent
- [ ] **TEST-13**: `@pytest.mark.mne` marker registered in pyproject.toml for `pytest -m "not mne"` exclusion

### BIDS Loader Tests

- [ ] **BIDS-01**: `load_bids_raw` returns valid `mne.io.BaseRaw` from synthetic BIDS dataset written via `write_raw_bids` to `tmp_path`
- [ ] **BIDS-02**: `load_bids_epochs` returns valid `mne.Epochs` from synthetic BIDS dataset
- [ ] **BIDS-03**: BIDS annotation edge case — handle `BAD_ACQ_SKIP` spans and non-trivial annotations without error

### Pipeline Scripts

- [ ] **PIPE-01**: Spectral DCM demo script — end-to-end: synthetic MNE Epochs → `epochs_to_csd` → SpectralDCMModel → SVI → posterior A matrix, with preprocessing guidance as comments
- [ ] **PIPE-02**: Task DCM demo script — end-to-end: synthetic MNE Epochs → `epochs_to_timeseries` → TaskDCMModel → SVI → posterior A + B matrices, with preprocessing guidance as comments

---

## Deferred to Future Milestone

- Documentation: IO quickstart doc, Sphinx build setup, docstring examples, DCM variant guide
- Input validation warnings in loaders (sfreq, highpass, channel count, reference, epoch count)
- Source-space integration test with real SourceSpaces + Labels
- End-to-end SVI recovery tests through MNE IO path (>3 min, cluster work)
- Full hosted API reference site
- BIDS round-trip integration test (BIDS → CSD → model)

## Out of Scope

- Wrapping MNE preprocessing (filtering, ICA, artifact rejection) — MNE owns this
- Building a parallel BIDS parser — mne-bids already solves this
- Downloading MNE sample data (1.9 GB) in tests — synthetic fixtures only
- Auto-detecting DCM variant from data properties — researcher's scientific judgment

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TEST-01 | — | Pending |
| TEST-02 | — | Pending |
| TEST-03 | — | Pending |
| TEST-04 | — | Pending |
| TEST-05 | — | Pending |
| TEST-06 | — | Pending |
| TEST-07 | — | Pending |
| TEST-08 | — | Pending |
| TEST-09 | — | Pending |
| TEST-10 | — | Pending |
| TEST-11 | — | Pending |
| TEST-12 | — | Pending |
| TEST-13 | — | Pending |
| BIDS-01 | — | Pending |
| BIDS-02 | — | Pending |
| BIDS-03 | — | Pending |
| PIPE-01 | — | Pending |
| PIPE-02 | — | Pending |

**Coverage:**
- v0.5.0 requirements: 18 total
- Mapped to phases: 0/18 (roadmap pending)
- Unmapped: 18

---
*Requirements defined: 2026-05-21*
*Last updated: 2026-05-21*
