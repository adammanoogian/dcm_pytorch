# Project Research Summary

**Project:** Pyro-DCM -- MNE-Python Integration (v0.5.0)
**Domain:** Scientific Python IO testing, pipeline scripting, and API documentation for M/EEG DCM
**Researched:** 2026-05-21
**Confidence:** HIGH (all four research areas verified against official MNE, pytest, and Sphinx sources)

## Executive Summary

Pyro-DCM v0.5.0 is a well-scoped integration milestone: the IO code already exists, the
DCM models already exist, and the task is to prove correctness via tests, demonstrate usage
via pipeline scripts, and explain the handoff to users via documentation. The research is
clear that this should proceed in strict dependency order -- fixtures before tests, tests
before scripts, scripts before narrative docs -- and that no new infrastructure decisions
are consequential enough to delay execution.

The recommended test strategy is entirely synthetic: MNE’s own `RawArray`, `EpochsArray`,
and `SourceEstimate` constructors suffice for every test fixture, requiring no data
downloads, no FreeSurfer directories, and no binary files in git. The single non-trivial
architectural decision is to mock `mne.extract_label_time_course` for the source-space
path rather than constructing synthetic SourceSpaces, which avoids a complex API surface
while still exercising the packaging logic that Pyro-DCM actually owns. For the
documentation layer, Sphinx with MyST-parser and the furo theme is the right call: it
consumes the existing Markdown docs without rewriting them and matches the scientific
Python ecosystem standard.

The three pitfalls that could produce silent scientific errors -- CSD frequency grid
mismatch between MNE and SPM conventions, CSD normalization inconsistency across
computation paths, and channel selection inconsistency between `Epochs` and `Evoked`
objects -- all have concrete, testable detection criteria. These must be addressed as
explicit test cases in Phase 1, not deferred. Every other pitfall is either minor (verbose
output, events array shape) or already mitigated by the existing IO code design.

---

## Key Findings

### Recommended Stack

The existing stack (`[dev]` + `[mne]` extras) is sufficient for tests and pipeline
scripts. Two new extras are needed:

- `pytest-cov` added to `[dev]`: coverage reporting to verify IO code is actually
  exercised. No alternatives warranted.
- New `[docs]` optional extra: `sphinx>=8.0`, `furo>=2024.1`, `myst-parser>=4.0`,
  `sphinx-autodoc-typehints>=3.10`. This combination consumes existing Markdown docs,
  renders NumPy docstrings natively via napoleon, cross-links to MNE/PyTorch/Pyro docs
  via intersphinx, and renders `from __future__ import annotations` type hints correctly.

Nothing else new is needed. `argparse` and `logging` are stdlib. MNE test fixtures are
built from bundled MNE constructors. Notebook infrastructure (`myst-nb`) is deferred
until notebooks exist.

One `pyproject.toml` change is also needed: register an `mne` pytest marker alongside
the existing `slow`, `spm`, and `tapas` markers.

**Version floors verified on PyPI as of 2026-05-21:**
- `sphinx>=8.0` (current: 9.1.0)
- `furo>=2024.1` (current: 2025.12.19)
- `myst-parser>=4.0` (current: 4.0.1)
- `sphinx-autodoc-typehints>=3.10` (current: 3.10.2)

### Expected Features

**Must have -- tests:**
- Shape validation for all 4 MNE loaders (`epochs_to_csd`, `epochs_to_timeseries`,
  `raw_to_timeseries`, `stc_to_roi_timeseries`). Users must trust `(T,N)` and `(F,N,N)`
  output contracts before relying on them for scientific analysis.
- dtype validation: loaders promise `torch.float64` / `torch.complex128` by default;
  these contracts must be tested and custom dtypes must propagate correctly.
- Channel picks tests: the `picks` parameter is central to user workflows. DCM requires
  3-8 channels, not 64+; correct subsetting must be verified by name and by type string.
- `pytest.importorskip("mne")` gating throughout: MNE is optional; the suite must run
  cleanly without it installed.
- CSD mathematical property tests: Hermitian symmetry and non-negative auto-spectra
  diagonal are physics constraints that the IO layer is responsible for preserving.
- Error handling: `_require_mne()` ImportError, invalid `method` arg, mismatched picks.

**Must have -- pipeline and docs:**
- At least one end-to-end demo script (spectral DCM path: epochs -> CSD -> model ->
  posterior).
- IO quickstart doc in `docs/02_pipeline_guide/` explaining all four loading paths and
  the output dict format.
- Docstring `>>>` examples on all loader functions.

**Should have -- differentiators:**
- CSD round-trip validation: inject a sine at known frequency, verify peak location in
  returned `freqs` vector. Standard spectral transform validation; proves the IO layer
  does not corrupt frequency information.
- Task DCM end-to-end demo script (`epochs_to_timeseries` path).
- "Choosing your DCM variant" cross-reference framed for MNE users.

**Defer to post-milestone:**
- Source-space integration test with full `SourceEstimate + SourceSpaces + Labels`:
  complex fixture setup, narrow audience. Mock-based unit test is sufficient for v0.5.0.
- BIDS round-trip integration test.
- Full hosted Sphinx API reference site.
- End-to-end SVI recovery tests through the MNE IO path: scientifically important but
  >3 minutes on CPU; route to cluster as `@pytest.mark.slow` work.

**Anti-features -- explicitly out of scope:**
- Wrapping MNE preprocessing (filtering, ICA, artifact rejection) -- scope creep.
- Building a parallel BIDS parser -- mne-bids already owns this.
- Downloading MNE sample data (1.9 GB) in tests.
- Auto-detecting DCM variant from data properties.

### Architecture Approach

The architecture is additive and flat. All existing structure is preserved; new components
are appended or created alongside. The critical data flow is:

```
MNE object (numpy internals)
    -> pyro_dcm.io loader
    -> dict[str, torch.Tensor]
    -> downstream DCM model
```

Tests validate this contract. Pipeline scripts demonstrate it. Docs explain it.

**New and modified components:**

| Component | Location | Action |
|-----------|----------|--------|
| MNE synthetic fixtures | `tests/conftest.py` | Append (5-7 fixtures) |
| MNE loader unit tests | `tests/test_mne_loader.py` | New file |
| BIDS loader unit tests | `tests/test_bids_loader.py` | New file |
| Demo pipeline script | `scripts/demo_mne_to_dcm.py` | New file |
| IO quickstart doc | `docs/02_pipeline_guide/mne_io_quickstart.md` | New file |
| pytest marker + extras | `pyproject.toml` | Modify |
| Cross-reference link | `docs/02_pipeline_guide/quickstart.md` | Modify |

The `stc_to_roi_timeseries` fixture uses `unittest.mock.patch` on
`mne.extract_label_time_course` -- the correct boundary because the function under test is
a 6-line packaging wrapper around that single MNE call. Constructing synthetic
SourceSpaces is unnecessary complexity for this milestone.

BIDS tests use `write_raw_bids` to a `tmp_path` -- a real BIDS round-trip from synthetic
data, no committed binary fixtures.

---

### Critical Pitfalls

1. **CSD frequency grid mismatch (CRITICAL -- P1)** -- `csd_multitaper` returns
   FFT-locked frequencies regardless of the `fmin`/`fmax` arguments. The returned
   `freqs` vector does not match user-requested frequencies. Test mitigation:
   inject a sine at a known frequency and assert its peak appears in the returned
   `freqs` tensor within one frequency bin. `csd_morlet` uses exact user-specified
   frequencies but is slower; both paths must be exercised in tests.

2. **CSD normalization mismatch (CRITICAL -- P2)** -- scipy Welch
   (scaling="density"), MNE multitaper, and SPM12 (C=1/256) all produce
   different absolute magnitudes for the same data. The IO layer must document
   which normalization convention it uses and tests must assert that Hermitian
   symmetry holds regardless of convention. Do not assert absolute magnitude
   values in unit tests -- assert structural properties only.

3. **Channel picks inconsistency (CRITICAL -- P3)** -- `Epochs.get_data(picks="eeg")`
   silently excludes channels marked as bad; `Evoked.get_data()` includes them.
   The IO loaders use Epochs; tests must include a fixture with bad-channel
   annotations and assert output shape matches explicitly-passed picks, not the
   full channel count. See MNE issue #12577.

4. **dtype / MPS incompatibility (MODERATE -- P4)** -- MNE uses float64 throughout;
   spectral DCM requires float64/complex128. Apple MPS and some CUDA configs do
   not support float64. Document this in the IO quickstart. Do not silently
   downcast; let the user choose device placement after loading.

5. **Optional dependency test isolation (MODERATE -- P6)** -- `pytest.importorskip`
   must appear at module level in every test file that imports MNE, not inside
   individual test functions. Module-level placement skips the entire file when
   MNE is absent; function-level placement causes 30+ individual SKIPs and an
   unreadable test output.

6. **BIDS annotation edge cases (MODERATE -- P7)** -- `load_bids_epochs` must handle
   empty events.tsv, zero-duration annotations, and BAD_ACQ_SKIP spans. The
   synthetic BIDS fixture must exercise at least one non-trivial annotation to
   catch event-parsing regressions.

7. **MNE API deprecation (MODERATE -- P8)** -- `pick_types()` and `pick_channels()`
   are deprecated since MNE 1.0. Use `inst.pick()` exclusively. Ruff cannot catch
   this; must be enforced in code review.

8. **Silent zeros from empty label intersection (MODERATE -- P9)** --
   `stc_to_roi_timeseries` returns a zero row if a label has no vertices in the
   source estimate. No error is raised. The test for this loader must explicitly
   assert that non-zero output is produced when the fixture data is non-zero.

---

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: Test Infrastructure (no pre-research needed)**

Deliver: `tests/conftest.py` fixtures + `test_mne_loader.py` +
`test_bids_loader.py` + pyproject.toml updates.

Rationale: Tests gate everything else. The three CRITICAL pitfalls (P1, P2, P3)
must be encoded as explicit test cases before any pipeline script or documentation
is written -- otherwise the scripts demonstrate potentially broken behavior.

Features from FEATURES.md delivered:
- Shape/dtype/picks/importorskip tests for all 6 loaders
- CSD Hermitian symmetry and non-negative diagonal property tests
- Error handling tests (`_require_mne()`, invalid method, mismatched picks)
- CSD round-trip sine-injection validation (differentiator)

Pitfalls to avoid:
- P6: module-level importorskip placement
- P5: synthetic fixtures only, no data downloads
- P13: events array must be shape (n_epochs, 3) exactly

Research flag: NONE. Patterns are standard pytest + MNE. Well-documented.

---

**Phase 2: Pipeline Scripts (no pre-research needed)**

Deliver: `scripts/demo_mne_to_dcm.py` (spectral DCM path) + optional
`scripts/demo_task_dcm_from_epochs.py` (task DCM path).

Rationale: Scripts depend on passing tests. Once tests confirm the IO contracts
are correct, scripts can demonstrate real usage. Scripts are the primary artifact
users will copy-paste.

Features from FEATURES.md delivered:
- Spectral DCM end-to-end demo (epochs -> CSD -> SpectralDCMModel -> posterior)
- Task DCM end-to-end demo (should-have)

Pitfalls to avoid:
- P4: document float64 / MPS incompatibility as an explicit comment in the script
- P10: script must include comments explaining preprocessing prerequisites
  (filtering, bad channel rejection) that users must perform before calling loaders

Research flag: NONE. Follow existing demo_bilinear_consumer.py pattern.

---

**Phase 3: Documentation (no pre-research needed)**

Deliver: `[docs]` extra in pyproject.toml + `docs/02_pipeline_guide/mne_io_quickstart.md`
+ docstring `>>>` examples on all loader functions + cross-reference link from
`quickstart.md` + Sphinx build configuration.

Rationale: Documentation is meaningful only after the code is tested and scripted.
The Sphinx build is a new build artifact; configure it last to avoid blocking earlier
phases on toolchain setup.

Features from FEATURES.md delivered:
- IO quickstart doc with all four loading paths and output dict format
- Docstring examples on all loaders
- Choosing DCM variant cross-reference framed for MNE users

Pitfalls to avoid:
- P10: quickstart must prominently note preprocessing prerequisites (filtering,
  artifact rejection, referencing) that MNE users are expected to have completed
  before calling Pyro-DCM loaders
- P4: dtype / device section required in quickstart

Research flag: NONE. Sphinx + MyST + furo is standard scientific Python stack.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified on PyPI 2026-05-21. Sphinx+napoleon+furo is the scientific Python standard. `pytest.importorskip` is an official stable pytest API. |
| Features | HIGH | IO loaders already exist; test contracts derived directly from the existing function signatures and return type annotations. |
| Architecture | HIGH | Additive changes only. No structural risk. One MEDIUM item: `extract_label_time_course` mock approach needs verification that the mock call signature matches MNE 1.6+. |
| Pitfalls | HIGH | P1/P2/P3 verified against MNE source and issue tracker. P4/P6/P7/P8 verified against MNE docs. |

**Overall: HIGH**

---

## Gaps to Address

1. **`extract_label_time_course` mock call signature** -- The mock approach for
   `stc_to_roi_timeseries` assumes a specific call signature. Verify against
   `mne.extract_label_time_course` in MNE 1.6+ before finalizing the test.
   Risk: LOW (function exists and is stable; only the mock setup needs care).

2. **CSD normalization convention** -- Research confirmed that MNE, scipy, and SPM12
   differ in normalization but did not resolve which convention the existing
   `epochs_to_csd` implementation uses. Inspect the implementation before writing
   the normalization-related docstring example.

3. **Sphinx build integration** -- A `docs/conf.py` and `docs/index.rst` (or
   `index.md`) need to be created for the Sphinx build. The research specified the
   toolchain but not the exact conf.py contents. Standard Sphinx+MyST config is
   well-documented and should not block Phase 3.

4. **`@pytest.mark.mne` vs `importorskip` dual gating** -- Research recommends both
   the marker and `importorskip`. The interaction (double-skip messages, marker
   with `-m "not mne"`) should be validated once during Phase 1 conftest setup.

---

## Sources

*Aggregated from all four research files.*

- MNE 1.12.1 documentation: EpochsArray, RawArray, SourceEstimate, create_info
- MNE tutorials: Creating data structures from scratch
- MNE-BIDS 0.18.0: write_raw_bids, make_dataset_description, BIDSPath
- pytest documentation: importorskip, skipping, markers
- pytest-cov 7.1.0 documentation
- Sphinx 9.1.0 documentation: napoleon, autodoc, intersphinx extensions
- sphinx-autodoc-typehints 3.10.2 (PyPI)
- furo 2025.12.19 (PyPI)
- myst-parser 4.0.1 (PyPI)
- MNE issue #12577: channel picks inconsistency between Epochs and Evoked
- MNE source: mne/time_frequency/csd.py (normalization behavior)
- SPM12 source: spm_csd_mtf.m (C=1/256 convention)
- scipy docs: signal.csd scaling parameter
