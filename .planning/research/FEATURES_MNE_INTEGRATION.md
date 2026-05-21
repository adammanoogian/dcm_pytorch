# Feature Landscape: MNE-Python Integration Testing, Pipelines, and Documentation

**Domain:** MNE-Python integration for a neuroimaging DCM framework (Pyro-DCM)
**Researched:** 2026-05-21
**Overall confidence:** MEDIUM-HIGH (patterns well-established in MNE ecosystem)

## Existing Features (Already Built)

For reference -- these IO loaders exist and are the foundation for all features below:

| Loader | Input | Output | DCM Variant |
|--------|-------|--------|-------------|
| `epochs_to_csd` | `mne.Epochs` | `(F,N,N)` complex CSD tensor | Spectral DCM |
| `epochs_to_timeseries` | `mne.Epochs` | `(T,N)` float time series | Task DCM |
| `raw_to_timeseries` | `mne.io.BaseRaw` | `(T,N)` float time series | Task DCM |
| `stc_to_roi_timeseries` | `mne.SourceEstimate` + labels | `(T,N)` float ROI series | Task DCM |
| `load_bids_raw` | `BIDSPath` or dict | `mne.io.BaseRaw` | N/A (bridge) |
| `load_bids_epochs` | `BIDSPath` or dict | `mne.Epochs` | N/A (bridge) |

---

## Table Stakes

Features users expect. Missing any of these means the integration feels unfinished and untrustworthy.

### Testing: Unit Tests for Each Loader

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Shape validation tests for all 4 MNE loaders | Every MNE ecosystem package validates output shapes. Users must trust `(T,N)` and `(F,N,N)` contracts. | Low | Synthetic MNE objects via `create_info`, `EpochsArray`, `RawArray` |
| dtype validation tests | Loaders promise `torch.float64` / `torch.complex128` by default. Must verify these contracts hold and that custom dtypes propagate. | Low | Same synthetic fixtures |
| Channel picks tests | MNE's `picks` parameter is central to user workflows (pick by name, by type string, by list). Loaders must correctly subset channels. | Medium | Synthetic data with mixed channel types (EEG + EOG) |
| Channel name / metadata preservation | Output dicts include `ch_names`, `sfreq`, `times`. Must verify these match input MNE objects exactly. | Low | Same fixtures |
| `pytest.importorskip("mne")` gating | MNE is an optional dependency (`pyro-dcm[mne]`). Tests must skip gracefully when MNE is not installed, not crash the test suite. | Low | pytest infrastructure only |
| `pytest.mark.mne` marker | Users running `pytest -m "not mne"` should be able to exclude MNE tests entirely. Standard pattern for optional-dependency test isolation. | Low | `pyproject.toml` marker registration |
| Error handling tests | `_require_mne()` ImportError, invalid `method` arg to `epochs_to_csd`, mismatched picks. Users expect clear error messages. | Low | Standard pytest assertions |

### Testing: Synthetic Data Fixtures in conftest.py

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| `mne_info` fixture | Session-scoped fixture creating `mne.create_info(ch_names, sfreq, ch_types)`. Reusable across all MNE tests. | Low | MNE import |
| `synthetic_epochs` fixture | `mne.EpochsArray` with known shape `(n_epochs, n_channels, n_samples)`, proper events array, and known frequency content for CSD validation. | Medium | `mne_info` fixture |
| `synthetic_raw` fixture | `mne.io.RawArray` with continuous data, known sfreq. | Low | `mne_info` fixture |
| Small fixture sizes | MNE ecosystem convention: test data should be minimal (e.g., 3-5 channels, 100-256 samples, 5 epochs). Large fixtures slow CI and are an anti-pattern. mne-bids uses a `tiny_bids` directory; mne-testing-data is separate from sample data precisely to keep test fixtures small. | Low | Design choice |

### Testing: CSD Round-Trip Validation

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Known-frequency CSD test | Inject a sine wave at a known frequency (e.g., 10 Hz) into synthetic epochs. Run `epochs_to_csd`. Verify CSD peak is at the expected frequency bin. This is the standard validation pattern for spectral transforms. | Medium | `synthetic_epochs` with controlled spectral content |
| Hermitian symmetry check | CSD matrices must be Hermitian (`csd[f,i,j] == conj(csd[f,j,i])`). Physics constraint that must be verified. | Low | Any CSD output |
| Positive semi-definite diagonal | Diagonal of CSD (`csd[f,i,i]`) must be real and non-negative (auto-spectral density). | Low | Any CSD output |

### Documentation: API Docstrings

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Complete NumPy-style docstrings on all public functions | Already done for the loaders (verified in source). This is table stakes for any scientific Python package. MNE, nilearn, scikit-learn all follow this pattern. | Done | N/A |
| Docstring examples (>>> blocks) | `load_bids_raw` has one already. The MNE loaders (`epochs_to_csd`, etc.) should also have short doctest-style examples showing minimal usage. | Low | Already-written functions |

---

## Differentiators

Features that set Pyro-DCM's MNE integration apart from a basic utility package. Not expected, but valued by researchers adopting the tool.

### Testing: Integration / End-to-End Recovery Tests

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| MNE-to-DCM round-trip recovery test | Load synthetic MNE data with **known ground-truth connectivity** through the full pipeline: `epochs_to_csd` -> `spectral_dcm_model` -> SVI -> posterior A. Verify A recovery. This proves the IO layer does not corrupt information. No competitor in the Python DCM space does this. | High | Existing recovery harness + synthetic MNE fixtures. Route to cluster (>3 min). |
| Task DCM MNE round-trip | Same pattern for task path: `epochs_to_timeseries` -> `task_dcm_model` -> SVI -> posterior A. | High | Existing task recovery tests. Route to cluster. |
| Source-space pipeline test | `stc_to_roi_timeseries` with synthetic SourceEstimate -> task DCM. Validates the least-common but scientifically important path. | High | Requires synthetic SourceEstimate + labels + SourceSpaces (complex to set up). |
| BIDS round-trip test | `load_bids_epochs` -> `epochs_to_csd` -> tensor dict. Validates the BIDS convenience path. Requires a tiny BIDS dataset fixture (like mne-bids's `tiny_bids`). | Medium | Tiny BIDS fixture or `pytest.importorskip("mne_bids")` |

### Pipeline Scripts: End-to-End Examples

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| `examples/01_mne_spectral_dcm.py` -- MNE Epochs to spectral DCM | Self-contained script showing: create synthetic MNE Epochs -> compute CSD via `epochs_to_csd` -> define spectral DCM model -> run SVI -> extract posterior -> plot connectivity matrix. This is what researchers look for first. MNE ecosystem packages (mne-connectivity, mne-nirs) all provide this pattern via sphinx-gallery or standalone scripts. | Medium | Existing spectral DCM pipeline |
| `examples/02_mne_task_dcm.py` -- MNE Epochs to task DCM | Same pattern for task DCM: create/load epochs -> extract timeseries via `epochs_to_timeseries` -> define model -> run SVI -> plot. | Medium | Existing task DCM pipeline |
| `examples/03_bids_to_dcm.py` -- BIDS dataset to DCM | Shows BIDS loading -> epoching -> DCM analysis. Differentiates from SPM's pipeline by being Python-native and BIDS-first. | Medium | Tiny BIDS example dataset |
| Pipeline scripts follow `{stage}_{verb}_{noun}.py` naming | Project convention from CLAUDE.md. | Low | Convention only |

### Documentation: Tutorials

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| "MNE-Python to DCM" tutorial (narrative + code) | Step-by-step guide: "I have MNE Epochs, how do I do DCM?" This bridges the MNE-familiar / DCM-unfamiliar user gap. MNE's documentation follows a four-section philosophy: Tutorials (narrative), Examples (code-focused), API Reference, Glossary. This tutorial fills the Tutorials slot. | Medium | Working pipeline code |
| "BIDS to DCM" tutorial | For fMRI researchers with BIDS datasets. Shows the BIDS -> MNE -> Pyro-DCM path. | Medium | BIDS pipeline code |
| "Choosing your DCM variant" guide for MNE users | Cross-reference with existing `guide_selection.md` but framed for MNE users: "I have resting-state EEG -> use spectral DCM"; "I have task-evoked MEG -> use task DCM"; "I have whole-brain fMRI -> use rDCM". | Low | Existing guide + IO loader knowledge |

### Documentation: API Reference Page

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| Dedicated `io` module API reference section in docs | Separate page or section listing all IO functions with rendered docstrings. MNE, nilearn, and scikit-learn all have grouped API reference pages by module. | Low-Medium | Sphinx/numpydoc setup or equivalent |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Wrapping MNE preprocessing (filtering, ICA, artifact rejection) | Scope creep. MNE already handles preprocessing excellently. Pyro-DCM is a DCM inference engine, not a preprocessing pipeline. Users should preprocess in MNE, then hand off to Pyro-DCM loaders. | Document the expected preprocessing steps as prerequisites in tutorials. Link to MNE's preprocessing tutorials. |
| Building a custom BIDS parser | mne-bids already solves BIDS loading comprehensively. Building a parallel implementation would be redundant, unmaintainable, and always inferior. | Keep the thin `load_bids_raw` / `load_bids_epochs` wrappers. They add value by providing DCM-relevant defaults (epoch window, baseline), not by reimplementing BIDS parsing. |
| Downloading MNE sample data in tests | MNE sample data is ~2 GB. Downloading it in CI or test runs is extremely slow, fragile, and violates the "small fixtures" principle used by all MNE ecosystem packages (mne-bids uses `tiny_bids`, mne-testing-data is separate from sample data). | Use `mne.create_info` + `EpochsArray` / `RawArray` with synthetic NumPy data. Keep fixtures under 1 MB. |
| Supporting file-format-specific loading | The loaders accept MNE objects (`Epochs`, `Raw`), not files. MNE handles format-specific reading (FIF, EDF, BDF, MEF, etc.). Adding format-specific code would duplicate MNE and break when MNE updates format support. | Keep the abstraction at the MNE object level. Document: "Use `mne.io.read_raw_*` for your format, then pass to `raw_to_timeseries`." |
| Auto-detecting DCM variant from data | Tempting to auto-select task vs. spectral DCM based on data properties. But this conflates scientific judgment with tooling. The researcher must choose their analysis approach. | Provide the "Choosing your DCM variant" decision guide instead. Let users make the scientific choice explicitly. |
| Sphinx-gallery documentation build | Full sphinx-gallery infrastructure is heavy engineering for a research package. The project already has markdown-based docs in `docs/`. Switching to sphinx-gallery would be a large infrastructure change for marginal benefit at this stage. | Use markdown tutorials with embedded code blocks. Graduate to sphinx-gallery only if/when the project reaches community-release stage. |
| GUI or interactive widgets | Some MNE ecosystem packages provide interactive browsers (mne-qt-browser). This is irrelevant for a Bayesian inference engine. | Provide matplotlib-based diagnostic plots (already exists in `utils/diagnostics.py`). |
| Real-time / streaming data support | mne-lsl handles real-time streaming. DCM is inherently an offline analysis method operating on completed recordings. | Explicitly document that Pyro-DCM operates on recorded (not streaming) data. |

---

## Feature Dependencies

```
pytest.importorskip("mne") gating  +  pytest.mark.mne marker
    |
    v
mne_info fixture (create_info)
    |
    +---> synthetic_epochs fixture (EpochsArray)
    |        |
    |        +---> epochs_to_csd shape/dtype tests
    |        +---> epochs_to_csd CSD validation (known-freq, Hermitian, PSD)
    |        +---> epochs_to_timeseries shape/dtype tests
    |        +---> epochs_to_csd/timeseries picks tests
    |        +---> [DIFFERENTIATOR] spectral DCM round-trip recovery
    |
    +---> synthetic_raw fixture (RawArray)
    |        |
    |        +---> raw_to_timeseries shape/dtype/picks tests
    |        +---> [DIFFERENTIATOR] task DCM round-trip recovery
    |
    +---> [COMPLEX] synthetic_stc + labels + src fixtures
             |
             +---> stc_to_roi_timeseries tests
             +---> [DIFFERENTIATOR] source-space pipeline test

load_bids_epochs/load_bids_raw tests
    |
    +---> pytest.importorskip("mne_bids") gating (separate from mne)
    +---> tiny BIDS fixture or mock

Pipeline examples depend on:
    - All table-stakes tests passing (proves loaders work)
    - Existing DCM inference pipeline (already built)

Documentation depends on:
    - Pipeline examples (provide code to narrate around)
    - Tests passing (proves correctness claims in docs)
```

---

## MVP Recommendation

For the immediate milestone, prioritize in this order:

### Phase 1: Test Infrastructure (must come first)

1. **pytest marker + importorskip gating** -- 1 hour. Enables all other MNE tests. Add `"mne: marks tests requiring MNE-Python"` to `pyproject.toml` markers.
2. **Synthetic MNE fixtures** (info, epochs, raw) in a new `tests/conftest_mne.py` or appended to `tests/conftest.py` with importorskip guard -- 2 hours. Foundation for all tests.
3. **Shape/dtype/picks tests for all 4 loaders** -- 3-4 hours. Cover: correct output shapes, default and custom dtypes, channel subsetting by name and by type string, metadata keys present.
4. **Error handling tests** -- 1 hour. Invalid method arg, missing MNE import, empty picks.
5. **CSD validation tests** (known-frequency peak, Hermitian symmetry, non-negative diagonal) -- 2 hours.

### Phase 2: Pipeline Examples (builds on passing tests)

6. **Spectral DCM end-to-end example** -- 3 hours. Highest-value single deliverable for MNE users.
7. **Task DCM end-to-end example** -- 2 hours (similar pattern, adapts from existing quickstart).

### Phase 3: Documentation (builds on working examples)

8. **"MNE to DCM" tutorial** -- 2-3 hours. Narrative wrapper around examples.
9. **DCM variant decision guide for MNE users** -- 1 hour.
10. **Docstring examples on loaders** -- 1 hour.

### Defer to Post-Milestone

- **Source-space pipeline test**: Complex fixture setup (need synthetic SourceEstimate, labels, SourceSpaces), narrow audience. Do when source-space DCM becomes a priority use case.
- **BIDS round-trip test**: Needs tiny BIDS fixture infrastructure. Valuable but lower priority than core loader tests.
- **Full API reference page**: Needs Sphinx infrastructure decision. Current markdown docs adequate for research stage.
- **End-to-end recovery tests through MNE path**: Scientifically valuable but computationally expensive (SVI fitting). Route to cluster per compute-routing rule. Consider as a `@pytest.mark.slow` integration test.

---

## Sources

### HIGH Confidence (official documentation, verified)
- [MNE data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html) -- `create_info`, `RawArray`, `EpochsArray`, `EvokedArray` patterns
- [MNE documentation overview](https://mne.tools/stable/documentation/index.html) -- Four-section structure: tutorials, examples, API reference, glossary
- [MNE datasets overview](https://mne.tools/stable/documentation/datasets.html) -- Sample data and testing data organization
- [MNE pick_types](https://mne.tools/stable/generated/mne.pick_types.html) -- Channel selection including bad channel handling
- [MNE handling bad channels](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html) -- Edge cases for picks
- [mne-testing-data](https://github.com/mne-tools/mne-testing-data) -- Separate repo for test data, minimal fixtures
- [pytest skipping](https://docs.pytest.org/en/stable/how-to/skipping.html) -- `importorskip` and `skipif` patterns
- [MNE CSD computation example](https://mne.tools/stable/auto_examples/time_frequency/compute_csd.html) -- CSD methods: multitaper, morlet, fourier

### MEDIUM Confidence (ecosystem packages, community patterns)
- [mne-connectivity examples](https://mne.tools/mne-connectivity/stable/auto_examples/index.html) -- Example gallery organization by functional category
- [mne-bids tests](https://github.com/mne-tools/mne-bids/tree/main/mne_bids/tests) -- `tiny_bids` fixture pattern, test file organization by function
- [mne-connectivity tests](https://github.com/mne-tools/mne-connectivity/tree/main/mne_connectivity/tests) -- Test organization: test_connectivity, test_effective, test_envelope, test_utils
- [nilearn documentation](https://nilearn.github.io/) -- Sphinx-gallery pattern, tutorial/example/API structure
- [numpydoc](https://numpydoc.readthedocs.io/) -- NumPy docstring Sphinx extension

### Project-Internal (verified by reading source)
- `src/pyro_dcm/io/mne_loader.py` -- All 4 loader implementations with full docstrings
- `src/pyro_dcm/io/bids_loader.py` -- BIDS loader implementations
- `src/pyro_dcm/io/__init__.py` -- Public API with `__all__`, conditional BIDS import
- `tests/conftest.py` -- Existing fixtures (hemo_params, test_A, test_C, device, dtype)
- `pyproject.toml` -- Optional `[mne]` dependency group, existing pytest markers (slow, spm, tapas)
- `docs/02_pipeline_guide/quickstart.md` -- Existing tutorial pattern (simulate -> infer -> plot)
- `docs/02_pipeline_guide/guide_selection.md` -- Existing DCM variant decision guide
