# Architecture Patterns: MNE-Python Integration (Tests, Pipelines, Docs)

**Domain:** MNE-Python IO integration into a Pyro-based DCM framework
**Researched:** 2026-05-21
**Scope:** How tests, end-to-end pipeline scripts, and documentation integrate with existing Pyro-DCM architecture

## Recommended Architecture

The MNE IO module (`src/pyro_dcm/io/`) is already implemented with 4 loader functions and 2 BIDS loaders. This milestone adds three integration layers: testing, pipeline scripts, and documentation. The architecture must respect existing project conventions observed in 45+ existing test files, 6 scripts, and 8 docs.

### Current Component Map

```
src/pyro_dcm/io/
  mne_loader.py       -- 4 functions: epochs_to_csd, epochs_to_timeseries,
                          raw_to_timeseries, stc_to_roi_timeseries
  bids_loader.py      -- 2 functions: load_bids_raw, load_bids_epochs
  __init__.py          -- re-exports mne_loader; try/except for bids_loader

tests/                 -- 45+ test files, flat structure, conftest.py with fixtures
scripts/               -- 6 scripts, demo/debug pattern
docs/02_pipeline_guide -- quickstart.md, guide_selection.md, consumer_bilinear_quickstart.md
```

### New Components Needed

| Component | Location | Purpose | Modifies Existing? |
|-----------|----------|---------|-------------------|
| MNE synthetic fixture factory | `tests/conftest.py` | pytest fixtures for synthetic MNE objects | Yes (append) |
| Unit tests for mne_loader | `tests/test_mne_loader.py` | Shape, dtype, key validation per function | No (new file) |
| Unit tests for bids_loader | `tests/test_bids_loader.py` | BIDS loading logic with mocked filesystem | No (new file) |
| Pipeline script: MNE-to-DCM | `scripts/demo_mne_to_dcm.py` | End-to-end: synthetic MNE data -> IO -> DCM fit | No (new file) |
| IO quickstart doc | `docs/02_pipeline_guide/mne_io_quickstart.md` | Usage guide for MNE loaders | No (new file) |
| pytest marker: `mne` | `pyproject.toml` | Skip MNE tests when MNE not installed | Yes (add marker) |

### Data Flow: MNE Objects -> DCM Tensors -> Models

```
MNE Epochs/Raw/STC (numpy internals)
        |
        v
  pyro_dcm.io.mne_loader  (4 loader functions)
        |
        v
  dict[str, torch.Tensor]  -- standardized output format
  {
    "csd": (F, N, N) complex128      -- for spectral_dcm_model
    "timeseries": (T, N) float64     -- for task_dcm_model
    "freqs": (F,) float64            -- frequency vector
    "times": (T,) float64            -- time vector
    "ch_names" / "roi_names": [str]  -- metadata
    "sfreq": float                   -- sampling frequency
  }
        |
        v
  Downstream DCM models consume tensors directly
  (spectral_dcm_model, task_dcm_model, rdcm_model)
```

The critical boundary: IO functions output `dict[str, torch.Tensor]`. Tests must validate this contract. Pipeline scripts demonstrate the full path. Docs explain how to connect MNE objects to DCM models.

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `tests/conftest.py` (MNE fixtures) | Generate synthetic MNE Epochs, Raw, STC, Labels, SourceSpaces | `test_mne_loader.py`, `test_bids_loader.py` |
| `tests/test_mne_loader.py` | Validate output dict structure, shapes, dtypes, edge cases | `pyro_dcm.io.mne_loader`, conftest fixtures |
| `tests/test_bids_loader.py` | Validate BIDS path construction, error handling | `pyro_dcm.io.bids_loader`, `tmp_path` fixture |
| `scripts/demo_mne_to_dcm.py` | Show complete MNE -> IO -> DCM workflow | `pyro_dcm.io`, `pyro_dcm.models`, `pyro_dcm.simulators` |
| `docs/02_pipeline_guide/mne_io_quickstart.md` | User-facing guide for IO module | References all IO functions |

## Patterns to Follow

### Pattern 1: Synthetic MNE Fixture Factory in conftest.py

**What:** Add MNE object fixtures to the existing `tests/conftest.py` using `mne.create_info`, `mne.io.RawArray`, `mne.EpochsArray`, and `mne.SourceEstimate` constructors. Guard all MNE fixtures behind `pytest.importorskip("mne")` so the test suite runs cleanly without MNE installed.

**Why:** The project already centralizes fixtures in `conftest.py` (see existing `hemo_params`, `test_A`, `test_C`, `device`, `dtype` fixtures). MNE fixtures follow the same pattern. Using synthetic data avoids external data dependencies and keeps tests fast (~100ms).

**Example:**
```python
# In tests/conftest.py, appended after existing fixtures

@pytest.fixture()
def mne_info_3ch():
    """3-channel EEG info for testing IO loaders."""
    mne = pytest.importorskip("mne")
    return mne.create_info(
        ch_names=["Fz", "Cz", "Pz"],
        sfreq=256.0,
        ch_types="eeg",
    )


@pytest.fixture()
def mne_raw_3ch(mne_info_3ch):
    """Synthetic Raw with 3 EEG channels, 10s duration."""
    mne = pytest.importorskip("mne")
    import numpy as np
    rng = np.random.RandomState(42)
    n_samples = int(256.0 * 10.0)
    data = rng.randn(3, n_samples) * 1e-6  # V scale for EEG
    return mne.io.RawArray(data, mne_info_3ch)


@pytest.fixture()
def mne_epochs_3ch(mne_info_3ch):
    """Synthetic Epochs: 5 epochs, 3 channels, 1s duration."""
    mne = pytest.importorskip("mne")
    import numpy as np
    rng = np.random.RandomState(42)
    n_epochs, n_channels, n_samples = 5, 3, 256
    data = rng.randn(n_epochs, n_channels, n_samples) * 1e-6
    events = np.column_stack([
        np.arange(0, n_epochs * n_samples, n_samples),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    return mne.EpochsArray(data, mne_info_3ch, events=events, tmin=0.0)
```

**Confidence:** HIGH -- `mne.io.RawArray`, `mne.EpochsArray`, `mne.create_info` are stable MNE APIs since v0.20+. Verified via MNE official tutorial.

### Pattern 2: SourceEstimate + Label Fixtures (Mock Strategy)

**What:** For `stc_to_roi_timeseries` testing, use `unittest.mock.patch` to mock `mne.extract_label_time_course` and test the dict-packaging logic rather than building complex synthetic SourceSpaces.

**Why:** Creating synthetic SourceSpaces is complex (requires hemisphere vertex arrays, surface triangulations). The function under test (`stc_to_roi_timeseries`) is thin -- it calls `mne.extract_label_time_course` and packages the result into the standard dict format. Testing the packaging logic via mock is simpler and more robust than building fake SourceSpaces.

**Alternative considered:** Using MNE's sample data with `pytest.mark.slow`. Rejected as primary strategy because it requires downloading ~2GB of sample data, but acceptable as one optional integration test.

**Recommended approach:**
- Primary: Mock `mne.extract_label_time_course` returning a known `(N, T)` numpy array, verify output dict has correct tensor shapes/dtypes/keys.
- Secondary (optional): One `@pytest.mark.slow` integration test using MNE sample data if available via `mne.datasets.sample.data_path()`.

**Confidence:** HIGH for the mock strategy. The function body is 6 lines of packaging logic around one MNE call.

### Pattern 3: Test File per Module (Flat tests/ Directory)

**What:** Create `tests/test_mne_loader.py` and `tests/test_bids_loader.py` as flat files in `tests/`, following the existing pattern.

**Why:** The project uses a flat test directory with no subdirectories. Each source module gets one test file. This is consistent with:
- `test_balloon.py` -> `forward_models/balloon_model.py`
- `test_spectral_simulator.py` -> `simulators/spectral_simulator.py`
- `test_task_dcm_recovery.py` -> `models/task_dcm_model.py`

**Test class organization within each file:**
```
test_mne_loader.py
  TestEpochsToCSD
    test_output_keys
    test_csd_shape
    test_csd_dtype_complex128
    test_freqs_shape_and_dtype
    test_ch_names_match_epochs
    test_sfreq_matches_epochs
    test_n_epochs_matches
    test_csd_hermitian              # mathematical property
    test_auto_spectra_positive      # diagonal CSD real > 0
  TestEpochsToCSDMethods
    test_multitaper_method
    test_morlet_method
    test_invalid_method_raises
  TestEpochsToCSDFrequencyGrid
    test_custom_n_freqs
    test_default_fmax_is_nyquist
    test_custom_fmin_fmax
  TestEpochsToTimeseries
    test_averaged_output_keys
    test_averaged_shape_2d
    test_non_averaged_shape_3d
    test_picks_selects_channels
    test_dtype_matches_request
  TestRawToTimeseries
    test_output_keys
    test_full_read_shape
    test_start_stop_trim
    test_picks_parameter
  TestStcToRoiTimeseries
    test_output_contract_with_mock
    test_roi_names_from_labels
    test_sfreq_from_tstep
  TestMneImportGuard
    test_require_mne_raises_without_mne

test_bids_loader.py
  TestLoadBidsRaw
    test_dict_to_bids_path_conversion
    test_returns_raw_type
    test_import_guard
  TestLoadBidsEpochs
    test_returns_epochs_type
    test_event_id_parameter
    test_import_guard
```

**Confidence:** HIGH -- directly observed from 45+ existing test files.

### Pattern 4: Pytest Marker for Optional MNE Dependency

**What:** Add an `mne` pytest marker in `pyproject.toml` and use `pytest.importorskip("mne")` in fixtures for actual skip behavior.

**Why:** The project already has `slow`, `spm`, and `tapas` markers. MNE tests follow the same optional-dependency pattern. `importorskip` handles automatic skipping; the marker enables `pytest -m mne` for targeted runs.

**Implementation in `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "spm: marks tests requiring MATLAB + SPM12",
    "tapas: marks tests requiring tapas rDCM MATLAB toolbox",
    "mne: marks tests requiring MNE-Python (auto-skipped if not installed)",
]
```

**Confidence:** HIGH -- follows existing marker pattern exactly.

### Pattern 5: Pipeline Script Following Project Convention

**What:** Create `scripts/demo_mne_to_dcm.py` following the existing script pattern (modeled on `scripts/demo_bilinear_consumer.py`).

**Why:** The `scripts/` directory contains demo and utility scripts. The naming uses descriptive verbs: `demo_bilinear_consumer.py`, `generate_training_data.py`. A demo script fits this convention.

**Script structure (following `demo_bilinear_consumer.py` template):**
```python
"""End-to-end demo: MNE data -> Pyro-DCM spectral analysis.

Demonstrates loading MNE Epochs, computing CSD via pyro_dcm.io,
and running spectral DCM inference. Uses synthetic data so no
external files needed. Runs in ~30s on CPU.
"""

def main() -> None:
    # 1. Create synthetic MNE Epochs (mne.EpochsArray)
    # 2. epochs_to_csd() -> dict with CSD tensor
    # 3. Run spectral_dcm_model SVI on the CSD
    # 4. Print recovery summary

if __name__ == "__main__":
    main()
```

**Key design decisions:**
- Self-contained: generates synthetic MNE data inline, no external files
- Fast: ~30-60s on CPU (matching `demo_bilinear_consumer.py` runtime target)
- Shows two paths: epochs -> CSD (spectral) and epochs -> timeseries (task)
- Uses `from pyro_dcm.io import epochs_to_csd, epochs_to_timeseries` (the public API)

**Confidence:** HIGH -- directly modeled on existing `demo_bilinear_consumer.py`.

### Pattern 6: Documentation in docs/02_pipeline_guide/

**What:** Create `docs/02_pipeline_guide/mne_io_quickstart.md` plus cross-reference from existing quickstart.

**Why:** The `02_pipeline_guide/` directory contains user-facing workflow guides. An MNE IO guide fills a gap: users with real M/EEG data need to know how to get it into DCM tensor format.

**Document structure:**
```
# MNE-Python IO Quickstart

## Installation
  pip install pyro-dcm[mne]

## Path 1: Epochs -> Spectral DCM (CSD)
  epochs_to_csd() usage with code example

## Path 2: Epochs -> Task DCM (Time Series)
  epochs_to_timeseries() usage with code example

## Path 3: Raw -> Task DCM
  raw_to_timeseries() usage with code example

## Path 4: Source-Space ROI Extraction
  stc_to_roi_timeseries() usage with code example

## BIDS Integration
  load_bids_raw() and load_bids_epochs() usage

## Complete Pipeline: MNE Epochs -> Spectral DCM Fit
  Full code block showing epochs -> CSD -> spectral_dcm_model -> posterior

## Output Format Reference
  Table of all output dict keys, shapes, dtypes per function
```

**Cross-reference:** Add a bullet point to the "Next Steps" section of `docs/02_pipeline_guide/quickstart.md` referencing the new guide.

**Confidence:** HIGH -- directly modeled on existing docs structure.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Subdirectory for MNE Tests

**What:** Creating `tests/io/test_mne_loader.py` with its own conftest.

**Why bad:** The project has 45+ test files in a flat `tests/` directory. Introducing subdirectories breaks the established pattern, requires `__init__.py` chains, and makes `pytest` discovery more complex.

**Instead:** `tests/test_mne_loader.py` and `tests/test_bids_loader.py` at the top level.

### Anti-Pattern 2: Real MNE Data Files as Test Fixtures

**What:** Shipping `.fif` files or downloading MNE sample data in tests.

**Why bad:** Adds large binary dependencies, requires network access, slows CI. The project uses synthetic fixture generation everywhere (see `benchmarks/fixtures.py`, `benchmarks/generate_fixtures.py`).

**Instead:** `mne.io.RawArray` and `mne.EpochsArray` with numpy random data. Creates valid MNE objects entirely in memory.

### Anti-Pattern 3: Testing MNE Internals

**What:** Testing whether `csd_multitaper` produces correct CSD values.

**Why bad:** That is MNE's responsibility. We test our adapter layer: correct keys, shapes, dtypes, metadata preservation.

**Instead:** Test the contract: "given valid MNE Epochs, does `epochs_to_csd` return a dict with correct structure?"

### Anti-Pattern 4: Pipeline Script Requiring External Data

**What:** A demo script that downloads real EEG datasets.

**Why bad:** Existing scripts (`demo_bilinear_consumer.py`) are self-contained with inline synthetic data generation.

**Instead:** Generate synthetic MNE objects within the script.

### Anti-Pattern 5: Putting Usage Docs in Docstrings Only

**What:** Relying on function docstrings as sole documentation.

**Why bad:** The project has a `docs/` directory with dedicated workflow guides alongside API docstrings. Both serve different purposes.

**Instead:** Comprehensive docstrings (already done) PLUS a workflow guide in `docs/02_pipeline_guide/`.

## Scalability Considerations

| Concern | Current (3 channels) | 64 channels | 256 channels |
|---------|----------------------|-------------|--------------|
| CSD computation time | <1s | ~5s | ~30s |
| CSD tensor memory | Negligible | ~500KB | ~8MB |
| Test fixture creation | <10ms | <10ms | <10ms |
| Test suite impact on CI | +2-5s per MNE test file | Same | Same |

The IO layer is not a scalability bottleneck. CSD computation scales as O(N^2 * F) in the MNE layer, but our adapter layer is O(1) dict packaging. Tests use 3 channels and complete in milliseconds.

## Build Order (Suggested Phase Structure)

Dependencies dictate this ordering:

```
Step 1: MNE fixtures in conftest.py
   |      (no dependencies, enables all downstream tests)
   v
Step 2: test_mne_loader.py  +  Step 3: test_bids_loader.py  (parallel)
   |      (depend on: conftest fixtures, respective source modules)
   v
Step 4: pytest marker in pyproject.toml
   |      (no code dependency, but should be in place before CI)
   v
Step 5: scripts/demo_mne_to_dcm.py
   |      (depends on: tested IO functions, models)
   v
Step 6: docs/02_pipeline_guide/mne_io_quickstart.md
   |      (depends on: working pipeline script for verified examples)
   v
Step 7: Update quickstart.md cross-references
```

**Rationale:**
- Fixtures first: all test files depend on them
- Steps 2 and 3 are parallelizable -- they test independent modules
- Pipeline script after tests pass: ensures the demonstrated workflow actually works
- Docs last: written from working, tested code

## Integration Points with Existing Architecture

### 1. conftest.py Extension (MODIFY)

The existing `tests/conftest.py` has 5 fixtures (67 lines). MNE fixtures add approximately 5-7 new fixtures (~60-80 lines). These are additive -- no modification to existing fixtures. All MNE fixtures use `pytest.importorskip("mne")` so existing tests are completely unaffected when MNE is not installed.

### 2. pyproject.toml Marker Addition (MODIFY)

Add one line to `[tool.pytest.ini_options].markers`. No changes to existing markers or configuration.

### 3. Package __init__.py (NO CHANGE)

The IO functions are intentionally NOT re-exported from `pyro_dcm.__init__` because MNE is optional. Users import from `pyro_dcm.io` directly. No changes needed.

### 4. config.py (NO CHANGE)

No new paths needed. The IO module converts in-memory MNE objects to in-memory tensors. The demo script may optionally use `FIGURES_DIR` for saving plots but does not require new path constants.

### 5. quickstart.md Cross-Reference (MODIFY)

Add one bullet point to the "Next Steps" section:
```markdown
- **MNE-Python integration:** Loading real M/EEG data? See
  [mne_io_quickstart.md](mne_io_quickstart.md) for converting MNE Epochs,
  Raw, and source estimates to DCM tensor format.
```

## Sources

- MNE: Creating data structures from scratch -- RawArray, EpochsArray constructors (HIGH confidence)
  https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html
- MNE: mne.io.RawArray API -- constructor signature (HIGH confidence)
  https://mne.tools/stable/generated/mne.io.RawArray.html
- MNE: mne.SourceEstimate API -- constructor: data, vertices, tmin, tstep (HIGH confidence)
  https://mne.tools/stable/generated/mne.SourceEstimate.html
- MNE: extract_label_time_course -- function signature, mode parameter (HIGH confidence)
  https://mne.tools/stable/generated/mne.extract_label_time_course.html
- Existing project files directly examined (HIGH confidence):
  - `tests/conftest.py` -- 5 existing fixtures, 67 lines
  - `tests/test_spectral_simulator.py` -- class-per-concern test organization
  - `tests/test_task_dcm_recovery.py` -- recovery test pattern with CI-fast + slow tiers
  - `scripts/demo_bilinear_consumer.py` -- demo script convention
  - `docs/02_pipeline_guide/quickstart.md` -- doc structure and cross-references
  - `benchmarks/config.py` -- dataclass config pattern
  - `benchmarks/fixtures.py` -- fixture loading pattern
  - `pyproject.toml` -- existing markers, optional deps, ruff/mypy config
  - `src/pyro_dcm/io/mne_loader.py` -- 4 loader functions under test
  - `src/pyro_dcm/io/bids_loader.py` -- 2 BIDS loader functions under test
  - `src/pyro_dcm/io/__init__.py` -- re-export pattern with try/except
