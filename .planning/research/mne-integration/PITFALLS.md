# Domain Pitfalls: MNE-Python Integration (Tests, Pipelines, Documentation)

**Domain:** Adding test suite, end-to-end pipelines, and documentation for MNE IO module in a DCM framework
**Researched:** 2026-05-21
**Scope:** Testing, pipeline, and documentation pitfalls specific to bridging MNE-Python and Pyro-DCM

---

## Critical Pitfalls

Mistakes that cause silent data corruption, wrong scientific results, or major rewrites.

---

### Pitfall 1: CSD Frequency Grid Mismatch Between MNE and SPM Convention

**What goes wrong:** The existing `epochs_to_csd` function requests `n_freqs` linearly-spaced frequency bins, but `csd_multitaper` returns CSD at FFT-determined frequency bins (dictated by `n_fft` and `sfreq`), NOT at user-specified frequencies. The code then interpolates the MNE CSD onto the target grid. This interpolation introduces spectral artifacts that do not exist in the original data, particularly at low frequencies where fMRI spectral DCM operates (1/128 to ~0.25 Hz). Meanwhile, `csd_morlet` accepts arbitrary `frequencies` arrays directly, producing CSD at exactly the requested bins -- no interpolation needed.

The downstream spectral DCM forward model (`spectral_dcm_forward`) uses `default_frequency_grid(TR, n_freqs=32)` which produces `linspace(1/128, 1/(2*TR), 32)`. If the MNE CSD frequencies do not match this grid exactly, the ELBO comparison between predicted and observed CSD is comparing quantities at different frequencies.

**Why it happens:** SPM12 uses MAR (multivariate autoregressive, order p=4) to estimate CSD, which can produce CSD at arbitrary frequencies analytically. MNE uses FFT-based methods where frequencies are locked to the DFT grid. The two approaches produce fundamentally different frequency sampling.

**Consequences:**
- Interpolation of CSD (especially complex-valued) introduces phase errors
- Low-frequency CSD values (critical for fMRI DCM) are most affected because DFT resolution is coarsest there
- Model comparison results become unreliable if observed CSD has interpolation artifacts

**Prevention:**
1. For `method='morlet'`: pass `freqs_target` directly as the `frequencies` parameter -- this is already done correctly in the existing code
2. For `method='multitaper'`: either (a) accept the native FFT grid and adjust the spectral DCM forward model to match, or (b) document clearly that interpolation is applied and add a test verifying interpolation error is below a threshold on a known-spectrum signal
3. Add a test: generate sinusoidal signal at known frequency, compute CSD via both methods, verify peak location and amplitude match within tolerance
4. Add a property to the returned dict: `'interpolated': bool` to flag when interpolation was applied

**Detection:** Test that `freqs` in the returned dict exactly matches `default_frequency_grid()` output to within floating-point tolerance. If not, the pipeline is using interpolated data.

**Confidence:** HIGH -- verified from MNE documentation and SPM12 source code.

**Sources:**
- [MNE csd_multitaper docs](https://mne.tools/stable/generated/mne.time_frequency.csd_multitaper.html) -- n_fft determines frequency bins
- [MNE csd_morlet docs](https://mne.tools/stable/generated/mne.time_frequency.csd_morlet.html) -- accepts arbitrary frequencies
- [SPM12 spm_dcm_fmri_csd_data.m](https://github.com/neurodebian/spm12/blob/master/spm_dcm_fmri_csd_data.m) -- MAR-based, p=4, N_w=32 linearly spaced

---

### Pitfall 2: CSD Normalization Convention Mismatch (MNE vs SPM vs scipy)

**What goes wrong:** Three different CSD computation paths exist in the codebase, each with different normalization:

| Source | Method | Normalization | Units |
|--------|--------|---------------|-------|
| `csd_computation.py` | scipy Welch | `scaling='density'` (power per Hz) | V^2/Hz |
| `mne_loader.py` | MNE multitaper/morlet | Undocumented in MNE API | Unknown |
| SPM12 | MAR analytical | Custom (with optional sqrt(sum) norm) | Arbitrary |

The spectral DCM forward model (`spectral_dcm_forward`) produces predicted CSD with SPM's internal scaling (C=1/256 constant, specific amplitude/exponent parameterization). If observed CSD from MNE has different normalization than what SPM's forward model expects, the likelihood comparison will be dominated by a global scale mismatch rather than by connectivity structure.

**Why it happens:** MNE's CrossSpectralDensity documentation does not specify normalization convention or units. The CSD object stores values internally as a vector but does not document whether it applies one-sided or two-sided normalization, or what scaling is used.

**Consequences:**
- SVI converges to wrong connectivity parameters because the noise parameters absorb the scale mismatch
- Apparent good ELBO but scientifically meaningless A matrix estimates
- Validation against SPM will show systematic offsets

**Prevention:**
1. Add a normalization test: generate white noise with known variance sigma^2, compute CSD via each method, verify that the diagonal (auto-spectral) values integrate to sigma^2 across the frequency range
2. Add an explicit `normalize` parameter to `epochs_to_csd` that rescales output to match the convention used by `spectral_dcm_forward`
3. Document the normalization convention in the function docstring: "Output CSD uses [specific convention], matching spectral_dcm_forward expectations"
4. In the pipeline, add an assertion that checks CSD magnitude is in the expected range before passing to the model

**Detection:** Diagonal of CSD at each frequency should be real and positive (auto-spectra). If diagonal values are orders of magnitude different from predicted CSD with default noise parameters, normalization is wrong.

**Confidence:** HIGH for the existence of the mismatch risk. MEDIUM for the specific MNE normalization (undocumented).

**Sources:**
- [MNE CrossSpectralDensity API](https://mne.tools/stable/generated/mne.time_frequency.CrossSpectralDensity.html) -- no normalization documented
- [scipy.signal.csd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) -- scaling='density' documented
- SPM12 spm_csd_fmri_mtf.m -- C=1/256 scaling constant

---

### Pitfall 3: Channel Selection Inconsistency Across MNE Data Containers

**What goes wrong:** The `picks` parameter behaves inconsistently between MNE object types:
- `Epochs.get_data(picks='eeg')` **excludes** bad channels
- `Evoked.get_data(picks='eeg')` **includes** bad channels
- `picks=None` in `Raw.get_data()` includes bad channels
- `epochs.copy().pick(picks)` vs `picks` parameter in `csd_multitaper` may produce different channel sets

The existing `epochs_to_timeseries` averages epochs then extracts data. If the Epochs object has bad channels marked, the `.average()` call excludes them but `.pick(picks)` may or may not depending on version.

**Why it happens:** This is a known MNE bug (GitHub issue #12577, reported April 2024). The `get_data()` method is inconsistent between `Epochs` and `Evoked`. Additionally, MNE's `pick()` method handles `exclude='bads'` differently depending on whether you specify channel names, types, or indices.

**Consequences:**
- CSD shape `(F, N, N)` and timeseries shape `(T, N)` have different N for the same data if one path excludes bads and another includes them
- Downstream DCM model receives mismatched tensor dimensions
- Silent errors: tensors have the right shapes but wrong channels, producing garbage connectivity estimates

**Prevention:**
1. Standardize channel selection: always use `epochs.copy().pick(picks)` FIRST, then operate on the picked copy. Never rely on `picks` parameter in downstream MNE functions.
2. The existing code already does this correctly for `epochs_to_timeseries` but `epochs_to_csd` passes `picks` directly to `csd_multitaper`/`csd_morlet`. Fix: pick first, then compute CSD on the picked epochs.
3. Add a test: create Epochs with a bad channel, call both `epochs_to_csd` and `epochs_to_timeseries`, verify `ch_names` lists are identical.
4. Add an assertion in the pipeline that CSD `ch_names` matches timeseries `ch_names`.

**Detection:** Compare `len(result['ch_names'])` against expected number of channels. If they differ between functions called on the same data, channel selection is inconsistent.

**Confidence:** HIGH -- verified via MNE issue #12577 and documentation.

**Sources:**
- [MNE Epochs/Evoked get_data inconsistency, Issue #12577](https://github.com/mne-tools/mne-python/issues/12577)
- [MNE bad channel handling tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html)

---

### Pitfall 4: dtype Mismatch Chain (MNE float64 -> PyTorch float32/float64 -> Pyro)

**What goes wrong:** MNE-Python internally typecasts ALL data to float64 upon loading. The existing IO module defaults to `dtype=torch.float64` for timeseries and `dtype=torch.complex128` for CSD, which matches the spectral DCM convention. However:

1. **Apple Silicon (MPS backend):** Does NOT support float64. Any user on macOS with `device='mps'` will get `TypeError: Cannot convert a MPS Tensor to float64 dtype`. This is a hard failure with no graceful degradation.

2. **Mixed precision traps:** If a user changes `dtype=torch.float32` for speed, the CSD should be `torch.complex64` but the spectral DCM forward model (`spectral_dcm_forward`) hardcodes `torch.complex128` and `torch.float64` throughout. Mixing dtypes causes silent precision loss or RuntimeError.

3. **SI unit scaling:** MNE stores EEG data in Volts (order 1e-6), MEG magnetometers in Tesla (order 1e-13). When converted to float32, very small values like 1e-13 are representable but matrix operations on them lose precision. The CSD of MEG data in SI units has values ~1e-26, which underflows float32.

**Why it happens:** MNE uses float64 as a deliberate design choice for numerical accuracy. The spectral DCM forward model also uses float64 for the same reason. But users or downstream code may request float32 for GPU compatibility.

**Consequences:**
- MPS users cannot use the IO module at all without modification
- CSD values underflow in float32 for MEG data, producing all-zero matrices
- Mixed dtype tensors cause PyTorch RuntimeErrors in matrix operations

**Prevention:**
1. Document the dtype constraint prominently: "spectral DCM requires float64/complex128; float32 is NOT supported"
2. Add a validation check in the pipeline: if device is MPS, warn that float64 is not supported and recommend CPU
3. Test with both float32 and float64 inputs to verify either (a) float32 works correctly or (b) a clear error is raised
4. Consider adding a `scale_si` parameter that converts EEG to microvolts and MEG to femtotesla before creating tensors, making float32 viable for sensor-space data

**Detection:** Check `tensor.dtype` at pipeline boundaries. Add `assert tensor.dtype == torch.float64` in the spectral DCM model's data preparation.

**Confidence:** HIGH -- MNE float64 convention is documented; MPS float64 limitation is well-known.

**Sources:**
- [MNE implementation details -- float64 convention](https://mne.tools/stable/documentation/implementation.html)
- [PyTorch MPS float64 issue](https://github.com/Lightning-AI/pytorch-lightning/issues/21261)

---

## Moderate Pitfalls

Mistakes that cause test failures, delays, or technical debt.

---

### Pitfall 5: Test Data Strategy -- Heavy MNE Objects vs Lightweight Mocks

**What goes wrong:** Tests that create real MNE objects (Raw, Epochs, SourceEstimate) with realistic parameters are slow, verbose, and fragile:
- Creating `mne.Epochs` requires events array, Info object with channel locations, and preloaded data
- `SourceEstimate` requires source spaces, which requires a FreeSurfer subject directory
- Full MNE objects pull in the entire MNE runtime, making tests take seconds instead of milliseconds

Teams either (a) skip testing MNE integration because fixtures are too complex, or (b) create monolithic test fixtures that are hard to maintain.

**Why it happens:** MNE data structures are rich objects with many metadata fields. Minimal construction requires knowing the exact constructor API, which changes between versions.

**Prevention:**
1. Use `mne.create_info()` + `mne.io.RawArray()` for minimal Raw objects: only need channel names, sfreq, and a numpy array
2. Use `mne.EpochsArray()` for minimal Epochs: need `(n_epochs, n_channels, n_samples)` array, info, and events
3. Create a `conftest.py` fixture factory that produces minimal MNE objects:
   ```python
   @pytest.fixture
   def make_epochs():
       def _make(n_channels=3, n_epochs=5, n_times=100, sfreq=250.0):
           info = mne.create_info(
               [f"EEG {i:03d}" for i in range(n_channels)],
               sfreq=sfreq, ch_types="eeg"
           )
           data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
           events = np.column_stack([
               np.arange(0, n_epochs * n_times, n_times),
               np.zeros(n_epochs, dtype=int),
               np.ones(n_epochs, dtype=int),
           ])
           return mne.EpochsArray(data, info, events)
       return _make
   ```
4. For `stc_to_roi_timeseries` tests: mock `mne.extract_label_time_course` rather than constructing full SourceEstimate + SourceSpaces + Labels, since the function is a thin wrapper
5. Mark tests requiring full MNE objects as `@pytest.mark.slow` and guard with `pytest.importorskip("mne")`

**Detection:** If MNE-related tests take >5 seconds each or require downloading sample datasets, the fixture strategy is too heavy.

**Confidence:** HIGH -- standard MNE testing pattern documented in tutorials.

**Sources:**
- [MNE creating data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html)
- [MNE EpochsArray API](https://mne.tools/stable/generated/mne.EpochsArray.html)

---

### Pitfall 6: Optional Dependency Test Isolation

**What goes wrong:** MNE is an optional dependency (`pip install pyro-dcm[mne]`). Tests for MNE functionality must:
1. Skip gracefully when MNE is not installed
2. Not import MNE at module level in test files (breaks `pytest --collect-only` without MNE)
3. Not break the existing test suite when MNE IS installed but a different optional dep is missing

The existing `io/__init__.py` already has a problem: it imports from `mne_loader` unconditionally at the top level, but `mne_loader.py` only imports MNE inside `_require_mne()`. If a user imports `from pyro_dcm.io import epochs_to_csd` without MNE installed, the import succeeds but the function fails at runtime. This is the correct behavior. BUT if tests import the module at collection time, and MNE is not installed, tests will still collect but fail confusingly.

**Why it happens:** Python's lazy-vs-eager import patterns interact poorly with pytest's collection phase and optional dependencies.

**Prevention:**
1. Use `pytest.importorskip("mne")` at the top of each MNE test file or in a conftest fixture
2. Add a custom pytest marker: `@pytest.mark.mne` and register it in `pyproject.toml`
3. In CI, run MNE tests separately: `pytest -m mne` with the `[mne]` extra installed
4. Test the "MNE not installed" path explicitly: verify `_require_mne()` raises `ImportError` with the correct install instructions
5. Do NOT use `try/except ImportError` at module level in test files; use `pytest.importorskip` which provides proper skip semantics

**Detection:** Run `pytest --collect-only` in an environment without MNE installed. If any test file fails to collect, the import isolation is broken.

**Confidence:** HIGH -- standard Python optional dependency testing pattern.

---

### Pitfall 7: BIDS Event Annotation Edge Cases

**What goes wrong:** The `load_bids_epochs` function uses `mne.events_from_annotations(raw)` to extract events. Several edge cases cause failures:

1. **No annotations:** Some BIDS datasets store events in `events.tsv` sidecar files, not as Raw annotations. `mne_bids.read_raw_bids` loads sidecar events as annotations, but if the sidecar is missing or malformed, `events_from_annotations` returns an empty events array, and `mne.Epochs()` silently creates an Epochs object with 0 epochs.

2. **Annotation duration filtering:** Annotations with duration shorter than the epoch's chunk duration are silently dropped during conversion.

3. **Event ID scrambling:** Converting annotations to events and back can scramble integer trigger codes unless `event_id` mapping is explicitly provided. Auto-detected `event_id` from `events_from_annotations` uses string-based annotation descriptions as keys, which may not match user expectations.

4. **Edge annotations from concatenation:** If raw files were concatenated, MNE inserts `BAD_ACQ_SKIP` annotations at boundaries. These get included in auto-detected events unless filtered.

**Why it happens:** BIDS stores events as TSV files with flexible schemas, while MNE internally uses integer-coded events arrays. The annotation-to-event conversion is inherently lossy.

**Consequences:**
- Zero-epoch Epochs objects propagate silently through the pipeline
- Wrong events produce epochs at wrong time points, corrupting the timeseries
- Pipeline appears to work but results are scientifically wrong

**Prevention:**
1. After `events_from_annotations`, check `len(events) > 0` and raise a descriptive error if empty
2. After `mne.Epochs()`, check `len(epochs) > 0` with a message: "No epochs created. Check event_id mapping and BIDS events.tsv file."
3. Log the auto-detected `event_id` mapping so users can verify it
4. Filter out `BAD_*` annotations before event extraction: `events_from_annotations(raw, regexp='^(?!BAD)')`
5. Test edge cases: empty events.tsv, missing events.tsv, mismatched event_id keys

**Detection:** `len(epochs) == 0` after creation. Also check `epochs.drop_log` for excessive rejection.

**Confidence:** MEDIUM -- based on mne-bids issues (#683, #332, #1084) and MNE annotation documentation.

**Sources:**
- [mne-bids issue #683: datasets without events.tsv](https://github.com/mne-tools/mne-bids/issues/683)
- [mne-bids issue #1084: annotations vs events conflict](https://github.com/mne-tools/mne-bids/issues/1084)
- [mne.events_from_annotations API](https://mne.tools/stable/generated/mne.events_from_annotations.html)

---

### Pitfall 8: MNE API Deprecation Churn in Channel Selection

**What goes wrong:** MNE-Python has been migrating channel selection APIs across several versions:
- `inst.pick_types()` and `inst.pick_channels()` were marked as legacy in v1.4 (2023)
- The unified `inst.pick()` method is the recommended replacement
- The `ordered` parameter default changed from `False` (v1.6) to `True` (v1.7)
- `pick()` got unexpected keyword errors in v1.7 (issue #12402)

The existing IO code uses `epochs.copy().pick(picks)` and `raw.copy().pick(picks)`, which is the correct modern API. But tests or documentation examples that use older patterns will break on newer MNE versions, and users following old tutorials may encounter confusing deprecation warnings.

**Why it happens:** MNE-Python follows a deprecation cycle policy but the migration from multiple pick functions to a unified `pick()` has been gradual across 4+ versions.

**Prevention:**
1. Pin minimum MNE version clearly: `mne>=1.6` is already in `pyproject.toml` -- verify all code works with 1.6 AND current (1.12+)
2. Use only `inst.pick()` in all code and documentation -- never `pick_types` or `pick_channels`
3. Test against both minimum supported version and latest version in CI
4. Document the minimum version requirement in the MNE loader module docstring
5. Add a version check at import time: `if mne.__version__ < '1.6': warn(...)`

**Detection:** Run tests with `mne==1.6` and latest version. Deprecation warnings (`DeprecationWarning`) should be treated as errors in CI: `filterwarnings = "error::DeprecationWarning"` in pytest config.

**Confidence:** MEDIUM -- MNE deprecation policy is documented but the specific behavior changes across versions need runtime verification.

**Sources:**
- [MNE pick() issue #12402](https://github.com/mne-tools/mne-python/issues/12402)
- [MNE v1.6 what's new](https://mne.tools/1.6/development/whats_new.html)
- [MNE v1.8 what's new](https://mne.tools/stable/changes/v1.8.html)

---

### Pitfall 9: stc_to_roi_timeseries Silent Zeros from Empty Labels

**What goes wrong:** The `stc_to_roi_timeseries` function calls `mne.extract_label_time_course(stc, labels, src, mode=mode)` without setting `allow_empty`. The default (`allow_empty=False`) raises an error if any label has no vertices in the source estimate. This seems safe, but:

1. If a user passes atlas labels for a parcellation that includes medial wall or subcortical regions not in the source space, the function crashes
2. If `allow_empty=True` is set to avoid the crash, the function silently returns all-zero time courses for those regions
3. Those zero time courses get passed to DCM, which interprets them as a disconnected region with no activity -- producing connectivity estimates of zero that look plausible

**Why it happens:** Neuroimaging atlases (Desikan-Killiany, Destrieux, Schaefer) include labels that may not overlap with the source space (e.g., medial wall, corpus callosum, subcortical structures). Users selecting "all labels from atlas" will hit this.

**Prevention:**
1. Set `allow_empty='ignore'` but check for all-zero columns in the output
2. After extraction, verify `(label_ts != 0).any(axis=0).all()` -- no column should be all zeros
3. If zero columns found, raise a warning listing which ROI names have no data
4. Add a test: create a label that has no overlap with the source space, verify the function raises an informative error (not just MNE's generic error)
5. Document: "Ensure all labels have vertices in the source space. Remove medial wall labels before calling this function."

**Detection:** Check for columns of zeros in the output timeseries. Any column where `max(abs(x)) < eps` is suspicious.

**Confidence:** HIGH -- `allow_empty` parameter behavior is clearly documented in MNE.

**Sources:**
- [mne.extract_label_time_course API](https://mne.tools/stable/generated/mne.extract_label_time_course.html) -- allow_empty parameter

---

### Pitfall 10: Pipeline Documentation Misleading Neuroscience Users

**What goes wrong:** The IO module bridges two communities with different expertise:
- **Neuroscience users** know MNE but not DCM internals, Pyro, or tensor conventions
- **DCM/ML users** know the math but not MNE preprocessing workflows

Common documentation mistakes:

1. **Assuming preprocessing is done:** The IO functions do NOT preprocess data (no filtering, no artifact rejection, no rereferencing). If documentation shows `raw_to_timeseries(raw)` without mentioning that the raw data should be preprocessed first, users will feed unfiltered 256-channel EEG into a 3-region DCM model.

2. **Hiding the picks requirement:** For DCM, users must select a small number of channels/ROIs (typically 3-8). If the default `picks=None` selects all 64 EEG channels, the user gets a 64x64 A matrix, which is unidentifiable. The documentation must make clear that picks selection is mandatory in practice, even though it is optional in the API.

3. **Not showing the full pipeline:** A quickstart that shows only `epochs_to_csd(epochs)` without showing how to go from CSD dict to spectral DCM model will leave users stuck at the handoff point.

4. **Wrong units in examples:** Showing `reject={'eeg': 100e-6}` in documentation but using arbitrary units in test fixtures (e.g., `np.random.randn() * 1.0` instead of `* 1e-6`) creates confusion about expected data ranges.

**Prevention:**
1. Every example must show the complete pipeline: load -> preprocess -> select channels -> compute CSD/timeseries -> pass to DCM model
2. Document preconditions prominently: "Data must be preprocessed (filtered, artifact-rejected) before using these functions"
3. Add a "Data Preparation Checklist" section to the IO module documentation
4. Use realistic units in ALL examples and test fixtures (EEG: ~1e-6 V, MEG mag: ~1e-13 T)
5. Add parameter validation: if `picks is None` and `len(epochs.ch_names) > 20`, emit a warning about DCM identifiability

**Detection:** Have a neuroscience user (not the developer) try to follow the documentation end-to-end. If they get stuck, the documentation has gaps.

**Confidence:** HIGH -- this is a universal documentation pitfall for cross-domain tools.

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable without redesign.

---

### Pitfall 11: MNE Verbose Output Polluting Test Logs

**What goes wrong:** MNE-Python defaults to `verbose=True` for many functions, producing INFO-level log output about channel selection, filtering, event detection, etc. This pollutes pytest output and makes it hard to find actual test failures.

**Prevention:**
1. Set `verbose=False` in all test fixtures
2. Consider adding `mne.set_log_level("WARNING")` in the MNE test conftest.py
3. The existing BIDS loader already accepts `verbose=False` -- ensure all MNE calls in tests use it

**Detection:** Run tests and check if MNE INFO messages appear in stdout.

---

### Pitfall 12: Interpolation of Complex CSD Values

**What goes wrong:** The current `epochs_to_csd` code interpolates real and imaginary parts of CSD independently using `np.interp`. This is mathematically incorrect for complex-valued signals because it does not preserve the magnitude-phase relationship. For example, if two adjacent frequency bins have CSD values with similar magnitude but different phase, linear interpolation of real and imaginary parts can produce a value with reduced magnitude (destructive interference in the interpolation).

**Prevention:**
1. Interpolate magnitude and phase separately (polar interpolation) instead of real and imaginary parts
2. Better yet: avoid interpolation entirely by using `csd_morlet` with the exact target frequencies, or by adjusting `n_fft` in `csd_multitaper` to produce frequencies close to the target grid
3. If interpolation is unavoidable, add a test: verify that interpolated CSD remains Hermitian (`csd[f,i,j] == conj(csd[f,j,i])`) and that auto-spectra (diagonal) remain real and non-negative after interpolation

**Detection:** Check `csd.diagonal().imag` -- should be zero. Check `torch.allclose(csd, csd.conj().transpose(-1,-2))` for Hermitian symmetry.

**Confidence:** HIGH -- mathematical fact about complex interpolation.

---

### Pitfall 13: Events Array Shape for EpochsArray in Tests

**What goes wrong:** When creating `mne.EpochsArray` in test fixtures, the events array must be `(n_events, 3)` with columns `[sample_onset, previous_event_id, event_id]`. Common mistakes:
- Omitting the middle column (previous event ID) -- it must be 0 for most use cases
- Using time values instead of sample indices for onset
- Not matching `n_events` to `n_epochs` in the data array

**Prevention:**
1. Use a fixture helper that creates the events array correctly:
   ```python
   events = np.column_stack([
       np.arange(0, n_epochs * n_times, n_times),
       np.zeros(n_epochs, dtype=int),
       np.ones(n_epochs, dtype=int),
   ])
   ```
2. Add explicit shape assertion: `assert events.shape == (n_epochs, 3)`

**Detection:** MNE raises `ValueError` with informative messages about events shape mismatches. These should appear immediately in tests.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Priority |
|-------------|---------------|------------|----------|
| Test fixtures | P5: Heavy MNE objects | Use RawArray/EpochsArray factories | Do first |
| Test fixtures | P6: Optional dep isolation | pytest.importorskip + markers | Do first |
| Test: CSD path | P1: Frequency grid mismatch | Test morlet vs multitaper grids match DCM convention | Critical test |
| Test: CSD path | P2: Normalization mismatch | White noise integration test | Critical test |
| Test: CSD path | P12: Complex interpolation | Hermitian symmetry assertion | Add to CSD tests |
| Test: channel picking | P3: Bad channel inconsistency | Pre-pick then compute; test with bads | Before pipeline |
| Test: STC path | P9: Empty label zeros | Test with non-overlapping label | Add to STC tests |
| Pipeline: CSD | P4: dtype chain | Assert float64 at boundaries; document MPS limitation | Pipeline validation |
| Pipeline: BIDS | P7: Empty events | Assert len(epochs) > 0 after creation | BIDS pipeline |
| Pipeline: end-to-end | P10: Documentation gaps | Full pipeline example with preprocessing context | Documentation phase |
| Documentation | P8: API deprecation | Test min+max MNE versions | CI matrix |
| Documentation | P10: Cross-domain confusion | Preprocessing checklist, picks guidance | Documentation phase |

---

## Sources

- [MNE csd_multitaper documentation](https://mne.tools/stable/generated/mne.time_frequency.csd_multitaper.html)
- [MNE csd_morlet documentation](https://mne.tools/stable/generated/mne.time_frequency.csd_morlet.html)
- [MNE CrossSpectralDensity API](https://mne.tools/stable/generated/mne.time_frequency.CrossSpectralDensity.html)
- [MNE creating data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html)
- [MNE bad channel handling](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html)
- [MNE implementation details (float64)](https://mne.tools/stable/documentation/implementation.html)
- [MNE extract_label_time_course](https://mne.tools/stable/generated/mne.extract_label_time_course.html)
- [MNE events_from_annotations](https://mne.tools/stable/generated/mne.events_from_annotations.html)
- [MNE Epochs/Evoked bad channel inconsistency, Issue #12577](https://github.com/mne-tools/mne-python/issues/12577)
- [MNE pick() unexpected keyword, Issue #12402](https://github.com/mne-tools/mne-python/issues/12402)
- [mne-bids missing events.tsv, Issue #683](https://github.com/mne-tools/mne-bids/issues/683)
- [mne-bids annotations vs events, Issue #1084](https://github.com/mne-tools/mne-bids/issues/1084)
- [SPM12 spm_dcm_fmri_csd_data.m](https://github.com/neurodebian/spm12/blob/master/spm_dcm_fmri_csd_data.m)
- [scipy.signal.csd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html)
- [PyTorch MPS float64 limitation](https://github.com/Lightning-AI/pytorch-lightning/issues/21261)
- [MNE CSD computation tutorial](https://mne.tools/stable/auto_examples/time_frequency/compute_csd.html)
