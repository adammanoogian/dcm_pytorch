# Phase 18: MNE/BIDS IO Test Suite - Research

**Researched:** 2026-05-21
**Domain:** MNE-Python testing, mne-bids testing, pytest optional-dependency patterns
**Confidence:** HIGH

## Summary

This phase creates a comprehensive test suite for the MNE and BIDS IO loaders
already implemented in `src/pyro_dcm/io/mne_loader.py` and `src/pyro_dcm/io/bids_loader.py`.
The test suite must validate shape contracts, mathematical properties (CSD Hermitian
symmetry, non-negative auto-spectra, sine-injection round-trip), error handling, and
BIDS round-trip correctness, while cleanly skipping when MNE is not installed.

The standard approach is to create synthetic MNE objects (RawArray, EpochsArray) from
NumPy arrays using `mne.create_info` and `mne.io.RawArray` / `mne.EpochsArray`, avoiding
any dependency on external data files. For BIDS tests, `mne_bids.write_raw_bids` writes
synthetic data to `tmp_path`, and `load_bids_raw` / `load_bids_epochs` reads it back. For
the source-estimate function (`stc_to_roi_timeseries`), `unittest.mock.patch` or
`monkeypatch` mocks `mne.extract_label_time_course` since constructing real source
spaces is prohibitively complex for unit tests.

**Primary recommendation:** Build all synthetic test data from NumPy arrays + `mne.create_info` --
never download or depend on external MNE datasets. Use `pytest.importorskip("mne")` at
module level for clean skip behavior. Register the `mne` marker in `pyproject.toml`.

## Standard Stack

### Core (Test Infrastructure)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | (project dep) | Test framework | Already used throughout project |
| mne | >=1.6 | Create synthetic test objects, CSD computation | Optional dep -- tests skip if absent |
| mne-bids | >=0.14 | Write/read synthetic BIDS datasets | Optional dep -- tests skip if absent |
| numpy | (project dep) | Synthetic data generation (sine waves, random noise) | Already used |
| torch | (project dep) | Assert output tensor shapes and dtypes | Already used |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mock `mne.extract_label_time_course` for TEST-04 | stc_to_roi_timeseries test only |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Synthetic RawArray/EpochsArray | MNE sample dataset | External download, slow CI, fragile |
| unittest.mock for stc test | Full inverse solution pipeline | Would need source spaces, forward model -- massive overhead |
| `pytest.importorskip` | `try/except/pytest.skip` | importorskip is cleaner, single line, standard pattern |

**Installation:**
```bash
pip install pyro-dcm[mne]  # installs mne>=1.6, mne-bids>=0.14
pip install pyro-dcm[dev]   # installs pytest
```

## Architecture Patterns

### Recommended Test File Structure

```
tests/
├── conftest.py                  # Existing -- no MNE fixtures here (optional dep)
├── test_mne_loader.py           # TEST-01 through TEST-11 (13 tests)
└── test_bids_loader.py          # BIDS-01 through BIDS-03 (3+ tests)
```

### Pattern 1: Module-Level Import Skip (TEST-12)

**What:** Skip the entire test file when MNE is not installed.
**When to use:** Every test file that imports MNE objects.
**Example:**
```python
# Source: https://docs.pytest.org/en/stable/how-to/skipping.html
from __future__ import annotations

import numpy as np
import pytest
import torch

mne = pytest.importorskip("mne")

# All tests below this line are skipped if MNE is not installed.
```

**Critical detail:** `pytest.importorskip` must be at module level (top of file,
after standard imports). It assigns the imported module to the variable, so you
can use `mne.create_info(...)` etc. directly.

### Pattern 2: Synthetic MNE Object Fixtures

**What:** Create MNE Info, Raw, Epochs from NumPy arrays for each test.
**When to use:** Every test needing MNE objects.
**Example:**
```python
# Source: https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html

@pytest.fixture()
def synthetic_epochs() -> mne.EpochsArray:
    """Create a 5-epoch, 3-channel EEG dataset at 256 Hz."""
    sfreq = 256.0
    n_epochs, n_channels, n_times = 5, 3, 256
    ch_names = ["EEG1", "EEG2", "EEG3"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6  # Volts
    epochs = mne.EpochsArray(data, info)
    return epochs


@pytest.fixture()
def synthetic_raw() -> mne.io.RawArray:
    """Create a 3-channel EEG Raw at 256 Hz, 10 seconds."""
    sfreq = 256.0
    n_channels, n_times = 3, 2560
    ch_names = ["EEG1", "EEG2", "EEG3"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6
    raw = mne.io.RawArray(data, info)
    return raw
```

**Key detail:** EEG data should be in Volts (multiply by `1e-6`).
Channel types must be valid MNE types (e.g., "eeg", "meg", "misc").

### Pattern 3: Sine-Injection CSD Round-Trip (TEST-09)

**What:** Inject a known-frequency sine wave, compute CSD, verify peak at that frequency.
**When to use:** Validating CSD computation correctness.
**Example:**
```python
def test_csd_sine_injection_roundtrip():
    """10 Hz sine -> CSD peak at 10 Hz bin."""
    sfreq = 256.0
    n_epochs, n_channels = 20, 2
    duration = 2.0  # seconds per epoch
    n_times = int(sfreq * duration)
    freq_target = 10.0  # Hz

    ch_names = ["EEG1", "EEG2"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    t = np.arange(n_times) / sfreq
    sine_10hz = np.sin(2 * np.pi * freq_target * t)

    # Both channels carry same 10 Hz signal + small noise
    rng = np.random.default_rng(42)
    data = np.zeros((n_epochs, n_channels, n_times))
    for ep in range(n_epochs):
        noise = rng.standard_normal((n_channels, n_times)) * 0.01
        data[ep, 0, :] = sine_10hz + noise[0]
        data[ep, 1, :] = sine_10hz + noise[1]
    data *= 1e-6  # Volts

    epochs = mne.EpochsArray(data, info)
    result = epochs_to_csd(epochs, fmin=1.0, fmax=50.0, n_freqs=50)

    csd = result["csd"]      # (F, N, N)
    freqs = result["freqs"]  # (F,)

    # Find bin closest to 10 Hz
    target_idx = torch.argmin(torch.abs(freqs - freq_target))
    auto_power = csd[:, 0, 0].real  # auto-spectrum channel 0

    # 10 Hz bin should have the highest power
    peak_idx = torch.argmax(auto_power)
    assert abs(peak_idx.item() - target_idx.item()) <= 1, (
        f"CSD peak at bin {peak_idx} (freq {freqs[peak_idx]:.1f} Hz), "
        f"expected near bin {target_idx} (freq {freqs[target_idx]:.1f} Hz)"
    )
```

**Key details:** Use >= 20 epochs for stable CSD estimate. Use >= 2s duration
per epoch for adequate frequency resolution. The tolerance of +/- 1 bin accounts
for interpolation in the loader's frequency-grid resampling step.

### Pattern 4: BIDS Round-Trip with write_raw_bids (BIDS-01, BIDS-02)

**What:** Write synthetic data to BIDS format, read it back with the loader.
**When to use:** BIDS loader tests.
**Example:**
```python
# Source: https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html

mne_bids = pytest.importorskip("mne_bids")

def test_load_bids_raw(tmp_path):
    """Round-trip: write synthetic BIDS, read back with load_bids_raw."""
    sfreq = 256.0
    ch_names = ["EEG1", "EEG2", "EEG3"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info["line_freq"] = 50.0  # Required by BIDS

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 2560)) * 1e-6
    raw = mne.io.RawArray(data, info)

    bids_path = mne_bids.BIDSPath(
        subject="01",
        task="test",
        datatype="eeg",
        root=tmp_path,
    )
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    from pyro_dcm.io.bids_loader import load_bids_raw
    raw_loaded = load_bids_raw(bids_path)

    assert isinstance(raw_loaded, mne.io.BaseRaw)
    assert len(raw_loaded.ch_names) >= len(ch_names)  # BIDS may add stim
```

**Key details:** `info["line_freq"]` must be set (BIDS requirement). The `datatype`
parameter should match the channel types. `write_raw_bids` may add a STIM channel
for events, so channel count comparison should use `>=`.

### Pattern 5: Mocking stc_to_roi_timeseries (TEST-04)

**What:** Mock `mne.extract_label_time_course` to avoid needing real source spaces.
**When to use:** stc_to_roi_timeseries test only.
**Example:**
```python
from unittest.mock import MagicMock, patch

def test_stc_to_roi_timeseries_shape():
    """Output is (T, N) float tensor with mocked source extraction."""
    n_times, n_labels = 100, 3

    # Create mock SourceEstimate
    mock_stc = MagicMock()
    mock_stc.times = np.linspace(0, 1, n_times)
    mock_stc.tstep = 1.0 / 256.0

    # Create mock labels
    mock_labels = [MagicMock(name=f"ROI_{i}") for i in range(n_labels)]
    # Fix: MagicMock(name=...) sets the mock's name attribute, not .name
    for i, label in enumerate(mock_labels):
        label.name = f"ROI_{i}"

    mock_src = MagicMock()

    # Mock return: (n_labels, n_times) array
    mock_ts = np.random.default_rng(42).standard_normal((n_labels, n_times))

    with patch("pyro_dcm.io.mne_loader._mne") as mock_mne_module:
        # The function does: import mne as _mne; _mne.extract_label_time_course(...)
        mock_mne_module.extract_label_time_course.return_value = mock_ts
        result = stc_to_roi_timeseries(mock_stc, mock_labels, mock_src)

    ts = result["timeseries"]
    assert ts.shape == (n_times, n_labels)
    assert ts.dtype == torch.float64
```

**Important caveat:** The function imports `mne` locally as `_mne`, so the mock
target must match the actual import path used inside the function. The actual code
does `import mne as _mne` at line 289 of mne_loader.py, so we need to patch the
module-level import that gets resolved. The cleanest approach is to use
`monkeypatch.setattr` on the function directly, or to restructure the mock to
intercept `mne.extract_label_time_course`.

### Anti-Patterns to Avoid

- **Downloading MNE sample data in tests:** Creates CI dependency on network,
  slow, fragile. Always use synthetic data from NumPy arrays.
- **Creating real SourceSpaces for stc tests:** Requires fsaverage, FreeSurfer
  subjects dir. Mock instead.
- **Hardcoding frequency bin indices in CSD tests:** CSD frequency grids depend
  on sfreq and n_fft; use `argmin(abs(freqs - target))` instead.
- **Asserting exact channel counts after BIDS round-trip:** `write_raw_bids` may
  add STIM channels from events; use `>=` or check specific channel names.
- **Using `picks="eeg"` and expecting consistent bad-channel behavior:** See
  Pitfall 2 below -- this behavior differs between Epochs and Evoked objects.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Synthetic MNE objects | Custom data classes mimicking MNE | `mne.create_info` + `mne.io.RawArray` / `mne.EpochsArray` | MNE's constructors handle Info validation, channel types, montage -- custom mocks miss edge cases |
| BIDS dataset creation | Manual file/directory structure | `mne_bids.write_raw_bids` to `tmp_path` | BIDS has strict directory naming, JSON sidecars, TSV files -- hand-rolling is error-prone |
| CSD computation reference | Custom FFT-based CSD | `mne.time_frequency.csd_multitaper` / `csd_morlet` (already used by loader) | The loader wraps these; test the wrapper, not re-implement CSD |
| Module-level test skipping | Custom import-try-except-skip blocks | `pytest.importorskip("mne")` | Single line, returns module reference, standard pytest pattern |

**Key insight:** The loader functions are thin wrappers around MNE's own
functions. Tests should validate the *wrapper behavior* (shape conversion,
dtype, dict structure) not re-test MNE's internals.

## Common Pitfalls

### Pitfall 1: CSD Frequency Grid Mismatch (STATE.md P1)

**What goes wrong:** `csd_multitaper` returns CSD at FFT-determined frequencies,
not the user-requested `n_freqs` grid. The loader interpolates to match the
requested grid, but the interpolated frequencies may not include exact target
frequencies.

**Why it happens:** `csd_multitaper(fmin, fmax)` computes CSD at FFT-native
frequencies within the range, which depend on `n_fft` (auto-computed from epoch
length and `sfreq`). The loader then interpolates to `np.linspace(fmin, fmax, n_freqs)`.

**How to avoid:** In sine-injection tests (TEST-09), don't assert exact frequency
match. Use `argmin(abs(freqs - target))` and allow +/- 1 bin tolerance. Use
long epochs (>= 2 seconds) for better frequency resolution.

**Warning signs:** Test passes locally but fails on different platform due to
different FFT rounding behavior.

### Pitfall 2: Epochs.get_data(picks="eeg") vs Evoked.get_data(picks="eeg") Bad Channel Inconsistency (STATE.md P3)

**What goes wrong:** `Epochs.get_data(picks="eeg")` silently EXCLUDES bad channels.
`Evoked.get_data(picks="eeg")` INCLUDES bad channels. This is a known MNE bug
(issue #12577).

**Why it happens:** `Epochs.get_data()` defaults to `exclude='bads'` while `Evoked`
does not consistently apply this.

**How to avoid:** In TEST-06, verify that the loader's output channel count is
correct when bads are present. The loader code uses `epochs.copy().pick(picks)`,
and `pick()` uses `exclude=()` by default (does NOT auto-exclude bads). So when
`picks=None`, `epochs.average()` produces an Evoked with all channels including
bads. The test should verify this behavior explicitly.

**Warning signs:** Channel count differs between averaged and unaveraged paths.

### Pitfall 3: EEG Units in Synthetic Data

**What goes wrong:** MNE expects EEG data in Volts. If test data uses arbitrary
units (e.g., values around 1.0), MNE may warn or reject the data.

**Why it happens:** `mne.create_info` with `ch_types="eeg"` sets expected SI
units. Values of 1.0 would be 1 Volt -- unreasonably large for EEG.

**How to avoid:** Multiply synthetic data by `1e-6` (microvolts to volts).

**Warning signs:** MNE deprecation warnings about data range in test output.

### Pitfall 4: Mock Name Attribute Collision

**What goes wrong:** `MagicMock(name="ROI_V1")` sets the mock's internal name (for
repr), not the `.name` attribute. Accessing `.name` on this mock returns another
MagicMock, not the string "ROI_V1".

**Why it happens:** `name` is a reserved constructor argument for `MagicMock`.

**How to avoid:** Create the mock first, then set `.name` as an attribute:
```python
label = MagicMock()
label.name = "ROI_V1"
```

**Warning signs:** `roi_names` in output contains MagicMock objects instead of strings.

### Pitfall 5: BIDS line_freq Requirement

**What goes wrong:** `write_raw_bids` requires `info["line_freq"]` to be set.
If not set, it raises a RuntimeError.

**Why it happens:** BIDS specification mandates power line frequency in the
sidecar JSON.

**How to avoid:** Always set `info["line_freq"] = 50.0` (or 60.0) in synthetic
data for BIDS tests.

**Warning signs:** RuntimeError about missing line_freq during BIDS write.

### Pitfall 6: Mocking the Local Import in stc_to_roi_timeseries

**What goes wrong:** `stc_to_roi_timeseries` does `import mne as _mne` inside
the function body (line 289). Standard `unittest.mock.patch("mne.extract_label_time_course")`
may not intercept this correctly because the local import creates a new reference.

**Why it happens:** The function lazily imports MNE to support the optional
dependency pattern. The mock must target the module lookup, not a pre-bound name.

**How to avoid:** Use `monkeypatch.setattr("mne.extract_label_time_course", mock_fn)`
which patches the actual `mne` module attribute. Since the function does
`import mne as _mne` each time it runs, the patched attribute is seen. Alternatively,
use `unittest.mock.patch("mne.extract_label_time_course")`.

**Warning signs:** Mock not called, real function runs (or ImportError if mne
not installed).

## Code Examples

### Creating Synthetic Epochs with Known Properties

```python
# Source: https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html
import mne
import numpy as np

def make_synthetic_epochs(
    n_epochs: int = 10,
    n_channels: int = 3,
    sfreq: float = 256.0,
    duration: float = 1.0,
    seed: int = 42,
) -> mne.EpochsArray:
    """Create synthetic EEG epochs for testing."""
    n_times = int(sfreq * duration)
    ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    return mne.EpochsArray(data, info)
```

### Registering Custom Pytest Marker (TEST-13)

```toml
# pyproject.toml [tool.pytest.ini_options] markers section
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "spm: marks tests requiring MATLAB + SPM12",
    "tapas: marks tests requiring tapas rDCM MATLAB toolbox",
    "mne: marks tests requiring MNE-Python (run with '-m mne', skip with '-m \"not mne\"')",
]
```

### Module-Level Import Skip (TEST-12)

```python
# test_mne_loader.py -- top of file
from __future__ import annotations

import numpy as np
import pytest
import torch

# Skip entire file if MNE not installed
mne = pytest.importorskip("mne")

from pyro_dcm.io.mne_loader import (
    _require_mne,
    epochs_to_csd,
    epochs_to_timeseries,
    raw_to_timeseries,
    stc_to_roi_timeseries,
)

pytestmark = pytest.mark.mne  # Apply marker to all tests in file
```

### CSD Hermitian Symmetry Test (TEST-07)

```python
def test_csd_hermitian_symmetry(synthetic_epochs):
    """CSD satisfies csd[f,i,j] == conj(csd[f,j,i]) for all f."""
    result = epochs_to_csd(synthetic_epochs, fmin=1.0, fmax=50.0, n_freqs=32)
    csd = result["csd"]  # (F, N, N)

    # Hermitian: C[f] == C[f].conj().T
    csd_H = csd.conj().transpose(-2, -1)
    assert torch.allclose(csd, csd_H, atol=1e-12), (
        f"CSD not Hermitian. Max deviation: {(csd - csd_H).abs().max():.2e}"
    )
```

### CSD Non-Negative Diagonal Test (TEST-08)

```python
def test_csd_nonnegative_diagonal(synthetic_epochs):
    """Auto-spectra (diagonal) are real and non-negative."""
    result = epochs_to_csd(synthetic_epochs, fmin=1.0, fmax=50.0, n_freqs=32)
    csd = result["csd"]  # (F, N, N)
    n_freqs, n_ch, _ = csd.shape

    for i in range(n_ch):
        diag = csd[:, i, i]
        # Imaginary part should be ~zero for auto-spectra
        assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=1e-10), (
            f"Channel {i} auto-spectrum has non-zero imaginary part"
        )
        # Real part should be non-negative
        assert torch.all(diag.real >= -1e-12), (
            f"Channel {i} auto-spectrum has negative values: min={diag.real.min():.2e}"
        )
```

### BIDS BAD_ACQ_SKIP Annotation Test (BIDS-03)

```python
# Source: https://mne.tools/stable/generated/mne.Annotations.html
def test_bids_bad_acq_skip_annotation(tmp_path):
    """load_bids_epochs handles BAD_ACQ_SKIP annotations."""
    sfreq = 256.0
    ch_names = ["EEG1", "EEG2"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info["line_freq"] = 50.0

    rng = np.random.default_rng(42)
    n_samples = int(sfreq * 30)  # 30 seconds
    data = rng.standard_normal((2, n_samples)) * 1e-6
    raw = mne.io.RawArray(data, info)

    # Add events as annotations (required for epoching)
    events_onset = np.arange(1.0, 25.0, 3.0)
    annotations = mne.Annotations(
        onset=list(events_onset) + [10.0],
        duration=[0.0] * len(events_onset) + [2.0],
        description=["stimulus"] * len(events_onset) + ["BAD_ACQ_SKIP"],
    )
    raw.set_annotations(annotations)

    bids_path = mne_bids.BIDSPath(
        subject="01", task="test", datatype="eeg", root=tmp_path
    )
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    from pyro_dcm.io.bids_loader import load_bids_epochs
    epochs = load_bids_epochs(bids_path, event_id={"stimulus": 1})

    # Epochs overlapping BAD_ACQ_SKIP should be rejected
    assert isinstance(epochs, mne.Epochs)
    # The epoch near t=10 should be dropped
    assert len(epochs) < len(events_onset)
```

### _require_mne ImportError Test (TEST-10)

```python
def test_require_mne_raises_import_error(monkeypatch):
    """_require_mne raises ImportError with install instructions."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mne":
            raise ImportError("No module named 'mne'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pip install pyro-dcm\\[mne\\]"):
        _require_mne()
```

**Alternative (simpler):** Since TEST-10 tests `_require_mne()` and this test
file already `importorskip("mne")`, the `_require_mne` test needs special
handling. Options:
1. Put this test in a separate file that does NOT importorskip mne.
2. Use monkeypatch to temporarily break the mne import (as above).
3. Test it directly since the happy path (mne installed) just passes silently.

The cleanest approach is option 2: monkeypatch `builtins.__import__` to
simulate mne being absent.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `mne.pick_types(info, eeg=True)` | `info.pick("eeg")` / `raw.pick("eeg")` | MNE 1.1+ | `pick_types` still works but `pick()` is preferred |
| `mne.io.RawArray` with string ch_types | `mne.create_info(..., ch_types="eeg")` | Stable | String broadcasts to all channels |
| `mne.find_events(raw)` | `mne.events_from_annotations(raw)` | MNE 0.24+ | BIDS uses annotations, not STIM channels |
| `csd_multitaper` single-frequency | `CrossSpectralDensity.get_data(frequency=f)` | MNE 0.20+ | Returns (n_ch, n_ch) array at requested frequency |

**Deprecated/outdated:**
- `mne.pick_types()` function: Still works but `inst.pick()` method preferred
- `mne.find_events()` for BIDS data: Use `mne.events_from_annotations()` instead

## Open Questions

1. **_require_mne test isolation**
   - What we know: TEST-10 requires simulating mne absence, but the test file
     uses `pytest.importorskip("mne")` which means mne IS installed when tests run.
   - What's unclear: Whether monkeypatching `builtins.__import__` is reliable
     across all Python implementations, or if a separate test file is cleaner.
   - Recommendation: Use monkeypatch approach in the main test file. It's the
     standard pytest pattern and avoids file proliferation. If it proves fragile,
     fall back to a separate `test_mne_import_guard.py` file without importorskip.

2. **stc_to_roi_timeseries mock target path**
   - What we know: The function does `import mne as _mne` locally (line 289).
     `unittest.mock.patch("mne.extract_label_time_course")` should work because
     it patches the `mne` module's attribute before the local import resolves it.
   - What's unclear: Edge cases with lazy import caching.
   - Recommendation: Test with `monkeypatch.setattr(mne, "extract_label_time_course", mock_fn)`.
     Since `mne` is already imported at module level via `importorskip`, patching
     the module attribute is straightforward and the local `import mne as _mne`
     will see the patched version.

3. **BIDS-02 event handling**
   - What we know: `load_bids_epochs` calls `mne.events_from_annotations(raw)`.
     The annotations are written by `write_raw_bids` from the events parameter
     or from `raw.annotations`.
   - What's unclear: Whether events round-trip perfectly through BIDS
     annotation serialization (onset times may have floating-point drift).
   - Recommendation: In BIDS-02, verify epoch count rather than exact event
     timing. Check `isinstance(epochs, mne.Epochs)` and `len(epochs) > 0`.

## Sources

### Primary (HIGH confidence)
- [MNE tutorial: Creating data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html) -- RawArray, EpochsArray creation patterns
- [MNE API: CrossSpectralDensity](https://mne.tools/stable/generated/mne.time_frequency.CrossSpectralDensity.html) -- CSD object interface, get_data() method
- [MNE API: csd_multitaper](https://mne.tools/stable/generated/mne.time_frequency.csd_multitaper.html) -- Function signature, parameter types
- [MNE API: extract_label_time_course](https://mne.tools/stable/generated/mne.extract_label_time_course.html) -- Return shape (n_labels, n_times), mode options
- [MNE API: Annotations](https://mne.tools/stable/generated/mne.Annotations.html) -- Constructor, BAD_ACQ_SKIP behavior
- [MNE-BIDS API: write_raw_bids](https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html) -- Function signature, events handling
- [MNE: Handling bad channels](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html) -- pick() vs pick_types() exclude behavior
- [pytest: skipping](https://docs.pytest.org/en/stable/how-to/skipping.html) -- importorskip, module-level skip

### Secondary (MEDIUM confidence)
- [MNE issue #12577](https://github.com/mne-tools/mne-python/issues/12577) -- Epochs vs Evoked bad channel inconsistency (confirmed bug)
- [MNE-BIDS example: convert_mne_sample](https://mne.tools/mne-bids/stable/auto_examples/convert_mne_sample.html) -- BIDS round-trip pattern

### Tertiary (LOW confidence)
- None -- all findings verified with official documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are already project dependencies, APIs verified with official docs
- Architecture: HIGH -- patterns derived from official MNE tutorials and pytest documentation
- Pitfalls: HIGH -- CSD frequency mismatch and bad-channel inconsistency confirmed via official docs and issue tracker
- Code examples: HIGH -- all patterns verified against official tutorials; sine-injection approach is standard signal processing

**Research date:** 2026-05-21
**Valid until:** 2026-07-21 (stable -- MNE 1.6+ API is mature, pytest patterns are stable)
