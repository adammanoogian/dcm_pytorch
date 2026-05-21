"""Tests for MNE-Python IO loader functions.

Validates shape, dtype, mathematical properties (CSD Hermitian symmetry,
non-negative auto-spectra, sine injection peak), channel subsetting, bad
channel behavior, and error paths for all four MNE loader functions.

TEST-12: ``pytest.importorskip("mne")`` at module level ensures the
entire file is skipped when MNE-Python is not installed.

TEST-13: ``pytestmark = pytest.mark.mne`` applies the ``mne`` marker
to every test in this file for selective execution via ``pytest -m mne``.
"""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

mne = pytest.importorskip("mne")

from pyro_dcm.io.mne_loader import (  # noqa: E402
    _require_mne,
    epochs_to_csd,
    epochs_to_timeseries,
    raw_to_timeseries,
    stc_to_roi_timeseries,
)

pytestmark = pytest.mark.mne


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_epochs() -> mne.EpochsArray:
    """Create 10 epochs, 3 EEG channels, 256 Hz, 1.0s duration.

    Data is random * 1e-6 (Volts). Seed=42.
    """
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_epochs = 10
    n_channels = 3
    n_times = int(sfreq * 1.0)  # 256 samples

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    # shape: (n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    return mne.EpochsArray(data, info)


@pytest.fixture()
def synthetic_raw() -> mne.io.RawArray:
    """Create Raw with 3 EEG channels, 256 Hz, 10 seconds.

    Data is random * 1e-6 (Volts). Seed=42.
    """
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_channels = 3
    n_times = int(sfreq * 10.0)  # 2560 samples

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    # shape: (n_channels, n_times)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6
    return mne.io.RawArray(data, info)


@pytest.fixture()
def sine_epochs() -> mne.EpochsArray:
    """Create epochs with 10 Hz sine wave for CSD peak detection.

    20 epochs, 2 EEG channels, 256 Hz, 2.0s per epoch.
    Both channels: 10 Hz sine + small noise (0.01 * randn).
    Data * 1e-6 (Volts). Seed=42.
    """
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_epochs = 20
    n_channels = 2
    duration = 2.0
    n_times = int(sfreq * duration)  # 512 samples

    info = mne.create_info(
        ch_names=["EEG1", "EEG2"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    t = np.arange(n_times) / sfreq
    sine_10hz = np.sin(2.0 * np.pi * 10.0 * t)

    data = np.zeros((n_epochs, n_channels, n_times))
    for ep in range(n_epochs):
        for ch in range(n_channels):
            noise = 0.01 * rng.standard_normal(n_times)
            data[ep, ch, :] = (sine_10hz + noise) * 1e-6

    return mne.EpochsArray(data, info)


# ---------------------------------------------------------------------------
# TEST-01: epochs_to_csd shape
# ---------------------------------------------------------------------------


def test_epochs_to_csd_shape(synthetic_epochs: mne.EpochsArray) -> None:
    """Verify CSD output shape, dtype, and metadata."""
    result = epochs_to_csd(synthetic_epochs, fmin=1.0, fmax=50.0, n_freqs=32)

    csd = result["csd"]
    freqs = result["freqs"]

    assert csd.shape == (32, 3, 3), f"Expected CSD shape (32, 3, 3), got {csd.shape}"
    assert csd.is_complex(), "CSD tensor must be complex"
    assert freqs.shape == (32,), f"Expected freqs shape (32,), got {freqs.shape}"
    assert len(result["ch_names"]) == 3, (
        f"Expected 3 channel names, got {len(result['ch_names'])}"
    )
    assert result["sfreq"] == 256.0, f"Expected sfreq=256.0, got {result['sfreq']}"
    assert result["n_epochs"] == 10, f"Expected n_epochs=10, got {result['n_epochs']}"


# ---------------------------------------------------------------------------
# TEST-02: epochs_to_timeseries shape (averaged and unaveraged)
# ---------------------------------------------------------------------------


def test_epochs_to_timeseries_shape_averaged(
    synthetic_epochs: mne.EpochsArray,
) -> None:
    """Verify averaged epochs produce (T, N) float tensor."""
    result = epochs_to_timeseries(synthetic_epochs, average=True)

    ts = result["timeseries"]
    assert ts.shape == (256, 3), f"Expected timeseries shape (256, 3), got {ts.shape}"
    assert ts.dtype == torch.float64, f"Expected dtype float64, got {ts.dtype}"
    assert result["times"].shape == (256,), (
        f"Expected times shape (256,), got {result['times'].shape}"
    )
    assert len(result["ch_names"]) == 3


def test_epochs_to_timeseries_shape_unaveraged(
    synthetic_epochs: mne.EpochsArray,
) -> None:
    """Verify unaveraged epochs produce (n_epochs, T, N) float tensor."""
    result = epochs_to_timeseries(synthetic_epochs, average=False)

    ts = result["timeseries"]
    assert ts.shape == (10, 256, 3), (
        f"Expected timeseries shape (10, 256, 3), got {ts.shape}"
    )
    assert ts.dtype == torch.float64, f"Expected dtype float64, got {ts.dtype}"
    assert result["times"].shape == (256,), (
        f"Expected times shape (256,), got {result['times'].shape}"
    )
    assert len(result["ch_names"]) == 3


# ---------------------------------------------------------------------------
# TEST-03: raw_to_timeseries shape
# ---------------------------------------------------------------------------


def test_raw_to_timeseries_shape(
    synthetic_raw: mne.io.RawArray,
) -> None:
    """Verify Raw produces (T, N) float tensor."""
    result = raw_to_timeseries(synthetic_raw)

    ts = result["timeseries"]
    assert ts.shape == (2560, 3), f"Expected timeseries shape (2560, 3), got {ts.shape}"
    assert ts.dtype == torch.float64, f"Expected dtype float64, got {ts.dtype}"
    assert result["times"].shape == (2560,), (
        f"Expected times shape (2560,), got {result['times'].shape}"
    )
    assert len(result["ch_names"]) == 3
    assert result["sfreq"] == 256.0, f"Expected sfreq=256.0, got {result['sfreq']}"


# ---------------------------------------------------------------------------
# TEST-04: stc_to_roi_timeseries shape
# ---------------------------------------------------------------------------


def test_stc_to_roi_timeseries_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SourceEstimate ROI extraction produces (T, N) tensor."""
    rng = np.random.default_rng(42)
    n_rois = 3
    n_times = 100

    # Mock SourceEstimate
    mock_stc = MagicMock()
    mock_stc.times = np.linspace(0, 1, n_times)
    mock_stc.tstep = 1.0 / 256.0

    # Mock labels -- set .name as attribute, NOT via MagicMock(name=...)
    mock_labels = []
    for i in range(n_rois):
        label = MagicMock()
        label.name = f"ROI_{i}"
        mock_labels.append(label)

    mock_src = MagicMock()

    def mock_extract(stc, labels, src, mode="mean_flip"):
        return rng.standard_normal((n_rois, n_times))

    monkeypatch.setattr(mne, "extract_label_time_course", mock_extract)

    result = stc_to_roi_timeseries(mock_stc, mock_labels, mock_src)

    ts = result["timeseries"]
    assert ts.shape == (100, 3), f"Expected timeseries shape (100, 3), got {ts.shape}"
    assert ts.dtype == torch.float64, f"Expected dtype float64, got {ts.dtype}"
    assert result["times"].shape == (100,), (
        f"Expected times shape (100,), got {result['times'].shape}"
    )
    assert result["roi_names"] == ["ROI_0", "ROI_1", "ROI_2"], (
        f"Expected ROI names ['ROI_0', 'ROI_1', 'ROI_2'], got {result['roi_names']}"
    )
    assert result["sfreq"] == 256.0, f"Expected sfreq=256.0, got {result['sfreq']}"


# ---------------------------------------------------------------------------
# TEST-05: channel picks subsetting
# ---------------------------------------------------------------------------


def test_channel_picks_subsetting() -> None:
    """Verify explicit picks reduce N dimension correctly."""
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_epochs = 5
    n_channels = 5
    n_times = int(sfreq * 1.0)

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4", "EEG5"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    epochs = mne.EpochsArray(data, info)

    # Pick 2 of 5 channels
    result = epochs_to_timeseries(epochs, picks=["EEG1", "EEG3"])
    assert result["timeseries"].shape[1] == 2, (
        f"Expected N=2 after picking 2 channels, got {result['timeseries'].shape[1]}"
    )
    assert result["ch_names"] == ["EEG1", "EEG3"]

    # Also test raw picks
    raw_info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4", "EEG5"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw_data = rng.standard_normal((n_channels, n_times * 10)) * 1e-6
    raw = mne.io.RawArray(raw_data, raw_info)

    result_raw = raw_to_timeseries(raw, picks=["EEG2"])
    assert result_raw["timeseries"].shape[1] == 1, (
        f"Expected N=1 after picking 1 channel, got {result_raw['timeseries'].shape[1]}"
    )


# ---------------------------------------------------------------------------
# TEST-06: bad channel exclusion behavior
# ---------------------------------------------------------------------------


def test_bad_channel_exclusion() -> None:
    """Document bad channel behavior with picks=None vs explicit picks.

    With picks=None, the raw/epochs functions pass through all channels
    (bads are NOT automatically excluded by our wrappers). With explicit
    picks, only the requested channels appear.
    """
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_epochs = 5
    n_channels = 3
    n_times = int(sfreq * 1.0)

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    epochs = mne.EpochsArray(data, info)

    # Mark EEG2 as bad
    epochs.info["bads"] = ["EEG2"]

    # With picks=None: all 3 channels pass through
    result_none = epochs_to_timeseries(epochs, picks=None)
    assert result_none["timeseries"].shape[1] == 3, (
        f"Expected 3 channels with picks=None (bads not excluded), "
        f"got {result_none['timeseries'].shape[1]}"
    )

    # With explicit picks excluding bad: 2 channels
    result_explicit = epochs_to_timeseries(epochs, picks=["EEG1", "EEG3"])
    assert result_explicit["timeseries"].shape[1] == 2, (
        f"Expected 2 channels with explicit picks, "
        f"got {result_explicit['timeseries'].shape[1]}"
    )


# ---------------------------------------------------------------------------
# TEST-07: CSD Hermitian symmetry
# ---------------------------------------------------------------------------


def test_csd_hermitian_symmetry(
    synthetic_epochs: mne.EpochsArray,
) -> None:
    """Verify CSD satisfies csd[f,i,j] == conj(csd[f,j,i])."""
    result = epochs_to_csd(synthetic_epochs, fmin=1.0, fmax=50.0, n_freqs=32)
    csd = result["csd"]

    csd_conj_transpose = csd.conj().transpose(-2, -1)
    assert torch.allclose(csd, csd_conj_transpose, atol=1e-10), (
        "CSD matrix is not Hermitian: "
        f"max deviation = {(csd - csd_conj_transpose).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# TEST-08: CSD non-negative auto-spectra
# ---------------------------------------------------------------------------


def test_csd_nonnegative_autospectra(
    synthetic_epochs: mne.EpochsArray,
) -> None:
    """Verify CSD diagonal is real and non-negative."""
    result = epochs_to_csd(synthetic_epochs, fmin=1.0, fmax=50.0, n_freqs=32)
    csd = result["csd"]
    n_channels = csd.shape[1]

    for i in range(n_channels):
        auto_spectrum = csd[:, i, i]

        # Real part must be non-negative
        min_real = auto_spectrum.real.min().item()
        assert min_real >= -1e-12, (
            f"Channel {i} auto-spectrum has negative real part: min = {min_real}"
        )

        # Imaginary part must be near zero
        assert torch.allclose(
            auto_spectrum.imag,
            torch.zeros_like(auto_spectrum.imag),
            atol=1e-10,
        ), (
            f"Channel {i} auto-spectrum has non-zero imaginary part: "
            f"max |imag| = {auto_spectrum.imag.abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# TEST-09: CSD sine injection round-trip
# ---------------------------------------------------------------------------


def test_csd_sine_injection_roundtrip(
    sine_epochs: mne.EpochsArray,
) -> None:
    """Verify 10 Hz sine produces CSD peak at 10 Hz bin (+/- 1 bin)."""
    result = epochs_to_csd(sine_epochs, fmin=1.0, fmax=50.0, n_freqs=50)
    csd = result["csd"]
    freqs = result["freqs"]

    # Find target and peak bins
    target_idx = torch.argmin(torch.abs(freqs - 10.0)).item()
    peak_idx = torch.argmax(csd[:, 0, 0].real).item()

    assert abs(peak_idx - target_idx) <= 1, (
        f"CSD peak at bin {peak_idx} "
        f"(freq={freqs[peak_idx].item():.2f} Hz) "
        f"is more than 1 bin from target bin {target_idx} "
        f"(freq={freqs[target_idx].item():.2f} Hz)"
    )


# ---------------------------------------------------------------------------
# TEST-10: _require_mne ImportError
# ---------------------------------------------------------------------------


def test_require_mne_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _require_mne raises ImportError with install instructions."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mne":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match=r"pip install pyro-dcm\[mne\]"):
        _require_mne()


# ---------------------------------------------------------------------------
# TEST-11: epochs_to_csd invalid method
# ---------------------------------------------------------------------------


def test_epochs_to_csd_invalid_method(
    synthetic_epochs: mne.EpochsArray,
) -> None:
    """Verify ValueError for invalid CSD method argument."""
    with pytest.raises(ValueError, match="method must be"):
        epochs_to_csd(synthetic_epochs, method="invalid")
