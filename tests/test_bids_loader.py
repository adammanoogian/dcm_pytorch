"""Tests for BIDS IO loader functions.

Round-trip tests that write synthetic EEG data to BIDS format via
``mne_bids.write_raw_bids`` and read it back via ``load_bids_raw``
and ``load_bids_epochs``.  Covers channel/sample-rate preservation,
event extraction, and BAD_ACQ_SKIP annotation handling.

Both ``mne`` and ``mne_bids`` are imported via ``pytest.importorskip``
so the entire file is skipped when either package is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
mne_bids = pytest.importorskip("mne_bids")

from pyro_dcm.io.bids_loader import load_bids_epochs, load_bids_raw  # noqa: E402

pytestmark = pytest.mark.mne


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_bids_dataset(
    tmp_path: Path,
    *,
    ch_names: list[str],
    sfreq: float = 256.0,
    duration_s: float = 10.0,
    annotations: mne.Annotations | None = None,
    seed: int = 42,
) -> mne_bids.BIDSPath:
    """Write a minimal synthetic BIDS dataset and return the BIDSPath.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory (pytest built-in fixture).
    ch_names : list[str]
        EEG channel names.
    sfreq : float
        Sampling frequency in Hz.
    duration_s : float
        Duration of the recording in seconds.
    annotations : mne.Annotations or None
        Optional annotations to attach before writing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mne_bids.BIDSPath
        Path to the written BIDS dataset.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration_s)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )
    info["line_freq"] = 50.0  # BIDS requirement

    data = rng.standard_normal((len(ch_names), n_samples)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)

    if annotations is not None:
        raw.set_annotations(annotations)

    bids_path = mne_bids.BIDSPath(
        subject="01",
        task="test",
        datatype="eeg",
        root=str(tmp_path),
    )
    mne_bids.write_raw_bids(
        raw,
        bids_path,
        allow_preload=True,
        format="BrainVision",
        overwrite=True,
        verbose=False,
    )
    return bids_path


# ---------------------------------------------------------------------------
# BIDS-01: load raw round-trip
# ---------------------------------------------------------------------------


def test_load_bids_raw(tmp_path: Path) -> None:
    """BIDS-01: write_raw_bids -> load_bids_raw preserves channels/sfreq."""
    ch_names = ["EEG1", "EEG2", "EEG3"]
    bids_path = _make_bids_dataset(
        tmp_path,
        ch_names=ch_names,
        sfreq=256.0,
        duration_s=10.0,
    )

    raw_loaded = load_bids_raw(bids_path)

    assert isinstance(raw_loaded, mne.io.BaseRaw), (
        f"Expected mne.io.BaseRaw, got {type(raw_loaded)}"
    )
    loaded_names = raw_loaded.ch_names
    for ch in ch_names:
        assert ch in loaded_names, (
            f"Channel {ch!r} missing; loaded channels: {loaded_names}"
        )
    assert raw_loaded.info["sfreq"] == 256.0, (
        f"Expected sfreq=256.0, got {raw_loaded.info['sfreq']}"
    )
    assert raw_loaded.n_times > 0, "Loaded raw has no time points"


# ---------------------------------------------------------------------------
# BIDS-02: load epochs round-trip
# ---------------------------------------------------------------------------


def test_load_bids_epochs(tmp_path: Path) -> None:
    """BIDS-02: write_raw_bids -> load_bids_epochs creates valid Epochs."""
    ch_names = ["EEG1", "EEG2"]
    onsets = np.arange(1.0, 25.0, 3.0)
    annotations = mne.Annotations(
        onset=onsets,
        duration=np.zeros(len(onsets)),
        description=["stimulus"] * len(onsets),
    )

    bids_path = _make_bids_dataset(
        tmp_path,
        ch_names=ch_names,
        sfreq=256.0,
        duration_s=30.0,
        annotations=annotations,
    )

    epochs = load_bids_epochs(
        bids_path,
        event_id=None,
        tmin=-0.1,
        tmax=0.4,
    )

    assert isinstance(epochs, mne.Epochs), f"Expected mne.Epochs, got {type(epochs)}"
    assert len(epochs) > 0, "No epochs created from stimulus annotations"
    assert epochs.info["sfreq"] == 256.0, (
        f"Expected sfreq=256.0, got {epochs.info['sfreq']}"
    )


# ---------------------------------------------------------------------------
# BIDS-03: BAD_ACQ_SKIP annotation handling
# ---------------------------------------------------------------------------


def test_bids_bad_acq_skip_annotation(tmp_path: Path) -> None:
    """BIDS-03: BAD_ACQ_SKIP annotation does not crash epoch creation."""
    ch_names = ["EEG1", "EEG2"]
    event_onsets = np.arange(1.0, 25.0, 3.0)

    annotations = mne.Annotations(
        onset=list(event_onsets) + [10.0],
        duration=[0.0] * len(event_onsets) + [2.0],
        description=["stimulus"] * len(event_onsets) + ["BAD_ACQ_SKIP"],
    )

    bids_path = _make_bids_dataset(
        tmp_path,
        ch_names=ch_names,
        sfreq=256.0,
        duration_s=30.0,
        annotations=annotations,
    )

    epochs = load_bids_epochs(
        bids_path,
        event_id=None,
        tmin=-0.1,
        tmax=0.4,
    )

    assert isinstance(epochs, mne.Epochs), f"Expected mne.Epochs, got {type(epochs)}"
    # Primary assertion: at least some epochs survive the BAD segment
    assert len(epochs) >= 1, (
        f"Expected at least 1 epoch after BAD_ACQ_SKIP rejection, got {len(epochs)}"
    )
