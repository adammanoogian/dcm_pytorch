"""Load MNE-Python objects into Pyro-DCM tensor formats.

Converts MNE Epochs, Raw, and SourceEstimate objects into the PyTorch tensor
formats expected by the spectral DCM (CSD matrices) and task DCM (time series)
models.

Tensor shape conventions follow the project standard (see CLAUDE.md):
    - CSD: ``(F, N, N)`` complex -- frequency bins x regions x regions
    - Time series: ``(T, N)`` float -- time points x regions
    - Frequency vector: ``(F,)`` float -- Hz
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import mne


def _require_mne() -> None:
    try:
        import mne as _mne  # noqa: F401
    except ImportError:
        raise ImportError(
            "MNE-Python is required for this function. "
            "Install with: pip install pyro-dcm[mne]"
        ) from None


def epochs_to_csd(
    epochs: mne.Epochs,
    *,
    fmin: float = 1.0 / 128.0,
    fmax: float | None = None,
    n_freqs: int = 32,
    method: str = "multitaper",
    picks: str | list[str] | None = None,
    dtype: torch.dtype = torch.complex128,
) -> dict[str, torch.Tensor]:
    """Compute cross-spectral density from MNE Epochs for spectral DCM.

    Wraps ``mne.time_frequency.csd_multitaper`` or ``csd_morlet`` to produce
    a CSD tensor in the format expected by ``spectral_dcm_model``.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched M/EEG data. Channels selected via ``picks``.
    fmin : float
        Minimum frequency in Hz. Default ``1/128`` matches SPM12 convention.
    fmax : float or None
        Maximum frequency in Hz. If None, uses Nyquist ``sfreq / 2``.
    n_freqs : int
        Number of frequency bins. Default 32 matches
        ``spectral_transfer.default_frequency_grid``.
    method : str
        CSD estimation method: ``'multitaper'`` (default) or ``'morlet'``.
    picks : str, list of str, or None
        Channel selection (MNE picks syntax). If None, uses all good data
        channels.
    dtype : torch.dtype
        Output complex dtype. Default ``torch.complex128`` matches spectral
        DCM convention.

    Returns
    -------
    dict
        Keys:

        - ``'csd'``: CSD tensor, shape ``(F, N, N)``, complex.
        - ``'freqs'``: Frequency vector, shape ``(F,)``, float64.
        - ``'ch_names'``: list of str, channel names (length N).
        - ``'sfreq'``: float, sampling frequency in Hz.
        - ``'n_epochs'``: int, number of epochs averaged.
    """
    _require_mne()
    from mne.time_frequency import csd_morlet, csd_multitaper

    if fmax is None:
        fmax = epochs.info["sfreq"] / 2.0

    freqs_target = np.linspace(fmin, fmax, n_freqs)

    if method == "multitaper":
        csd_obj = csd_multitaper(
            epochs,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
            n_jobs=1,
        )
    elif method == "morlet":
        csd_obj = csd_morlet(
            epochs,
            frequencies=freqs_target,
            picks=picks,
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"method must be 'multitaper' or 'morlet', got '{method}'"
        )

    csd_freqs = csd_obj.frequencies
    n_channels = len(csd_obj.ch_names)
    n_freq_bins = len(csd_freqs)
    csd_array = np.zeros(
        (n_freq_bins, n_channels, n_channels), dtype=np.complex128
    )

    for fi, freq in enumerate(csd_freqs):
        csd_at_freq = csd_obj.get_data(frequency=freq)
        csd_array[fi] = csd_at_freq

    if n_freq_bins != n_freqs:
        csd_interp = np.zeros(
            (n_freqs, n_channels, n_channels), dtype=np.complex128
        )
        for i in range(n_channels):
            for j in range(n_channels):
                real_interp = np.interp(
                    freqs_target, csd_freqs, csd_array[:, i, j].real
                )
                imag_interp = np.interp(
                    freqs_target, csd_freqs, csd_array[:, i, j].imag
                )
                csd_interp[:, i, j] = real_interp + 1j * imag_interp
        csd_array = csd_interp
        csd_freqs = freqs_target

    return {
        "csd": torch.tensor(csd_array, dtype=dtype),
        "freqs": torch.tensor(csd_freqs, dtype=torch.float64),
        "ch_names": list(csd_obj.ch_names),
        "sfreq": float(epochs.info["sfreq"]),
        "n_epochs": len(epochs),
    }


def epochs_to_timeseries(
    epochs: mne.Epochs,
    *,
    picks: str | list[str] | None = None,
    average: bool = True,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Extract time series from MNE Epochs for task DCM.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched M/EEG data.
    picks : str, list of str, or None
        Channel selection. If None, uses all good data channels.
    average : bool
        If True, return the evoked (trial-averaged) time series.
        If False, return all trials stacked: shape ``(n_epochs, T, N)``.
    dtype : torch.dtype
        Output dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Keys:

        - ``'timeseries'``: shape ``(T, N)`` if averaged, ``(n_epochs, T, N)``
          otherwise.
        - ``'times'``: shape ``(T,)``, time vector in seconds.
        - ``'ch_names'``: list of str, channel names.
        - ``'sfreq'``: float, sampling frequency.
    """
    _require_mne()

    epochs_picked = epochs.copy().pick(picks) if picks is not None else epochs
    ch_names = list(epochs_picked.ch_names)

    if average:
        evoked = epochs_picked.average()
        data = evoked.data.T  # (T, N)
        times = evoked.times
    else:
        data = epochs_picked.get_data()  # (n_epochs, N, T)
        data = np.transpose(data, (0, 2, 1))  # (n_epochs, T, N)
        times = epochs_picked.times

    return {
        "timeseries": torch.tensor(data, dtype=dtype),
        "times": torch.tensor(times, dtype=torch.float64),
        "ch_names": ch_names,
        "sfreq": float(epochs_picked.info["sfreq"]),
    }


def raw_to_timeseries(
    raw: mne.io.BaseRaw,
    *,
    picks: str | list[str] | None = None,
    start: float = 0.0,
    stop: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Extract continuous time series from MNE Raw for task DCM.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw M/EEG data.
    picks : str, list of str, or None
        Channel selection. If None, uses all good data channels.
    start : float
        Start time in seconds.
    stop : float or None
        Stop time in seconds. If None, reads to end.
    dtype : torch.dtype
        Output dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Keys:

        - ``'timeseries'``: shape ``(T, N)``, float.
        - ``'times'``: shape ``(T,)``, time vector in seconds.
        - ``'ch_names'``: list of str, channel names.
        - ``'sfreq'``: float, sampling frequency.
    """
    _require_mne()

    raw_picked = raw.copy().pick(picks) if picks is not None else raw
    sfreq = raw_picked.info["sfreq"]
    start_samp = int(start * sfreq)
    stop_samp = int(stop * sfreq) if stop is not None else None

    data, times = raw_picked.get_data(
        start=start_samp, stop=stop_samp, return_times=True
    )
    data = data.T  # (T, N)

    return {
        "timeseries": torch.tensor(data, dtype=dtype),
        "times": torch.tensor(times, dtype=torch.float64),
        "ch_names": list(raw_picked.ch_names),
        "sfreq": float(sfreq),
    }


def stc_to_roi_timeseries(
    stc: mne.SourceEstimate,
    labels: list[mne.Label],
    src: mne.SourceSpaces,
    *,
    mode: str = "mean_flip",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Extract ROI time series from source estimate for task DCM.

    Extracts label time courses from a source estimate, producing the
    ``(T, N)`` format expected by task DCM where N = number of ROIs.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source-space activation estimate.
    labels : list of mne.Label
        Anatomical labels defining ROIs (e.g., from an atlas).
    src : mne.SourceSpaces
        Source space used for the inverse solution.
    mode : str
        Extraction mode passed to ``mne.extract_label_time_course``.
        Default ``'mean_flip'`` (sign-flipped mean within label).
    dtype : torch.dtype
        Output dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Keys:

        - ``'timeseries'``: shape ``(T, N)``, float. N = len(labels).
        - ``'times'``: shape ``(T,)``, time vector in seconds.
        - ``'roi_names'``: list of str, label names.
        - ``'sfreq'``: float, sampling frequency (1 / tstep).
    """
    _require_mne()
    import mne as _mne

    label_ts = _mne.extract_label_time_course(
        stc, labels, src, mode=mode
    )  # (N, T)
    label_ts = label_ts.T  # (T, N)

    return {
        "timeseries": torch.tensor(label_ts, dtype=dtype),
        "times": torch.tensor(stc.times, dtype=torch.float64),
        "roi_names": [label.name for label in labels],
        "sfreq": 1.0 / stc.tstep,
    }
