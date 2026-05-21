"""Load BIDS-formatted M/EEG datasets via mne-bids.

Wraps ``mne_bids.read_raw_bids`` to provide a clean interface for loading
BIDS datasets into MNE objects, which can then be passed to the MNE loader
functions (``epochs_to_csd``, ``epochs_to_timeseries``, etc.).

Requires: ``pip install pyro-dcm[mne]`` (includes mne-bids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne
    import mne_bids


def _require_mne_bids() -> None:
    try:
        import mne_bids as _mb  # noqa: F401
    except ImportError:
        raise ImportError(
            "mne-bids is required for BIDS loading. "
            "Install with: pip install pyro-dcm[mne]"
        ) from None


def load_bids_raw(
    bids_path: mne_bids.BIDSPath | dict,
    *,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Load a raw M/EEG recording from a BIDS dataset.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath or dict
        BIDS path object or dict of keyword arguments to construct one.
        Dict keys: ``subject``, ``session``, ``task``, ``run``,
        ``datatype``, ``root``, etc.
    verbose : bool
        MNE verbosity flag.

    Returns
    -------
    mne.io.BaseRaw
        Raw MNE object with BIDS metadata applied.

    Examples
    --------
    >>> from pyro_dcm.io import load_bids_raw
    >>> raw = load_bids_raw({
    ...     "subject": "01",
    ...     "task": "rest",
    ...     "datatype": "eeg",
    ...     "root": "/data/bids_dataset",
    ... })
    """
    _require_mne_bids()
    import mne_bids

    if isinstance(bids_path, dict):
        bids_path = mne_bids.BIDSPath(**bids_path)

    raw = mne_bids.read_raw_bids(bids_path, verbose=verbose)
    return raw


def load_bids_epochs(
    bids_path: mne_bids.BIDSPath | dict,
    *,
    event_id: dict[str, int] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: tuple[float | None, float] | None = (None, 0),
    picks: str | list[str] | None = None,
    reject: dict[str, float] | None = None,
    verbose: bool = False,
) -> mne.Epochs:
    """Load a BIDS recording and epoch it around events.

    Convenience function that loads raw BIDS data, finds events from
    annotations (BIDS standard), and creates MNE Epochs ready for
    DCM analysis.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath or dict
        BIDS path (see ``load_bids_raw``).
    event_id : dict or None
        Mapping of event names to integer codes. If None, uses all
        events found in annotations.
    tmin : float
        Epoch start time relative to event (seconds). Default -0.2.
    tmax : float
        Epoch end time relative to event (seconds). Default 0.5.
    baseline : tuple or None
        Baseline correction window. Default ``(None, 0)`` = start to
        event onset. Pass None to skip baseline correction.
    picks : str, list of str, or None
        Channel selection.
    reject : dict or None
        Peak-to-peak rejection thresholds per channel type.
        Example: ``{"eeg": 100e-6, "mag": 4000e-13}``.
    verbose : bool
        MNE verbosity flag.

    Returns
    -------
    mne.Epochs
        Epoched data ready for ``epochs_to_csd`` or ``epochs_to_timeseries``.
    """
    _require_mne_bids()
    import mne

    raw = load_bids_raw(bids_path, verbose=verbose)

    events, auto_event_id = mne.events_from_annotations(raw, verbose=verbose)
    if event_id is None:
        event_id = auto_event_id

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        reject=reject,
        preload=True,
        verbose=verbose,
    )

    return epochs
