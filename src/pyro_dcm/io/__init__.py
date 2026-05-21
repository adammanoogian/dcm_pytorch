"""I/O utilities for loading neuroimaging data into Pyro-DCM tensor formats.

Provides loaders for MNE-Python objects (Raw, Epochs, SourceEstimate) and
BIDS-formatted datasets (via mne-bids). All loaders return dictionaries of
PyTorch tensors matching the shapes expected by the spectral and task DCM
models.

Requires optional dependencies: ``pip install pyro-dcm[mne]``
"""

from __future__ import annotations

from pyro_dcm.io.mne_loader import (
    epochs_to_csd,
    epochs_to_timeseries,
    raw_to_timeseries,
    stc_to_roi_timeseries,
)

__all__ = [
    "epochs_to_csd",
    "epochs_to_timeseries",
    "raw_to_timeseries",
    "stc_to_roi_timeseries",
]

try:
    from pyro_dcm.io.bids_loader import (
        load_bids_epochs,
        load_bids_raw,
    )

    __all__ += ["load_bids_raw", "load_bids_epochs"]
except ImportError:
    pass
