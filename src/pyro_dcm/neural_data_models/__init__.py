"""Neural data models for MEG/EEG timeseries (Phase 22, v0.6.0).

Provides temporal autoencoder models whose latent dynamics are analyzed
by spectral DCM for interpretability.
"""
from __future__ import annotations

from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder

__all__ = [
    "MEGAutoencoder",
]
