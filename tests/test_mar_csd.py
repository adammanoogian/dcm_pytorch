"""Tests for the CSD <-> MAR round-trip (spm_mar2csd / spm_csd2mar parity)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.mar_csd import ccf2mar, csd2ccf, mar2csd
from pyro_dcm.forward_models.spectral_transfer import (
    csd_mar_roundtrip,
    default_frequency_grid,
)


def _synthetic_csd(freqs: np.ndarray, n: int = 2) -> np.ndarray:
    """A smooth, Hermitian, positive-definite CSD independent of any engine."""
    csd = np.zeros((len(freqs), n, n), dtype=np.complex128)
    for k, f in enumerate(freqs):
        v = np.array([1.0, 0.6])[:n] / (1.0 + f)
        csd[k] = np.outer(v, v.conj()) + 0.1 * np.eye(n)
    return csd


class TestMarRoundTripScale:
    """The CSD -> MAR -> CSD round-trip must preserve scale.

    SPM's ``spm_mar2csd.m`` ends with ``csd = 2*csd/ns`` (ns = 2*freqs(end)).
    Omitting that normalisation makes the reconstructed CSD too small by
    ``2/ns`` -- a factor of 4 on the standard fMRI grid (freqs(end) = 0.25 Hz).

    That defect made the VL spectral forward predict exactly 1/4 of SPM's CSD
    at identical parameters, which was the residual VL-vs-SPM free-energy
    discrepancy (~52 nats) after the Y.Q noise-model mismatch was fixed.
    """

    def test_roundtrip_preserves_scale(self) -> None:
        """A smooth CSD must survive the round-trip at ~unit gain."""
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        csd = _synthetic_csd(freqs.numpy())
        out = csd_mar_roundtrip(
            torch.tensor(csd, dtype=torch.complex128), freqs, 7
        ).numpy()
        ratio = np.abs(out) / np.abs(csd)
        assert 0.7 < ratio.mean() < 1.3, (
            f"round-trip gain {ratio.mean():.4f} is not ~1.0 "
            f"(min {ratio.min():.4f}, max {ratio.max():.4f}). A gain near 0.25 "
            f"means mar2csd is missing SPM's 2/ns normalisation."
        )

    def test_mar2csd_applies_spm_normalisation(self) -> None:
        """mar2csd must scale by 2/fs, matching spm_mar2csd.m."""
        freqs_t = default_frequency_grid(TR=2.0, n_freqs=32)
        freqs = freqs_t.numpy()
        fs = 2.0 * freqs[-1]
        csd = _synthetic_csd(freqs)
        mar = ccf2mar(csd2ccf(csd, freqs), 7)

        got = mar2csd(mar, freqs, fs)

        # Reconstruct the unnormalised transfer form and apply 2/fs by hand.
        n = csd.shape[1]
        eye = np.eye(n)
        expect = np.zeros_like(got)
        for fi, f in enumerate(freqs):
            h_inv = eye.astype(complex)
            for k, a_k in enumerate(mar["A"]):
                h_inv = h_inv - a_k * np.exp(-2j * np.pi * f * (k + 1) / fs)
            h = np.linalg.inv(h_inv)
            expect[fi] = h @ mar["noise_cov"] @ h.conj().T
        expect = 2.0 * expect / fs

        assert np.allclose(got, expect, rtol=1e-10, atol=1e-14), (
            f"mar2csd does not match the spm_mar2csd 2/ns form; "
            f"max abs diff {np.abs(got - expect).max():.3e}"
        )

    @pytest.mark.parametrize("f_end", [0.25, 0.5])
    def test_normalisation_tracks_fs_not_a_fixed_four(self, f_end: float) -> None:
        """The factor is 2/fs, so it must move with the frequency grid."""
        freqs = np.linspace(f_end / 32, f_end, 32)
        fs = 2.0 * freqs[-1]
        csd = _synthetic_csd(freqs)
        mar = ccf2mar(csd2ccf(csd, freqs), 7)
        out = mar2csd(mar, freqs, fs)
        assert np.isfinite(out).all()
        # Scale check against the same model evaluated at a doubled fs.
        out2 = mar2csd(mar, freqs, 2.0 * fs)
        ratio = np.abs(out).max() / np.abs(out2).max()
        assert ratio > 1.0, (
            "halving fs must increase the 2/fs normalisation; "
            f"got ratio {ratio:.4f}"
        )
