"""Unit tests for empirical CSD computation from BOLD time series.

Tests validate:
- Correct output shape and dtype
- Hermitian symmetry of CSD matrices
- Real, positive auto-spectra (diagonal)
- Approximately flat spectrum for white noise
- Peak detection for sinusoidal input
- Frequency interpolation to target grid
- Short time series handling
- Torch wrapper consistency with numpy
- Default Welch parameter selection
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.csd_computation import (
    bold_to_csd_torch,
    compute_empirical_csd,
    default_welch_params,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


class TestComputeEmpiricalCSD:
    """Tests for compute_empirical_csd."""

    def test_csd_shape(self, rng: np.random.Generator) -> None:
        """Output shape is (F, N, N) for given input dimensions."""
        T, N, F = 500, 3, 32
        bold = rng.standard_normal((T, N))
        freqs = np.linspace(0.01, 0.25, F)
        csd = compute_empirical_csd(bold, fs=0.5, freqs=freqs)
        assert csd.shape == (F, N, N)

    def test_csd_complex_dtype(self, rng: np.random.Generator) -> None:
        """Output dtype is complex128."""
        bold = rng.standard_normal((500, 3))
        freqs = np.linspace(0.01, 0.25, 32)
        csd = compute_empirical_csd(bold, fs=0.5, freqs=freqs)
        assert csd.dtype == np.complex128

    def test_csd_hermitian(self, rng: np.random.Generator) -> None:
        """CSD is Hermitian: csd[:, i, j] == conj(csd[:, j, i])."""
        T, N = 500, 4
        bold = rng.standard_normal((T, N))
        freqs = np.linspace(0.01, 0.25, 32)
        csd = compute_empirical_csd(bold, fs=0.5, freqs=freqs)

        for i in range(N):
            for j in range(N):
                np.testing.assert_allclose(
                    csd[:, i, j],
                    np.conj(csd[:, j, i]),
                    atol=1e-10,
                    err_msg=f"Hermitian violation at ({i}, {j})",
                )

    def test_csd_auto_spectra_real_positive(
        self, rng: np.random.Generator
    ) -> None:
        """Diagonal elements are real (negligible imag) and positive."""
        T, N = 500, 3
        bold = rng.standard_normal((T, N))
        freqs = np.linspace(0.01, 0.25, 32)
        csd = compute_empirical_csd(bold, fs=0.5, freqs=freqs)

        for i in range(N):
            auto = csd[:, i, i]
            # Imaginary part should be negligible
            assert np.max(np.abs(auto.imag)) < 1e-10, (
                f"Auto-spectrum [{i}] has non-negligible imaginary part"
            )
            # Real part should be positive
            assert np.all(auto.real > 0), (
                f"Auto-spectrum [{i}] has non-positive values"
            )

    def test_csd_white_noise_flat(self, rng: np.random.Generator) -> None:
        """White noise auto-spectrum is approximately flat.

        For white Gaussian noise with unit variance, the PSD should be
        approximately constant across frequency. We check that the
        coefficient of variation (CV = std/mean) is below 0.5 to allow
        for finite-sample Welch estimation noise.
        """
        T, N = 2000, 2
        fs = 0.5
        bold = rng.standard_normal((T, N))
        freqs = np.linspace(0.01, fs / 2 - 0.01, 64)
        csd = compute_empirical_csd(bold, fs=fs, freqs=freqs)

        for i in range(N):
            auto = csd[:, i, i].real
            cv = np.std(auto) / np.mean(auto)
            assert cv < 0.5, (
                f"Auto-spectrum [{i}] not flat: CV={cv:.3f} > 0.5"
            )

    def test_csd_sinusoidal_peak(self) -> None:
        """CSD of sinusoidal signal shows peak at input frequency.

        A 0.1 Hz sinusoid sampled at 0.5 Hz should produce a clear
        spectral peak near 0.1 Hz.
        """
        T = 1000
        fs = 0.5
        t = np.arange(T) / fs
        freq_signal = 0.1  # Hz
        bold = np.column_stack([
            np.sin(2 * np.pi * freq_signal * t),
            0.5 * np.sin(2 * np.pi * freq_signal * t + np.pi / 4),
        ])

        freqs = np.linspace(0.01, fs / 2 - 0.01, 128)
        csd = compute_empirical_csd(bold, fs=fs, freqs=freqs)

        # Check auto-spectrum of region 0
        auto = csd[:, 0, 0].real
        peak_idx = np.argmax(auto)
        peak_freq = freqs[peak_idx]

        # Peak should be within 1 frequency bin of 0.1 Hz
        freq_resolution = freqs[1] - freqs[0]
        assert abs(peak_freq - freq_signal) <= freq_resolution, (
            f"Peak at {peak_freq:.4f} Hz, expected ~{freq_signal} Hz "
            f"(resolution={freq_resolution:.4f} Hz)"
        )

    def test_csd_frequency_interpolation(
        self, rng: np.random.Generator
    ) -> None:
        """Interpolation to target grid produces reasonable values.

        Both coarse and fine grids interpolate from the same underlying
        Welch CSD. The coarse-grid values should match what we get by
        interpolating the fine-grid CSD back to the coarse frequencies,
        since both derive from the same raw Welch output.
        """
        T, N = 500, 2
        bold = rng.standard_normal((T, N))
        fs = 0.5

        # Coarse grid
        freqs_coarse = np.linspace(0.02, 0.23, 8)
        csd_coarse = compute_empirical_csd(bold, fs=fs, freqs=freqs_coarse)

        # Fine grid covering same range
        freqs_fine = np.linspace(0.02, 0.23, 64)
        csd_fine = compute_empirical_csd(bold, fs=fs, freqs=freqs_fine)

        # Interpolate fine-grid auto-spectrum to coarse grid points
        auto_fine = csd_fine[:, 0, 0].real
        auto_coarse = csd_coarse[:, 0, 0].real
        auto_fine_at_coarse = np.interp(
            freqs_coarse, freqs_fine, auto_fine
        )

        # Both derive from same Welch CSD via np.interp, so they
        # should agree well (not exactly, due to grid alignment)
        np.testing.assert_allclose(
            auto_coarse,
            auto_fine_at_coarse,
            rtol=0.15,
            err_msg="Coarse and fine interpolations disagree",
        )

        # Also verify fine grid has no NaN or negative auto-spectra
        assert np.all(np.isfinite(auto_fine))
        assert np.all(auto_fine > 0)

    def test_csd_short_time_series(
        self, rng: np.random.Generator
    ) -> None:
        """Short time series (T=64) handled without errors.

        nperseg should auto-adjust to min(256, T)=64.
        """
        T, N = 64, 2
        bold = rng.standard_normal((T, N))
        freqs = np.linspace(0.01, 0.24, 16)
        # Should not raise
        csd = compute_empirical_csd(bold, fs=0.5, freqs=freqs)
        assert csd.shape == (16, N, N)
        # Auto-spectra should still be valid
        assert np.all(np.isfinite(csd))
        assert np.all(csd[:, 0, 0].real > 0)


class TestBoldToCsdTorch:
    """Tests for bold_to_csd_torch wrapper."""

    def test_bold_to_csd_torch_roundtrip(
        self, rng: np.random.Generator
    ) -> None:
        """Torch wrapper produces same result as numpy version."""
        T, N = 500, 3
        bold_np = rng.standard_normal((T, N))
        freqs_np = np.linspace(0.01, 0.25, 32)

        csd_np = compute_empirical_csd(bold_np, fs=0.5, freqs=freqs_np)

        bold_t = torch.as_tensor(bold_np, dtype=torch.float64)
        freqs_t = torch.as_tensor(freqs_np, dtype=torch.float64)
        csd_t = bold_to_csd_torch(bold_t, fs=0.5, freqs=freqs_t)

        np.testing.assert_allclose(
            csd_t.numpy(), csd_np, atol=1e-12,
            err_msg="Torch wrapper result differs from numpy",
        )

    def test_bold_to_csd_torch_complex128(
        self, rng: np.random.Generator
    ) -> None:
        """Torch wrapper output is complex128."""
        bold = torch.randn(500, 3, dtype=torch.float64)
        freqs = torch.linspace(0.01, 0.25, 32, dtype=torch.float64)
        csd = bold_to_csd_torch(bold, fs=0.5, freqs=freqs)
        assert csd.dtype == torch.complex128


class TestDefaultWelchParams:
    """Tests for default_welch_params helper."""

    def test_default_welch_params(self) -> None:
        """Standard case: T=500, fs=0.5 -> nperseg=256."""
        params = default_welch_params(T=500, fs=0.5)
        assert params["nperseg"] == 256
        assert params["noverlap"] == 128
        assert params["window"] == "hann"

    def test_default_welch_params_short(self) -> None:
        """Short series: T=100, fs=0.5 -> nperseg=100 (clamped)."""
        params = default_welch_params(T=100, fs=0.5)
        assert params["nperseg"] == 100
        assert params["noverlap"] == 50
        assert params["window"] == "hann"
