"""Tests for MEG spectral DCM adaptations.

Validates:
- MEG frequency grid shape, range, and Nyquist validation
- Eigenvalue clamp parameterization (backward compat, None, MEG threshold)
- spectral_dcm_model prior_a_var and eig_clamp keyword passthrough
"""

from __future__ import annotations

import pyro
import pytest
import torch

from pyro_dcm.forward_models.spectral_noise import (
    neuronal_noise_csd,
    observation_noise_csd,
)
from pyro_dcm.forward_models.spectral_transfer import (
    compute_transfer_function,
    default_frequency_grid,
    default_frequency_grid_meg,
    predicted_csd,
    spectral_dcm_forward,
)
from pyro_dcm.models.spectral_dcm_model import spectral_dcm_model
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


class TestDefaultFrequencyGridMeg:
    """Tests for default_frequency_grid_meg."""

    def test_shape_and_range(self) -> None:
        """Shape is (64,) spanning 1.0 to 45.0 Hz."""
        freqs = default_frequency_grid_meg()
        assert freqs.shape == (64,)
        assert freqs.dtype == torch.float64
        assert torch.isclose(
            freqs[0], torch.tensor(1.0, dtype=torch.float64)
        )
        assert torch.isclose(
            freqs[-1], torch.tensor(45.0, dtype=torch.float64)
        )

    def test_respects_nyquist(self) -> None:
        """sfreq=50 Hz raises ValueError (fmax=45 > Nyquist=25)."""
        with pytest.raises(ValueError, match="Nyquist"):
            default_frequency_grid_meg(sfreq=50.0)

    def test_custom_n_freqs(self) -> None:
        """Custom n_freqs changes output length."""
        freqs = default_frequency_grid_meg(n_freqs=128)
        assert freqs.shape == (128,)
        assert torch.isclose(
            freqs[0], torch.tensor(1.0, dtype=torch.float64)
        )
        assert torch.isclose(
            freqs[-1], torch.tensor(45.0, dtype=torch.float64)
        )


class TestEigClampParameterization:
    """Tests for eigenvalue clamp in compute_transfer_function."""

    @pytest.fixture()
    def stable_setup(self) -> dict[str, torch.Tensor]:
        """Set up 2-region system with moderately negative eigenvalues."""
        A = torch.diag(
            torch.tensor([-0.5, -0.5], dtype=torch.float64)
        )
        C = torch.eye(2, dtype=torch.float64)
        freqs = default_frequency_grid(TR=2.0, n_freqs=16)
        return {"A": A, "C": C, "freqs": freqs}

    def test_backward_compat(
        self, stable_setup: dict[str, torch.Tensor]
    ) -> None:
        """Default eig_clamp=-1/32 produces identical output to hardcoded.

        Since eigenvalues at -0.5 are already below -1/32, clamping has
        no effect. Verify default call matches explicit eig_clamp=-1/32.
        """
        d = stable_setup
        H_default = compute_transfer_function(
            d["A"], d["C"], d["C"], d["freqs"]
        )
        H_explicit = compute_transfer_function(
            d["A"], d["C"], d["C"], d["freqs"], eig_clamp=-1.0 / 32.0
        )
        assert torch.allclose(H_default, H_explicit, atol=1e-15)

    def test_none_disables(self) -> None:
        """eig_clamp=None preserves eigenvalues without clamping.

        A matrix with eigenvalue at -0.01 (less negative than -1/32)
        normally gets clamped to -1/32. With eig_clamp=None, the
        original eigenvalue is preserved, producing different H.
        """
        # A with near-zero eigenvalue: -0.01 > -1/32 = -0.03125
        # so clamping pulls it to -0.03125, but None leaves it at -0.01
        A = torch.diag(
            torch.tensor([-0.01, -0.5], dtype=torch.float64)
        )
        C = torch.eye(2, dtype=torch.float64)
        freqs = default_frequency_grid(TR=2.0, n_freqs=16)

        H_clamped = compute_transfer_function(
            A, C, C, freqs, eig_clamp=-1.0 / 32.0
        )
        H_unclamped = compute_transfer_function(
            A, C, C, freqs, eig_clamp=None
        )

        # They should differ because -0.01 gets clamped to -0.03125
        # in the clamped case, but stays at -0.01 when unclamped
        assert not torch.allclose(H_clamped, H_unclamped, atol=1e-6)

        # Verify unclamped is finite
        assert torch.all(torch.isfinite(H_unclamped.real))
        assert torch.all(torch.isfinite(H_unclamped.imag))

    def test_meg_threshold(self) -> None:
        """eig_clamp=-1.0 preserves eigenvalues more negative than -1.0.

        With A having eigenvalue at -2.0 and eig_clamp=-1.0, the
        eigenvalue is clamped to -1.0 (not -1/32). The result differs
        from the fMRI default.
        """
        A = torch.diag(
            torch.tensor([-2.0, -0.5], dtype=torch.float64)
        )
        C = torch.eye(2, dtype=torch.float64)
        freqs = default_frequency_grid(TR=2.0, n_freqs=16)

        H_fmri = compute_transfer_function(
            A, C, C, freqs, eig_clamp=-1.0 / 32.0
        )
        H_meg = compute_transfer_function(
            A, C, C, freqs, eig_clamp=-1.0
        )
        H_none = compute_transfer_function(
            A, C, C, freqs, eig_clamp=None
        )

        # MEG clamp should differ from fMRI default (different thresholds)
        assert not torch.allclose(H_fmri, H_meg, atol=1e-6)

        # MEG clamp should also differ from no-clamp for this A
        # (since -2.0 < -1.0, it gets clamped to -1.0)
        assert not torch.allclose(H_meg, H_none, atol=1e-6)

        # All results should be finite
        assert torch.all(torch.isfinite(H_meg.real))
        assert torch.all(torch.isfinite(H_meg.imag))

    def test_eig_clamp_propagates_through_forward(self) -> None:
        """spectral_dcm_forward passes eig_clamp to transfer function."""
        A = torch.diag(
            torch.tensor([-2.0, -0.5], dtype=torch.float64)
        )
        freqs = default_frequency_grid(TR=2.0, n_freqs=16)
        a = torch.zeros(2, 2, dtype=torch.float64)
        b = torch.zeros(2, 1, dtype=torch.float64)
        c = torch.zeros(2, 2, dtype=torch.float64)

        csd_fmri = spectral_dcm_forward(A, freqs, a, b, c)
        csd_meg = spectral_dcm_forward(
            A, freqs, a, b, c, eig_clamp=-1.0
        )

        # Differ because eigenvalue -2.0 is clamped differently
        assert not torch.allclose(csd_fmri, csd_meg, atol=1e-6)


class TestSpectralDcmModelMeg:
    """Tests for MEG-related parameters in spectral_dcm_model."""

    @pytest.fixture()
    def spectral_data(self) -> dict:
        """Generate synthetic spectral DCM data for 3 regions."""
        A = make_stable_A_spectral(3, seed=42)
        result = simulate_spectral_dcm(A, TR=2.0, n_freqs=32, seed=42)
        N = 3
        a_mask = torch.ones(N, N, dtype=torch.float64)
        return {
            "observed_csd": result["csd"],
            "freqs": result["freqs"],
            "a_mask": a_mask,
            "N": N,
        }

    def test_prior_a_var_default(self, spectral_data: dict) -> None:
        """Default prior_a_var=1/64 gives A_free prior std=(1/64)**0.5."""
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=spectral_data["observed_csd"],
            freqs=spectral_data["freqs"],
            a_mask=spectral_data["a_mask"],
            N=spectral_data["N"],
        )
        # Check A_free prior distribution scale
        a_free_fn = trace.nodes["A_free"]["fn"]
        # Unwrap Independent wrapper to get base Normal
        base_dist = a_free_fn.base_dist
        expected_std = (1.0 / 64.0) ** 0.5
        assert torch.allclose(
            base_dist.scale,
            torch.full_like(base_dist.scale, expected_std),
            atol=1e-10,
        )

    def test_meg_prior(self, spectral_data: dict) -> None:
        """prior_a_var=1/16 gives A_free prior std=0.25."""
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=spectral_data["observed_csd"],
            freqs=spectral_data["freqs"],
            a_mask=spectral_data["a_mask"],
            N=spectral_data["N"],
            prior_a_var=1.0 / 16.0,
        )
        a_free_fn = trace.nodes["A_free"]["fn"]
        base_dist = a_free_fn.base_dist
        expected_std = 0.25  # (1/16)**0.5
        assert torch.allclose(
            base_dist.scale,
            torch.full_like(base_dist.scale, expected_std),
            atol=1e-10,
        )

    def test_eig_clamp_passthrough(self, spectral_data: dict) -> None:
        """Model called with eig_clamp=-1.0 runs without error on MEG grid.

        Uses MEG frequency grid and verifies the model produces finite
        predicted CSD.
        """
        meg_freqs = default_frequency_grid_meg(n_freqs=32)
        # Simulate CSD on MEG grid for obs (shape must match)
        A = make_stable_A_spectral(3, seed=42)
        N = 3
        C = torch.eye(N, dtype=torch.float64)

        H = compute_transfer_function(
            A, C, C, meg_freqs, eig_clamp=-1.0
        )
        a = torch.zeros(2, N, dtype=torch.float64)
        b = torch.zeros(2, 1, dtype=torch.float64)
        c_noise = torch.zeros(2, N, dtype=torch.float64)
        Gu = neuronal_noise_csd(meg_freqs, a)
        Gn = observation_noise_csd(meg_freqs, b, c_noise, N)
        obs_csd = predicted_csd(H, Gu, Gn)

        a_mask = torch.ones(N, N, dtype=torch.float64)
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=obs_csd,
            freqs=meg_freqs,
            a_mask=a_mask,
            N=N,
            eig_clamp=-1.0,
        )
        pred = trace.nodes["predicted_csd"]["value"]
        assert torch.all(torch.isfinite(pred.real))
        assert torch.all(torch.isfinite(pred.imag))
