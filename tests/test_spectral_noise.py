"""Unit tests for spectral noise models.

Tests validate:
- Neuronal noise CSD shape, dtype, diagonal structure, and 1/f behavior
- Observation noise CSD shape, global fill, and regional diagonal
- Default noise prior shapes, values, and total parameter count
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.forward_models.spectral_noise import (
    default_noise_priors,
    neuronal_noise_csd,
    observation_noise_csd,
)
from pyro_dcm.forward_models.spectral_transfer import default_frequency_grid


class TestNeuronalNoiseCSD:
    """Tests for neuronal_noise_csd."""

    @pytest.fixture()
    def setup(self) -> dict[str, torch.Tensor]:
        """Set up test data."""
        N = 3
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        a = torch.zeros(2, N, dtype=torch.float64)
        return {"freqs": freqs, "a": a, "N": N}

    def test_shape(self, setup: dict[str, torch.Tensor]) -> None:
        """Output shape is (F, N, N) = (32, 3, 3) complex128."""
        d = setup
        Gu = neuronal_noise_csd(d["freqs"], d["a"])
        assert Gu.shape == (32, d["N"], d["N"])
        assert Gu.dtype == torch.complex128

    def test_diagonal(self, setup: dict[str, torch.Tensor]) -> None:
        """Off-diagonal elements are zero."""
        d = setup
        Gu = neuronal_noise_csd(d["freqs"], d["a"])
        for i in range(d["N"]):
            for j in range(d["N"]):
                if i != j:
                    assert torch.all(Gu[:, i, j] == 0)

    def test_positive_real(self, setup: dict[str, torch.Tensor]) -> None:
        """Diagonal elements have positive real parts and zero imaginary."""
        d = setup
        Gu = neuronal_noise_csd(d["freqs"], d["a"])
        for i in range(d["N"]):
            assert torch.all(Gu[:, i, i].real > 0)
            assert torch.allclose(
                Gu[:, i, i].imag,
                torch.zeros(32, dtype=torch.float64),
                atol=1e-15,
            )

    def test_decreasing_with_frequency(
        self, setup: dict[str, torch.Tensor]
    ) -> None:
        """For default params (a=zeros), power decreases with frequency."""
        d = setup
        Gu = neuronal_noise_csd(d["freqs"], d["a"])
        for i in range(d["N"]):
            # First frequency bin has more power than last (1/f)
            assert Gu[0, i, i].real > Gu[-1, i, i].real

    def test_amplitude_scaling(self) -> None:
        """Doubling log amplitude doubles the noise power."""
        N = 3
        freqs = default_frequency_grid()
        a_base = torch.zeros(2, N, dtype=torch.float64)
        a_double = a_base.clone()
        a_double[0, :] += math.log(2.0)

        Gu_base = neuronal_noise_csd(freqs, a_base)
        Gu_double = neuronal_noise_csd(freqs, a_double)

        for i in range(N):
            assert torch.allclose(
                Gu_double[:, i, i].real,
                2.0 * Gu_base[:, i, i].real,
                rtol=1e-10,
            )

    def test_exponent_effect(self) -> None:
        """Increasing exponent steepens the spectrum."""
        N = 1
        freqs = default_frequency_grid()

        # Low exponent
        a_low = torch.zeros(2, N, dtype=torch.float64)
        a_low[1, :] = 0.0  # exp(0) = 1 -> w^(-1)

        # High exponent
        a_high = torch.zeros(2, N, dtype=torch.float64)
        a_high[1, :] = math.log(2.0)  # exp(ln2) = 2 -> w^(-2)

        Gu_low = neuronal_noise_csd(freqs, a_low)
        Gu_high = neuronal_noise_csd(freqs, a_high)

        # Ratio of first to last frequency bin
        ratio_low = (
            Gu_low[0, 0, 0].real / Gu_low[-1, 0, 0].real
        )
        ratio_high = (
            Gu_high[0, 0, 0].real / Gu_high[-1, 0, 0].real
        )

        # Higher exponent -> steeper slope -> larger ratio
        assert ratio_high > ratio_low


class TestObservationNoiseCSD:
    """Tests for observation_noise_csd."""

    @pytest.fixture()
    def setup(self) -> dict[str, torch.Tensor]:
        """Set up test data."""
        N = 3
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        b = torch.zeros(2, 1, dtype=torch.float64)
        c = torch.zeros(2, N, dtype=torch.float64)
        return {"freqs": freqs, "b": b, "c": c, "N": N}

    def test_shape(self, setup: dict[str, torch.Tensor]) -> None:
        """Output shape is (F, N, N) = (32, 3, 3) complex128."""
        d = setup
        Gn = observation_noise_csd(d["freqs"], d["b"], d["c"], d["N"])
        assert Gn.shape == (32, d["N"], d["N"])
        assert Gn.dtype == torch.complex128

    def test_global_fills_all(
        self, setup: dict[str, torch.Tensor]
    ) -> None:
        """Global noise contributes to all entries (i,j), not just diag."""
        d = setup
        Gn = observation_noise_csd(d["freqs"], d["b"], d["c"], d["N"])
        # Off-diagonal entries should be non-zero (global fills all)
        for i in range(d["N"]):
            for j in range(d["N"]):
                if i != j:
                    assert torch.any(Gn[:, i, j].real.abs() > 1e-15)

    def test_regional_diagonal_only(self) -> None:
        """Regional noise adds to diagonal only."""
        N = 3
        freqs = default_frequency_grid()
        # Set b to very negative (effectively zero global noise)
        b = torch.tensor([[-20.0], [-20.0]], dtype=torch.float64)
        # Non-zero regional noise
        c = torch.zeros(2, N, dtype=torch.float64)

        Gn = observation_noise_csd(freqs, b, c, N)

        # Off-diagonal entries should be near-zero (only global fills them)
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert torch.allclose(
                        Gn[:, i, j].real,
                        torch.zeros(32, dtype=torch.float64),
                        atol=1e-10,
                    )


class TestDefaultNoisePriors:
    """Tests for default_noise_priors."""

    def test_shapes(self) -> None:
        """For n_regions=5, all shapes are correct."""
        priors = default_noise_priors(5)
        assert priors["a_prior_mean"].shape == (2, 5)
        assert priors["a_prior_var"].shape == (2, 5)
        assert priors["b_prior_mean"].shape == (2, 1)
        assert priors["b_prior_var"].shape == (2, 1)
        assert priors["c_prior_mean"].shape == (2, 5)
        assert priors["c_prior_var"].shape == (2, 5)

    def test_values(self) -> None:
        """Prior means are zeros, variances are 1/64."""
        priors = default_noise_priors(3)
        for key in ["a_prior_mean", "b_prior_mean", "c_prior_mean"]:
            assert torch.all(priors[key] == 0.0)
        for key in ["a_prior_var", "b_prior_var", "c_prior_var"]:
            assert torch.allclose(
                priors[key],
                torch.full_like(priors[key], 1.0 / 64.0),
            )

    def test_dtypes(self) -> None:
        """All tensors are float64."""
        priors = default_noise_priors(3)
        for val in priors.values():
            assert val.dtype == torch.float64

    def test_total_param_count(self) -> None:
        """Total params = 4N + 2 for N regions."""
        for N in [1, 3, 5, 10]:
            priors = default_noise_priors(N)
            total = sum(t.numel() for t in priors.values()) // 2
            # Divide by 2 because we have mean+var for each param
            expected = 4 * N + 2
            assert total == expected, (
                f"Expected {expected} params for N={N}, got {total}"
            )
