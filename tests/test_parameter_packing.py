"""Tests for parameter packing utilities.

Validates pack/unpack round-trips, log-space contracts, batch
dimensions, standardization, and A_free inversion for both task
and spectral DCM packers.
"""

from __future__ import annotations

import torch
import pytest

from pyro_dcm.guides.parameter_packing import (
    TaskDCMPacker,
    SpectralDCMPacker,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.simulators.task_simulator import make_random_stable_A


class TestTaskDCMPacker:
    """Tests for TaskDCMPacker."""

    @pytest.fixture()
    def packer(self) -> TaskDCMPacker:
        """Create a 3-region, 1-input task packer."""
        N, M = 3, 1
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        return TaskDCMPacker(N, M, a_mask, c_mask)

    @pytest.fixture()
    def sample_params(self) -> dict[str, torch.Tensor]:
        """Create sample task DCM parameters."""
        return {
            "A_free": torch.randn(3, 3, dtype=torch.float64),
            "C": torch.randn(3, 1, dtype=torch.float64),
            "noise_prec": torch.tensor(5.0, dtype=torch.float64),
        }

    def test_round_trip(
        self, packer: TaskDCMPacker,
        sample_params: dict[str, torch.Tensor],
    ) -> None:
        """Pack then unpack recovers original params."""
        packed = packer.pack(sample_params)
        unpacked = packer.unpack(packed)

        torch.testing.assert_close(
            unpacked["A_free"], sample_params["A_free"],
        )
        torch.testing.assert_close(
            unpacked["C"], sample_params["C"],
        )
        # noise_prec is in log-space after unpack, so exp() to recover
        torch.testing.assert_close(
            unpacked["noise_prec"].exp(),
            sample_params["noise_prec"],
            atol=1e-12, rtol=1e-12,
        )

    def test_batch_unpack(
        self, packer: TaskDCMPacker,
        sample_params: dict[str, torch.Tensor],
    ) -> None:
        """Unpack works with batch dimensions."""
        packed = packer.pack(sample_params)
        # Create batch: (batch=4, n_features)
        batch_packed = packed.unsqueeze(0).expand(4, -1)
        unpacked = packer.unpack(batch_packed)

        assert unpacked["A_free"].shape == (4, 3, 3)
        assert unpacked["C"].shape == (4, 3, 1)
        assert unpacked["noise_prec"].shape == (4,)

    def test_n_features(self, packer: TaskDCMPacker) -> None:
        """Feature count matches expected: N*N + N*M + 1."""
        N, M = 3, 1
        expected = N * N + N * M + 1  # 9 + 3 + 1 = 13
        assert packer.n_features == expected

    def test_noise_prec_log_space(
        self, packer: TaskDCMPacker,
        sample_params: dict[str, torch.Tensor],
    ) -> None:
        """After pack(), last element equals log(noise_prec)."""
        packed = packer.pack(sample_params)
        expected_log = torch.log(sample_params["noise_prec"])
        torch.testing.assert_close(
            packed[-1], expected_log, atol=1e-14, rtol=1e-14,
        )


class TestSpectralDCMPacker:
    """Tests for SpectralDCMPacker."""

    @pytest.fixture()
    def packer(self) -> SpectralDCMPacker:
        """Create a 3-region spectral packer."""
        return SpectralDCMPacker(n_regions=3)

    @pytest.fixture()
    def sample_params(self) -> dict[str, torch.Tensor]:
        """Create sample spectral DCM parameters."""
        N = 3
        return {
            "A_free": torch.randn(N, N, dtype=torch.float64),
            "noise_a": torch.randn(2, N, dtype=torch.float64),
            "noise_b": torch.randn(2, 1, dtype=torch.float64),
            "noise_c": torch.randn(2, N, dtype=torch.float64),
            "csd_noise_scale": torch.tensor(
                1.5, dtype=torch.float64,
            ),
        }

    def test_round_trip(
        self, packer: SpectralDCMPacker,
        sample_params: dict[str, torch.Tensor],
    ) -> None:
        """Pack then unpack recovers original params."""
        packed = packer.pack(sample_params)
        unpacked = packer.unpack(packed)

        torch.testing.assert_close(
            unpacked["A_free"], sample_params["A_free"],
        )
        torch.testing.assert_close(
            unpacked["noise_a"], sample_params["noise_a"],
        )
        torch.testing.assert_close(
            unpacked["noise_b"], sample_params["noise_b"],
        )
        torch.testing.assert_close(
            unpacked["noise_c"], sample_params["noise_c"],
        )
        # csd_noise_scale is in log-space after unpack
        torch.testing.assert_close(
            unpacked["csd_noise_scale"].exp(),
            sample_params["csd_noise_scale"],
            atol=1e-12, rtol=1e-12,
        )

    def test_n_features(self, packer: SpectralDCMPacker) -> None:
        """Feature count: N*N + 2*N + 2 + 2*N + 1."""
        N = 3
        expected = N * N + 2 * N + 2 + 2 * N + 1  # 9+6+2+6+1=24
        assert packer.n_features == expected


class TestStandardization:
    """Tests for standardization round-trips and range."""

    def test_round_trip(self) -> None:
        """Standardize then unstandardize recovers original."""
        N, M = 3, 1
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        packer = TaskDCMPacker(N, M, a_mask, c_mask)

        # Generate synthetic training data
        torch.manual_seed(42)
        dataset = []
        for _ in range(200):
            dataset.append({
                "A_free": torch.randn(N, N, dtype=torch.float64)
                * 0.125,
                "C": torch.randn(N, M, dtype=torch.float64),
                "noise_prec": torch.rand(1, dtype=torch.float64)
                .squeeze() * 10 + 0.1,
            })

        packer.fit_standardization(dataset)

        # Test round-trip on a sample
        z = packer.pack(dataset[0])
        z_std = packer.standardize(z)
        z_rec = packer.unstandardize(z_std)

        torch.testing.assert_close(z, z_rec, atol=1e-12, rtol=1e-12)

    def test_range(self) -> None:
        """After standardization, >99% of values in [-5, 5]."""
        N, M = 3, 1
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        packer = TaskDCMPacker(N, M, a_mask, c_mask)

        # Generate data with realistic distributions
        torch.manual_seed(42)
        dataset = []
        for _ in range(1000):
            dataset.append({
                "A_free": torch.randn(N, N, dtype=torch.float64)
                * 0.125,
                "C": torch.randn(N, M, dtype=torch.float64),
                "noise_prec": torch.rand(1, dtype=torch.float64)
                .squeeze() * 10 + 0.1,
            })

        packer.fit_standardization(dataset)

        # Standardize all
        all_std = torch.stack([
            packer.standardize(packer.pack(d)) for d in dataset
        ])

        # Check >99% in [-5, 5]
        in_range = (all_std.abs() <= 5.0).float().mean()
        assert in_range > 0.99, (
            f"Only {in_range:.3f} of values in [-5, 5]"
        )


class TestAFreeInversion:
    """Tests for A_free inversion formula."""

    def test_round_trip(self) -> None:
        """A_free inversion through parameterize_A recovers A."""
        from scripts.generate_training_data import invert_A_to_A_free

        for seed in [42, 123, 789]:
            A = make_random_stable_A(3, seed=seed)
            A_free = invert_A_to_A_free(A)
            A_recovered = parameterize_A(A_free)

            torch.testing.assert_close(
                A_recovered, A, atol=1e-12, rtol=1e-12,
            )


class TestTaskDCMPackerBilinearRefusal:
    """MODEL-07: TaskDCMPacker.pack refuses bilinear sites per D5.

    Amortized bilinear inference is deferred to v0.3.1. The packer's fixed
    n_features = N*N + N*M + 1 cannot accommodate J*N*N bilinear terms.
    pack() raises NotImplementedError (referencing v0.3.1) on any
    ``B_free_*`` key; the linear pack/unpack path is unchanged.
    """

    def test_packer_refuses_bilinear_keys(self) -> None:
        """TaskDCMPacker.pack raises NotImplementedError on 'B_free_*' keys."""
        packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
        params_bilinear = {
            "A_free": torch.zeros(3, 3),
            "C": torch.zeros(3, 1),
            "noise_prec": torch.tensor(10.0),
            "B_free_0": torch.zeros(3, 3),  # the trigger
        }
        with pytest.raises(NotImplementedError, match=r"v0\.3\.1"):
            packer.pack(params_bilinear)

    def test_packer_accepts_linear_keys_after_bilinear_guard(self) -> None:
        """Regression: linear pack/unpack still works after the guard added."""
        packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
        params_linear = {
            "A_free": torch.zeros(3, 3),
            "C": torch.zeros(3, 1),
            "noise_prec": torch.tensor(10.0),
        }
        z = packer.pack(params_linear)
        assert z.shape == (13,)
        unpacked = packer.unpack(z)
        assert unpacked["A_free"].shape == (3, 3)
        assert unpacked["C"].shape == (3, 1)
