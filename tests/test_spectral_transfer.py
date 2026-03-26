"""Unit tests for spectral transfer function and predicted CSD.

Tests validate:
- Default frequency grid shape, range, and dtype
- Transfer function shape, dtype, and mathematical correctness
- Eigenvalue stabilization for near-unstable and unstable A matrices
- Predicted CSD Hermitian symmetry and positive semi-definiteness
- Full pipeline integration (spectral_dcm_forward)
- Autograd differentiability through the forward model
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.forward_models.spectral_transfer import (
    compute_transfer_function,
    default_frequency_grid,
    predicted_csd,
    spectral_dcm_forward,
)


class TestDefaultFrequencyGrid:
    """Tests for default_frequency_grid."""

    def test_shape_and_range(self) -> None:
        """For TR=2.0, n_freqs=32, shape is (32,) with correct bounds."""
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        assert freqs.shape == (32,)
        assert freqs.dtype == torch.float64
        assert torch.isclose(
            freqs[0], torch.tensor(1.0 / 128.0, dtype=torch.float64)
        )
        assert torch.isclose(
            freqs[-1], torch.tensor(0.25, dtype=torch.float64)
        )

    def test_custom_TR(self) -> None:
        """For TR=0.72 (fast fMRI), upper bound is ~0.694 Hz."""
        freqs = default_frequency_grid(TR=0.72, n_freqs=32)
        expected_upper = 1.0 / (2.0 * 0.72)
        assert torch.isclose(
            freqs[-1],
            torch.tensor(expected_upper, dtype=torch.float64),
        )

    def test_monotonically_increasing(self) -> None:
        """Frequency grid is monotonically increasing."""
        freqs = default_frequency_grid()
        assert torch.all(freqs[1:] > freqs[:-1])


class TestComputeTransferFunction:
    """Tests for compute_transfer_function."""

    @pytest.fixture()
    def setup_3region(self) -> dict[str, torch.Tensor]:
        """Set up 3-region test configuration."""
        A = torch.tensor(
            [[-0.5, 0.1, 0.0], [0.2, -0.5, 0.1], [0.0, 0.3, -0.5]],
            dtype=torch.float64,
        )
        N = 3
        C_in = torch.eye(N, dtype=torch.float64)
        C_out = torch.eye(N, dtype=torch.float64)
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        return {"A": A, "C_in": C_in, "C_out": C_out, "freqs": freqs}

    def test_shape(self, setup_3region: dict[str, torch.Tensor]) -> None:
        """Output shape is (F, nn, nu) = (32, 3, 3) complex128."""
        d = setup_3region
        H = compute_transfer_function(d["A"], d["C_in"], d["C_out"], d["freqs"])
        assert H.shape == (32, 3, 3)
        assert H.dtype == torch.complex128

    def test_identity_matrices(
        self, setup_3region: dict[str, torch.Tensor]
    ) -> None:
        """With C_in=C_out=I, H(w) matches direct (iwI - A)^-1."""
        d = setup_3region
        H = compute_transfer_function(d["A"], d["C_in"], d["C_out"], d["freqs"])

        # Direct inversion reference
        N = d["A"].shape[0]
        A_c = d["A"].to(torch.complex128)
        I_mat = torch.eye(N, dtype=torch.complex128)
        for f_idx in range(0, 32, 8):  # Check at a few frequencies
            w = d["freqs"][f_idx].to(torch.complex128)
            M = 1j * 2.0 * math.pi * w * I_mat - A_c
            # Since eigenvalue stabilization may modify eigenvalues,
            # we stabilize A for fair comparison
            eigvals, eigvecs = torch.linalg.eig(A_c)
            eigvals_stable = torch.complex(
                torch.clamp(eigvals.real, max=-1.0 / 32.0),
                eigvals.imag,
            )
            A_stable = (
                eigvecs
                @ torch.diag(eigvals_stable)
                @ torch.linalg.inv(eigvecs)
            )
            M_stable = 1j * 2.0 * math.pi * w * I_mat - A_stable
            H_direct = torch.linalg.inv(M_stable)
            assert torch.allclose(H[f_idx], H_direct, atol=1e-10)

    def test_eigenvalue_stabilization(self) -> None:
        """Near-zero eigenvalue is clamped; output is finite."""
        # A with near-zero eigenvalue (-0.01)
        A = torch.diag(
            torch.tensor([-0.01, -0.5, -0.5], dtype=torch.float64)
        )
        N = 3
        C_in = torch.eye(N, dtype=torch.float64)
        C_out = torch.eye(N, dtype=torch.float64)
        freqs = default_frequency_grid()
        H = compute_transfer_function(A, C_in, C_out, freqs)
        assert torch.all(torch.isfinite(H.real))
        assert torch.all(torch.isfinite(H.imag))

    def test_unstable_A(self) -> None:
        """Positive eigenvalue is clamped; output is finite complex128."""
        # A with positive eigenvalue (+0.1)
        A = torch.diag(
            torch.tensor([0.1, -0.5, -0.5], dtype=torch.float64)
        )
        N = 3
        C_in = torch.eye(N, dtype=torch.float64)
        C_out = torch.eye(N, dtype=torch.float64)
        freqs = default_frequency_grid()
        H = compute_transfer_function(A, C_in, C_out, freqs)
        assert H.dtype == torch.complex128
        assert torch.all(torch.isfinite(H.real))
        assert torch.all(torch.isfinite(H.imag))

    def test_known_1d(self) -> None:
        """For 1-region A=[[-0.5]], H(0.1) = 1/(i*2*pi*0.1 + 0.5)."""
        A = torch.tensor([[-0.5]], dtype=torch.float64)
        C_in = torch.eye(1, dtype=torch.float64)
        C_out = torch.eye(1, dtype=torch.float64)
        freqs = torch.tensor([0.1], dtype=torch.float64)
        H = compute_transfer_function(A, C_in, C_out, freqs)

        w = 0.1
        # The stabilization clamps -0.5 to max(-1/32) = -0.03125,
        # but -0.5 < -0.03125, so it stays at -0.5
        expected = 1.0 / (1j * 2.0 * math.pi * w + 0.5)
        assert torch.isclose(
            H[0, 0, 0],
            torch.tensor(expected, dtype=torch.complex128),
            atol=1e-12,
        )


class TestPredictedCSD:
    """Tests for predicted_csd."""

    @pytest.fixture()
    def setup_csd(self) -> dict[str, torch.Tensor]:
        """Set up test data for predicted CSD."""
        F, nn, nu = 32, 3, 3
        # Random positive-definite H for testing
        torch.manual_seed(42)
        H = torch.randn(F, nn, nu, dtype=torch.float64).to(
            torch.complex128
        ) + 1j * torch.randn(F, nn, nu, dtype=torch.float64)

        # Diagonal positive Gu
        Gu = torch.zeros(F, nu, nu, dtype=torch.complex128)
        for i in range(nu):
            Gu[:, i, i] = torch.rand(F, dtype=torch.float64).to(
                torch.complex128
            ) + 0.1

        # Diagonal positive Gn
        Gn = torch.zeros(F, nn, nn, dtype=torch.complex128)
        for i in range(nn):
            Gn[:, i, i] = torch.rand(F, dtype=torch.float64).to(
                torch.complex128
            ) * 0.01 + 0.001

        return {"H": H, "Gu": Gu, "Gn": Gn, "F": F, "nn": nn}

    def test_shape(self, setup_csd: dict[str, torch.Tensor]) -> None:
        """Output shape is (F, nn, nn) complex128."""
        d = setup_csd
        S = predicted_csd(d["H"], d["Gu"], d["Gn"])
        assert S.shape == (d["F"], d["nn"], d["nn"])
        assert S.dtype == torch.complex128

    def test_hermitian(self, setup_csd: dict[str, torch.Tensor]) -> None:
        """S(w) is Hermitian at each frequency: S[f] == S[f].conj().T."""
        d = setup_csd
        S = predicted_csd(d["H"], d["Gu"], d["Gn"])
        S_H = S.conj().transpose(-2, -1)
        assert torch.allclose(S, S_H, atol=1e-10)

    def test_positive_semidefinite(
        self, setup_csd: dict[str, torch.Tensor]
    ) -> None:
        """Diagonal elements have non-negative real parts (power spectra)."""
        d = setup_csd
        S = predicted_csd(d["H"], d["Gu"], d["Gn"])
        for i in range(d["nn"]):
            assert torch.all(S[:, i, i].real >= -1e-10)


class TestSpectralDCMForward:
    """Tests for spectral_dcm_forward convenience function."""

    @pytest.fixture()
    def setup_forward(self) -> dict[str, torch.Tensor]:
        """Set up test data for full forward pipeline."""
        N = 3
        A = torch.tensor(
            [[-0.5, 0.1, 0.0], [0.2, -0.5, 0.1], [0.0, 0.3, -0.5]],
            dtype=torch.float64,
        )
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        a = torch.zeros(2, N, dtype=torch.float64)
        b = torch.zeros(2, 1, dtype=torch.float64)
        c = torch.zeros(2, N, dtype=torch.float64)
        return {"A": A, "freqs": freqs, "a": a, "b": b, "c": c, "N": N}

    def test_shape(
        self, setup_forward: dict[str, torch.Tensor]
    ) -> None:
        """Full pipeline produces (32, 3, 3) complex128."""
        d = setup_forward
        S = spectral_dcm_forward(d["A"], d["freqs"], d["a"], d["b"], d["c"])
        assert S.shape == (32, d["N"], d["N"])
        assert S.dtype == torch.complex128

    def test_no_nan(
        self, setup_forward: dict[str, torch.Tensor]
    ) -> None:
        """Full pipeline with default noise params has no NaN/Inf."""
        d = setup_forward
        S = spectral_dcm_forward(d["A"], d["freqs"], d["a"], d["b"], d["c"])
        assert torch.all(torch.isfinite(S.real))
        assert torch.all(torch.isfinite(S.imag))

    def test_differentiable(self) -> None:
        """Autograd flows through the forward model w.r.t. A."""
        N = 3
        A = torch.tensor(
            [[-0.5, 0.1, 0.0], [0.2, -0.5, 0.1], [0.0, 0.3, -0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        freqs = default_frequency_grid(TR=2.0, n_freqs=32)
        a = torch.zeros(2, N, dtype=torch.float64)
        b = torch.zeros(2, 1, dtype=torch.float64)
        c = torch.zeros(2, N, dtype=torch.float64)

        S = spectral_dcm_forward(A, freqs, a, b, c)
        loss = S.abs().sum()
        grads = torch.autograd.grad(loss, A)
        assert grads[0] is not None
        assert torch.all(torch.isfinite(grads[0]))
