"""Tests for summary networks (BoldSummaryNet, CsdSummaryNet).

Validates output shapes, variable-length handling, gradient flow,
and complex input decomposition for the amortized inference summary
networks.
"""

from __future__ import annotations

import torch
import pytest

from pyro_dcm.guides.summary_networks import BoldSummaryNet, CsdSummaryNet


class TestBoldSummaryNet:
    """Tests for BoldSummaryNet 1D-CNN summary network."""

    def test_output_shape_unbatched(self) -> None:
        """Unbatched (T, N) input produces (embed_dim,) output."""
        net = BoldSummaryNet(n_regions=3, embed_dim=128)
        bold = torch.randn(100, 3, dtype=torch.float64)
        out = net(bold)
        assert out.shape == (128,)
        assert out.dtype == torch.float64

    def test_output_shape_batched(self) -> None:
        """Batched (batch, T, N) input produces (batch, embed_dim)."""
        net = BoldSummaryNet(n_regions=3, embed_dim=128)
        bold = torch.randn(8, 100, 3, dtype=torch.float64)
        out = net(bold)
        assert out.shape == (8, 128)
        assert out.dtype == torch.float64

    def test_variable_length(self) -> None:
        """Different T values produce same output shape."""
        net = BoldSummaryNet(n_regions=3, embed_dim=128)
        net.eval()  # BatchNorm needs eval for single samples

        bold_short = torch.randn(50, 3, dtype=torch.float64)
        bold_long = torch.randn(200, 3, dtype=torch.float64)

        out_short = net(bold_short)
        out_long = net(bold_long)

        assert out_short.shape == (128,)
        assert out_long.shape == (128,)

    def test_gradient_flow(self) -> None:
        """Gradients flow through all conv and linear parameters."""
        net = BoldSummaryNet(n_regions=3, embed_dim=128)
        bold = torch.randn(8, 100, 3, dtype=torch.float64)

        out = net(bold)
        loss = out.sum()
        loss.backward()

        # Check all named parameters have gradients
        for name, param in net.named_parameters():
            assert param.grad is not None, (
                f"No gradient for {name}"
            )
            assert not torch.all(param.grad == 0), (
                f"Zero gradient for {name}"
            )

    def test_custom_embed_dim(self) -> None:
        """Non-default embed_dim works correctly."""
        net = BoldSummaryNet(n_regions=5, embed_dim=64)
        bold = torch.randn(100, 5, dtype=torch.float64)
        out = net(bold)
        assert out.shape == (64,)


class TestCsdSummaryNet:
    """Tests for CsdSummaryNet MLP summary network."""

    def test_output_shape_unbatched(self) -> None:
        """Unbatched (F, N, N) input produces (embed_dim,) output."""
        net = CsdSummaryNet(n_regions=3, n_freqs=32, embed_dim=128)
        csd = torch.randn(32, 3, 3, dtype=torch.complex128)
        out = net(csd)
        assert out.shape == (128,)
        assert out.dtype == torch.float64

    def test_output_shape_batched(self) -> None:
        """Batched (batch, F, N, N) input produces (batch, embed_dim)."""
        net = CsdSummaryNet(n_regions=3, n_freqs=32, embed_dim=128)
        csd = torch.randn(8, 32, 3, 3, dtype=torch.complex128)
        out = net(csd)
        assert out.shape == (8, 128)
        assert out.dtype == torch.float64

    def test_gradient_flow(self) -> None:
        """Gradients flow through all linear parameters."""
        net = CsdSummaryNet(n_regions=3, n_freqs=32, embed_dim=128)
        csd = torch.randn(8, 32, 3, 3, dtype=torch.complex128)

        out = net(csd)
        loss = out.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, (
                f"No gradient for {name}"
            )
            assert not torch.all(param.grad == 0), (
                f"Zero gradient for {name}"
            )

    def test_complex_input_decomposition(self) -> None:
        """Complex128 input is correctly decomposed to real/imag."""
        net = CsdSummaryNet(n_regions=3, n_freqs=32, embed_dim=128)

        # Create CSD with known real/imag parts
        real = torch.randn(32, 3, 3, dtype=torch.float64)
        imag = torch.randn(32, 3, 3, dtype=torch.float64)
        csd = torch.complex(real, imag)

        assert csd.dtype == torch.complex128
        out = net(csd)
        assert out.dtype == torch.float64
        assert out.shape == (128,)

    def test_custom_dims(self) -> None:
        """Non-default n_freqs and embed_dim work correctly."""
        net = CsdSummaryNet(n_regions=5, n_freqs=16, embed_dim=64)
        csd = torch.randn(16, 5, 5, dtype=torch.complex128)
        out = net(csd)
        assert out.shape == (64,)
