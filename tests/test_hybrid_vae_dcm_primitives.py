"""Tests for hybrid VAE-DCM primitive components.

Covers LatentCircuitDCMPacker (parameter packing for sparse masks)
and DCMEncoderNet (1D-CNN encoder for timeseries-to-posterior mapping).
"""

from __future__ import annotations

import torch
import pytest

from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def masks_4x1():
    """4-region, 1-input masks with sparse connectivity."""
    a_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32)
    c_mask = torch.tensor([[1], [0], [0], [0]], dtype=torch.float32)
    return a_mask, c_mask


@pytest.fixture()
def full_masks_3x1():
    """3-region, 1-input fully connected masks."""
    a_mask = torch.ones(3, 3)
    c_mask = torch.ones(3, 1)
    return a_mask, c_mask


# ---------------------------------------------------------------------------
# LatentCircuitDCMPacker tests
# ---------------------------------------------------------------------------

class TestLatentCircuitDCMPacker:
    """Tests for LatentCircuitDCMPacker."""

    def test_packer_total_dim(self, masks_4x1):
        """Total dim = n_a + n_c + n_regions + 1."""
        a_mask, c_mask = masks_4x1
        packer = LatentCircuitDCMPacker(4, 1, a_mask, c_mask)

        n_a = int(a_mask.sum().item())  # 10
        n_c = int(c_mask.sum().item())  # 1
        expected = n_a + n_c + 4 + 1  # 10 + 1 + 4 + 1 = 16
        assert packer.total_dim == expected

    def test_packer_total_dim_full(self, full_masks_3x1):
        """Full masks: n_a=9, n_c=3, N=3, total=9+3+3+1=16."""
        a_mask, c_mask = full_masks_3x1
        packer = LatentCircuitDCMPacker(3, 1, a_mask, c_mask)
        assert packer.total_dim == 9 + 3 + 3 + 1

    def test_packer_round_trip(self, masks_4x1):
        """pack -> unpack reproduces exact values at mask positions."""
        a_mask, c_mask = masks_4x1
        packer = LatentCircuitDCMPacker(4, 1, a_mask, c_mask)

        a_free = torch.randn(4, 4)
        c = torch.randn(4, 1)
        x0 = torch.randn(4)
        noise_prec = torch.tensor(5.0)

        flat = packer.pack(a_free, c, x0, noise_prec)
        assert flat.shape == (packer.total_dim,)

        params = packer.unpack(flat)

        # A_free: check values at mask positions match
        torch.testing.assert_close(
            params["A_free"][a_mask.bool()],
            a_free[a_mask.bool()],
        )
        # A_free: zeros at non-mask positions
        assert (params["A_free"][~a_mask.bool()] == 0).all()

        # C: check values at mask positions match
        torch.testing.assert_close(
            params["C"][c_mask.bool()],
            c[c_mask.bool()],
        )
        # C: zeros at non-mask positions
        assert (params["C"][~c_mask.bool()] == 0).all()

        # x0
        torch.testing.assert_close(params["x0"], x0)

        # noise_prec in log-space
        torch.testing.assert_close(
            params["noise_prec"],
            torch.log(noise_prec),
        )

    def test_packer_round_trip_full(self, full_masks_3x1):
        """Full masks: pack -> unpack reproduces all values exactly."""
        a_mask, c_mask = full_masks_3x1
        packer = LatentCircuitDCMPacker(3, 1, a_mask, c_mask)

        a_free = torch.randn(3, 3)
        c = torch.randn(3, 1)
        x0 = torch.randn(3)
        noise_prec = torch.tensor(2.5)

        flat = packer.pack(a_free, c, x0, noise_prec)
        params = packer.unpack(flat)

        torch.testing.assert_close(params["A_free"], a_free)
        torch.testing.assert_close(params["C"], c)
        torch.testing.assert_close(params["x0"], x0)
        torch.testing.assert_close(
            params["noise_prec"], torch.log(noise_prec),
        )

    def test_packer_standardization(self, masks_4x1):
        """fit -> standardize -> unstandardize round-trips."""
        a_mask, c_mask = masks_4x1
        packer = LatentCircuitDCMPacker(4, 1, a_mask, c_mask)

        # Generate some packed samples
        n_samples = 50
        samples = []
        for _ in range(n_samples):
            flat = packer.pack(
                torch.randn(4, 4),
                torch.randn(4, 1),
                torch.randn(4),
                torch.tensor(1.0 + torch.rand(1).item() * 9),
            )
            samples.append(flat)

        stacked = torch.stack(samples)
        packer.fit_standardization(stacked)

        # Check mean_ and std_ shapes
        assert packer.mean_ is not None
        assert packer.std_ is not None
        assert packer.mean_.shape == (packer.total_dim,)
        assert packer.std_.shape == (packer.total_dim,)

        # Round-trip: standardize -> unstandardize
        test_flat = samples[0]
        standardized = packer.standardize(test_flat)
        recovered = packer.unstandardize(standardized)
        torch.testing.assert_close(recovered, test_flat)

    def test_packer_standardization_not_fitted(self, masks_4x1):
        """standardize/unstandardize raise before fit."""
        a_mask, c_mask = masks_4x1
        packer = LatentCircuitDCMPacker(4, 1, a_mask, c_mask)
        flat = torch.randn(packer.total_dim)

        with pytest.raises(AssertionError, match="fit_standardization"):
            packer.standardize(flat)

        with pytest.raises(AssertionError, match="fit_standardization"):
            packer.unstandardize(flat)


# ---------------------------------------------------------------------------
# DCMEncoderNet tests
# ---------------------------------------------------------------------------

class TestDCMEncoderNet:
    """Tests for DCMEncoderNet."""

    def test_encoder_output_shapes(self):
        """(batch, T, N) -> (z_loc, z_scale) with correct shapes."""
        n_regions = 4
        latent_dim = 21
        batch_size = 8
        t_len = 100

        enc = DCMEncoderNet(n_regions=n_regions, latent_dim=latent_dim)
        x = torch.randn(batch_size, t_len, n_regions)

        z_loc, z_scale = enc(x)

        assert z_loc.shape == (batch_size, latent_dim)
        assert z_scale.shape == (batch_size, latent_dim)

    def test_encoder_unbatched(self):
        """(T, N) -> (z_loc, z_scale) with 1D shapes."""
        n_regions = 3
        latent_dim = 10
        t_len = 50

        enc = DCMEncoderNet(n_regions=n_regions, latent_dim=latent_dim)
        x = torch.randn(t_len, n_regions)

        z_loc, z_scale = enc(x)

        assert z_loc.shape == (latent_dim,)
        assert z_scale.shape == (latent_dim,)

    def test_encoder_initial_output_near_zero(self):
        """Fresh encoder z_loc should be near 0 (weights init ~0)."""
        n_regions = 4
        latent_dim = 16

        enc = DCMEncoderNet(n_regions=n_regions, latent_dim=latent_dim)
        enc.eval()

        x = torch.randn(8, 100, n_regions)
        with torch.no_grad():
            z_loc, _ = enc(x)

        # z_loc should be near zero due to small weight init
        assert z_loc.abs().max().item() < 1.0, (
            f"Initial z_loc max abs = {z_loc.abs().max().item():.4f}, "
            f"expected < 1.0"
        )

    def test_encoder_scale_positive(self):
        """z_scale always positive (softplus + epsilon)."""
        n_regions = 4
        latent_dim = 16
        batch_size = 16

        enc = DCMEncoderNet(n_regions=n_regions, latent_dim=latent_dim)
        x = torch.randn(batch_size, 50, n_regions)

        z_loc, z_scale = enc(x)

        assert (z_scale > 0).all(), (
            f"z_scale has non-positive entries: min={z_scale.min().item()}"
        )
        # Minimum should be at least 1e-5 (our epsilon floor)
        assert z_scale.min().item() >= 1e-5

    def test_encoder_custom_hidden_channels(self):
        """Custom hidden_channels architecture works."""
        enc = DCMEncoderNet(
            n_regions=4,
            latent_dim=10,
            hidden_channels=[16, 32],
        )
        x = torch.randn(4, 80, 4)
        z_loc, z_scale = enc(x)

        assert z_loc.shape == (4, 10)
        assert z_scale.shape == (4, 10)

    def test_encoder_variable_length(self):
        """Encoder handles different T lengths (AdaptiveAvgPool)."""
        enc = DCMEncoderNet(n_regions=3, latent_dim=8)

        for t_len in [20, 50, 100, 200]:
            x = torch.randn(2, t_len, 3)
            z_loc, z_scale = enc(x)
            assert z_loc.shape == (2, 8)
            assert z_scale.shape == (2, 8)
