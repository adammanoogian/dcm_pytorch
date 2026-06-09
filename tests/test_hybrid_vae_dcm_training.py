"""Tests for hybrid VAE-DCM training infrastructure.

Integration tests for ``generate_synthetic_vae_dataset`` and
``train_hybrid_vae_dcm``. Covers dataset shapes, A-matrix stability,
training loop convergence, KL annealing schedule, and amortized
sign recovery.
"""

from __future__ import annotations

import pyro
import pytest
import torch

from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
from pyro_dcm.models.hybrid_vae_dcm import (
    HybridVAEDCMGuide,
    generate_synthetic_vae_dataset,
    hybrid_vae_dcm_model,
    masked_sign_recovery,
    train_hybrid_vae_dcm,
)


class TestMaskedSignRecovery:
    """masked_sign_recovery ignores structural zeros (HVAE-02 metric fix)."""

    def test_ignores_structural_zeros(self) -> None:
        """Unmasked deflates on sparse A; masked scores only real connections."""
        # 4 of 6 entries are exactly zero (absent connections).
        true = torch.tensor([-0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        # All non-zero signs correct; predictions on zeros are non-zero noise.
        pred = torch.tensor([-0.4, 0.02, 0.25, -0.01, 0.03, -0.02])
        masked = masked_sign_recovery(pred, true)
        unmasked = float(
            (torch.sign(pred) == torch.sign(true)).float().mean().item()
        )
        assert masked == 1.0          # both real connections signed correctly
        assert unmasked < 0.5         # 4 structural zeros are guaranteed misses

    def test_counts_only_wrong_nonzero_signs(self) -> None:
        """A flipped sign on a real connection is penalised."""
        true = torch.tensor([-0.5, 0.3, 0.0, 0.0])
        pred = torch.tensor([-0.4, -0.2, 0.1, 0.1])  # 2nd sign flipped
        assert masked_sign_recovery(true=true, pred=pred) == 0.5

    def test_all_zero_returns_nan(self) -> None:
        """No eligible entry -> nan (so it can be dropped from aggregation)."""
        import math
        true = torch.zeros(4)
        assert math.isnan(masked_sign_recovery(torch.randn(4), true))

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_param_store():
    """Clear Pyro param store before each test."""
    pyro.clear_param_store()


@pytest.fixture()
def small_dataset():
    """Generate a small dataset for shape and stability tests.

    10 samples, N=4, M=1, duration=2.0s, dt=0.01s.
    """
    return generate_synthetic_vae_dataset(
        10,
        n_regions=4,
        n_inputs=1,
        duration=2.0,
        dt=0.01,
        seed=42,
    )


def _make_packer_and_guide(
    dataset: list[dict[str, torch.Tensor]],
    n_regions: int,
    n_inputs: int,
) -> tuple[LatentCircuitDCMPacker, HybridVAEDCMGuide]:
    """Create packer with fitted standardization and guide.

    Parameters
    ----------
    dataset : list of dict
        Training dataset.
    n_regions : int
        Number of latent dimensions.
    n_inputs : int
        Number of driving inputs.

    Returns
    -------
    tuple
        (packer, guide) with packer standardization fitted.
    """
    a_mask = dataset[0]["a_mask"]
    c_mask = dataset[0]["c_mask"]
    packer = LatentCircuitDCMPacker(n_regions, n_inputs, a_mask, c_mask)

    packed = []
    for ex in dataset:
        flat = packer.pack(
            ex["A"], ex["C"], ex["x0"], ex["noise_prec"],
        )
        packed.append(flat)
    samples = torch.stack(packed)
    packer.fit_standardization(samples)

    encoder = DCMEncoderNet(n_regions, packer.total_dim).double()
    guide = HybridVAEDCMGuide(encoder, packer)
    return packer, guide


# -------------------------------------------------------------------
# Dataset shape tests
# -------------------------------------------------------------------


class TestSyntheticDataset:
    """Tests for generate_synthetic_vae_dataset."""

    def test_generate_synthetic_dataset_shapes(self, small_dataset):
        """Each dict has correct tensor shapes for N=4, M=1."""
        n_regions = 4
        n_inputs = 1
        duration = 2.0
        dt = 0.01
        expected_T = int(duration / dt)

        assert len(small_dataset) == 10

        for ex in small_dataset:
            assert ex["observed"].shape == (expected_T, n_regions)
            assert ex["A"].shape == (n_regions, n_regions)
            assert ex["C"].shape == (n_regions, n_inputs)
            assert ex["x0"].shape == (n_regions,)
            assert ex["noise_prec"].shape == ()
            assert ex["t_eval"].shape == (expected_T,)
            assert ex["a_mask"].shape == (n_regions, n_regions)
            assert ex["c_mask"].shape == (n_regions, n_inputs)

            # Verify dtype
            assert ex["observed"].dtype == torch.float64
            assert ex["A"].dtype == torch.float64

    def test_generate_synthetic_dataset_all_stable(
        self, small_dataset,
    ):
        """All generated A matrices have max(Re(eig)) < 0."""
        for i, ex in enumerate(small_dataset):
            eigvals = torch.linalg.eigvals(ex["A"]).real
            max_eig = eigvals.max().item()
            assert max_eig < 0.0, (
                f"Sample {i}: max eigenvalue real part = {max_eig}, "
                "expected < 0 for stability"
            )


# -------------------------------------------------------------------
# Training loop tests
# -------------------------------------------------------------------


class TestTrainingLoop:
    """Tests for train_hybrid_vae_dcm."""

    def test_training_loop_smoke(self):
        """Training for 10 epochs produces decreasing finite losses.

        Uses N=3, M=1, duration=2.0, 20 samples, 10 epochs.
        """
        n_regions, n_inputs = 3, 1
        dataset = generate_synthetic_vae_dataset(
            20,
            n_regions=n_regions,
            n_inputs=n_inputs,
            duration=2.0,
            dt=0.01,
            seed=123,
        )
        _, guide = _make_packer_and_guide(
            dataset, n_regions, n_inputs,
        )

        result = train_hybrid_vae_dcm(
            hybrid_vae_dcm_model,
            guide,
            dataset,
            n_epochs=10,
            warmup_epochs=3,
            lr=0.005,
            clip_norm=5.0,
            log_every=5,
        )

        # Check length
        assert len(result["losses"]) == 10

        # All losses should be finite
        finite_losses = [
            val
            for val in result["losses"]
            if torch.isfinite(torch.tensor(val))
        ]
        assert len(finite_losses) >= 5, (
            f"Too few finite losses: {len(finite_losses)}/10"
        )

        # Loss should decrease: compare early vs late window
        # (skip first 2 epochs which have low beta)
        late = finite_losses[-3:]
        early = finite_losses[:3]
        late_avg = sum(late) / len(late)
        early_avg = sum(early) / len(early)
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.2f}, "
            f"late_avg={late_avg:.2f}"
        )

    def test_kl_annealing_schedule(self):
        """Beta schedule ramps correctly from ~0 to 1."""
        n_regions, n_inputs = 3, 1
        dataset = generate_synthetic_vae_dataset(
            5,
            n_regions=n_regions,
            n_inputs=n_inputs,
            duration=2.0,
            dt=0.01,
            seed=456,
        )
        _, guide = _make_packer_and_guide(
            dataset, n_regions, n_inputs,
        )

        warmup_epochs = 5
        n_epochs = 10

        result = train_hybrid_vae_dcm(
            hybrid_vae_dcm_model,
            guide,
            dataset,
            n_epochs=n_epochs,
            warmup_epochs=warmup_epochs,
            lr=0.005,
            clip_norm=5.0,
            log_every=100,  # suppress logging
        )

        betas = result["betas"]
        assert len(betas) == n_epochs

        # First beta should be ~0 (clamped to 1e-6)
        assert betas[0] < 0.01, (
            f"First beta should be ~0, got {betas[0]}"
        )

        # Beta at warmup_epochs should be 1.0
        assert betas[warmup_epochs] == 1.0, (
            f"Beta at warmup_epochs should be 1.0, "
            f"got {betas[warmup_epochs]}"
        )

        # All subsequent betas should be 1.0
        for i in range(warmup_epochs, n_epochs):
            assert betas[i] == 1.0, (
                f"Beta[{i}] should be 1.0, got {betas[i]}"
            )

        # Betas should be monotonically non-decreasing
        for i in range(1, n_epochs):
            assert betas[i] >= betas[i - 1], (
                f"Beta not monotonic: beta[{i - 1}]={betas[i - 1]}, "
                f"beta[{i}]={betas[i]}"
            )


# -------------------------------------------------------------------
# Amortized inference recovery test
# -------------------------------------------------------------------


class TestAmortizedRecovery:
    """Tests for amortized inference sign pattern recovery."""

    @pytest.mark.slow
    def test_amortized_inference_recovers_sign_pattern(self):
        """Trained encoder recovers A_free sign pattern > chance.

        Trains on 50 examples (N=3, M=1, duration=3.0) for 30 epochs.
        Tests on 5 held-out examples. Mean sign recovery should be
        > 0.5 (better than random binary chance).
        """
        n_regions, n_inputs = 3, 1
        train_data = generate_synthetic_vae_dataset(
            50,
            n_regions=n_regions,
            n_inputs=n_inputs,
            duration=3.0,
            dt=0.01,
            seed=789,
        )
        test_data = generate_synthetic_vae_dataset(
            5,
            n_regions=n_regions,
            n_inputs=n_inputs,
            duration=3.0,
            dt=0.01,
            seed=10_789,
        )
        packer, guide = _make_packer_and_guide(
            train_data, n_regions, n_inputs,
        )

        train_hybrid_vae_dcm(
            hybrid_vae_dcm_model,
            guide,
            train_data,
            n_epochs=30,
            warmup_epochs=5,
            lr=0.005,
            clip_norm=5.0,
            log_every=100,  # suppress logging
        )

        # Evaluate amortized inference on test data
        guide.eval()
        sign_recoveries = []

        with torch.no_grad():
            for ex in test_data:
                z_loc, _ = guide.encoder_net(ex["observed"])
                z = packer.unstandardize(z_loc)
                params = packer.unpack(z)
                a_pred = params["A_free"]
                a_true = ex["A"]

                sign_match = (
                    torch.sign(a_pred) == torch.sign(a_true)
                ).float().mean().item()
                sign_recoveries.append(sign_match)

        mean_recovery = sum(sign_recoveries) / len(sign_recoveries)
        assert mean_recovery > 0.5, (
            f"Mean sign recovery {mean_recovery:.3f} <= 0.5 "
            "(not better than chance)"
        )
