"""Recovery validation tests for trained hybrid VAE-DCM encoder.

These tests load a trained encoder checkpoint and recovery report from
``results/hybrid_vae_dcm/`` and validate that the amortized inference
meets acceptance thresholds. All tests are skipped if the checkpoint
or report files are not found (i.e., before the cluster training job
has completed).

Markers
-------
- ``@pytest.mark.slow``: skipped in fast CI runs (``-m "not slow"``).
- All tests skip gracefully when artifacts are missing.

References
----------
25-04-PLAN.md: Full-scale cluster training and recovery validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# Default results directory
RESULTS_DIR = Path("results/hybrid_vae_dcm")
CHECKPOINT_PATH = RESULTS_DIR / "encoder_checkpoint.pt"
REPORT_PATH = RESULTS_DIR / "recovery_report.json"


def _load_report() -> dict:
    """Load recovery report JSON, skip if not found."""
    if not REPORT_PATH.exists():
        pytest.skip(
            f"Recovery report not found: {REPORT_PATH}. "
            "Run cluster training first."
        )
    with open(REPORT_PATH) as f:
        return json.load(f)


@pytest.mark.slow
def test_load_trained_encoder() -> None:
    """Load encoder state_dict and verify it produces valid outputs."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run cluster training first."
        )

    from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    n_regions = ckpt["n_regions"]
    latent_dim = ckpt["latent_dim"]

    encoder = DCMEncoderNet(n_regions, latent_dim).double()
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    # Create dummy input and verify output shapes
    T = 200
    x = torch.randn(T, n_regions, dtype=torch.float64)
    with torch.no_grad():
        z_loc, z_scale = encoder(x)

    assert z_loc.shape == (latent_dim,), (
        f"Expected z_loc shape ({latent_dim},), got {z_loc.shape}"
    )
    assert z_scale.shape == (latent_dim,), (
        f"Expected z_scale shape ({latent_dim},), got {z_scale.shape}"
    )
    assert torch.isfinite(z_loc).all(), "z_loc contains non-finite values"
    assert (z_scale > 0).all(), "z_scale must be positive"


@pytest.mark.slow
def test_recovery_a_rmse_below_threshold() -> None:
    """Mean A_free RMSE should be below 0.3 on held-out test set."""
    report = _load_report()
    a_rmse = report["aggregated"]["A_free_rmse_mean"]
    threshold = 0.3
    assert a_rmse < threshold, (
        f"A_free RMSE {a_rmse:.4f} exceeds threshold {threshold}. "
        f"May need more training epochs or architecture tuning."
    )


@pytest.mark.slow
def test_recovery_sign_accuracy_above_chance() -> None:
    """Mean A_free sign recovery should be above 0.6 (chance = 0.5)."""
    report = _load_report()
    sign_acc = report["aggregated"]["A_sign_recovery_mean"]
    threshold = 0.6
    assert sign_acc > threshold, (
        f"A sign recovery {sign_acc:.4f} below threshold {threshold}. "
        f"Chance level is 0.5; encoder should do meaningfully better."
    )


@pytest.mark.slow
def test_amortized_inference_timing() -> None:
    """Mean inference time should be under 1.0 second per example."""
    report = _load_report()
    mean_time = report["inference_timing"]["mean_seconds"]
    threshold = 1.0
    assert mean_time < threshold, (
        f"Mean inference time {mean_time:.4f}s exceeds {threshold}s. "
        f"Amortized inference should be a single forward pass."
    )


@pytest.mark.slow
def test_kl_not_collapsed() -> None:
    """Final KL divergence should be above 0.1 (no posterior collapse).

    If KL drops to near zero, the encoder has collapsed to the prior
    and is not encoding subject-specific information (Pitfall 2 from
    25-RESEARCH.md).
    """
    report = _load_report()
    kl = report["kl_divergence"]["final_epoch_mean"]
    threshold = 0.1
    assert kl > threshold, (
        f"KL divergence {kl:.4f} below {threshold}. "
        f"Posterior may have collapsed to the prior."
    )
