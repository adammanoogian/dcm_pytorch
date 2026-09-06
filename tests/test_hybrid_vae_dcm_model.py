"""Tests for hybrid VAE-DCM model/guide integration.

Covers the Pyro model/guide pair: ``hybrid_vae_dcm_model`` (decoder)
and ``HybridVAEDCMGuide`` (encoder). Tests verify site matching,
shape correctness, SVI compatibility, and NaN guard behavior.
"""

from __future__ import annotations

import pyro
import pyro.poutine as poutine
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
from pyro_dcm.models.hybrid_vae_dcm import (
    HybridVAEDCMGuide,
    hybrid_vae_dcm_model,
)
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_REGIONS = 4
N_INPUTS = 1
DURATION = 2.0
DT = 0.01
N_STANDARDIZATION_SAMPLES = 50


@pytest.fixture()
def synthetic_fixture():
    """Generate synthetic data and pre-fit packer for testing.

    Creates a 4-region, 1-input latent circuit with 2s duration at
    dt=0.01. Pre-fits packer standardization on 50 simulated
    parameter sets. Returns all components needed for model/guide
    testing.
    """
    pyro.clear_param_store()
    torch.manual_seed(42)

    # Masks
    a_mask = torch.ones(N_REGIONS, N_REGIONS, dtype=torch.float64)
    c_mask = torch.zeros(N_REGIONS, N_INPUTS, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    # Stimulus
    stim = make_block_stimulus(
        n_blocks=2, block_duration=0.5, rest_duration=0.5,
    )

    # Ground truth simulation
    A_true = make_stable_latent_circuit_A(N_REGIONS, seed=42)
    C_true = torch.zeros(N_REGIONS, N_INPUTS, dtype=torch.float64)
    C_true[0, 0] = 0.5

    sim_result = simulate_latent_circuit(
        A_true, C_true, stim, duration=DURATION, dt=DT,
        SNR=10.0, seed=42,
    )
    observed = sim_result["trajectories"]
    t_eval = sim_result["times"]

    # Packer
    packer = LatentCircuitDCMPacker(
        N_REGIONS, N_INPUTS, a_mask, c_mask,
    )

    # Fit standardization on random parameter sets
    samples_list = []
    for i in range(N_STANDARDIZATION_SAMPLES):
        torch.manual_seed(i + 100)
        a_rand = torch.randn(N_REGIONS, N_REGIONS, dtype=torch.float64)
        c_rand = torch.randn(N_REGIONS, N_INPUTS, dtype=torch.float64)
        x0_rand = torch.randn(N_REGIONS, dtype=torch.float64) * 0.01
        prec_rand = torch.tensor(
            5.0 + torch.randn(1).item(), dtype=torch.float64,
        ).abs() + 0.1
        flat = packer.pack(a_rand, c_rand, x0_rand, prec_rand)
        samples_list.append(flat)

    samples = torch.stack(samples_list)
    packer.fit_standardization(samples)

    # Encoder and guide
    encoder = DCMEncoderNet(
        n_regions=N_REGIONS, latent_dim=packer.total_dim,
    ).double()
    guide = HybridVAEDCMGuide(encoder, packer)

    return {
        "observed": observed,
        "stimulus": stim,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "dt": DT,
        "packer": packer,
        "encoder": encoder,
        "guide": guide,
    }


def _model_args(fixture):
    """Extract model arguments from fixture dict."""
    return (
        fixture["observed"],
        fixture["stimulus"],
        fixture["a_mask"],
        fixture["c_mask"],
        fixture["t_eval"],
        fixture["dt"],
        fixture["packer"],
    )


# ---------------------------------------------------------------------------
# Trace / site-matching tests
# ---------------------------------------------------------------------------


class TestSiteMatching:
    """Tests for Pyro sample site naming and matching."""

    def test_model_trace_has_latent_and_obs_sites(
        self, synthetic_fixture,
    ):
        """Model trace has _latent, obs, predicted_trajectories, A."""
        pyro.clear_param_store()
        trace = poutine.trace(hybrid_vae_dcm_model).get_trace(
            *_model_args(synthetic_fixture),
        )
        site_names = set(trace.nodes.keys())
        assert "_latent" in site_names
        assert "obs" in site_names
        assert "predicted_trajectories" in site_names
        assert "A" in site_names

    def test_guide_trace_has_latent_site(self, synthetic_fixture):
        """Guide trace has _latent site."""
        pyro.clear_param_store()
        guide = synthetic_fixture["guide"]
        trace = poutine.trace(guide).get_trace(
            *_model_args(synthetic_fixture),
        )
        site_names = set(trace.nodes.keys())
        assert "_latent" in site_names

    def test_model_guide_site_names_match(self, synthetic_fixture):
        """Model and guide share exactly one stochastic site: _latent."""
        pyro.clear_param_store()

        model_trace = poutine.trace(hybrid_vae_dcm_model).get_trace(
            *_model_args(synthetic_fixture),
        )
        guide = synthetic_fixture["guide"]
        guide_trace = poutine.trace(guide).get_trace(
            *_model_args(synthetic_fixture),
        )

        # Get stochastic sample sites (not deterministic, not _RETURN)
        model_stochastic = {
            name
            for name, node in model_trace.nodes.items()
            if node.get("type") == "sample"
            and not node.get("is_observed", False)
            and not name.startswith("_RETURN")
        }
        guide_stochastic = {
            name
            for name, node in guide_trace.nodes.items()
            if node.get("type") == "sample"
            and not name.startswith("_RETURN")
        }

        # Both should have _latent
        assert "_latent" in model_stochastic
        assert "_latent" in guide_stochastic

        # Guide should only have _latent (no obs)
        # Note: "A" and "predicted_trajectories" are deterministic sites
        # that Pyro 1.9+ reports as type="sample"; filter by name.
        guide_non_module = {
            n for n in guide_stochastic
            if not n.startswith("hybrid_vae_dcm_encoder")
        }
        assert guide_non_module == {"_latent"}


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestShapes:
    """Tests for output tensor shapes."""

    def test_model_predicted_trajectories_shape(
        self, synthetic_fixture,
    ):
        """predicted_trajectories has shape (T, N)."""
        pyro.clear_param_store()
        trace = poutine.trace(hybrid_vae_dcm_model).get_trace(
            *_model_args(synthetic_fixture),
        )
        pred = trace.nodes["predicted_trajectories"]["value"]
        obs = synthetic_fixture["observed"]
        assert pred.shape == obs.shape

    def test_guide_sample_posterior_shapes(self, synthetic_fixture):
        """sample_posterior returns correctly shaped parameter dicts."""
        pyro.clear_param_store()
        guide = synthetic_fixture["guide"]
        obs = synthetic_fixture["observed"]

        n_samples = 10
        samples = guide.sample_posterior(obs, n_samples=n_samples)

        assert samples["A_free"].shape == (
            n_samples, N_REGIONS, N_REGIONS,
        )
        assert samples["C"].shape == (
            n_samples, N_REGIONS, N_INPUTS,
        )
        assert samples["x0"].shape == (n_samples, N_REGIONS)
        assert samples["noise_prec"].shape == (n_samples,)


# ---------------------------------------------------------------------------
# SVI smoke test
# ---------------------------------------------------------------------------


class TestSVI:
    """SVI integration tests."""

    @pytest.mark.svi_legacy
    @pytest.mark.xfail(
        reason=(
            "Mean-field SVI diverges to NaN on this problem. SUPERSEDED by the "
            "Variational Laplace engine -- see docs/03_methods_reference/svi_status.md. "
            "This is the v0.3.0 RECOV-04 finding (SVI B-RMSE 0.3467 vs VL 0.0170), "
            "not a regression. Kept as a documented baseline, not a quality gate."
        ),
        strict=False,
    )
    def test_svi_smoke_elbo_decreases(self, synthetic_fixture):
        """SVI ELBO decreases over 80 steps on synthetic data."""
        pyro.clear_param_store()
        guide = synthetic_fixture["guide"]

        optimizer = ClippedAdam({"lr": 0.005, "clip_norm": 5.0})
        svi = SVI(
            hybrid_vae_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        args = _model_args(synthetic_fixture)
        losses = []
        for _ in range(80):
            loss = svi.step(*args)
            losses.append(loss)

        # Filter out NaN losses (ODE divergence during early training)
        finite_losses = [
            val for val in losses
            if torch.isfinite(torch.tensor(val))
        ]

        # Need at least 10 finite losses to check trend
        assert len(finite_losses) >= 10, (
            f"Too few finite losses ({len(finite_losses)}/80); "
            "model/guide pair may have fundamental issue"
        )

        # Compare first 5 finite vs last 5 finite (windowed avg)
        first_5 = sum(finite_losses[:5]) / 5
        last_5 = sum(finite_losses[-5:]) / 5
        assert last_5 < first_5, (
            f"ELBO did not decrease: first_5_avg={first_5:.2f}, "
            f"last_5_avg={last_5:.2f}"
        )


# ---------------------------------------------------------------------------
# NaN guard test
# ---------------------------------------------------------------------------


class TestNaNGuard:
    """Tests for NaN guard behavior during ODE divergence."""

    def test_nan_guard_produces_finite_loss(self, synthetic_fixture):
        """NaN guard produces finite loss even with extreme params."""
        pyro.clear_param_store()

        packer = synthetic_fixture["packer"]

        # Create a guide that always produces extreme A_free values
        # by overriding the packer's mean_ to be very large
        original_mean = packer.mean_.clone()
        original_std = packer.std_.clone()

        # Set mean to extreme values so unstandardized params are huge
        packer.mean_ = torch.full_like(packer.mean_, 10.0)
        packer.std_ = torch.ones_like(packer.std_)

        guide = synthetic_fixture["guide"]
        optimizer = ClippedAdam({"lr": 0.001, "clip_norm": 1.0})
        svi = SVI(
            hybrid_vae_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        args = _model_args(synthetic_fixture)

        try:
            loss = svi.step(*args)
            assert torch.isfinite(torch.tensor(loss)), (
                f"Loss is not finite: {loss}"
            )
        finally:
            # Restore original packer stats
            packer.mean_ = original_mean
            packer.std_ = original_std
