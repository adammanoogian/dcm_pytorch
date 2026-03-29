"""Tests for amortized spectral DCM inference with normalizing flow guide.

Verifies that the AmortizedFlowGuide + CsdSummaryNet + wrapper model pattern
produces finite, decreasing ELBO on spectral DCM data, and that posterior
sampling returns correctly shaped tensors in under 1 second.

The spectral DCM forward model is algebraic (no ODE), so each SVI step is
fast (~10ms). This enables more SVI steps and larger test configurations
than task DCM tests without exceeding CI time limits.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting state
    fMRI. NeuroImage, 94, 396-407. Eq. 3-10.
[REF-042] Radev et al. (2020). BayesFlow.
[REF-043] Cranmer, Brehmer & Louppe (2020). SBI frontier.
07-03-PLAN.md: Task 1 test specifications.
"""

from __future__ import annotations

import math
import time

import pyro
import pyro.poutine
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_dcm.guides import (
    AmortizedFlowGuide,
    CsdSummaryNet,
    SpectralDCMPacker,
)
from pyro_dcm.models.amortized_wrappers import amortized_spectral_dcm_model
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.forward_models.spectral_noise import default_noise_priors
from scripts.generate_training_data import invert_A_to_A_free


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def spectral_test_data():
    """Generate a single spectral DCM dataset for testing.

    Uses make_stable_A_spectral(3, seed=42) and simulate_spectral_dcm
    to produce a deterministic CSD tensor with known true parameters.
    """
    torch.manual_seed(42)
    A = make_stable_A_spectral(3, seed=42)
    result = simulate_spectral_dcm(A, seed=42)
    csd = result["csd"]
    freqs = result["freqs"]
    A_free = invert_A_to_A_free(A)

    # Default noise params (all zeros in log-space)
    priors = default_noise_priors(3)
    true_params = {
        "A_free": A_free,
        "noise_a": priors["a_prior_mean"],
        "noise_b": priors["b_prior_mean"],
        "noise_c": priors["c_prior_mean"],
        "csd_noise_scale": torch.tensor(1.0, dtype=torch.float64),
    }
    return {
        "csd": csd,
        "freqs": freqs,
        "A": A,
        "true_params": true_params,
    }


@pytest.fixture(scope="module")
def spectral_packer():
    """Create SpectralDCMPacker(n_regions=3) fitted on 20 simulations.

    Spectral DCM simulation is fast (~0.01s each), so generating 20
    datasets in a fixture is acceptable for CI.
    """
    n_regions = 3
    packer = SpectralDCMPacker(n_regions)

    params_list = []
    for seed_i in range(20):
        A = make_stable_A_spectral(n_regions, seed=100 + seed_i)
        torch.manual_seed(200 + seed_i)
        priors = default_noise_priors(n_regions)
        noise_a = priors["a_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )
        noise_b = priors["b_prior_mean"] + 0.1 * torch.randn(
            2, 1, dtype=torch.float64,
        )
        noise_c = priors["c_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )
        A_free = invert_A_to_A_free(A)
        csd_noise_scale = torch.tensor(1.0, dtype=torch.float64)

        params_list.append({
            "A_free": A_free,
            "noise_a": noise_a,
            "noise_b": noise_b,
            "noise_c": noise_c,
            "csd_noise_scale": csd_noise_scale,
        })

    packer.fit_standardization(params_list)
    return packer


@pytest.fixture(scope="module")
def spectral_guide(spectral_packer):
    """Create AmortizedFlowGuide with CsdSummaryNet for 3-region spectral DCM.

    Uses small config for CI speed: n_transforms=2, hidden_features=[64, 64].
    Production uses n_transforms=5, hidden_features=[256, 256].
    """
    packer = spectral_packer
    n_regions = 3
    n_freqs = 32

    summary_net = CsdSummaryNet(
        n_regions, n_freqs=n_freqs, embed_dim=128,
    ).double()

    guide = AmortizedFlowGuide(
        summary_net, packer.n_features,
        embed_dim=128,
        n_transforms=2,
        hidden_features=[64, 64],
        packer=packer,
    ).double()

    return guide


# ---------- tests ----------

class TestSpectralGuideConstruction:
    """Tests for spectral guide construction and basic properties."""

    def test_spectral_guide_construction(self):
        """AmortizedFlowGuide with CsdSummaryNet(3) has n_features=24.

        n_features = N*N + 2*N + 2 + 2*N + 1 = 9 + 6 + 2 + 6 + 1 = 24
        for N=3.
        """
        n_regions = 3
        packer = SpectralDCMPacker(n_regions)

        # Fit dummy standardization
        dummy = []
        for _ in range(10):
            dummy.append({
                "A_free": torch.randn(3, 3, dtype=torch.float64),
                "noise_a": torch.randn(2, 3, dtype=torch.float64),
                "noise_b": torch.randn(2, 1, dtype=torch.float64),
                "noise_c": torch.randn(2, 3, dtype=torch.float64),
                "csd_noise_scale": torch.tensor(
                    1.0, dtype=torch.float64,
                ),
            })
        packer.fit_standardization(dummy)

        net = CsdSummaryNet(
            n_regions, n_freqs=32, embed_dim=128,
        ).double()

        guide = AmortizedFlowGuide(
            net, packer.n_features, packer=packer,
        ).double()

        # n_features = 3*3 + 2*3 + 2 + 2*3 + 1 = 24
        assert packer.n_features == 24
        assert guide.n_features == 24


class TestSpectralWrapperModelTrace:
    """Tests for the spectral wrapper model Pyro trace structure."""

    def test_spectral_wrapper_model_trace(
        self, spectral_test_data, spectral_packer,
    ):
        """Wrapper model trace has _latent, obs_csd (observed), A, and predicted_csd."""
        pyro.clear_param_store()
        data = spectral_test_data
        packer = spectral_packer

        # Condition on a specific _latent value
        z_val = torch.zeros(packer.n_features, dtype=torch.float64)
        conditioned = pyro.poutine.condition(
            amortized_spectral_dcm_model,
            data={"_latent": z_val},
        )

        a_mask = torch.ones(3, 3, dtype=torch.float64)
        trace = pyro.poutine.trace(conditioned).get_trace(
            data["csd"],
            data["freqs"],
            a_mask,
            packer,
        )

        # Check _latent site exists and has finite log_prob
        assert "_latent" in trace.nodes
        latent_lp = trace.nodes["_latent"]["fn"].log_prob(z_val)
        assert torch.isfinite(latent_lp).all()

        # Check obs_csd site exists and is observed
        assert "obs_csd" in trace.nodes
        assert trace.nodes["obs_csd"]["is_observed"]

        # Check A deterministic site exists
        assert "A" in trace.nodes

        # Check predicted_csd deterministic site
        assert "predicted_csd" in trace.nodes
        pred = trace.nodes["predicted_csd"]["value"]
        assert torch.isfinite(pred.real).all()
        assert torch.isfinite(pred.imag).all()


class TestSpectralSiteMatching:
    """Tests for guide-model site compatibility."""

    def test_spectral_guide_model_site_matching(
        self, spectral_test_data, spectral_packer, spectral_guide,
    ):
        """Guide and wrapper model both have exactly '_latent' site."""
        pyro.clear_param_store()
        data = spectral_test_data
        packer = spectral_packer
        guide = spectral_guide
        csd = data["csd"]
        freqs = data["freqs"]
        a_mask = torch.ones(3, 3, dtype=torch.float64)

        # Get guide trace
        guide_trace = pyro.poutine.trace(guide).get_trace(
            csd, freqs, a_mask, packer,
        )

        # Get model trace (unconditioned)
        model_trace = pyro.poutine.trace(
            amortized_spectral_dcm_model,
        ).get_trace(
            csd, freqs, a_mask, packer,
        )

        # Extract latent sample site names (non-observed)
        guide_sites = {
            name for name, node in guide_trace.nodes.items()
            if node.get("type") == "sample"
            and not node.get("is_observed", False)
        }
        model_sites = {
            name for name, node in model_trace.nodes.items()
            if node.get("type") == "sample"
            and not node.get("is_observed", False)
        }

        assert guide_sites == {"_latent"}
        assert model_sites == {"_latent"}


class TestSpectralSVIConvergence:
    """Tests for SVI training convergence on spectral DCM."""

    def test_spectral_svi_convergence(
        self, spectral_test_data, spectral_packer,
    ):
        """200 SVI steps produce finite, decreasing ELBO.

        Spectral DCM is algebraic (no ODE), so each step is ~10ms.
        200 steps should take < 5 seconds. Uses num_particles=4 for
        better gradient estimates.
        """
        pyro.clear_param_store()

        data = spectral_test_data
        packer = spectral_packer
        csd = data["csd"]
        freqs = data["freqs"]
        a_mask = torch.ones(3, 3, dtype=torch.float64)

        # Create a fresh guide for this test
        net = CsdSummaryNet(3, n_freqs=32, embed_dim=128).double()
        fresh_guide = AmortizedFlowGuide(
            net, packer.n_features,
            embed_dim=128,
            n_transforms=2,
            hidden_features=[64, 64],
            packer=packer,
        ).double()

        svi = SVI(
            amortized_spectral_dcm_model,
            fresh_guide,
            ClippedAdam({"lr": 1e-3, "clip_norm": 10.0}),
            loss=Trace_ELBO(num_particles=4, vectorize_particles=False),
        )

        losses = []
        for step in range(200):
            loss = svi.step(csd, freqs, a_mask, packer)
            losses.append(loss)

        # All losses should be finite (no ODE to diverge)
        finite = [lo for lo in losses if not math.isnan(lo)]
        assert len(finite) >= 180, (
            f"Too many NaN losses: {len(losses) - len(finite)}/200"
        )

        # Compare early vs late finite losses
        early = finite[:len(finite) // 4]
        late = finite[-(len(finite) // 4):]
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        assert late_avg < early_avg, (
            f"ELBO not decreasing: early={early_avg:.2f}, "
            f"late={late_avg:.2f}"
        )


class TestSpectralPosteriorSampling:
    """Tests for posterior sample shapes and validity."""

    def test_spectral_posterior_sampling(
        self, spectral_test_data, spectral_packer,
    ):
        """sample_posterior returns dict with correct shapes, no NaN.

        After 50 SVI steps (brief training), verify that posterior
        samples have the expected keys and shapes. Uses a fresh guide.
        """
        pyro.clear_param_store()

        data = spectral_test_data
        packer = spectral_packer
        csd = data["csd"]
        freqs = data["freqs"]
        a_mask = torch.ones(3, 3, dtype=torch.float64)

        # Brief training (50 steps)
        net = CsdSummaryNet(3, n_freqs=32, embed_dim=128).double()
        guide = AmortizedFlowGuide(
            net, packer.n_features,
            embed_dim=128,
            n_transforms=2,
            hidden_features=[64, 64],
            packer=packer,
        ).double()

        svi = SVI(
            amortized_spectral_dcm_model,
            guide,
            ClippedAdam({"lr": 1e-3}),
            loss=Trace_ELBO(num_particles=1),
        )
        for _ in range(50):
            svi.step(csd, freqs, a_mask, packer)

        n_samples = 100
        samples = guide.sample_posterior(csd, n_samples=n_samples)

        # Check keys
        assert "A_free" in samples
        assert "noise_a" in samples
        assert "noise_b" in samples
        assert "noise_c" in samples
        assert "csd_noise_scale" in samples

        # Check shapes
        assert samples["A_free"].shape == (100, 3, 3)
        assert samples["noise_a"].shape == (100, 2, 3)
        assert samples["noise_b"].shape == (100, 2, 1)
        assert samples["noise_c"].shape == (100, 2, 3)
        assert samples["csd_noise_scale"].shape == (100,)

        # No NaN/Inf
        for key, val in samples.items():
            assert torch.isfinite(val).all(), f"NaN/Inf in {key}"

    def test_spectral_inference_speed(
        self, spectral_test_data, spectral_guide,
    ):
        """Single forward pass for 1000 samples takes < 1 second."""
        csd = spectral_test_data["csd"]
        guide = spectral_guide

        # Warm-up forward pass
        guide.eval()
        with torch.no_grad():
            _ = guide.sample_posterior(csd, n_samples=10)

        # Timed run
        t0 = time.time()
        with torch.no_grad():
            samples = guide.sample_posterior(csd, n_samples=1000)
        elapsed = time.time() - t0

        assert elapsed < 1.0, (
            f"Posterior sampling too slow: {elapsed:.3f}s > 1.0s"
        )
        assert samples["A_free"].shape[0] == 1000
