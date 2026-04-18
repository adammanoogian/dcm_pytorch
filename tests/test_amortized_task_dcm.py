"""Tests for amortized task DCM inference with normalizing flow guide.

Verifies that the AmortizedFlowGuide + wrapper model pattern produces
finite, decreasing ELBO on task DCM data, and that posterior sampling
returns correctly shaped tensors in under 1 second.

References
----------
[REF-042] Radev et al. (2020). BayesFlow.
[REF-043] Cranmer, Brehmer & Louppe (2020). SBI frontier.
07-02-PLAN.md: Task 2 test specifications.
"""

from __future__ import annotations

import math
import time

import pyro
import pyro.distributions as dist
import pyro.poutine
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_dcm.guides import (
    AmortizedFlowGuide,
    BoldSummaryNet,
    TaskDCMPacker,
)
from pyro_dcm.models.amortized_wrappers import amortized_task_dcm_model
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from scripts.generate_training_data import invert_A_to_A_free


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def small_task_setup():
    """Create small task DCM setup for fast CI testing.

    Uses 3 regions, 1 input, short duration (60s), coarse dt (1.0)
    to minimize ODE computation time.
    """
    torch.manual_seed(42)
    pyro.set_rng_seed(42)

    n_regions = 3
    n_inputs = 1
    TR = 2.0
    dt = 0.5  # Trade-off: 2x slower ODE but numerically stable
    n_blocks = 2
    block_duration = 15.0
    rest_duration = 15.0
    duration = n_blocks * (block_duration + rest_duration)

    stimulus = make_block_stimulus(
        n_blocks=n_blocks,
        block_duration=block_duration,
        rest_duration=rest_duration,
        n_inputs=n_inputs,
    )

    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.ones(n_regions, n_inputs, dtype=torch.float64)
    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)

    # Generate a few simulations for fitting standardization
    data_list = []
    params_list = []
    for seed_i in range(10):
        A = make_random_stable_A(n_regions, seed=100 + seed_i)
        torch.manual_seed(200 + seed_i)
        C = torch.randn(n_regions, n_inputs, dtype=torch.float64) * c_mask
        try:
            result = simulate_task_dcm(
                A, C, stimulus,
                duration=duration, dt=0.01, TR=TR, SNR=5.0,
                seed=300 + seed_i,
            )
        except Exception:
            continue
        bold = result["bold"]
        if torch.isnan(bold).any() or torch.isinf(bold).any():
            continue
        A_free = invert_A_to_A_free(A)
        signal_std = result["bold_clean"].std(dim=0)
        signal_var = signal_std.pow(2).mean()
        noise_prec = torch.tensor(
            25.0 / signal_var.item() if signal_var.item() > 0 else 1.0,
            dtype=torch.float64,
        )
        data_list.append(bold)
        params_list.append({
            "A_free": A_free, "C": C, "noise_prec": noise_prec,
        })

    assert len(data_list) >= 3, (
        f"Need at least 3 valid simulations, got {len(data_list)}"
    )

    packer = TaskDCMPacker(n_regions, n_inputs, a_mask, c_mask)
    packer.fit_standardization(params_list)

    summary_net = BoldSummaryNet(n_regions, embed_dim=64).double()
    guide = AmortizedFlowGuide(
        summary_net, packer.n_features,
        embed_dim=64,
        n_transforms=3,  # Fewer transforms for speed
        n_bins=4,         # Fewer bins for speed
        hidden_features=[64, 64],  # Smaller hidden layers
        packer=packer,
    ).double()

    return {
        "guide": guide,
        "packer": packer,
        "data_list": data_list,
        "params_list": params_list,
        "stimulus": stimulus,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "TR": TR,
        "dt": dt,
        "n_regions": n_regions,
        "n_inputs": n_inputs,
    }


# ---------- tests ----------

class TestAmortizedGuideConstruction:
    """Tests for guide construction and basic properties."""

    def test_amortized_guide_construction(self):
        """AmortizedFlowGuide with BoldSummaryNet(3) has n_features=13."""
        n_regions, n_inputs = 3, 1
        a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
        c_mask = torch.ones(n_regions, n_inputs, dtype=torch.float64)

        packer = TaskDCMPacker(n_regions, n_inputs, a_mask, c_mask)
        # Fit dummy standardization
        dummy = [
            {
                "A_free": torch.randn(3, 3, dtype=torch.float64),
                "C": torch.randn(3, 1, dtype=torch.float64),
                "noise_prec": torch.tensor(1.0, dtype=torch.float64),
            }
            for _ in range(10)
        ]
        packer.fit_standardization(dummy)

        net = BoldSummaryNet(n_regions, embed_dim=128).double()
        guide = AmortizedFlowGuide(
            net, packer.n_features, packer=packer,
        ).double()

        # n_features = 3*3 + 3*1 + 1 = 13
        assert packer.n_features == 13
        assert guide.n_features == 13


class TestWrapperModelTrace:
    """Tests for the wrapper model Pyro trace structure."""

    def test_wrapper_model_trace(self, small_task_setup):
        """Wrapper model trace has _latent, obs (observed), and A sites."""
        pyro.clear_param_store()
        setup = small_task_setup
        packer = setup["packer"]

        # Condition on a specific _latent value
        z_val = torch.zeros(packer.n_features, dtype=torch.float64)
        conditioned = pyro.poutine.condition(
            amortized_task_dcm_model,
            data={"_latent": z_val},
        )

        trace = pyro.poutine.trace(conditioned).get_trace(
            setup["data_list"][0],
            setup["stimulus"],
            setup["a_mask"],
            setup["c_mask"],
            setup["t_eval"],
            setup["TR"],
            setup["dt"],
            packer,
        )

        # Check _latent site exists and has finite log_prob
        assert "_latent" in trace.nodes
        latent_lp = trace.nodes["_latent"]["fn"].log_prob(z_val)
        assert torch.isfinite(latent_lp).all()

        # Check obs site exists and is observed
        assert "obs" in trace.nodes
        assert trace.nodes["obs"]["is_observed"]

        # Check A deterministic site exists
        assert "A" in trace.nodes

        # Check predicted_bold deterministic site
        assert "predicted_bold" in trace.nodes
        pred = trace.nodes["predicted_bold"]["value"]
        assert torch.isfinite(pred).all()


class TestSiteMatching:
    """Tests for guide-model site compatibility."""

    def test_guide_model_site_matching(self, small_task_setup):
        """Guide and wrapper model both have exactly '_latent' site."""
        pyro.clear_param_store()
        setup = small_task_setup
        guide = setup["guide"]
        packer = setup["packer"]
        bold = setup["data_list"][0]

        # Get guide trace
        guide_trace = pyro.poutine.trace(guide).get_trace(
            bold,
            setup["stimulus"],
            setup["a_mask"],
            setup["c_mask"],
            setup["t_eval"],
            setup["TR"],
            setup["dt"],
            packer,
        )

        # Get model trace (unconditioned -- shows true sample sites)
        model_trace = pyro.poutine.trace(
            amortized_task_dcm_model,
        ).get_trace(
            bold,
            setup["stimulus"],
            setup["a_mask"],
            setup["c_mask"],
            setup["t_eval"],
            setup["TR"],
            setup["dt"],
            packer,
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


class TestSVIConvergence:
    """Tests for SVI training convergence."""

    @pytest.mark.slow
    def test_svi_elbo_convergence_small(self, small_task_setup):
        """100 SVI steps produce finite, decreasing ELBO.

        Task DCM ODE integration can produce NaN for some latent
        samples. The test filters NaN losses and checks that finite
        losses decrease. Uses a fresh guide to avoid state pollution.
        """
        pyro.clear_param_store()
        pyro.enable_validation(False)

        try:
            setup = small_task_setup
            packer = setup["packer"]
            bold = setup["data_list"][0]

            # Create a fresh guide to avoid state from other tests
            fresh_net = BoldSummaryNet(
                setup["n_regions"], embed_dim=64,
            ).double()
            fresh_guide = AmortizedFlowGuide(
                fresh_net, packer.n_features,
                embed_dim=64,
                n_transforms=3,
                n_bins=4,
                hidden_features=[64, 64],
                packer=packer,
            ).double()

            svi = SVI(
                amortized_task_dcm_model,
                fresh_guide,
                ClippedAdam({"lr": 1e-3, "clip_norm": 10.0}),
                loss=Trace_ELBO(num_particles=1),
            )

            losses = []
            for step in range(100):
                loss = svi.step(
                    bold,
                    setup["stimulus"],
                    setup["a_mask"],
                    setup["c_mask"],
                    setup["t_eval"],
                    setup["TR"],
                    setup["dt"],
                    packer,
                )
                losses.append(loss)

            # Filter finite losses (ODE can produce NaN occasionally)
            finite = [l for l in losses if not math.isnan(l)]
            assert len(finite) >= 50, (
                f"Too many NaN losses: {len(losses) - len(finite)}/100"
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
        finally:
            pyro.enable_validation(True)


class TestPosteriorSampling:
    """Tests for posterior sample shapes and validity."""

    def test_posterior_sampling(self, small_task_setup):
        """sample_posterior returns dict with correct shapes, no NaN.

        Tests the forward-pass sampling without SVI training.
        A randomly initialized flow should still produce finite
        samples (prior-like but structurally correct).
        """
        setup = small_task_setup
        guide = setup["guide"]
        bold = setup["data_list"][0]

        n_samples = 100
        samples = guide.sample_posterior(bold, n_samples=n_samples)

        # Check keys
        assert "A_free" in samples
        assert "C" in samples
        assert "noise_prec" in samples

        # Check shapes
        n_r = setup["n_regions"]
        n_i = setup["n_inputs"]
        assert samples["A_free"].shape == (n_samples, n_r, n_r)
        assert samples["C"].shape == (n_samples, n_r, n_i)
        assert samples["noise_prec"].shape == (n_samples,)

        # No NaN/Inf (flow outputs should always be finite)
        for key, val in samples.items():
            assert torch.isfinite(val).all(), f"NaN/Inf in {key}"

    def test_inference_speed(self, small_task_setup):
        """Single forward pass for 1000 samples takes < 1 second."""
        setup = small_task_setup
        guide = setup["guide"]
        bold = setup["data_list"][0]

        # Warm-up forward pass
        guide.eval()
        with torch.no_grad():
            _ = guide.sample_posterior(bold, n_samples=10)

        # Timed run
        t0 = time.time()
        with torch.no_grad():
            samples = guide.sample_posterior(bold, n_samples=1000)
        elapsed = time.time() - t0

        assert elapsed < 1.0, (
            f"Posterior sampling too slow: {elapsed:.3f}s > 1.0s"
        )
        assert samples["A_free"].shape[0] == 1000


# ------------------------------------------------------------------
# Bilinear refusal (MODEL-07, Phase 15-03)
# ------------------------------------------------------------------


class TestAmortizedRefusesBilinear:
    """MODEL-07: amortized_task_dcm_model refuses bilinear kwargs per D5.

    v0.3.0 defers bilinear amortized inference to v0.3.1. The wrapper raises
    NotImplementedError with a message explicitly referencing 'v0.3.1' when
    b_masks is non-empty. Linear behavior (b_masks=None or []) is unchanged.
    """

    def test_amortized_wrapper_refuses_bilinear_kwargs(self) -> None:
        """Non-empty b_masks -> NotImplementedError with v0.3.1 in the message."""
        from pyro_dcm.guides.parameter_packing import TaskDCMPacker
        from pyro_dcm.models.amortized_wrappers import (
            amortized_task_dcm_model,
        )
        from pyro_dcm.simulators.task_simulator import (
            make_block_stimulus,
            make_epoch_stimulus,
        )
        from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

        N, M = 3, 1
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        packer = TaskDCMPacker(N, M, a_mask, c_mask)

        # Minimal linear-compatible kwargs; the guard fires BEFORE any other work.
        stim_dict = make_block_stimulus(
            n_blocks=1, block_duration=8.0, rest_duration=7.0, n_inputs=M,
        )
        stimulus = PiecewiseConstantInput(
            stim_dict["times"], stim_dict["values"],
        )
        t_eval = torch.arange(0, 20.0, 0.5, dtype=torch.float64)
        observed_bold = torch.zeros(10, N, dtype=torch.float64)

        # Bilinear kwargs that should trip the guard.
        b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
        b_mask_0[1, 0] = 1.0
        stim_mod_dict = make_epoch_stimulus(
            event_times=[5.0], event_durations=[5.0], event_amplitudes=[1.0],
            duration=20.0, dt=0.01, n_inputs=1,
        )
        stim_mod = PiecewiseConstantInput(
            stim_mod_dict["times"], stim_mod_dict["values"],
        )

        with pytest.raises(NotImplementedError, match=r"v0\.3\.1"):
            amortized_task_dcm_model(
                observed_bold=observed_bold,
                stimulus=stimulus,
                a_mask=a_mask,
                c_mask=c_mask,
                t_eval=t_eval,
                TR=2.0,
                dt=0.5,
                packer=packer,
                b_masks=[b_mask_0],  # the trigger
                stim_mod=stim_mod,
            )

    def test_amortized_wrapper_linear_mode_unchanged(self) -> None:
        """b_masks=None + b_masks=[] both pass through to the linear body.

        MODEL-07 regression gate: adding the keyword-only kwargs must NOT
        break the linear amortized path. We verify the wrapper runs without
        raising through a poutine.trace on linear-compatible kwargs.
        """
        from pyro_dcm.guides.parameter_packing import TaskDCMPacker
        from pyro_dcm.models.amortized_wrappers import (
            amortized_task_dcm_model,
        )
        from pyro_dcm.simulators.task_simulator import (
            make_block_stimulus,
            make_random_stable_A,
            simulate_task_dcm,
        )

        N, M = 3, 1
        A = make_random_stable_A(N, density=0.5, seed=42)
        C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
        stim = make_block_stimulus(
            n_blocks=1, block_duration=8.0, rest_duration=7.0, n_inputs=M,
        )
        res = simulate_task_dcm(
            A, C, stim, duration=20.0, dt=0.01, TR=2.0, SNR=5.0, seed=7,
        )
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        t_eval = torch.arange(0, 20.0, 0.5, dtype=torch.float64)
        packer = TaskDCMPacker(N, M, a_mask, c_mask)
        # Fit standardization with a trivial dataset (single point).
        packer.fit_standardization([{
            "A_free": torch.zeros(N, N, dtype=torch.float64),
            "C": torch.zeros(N, M, dtype=torch.float64),
            "noise_prec": torch.tensor(10.0, dtype=torch.float64),
        }])
        # Above dataset has std=0 which clamps to 1e-6 (fit_standardization.clamp).

        pyro.clear_param_store()

        # Case 1: b_masks=None (default) -- should run without error.
        try:
            pyro.poutine.trace(amortized_task_dcm_model).get_trace(
                res["bold"],
                res["stimulus"],
                a_mask,
                c_mask,
                t_eval,
                2.0,
                0.5,
                packer,
                # b_masks / stim_mod omitted -> None defaults
            )
        except NotImplementedError:
            pytest.fail(
                "amortized_task_dcm_model raised NotImplementedError on "
                "linear (b_masks=None) path -- regression vs pre-15-03 "
                "behavior."
            )

        pyro.clear_param_store()

        # Case 2: b_masks=[] -- should also pass through per API.
        try:
            pyro.poutine.trace(amortized_task_dcm_model).get_trace(
                res["bold"],
                res["stimulus"],
                a_mask,
                c_mask,
                t_eval,
                2.0,
                0.5,
                packer,
                b_masks=[],
                stim_mod=None,
            )
        except NotImplementedError:
            pytest.fail(
                "amortized_task_dcm_model raised NotImplementedError on "
                "b_masks=[] path -- empty list must pass through per API."
            )
