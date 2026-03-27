"""Integration tests for SVI with all three DCM models.

Tests cover:
- Guide factory (create_guide) returns AutoNormal with correct init_scale
- SVI runner (run_svi) returns expected result format
- End-to-end SVI for all three models: no NaN, loss decreases, posterior extractable
- Prior samples from all three models produce finite values
"""

from __future__ import annotations

import math

import pytest
import torch
import pyro
from pyro.infer.autoguide import AutoNormal

from pyro_dcm.models import (
    task_dcm_model,
    spectral_dcm_model,
    rdcm_model,
    create_guide,
    run_svi,
    extract_posterior_params,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.rdcm_simulator import (
    make_stable_A_rdcm,
    make_block_stimulus_rdcm,
)
from pyro_dcm.forward_models.rdcm_forward import (
    create_regressors,
    generate_bold,
    get_hrf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_data() -> dict:
    """Generate synthetic task DCM data for integration testing."""
    N, M = 3, 1
    A = make_random_stable_A(N, density=0.5, seed=42)
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)

    stim = make_block_stimulus(
        n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
    )

    duration = 30.0
    TR = 2.0
    dt = 0.5

    result = simulate_task_dcm(
        A, C, stim, duration=duration, dt=0.01, TR=TR, SNR=5.0, seed=7,
    )

    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.ones(N, M, dtype=torch.float64)

    return {
        "model": task_dcm_model,
        "model_args": (
            result["bold"],
            result["stimulus"],
            a_mask,
            c_mask,
            t_eval,
            TR,
            dt,
        ),
        "N": N,
        "M": M,
    }


@pytest.fixture()
def spectral_data() -> dict:
    """Generate synthetic spectral DCM data for integration testing."""
    N = 3
    A = make_stable_A_spectral(N, seed=42)
    result = simulate_spectral_dcm(A, TR=2.0, n_freqs=32, seed=42)

    a_mask = torch.ones(N, N, dtype=torch.float64)

    return {
        "model": spectral_dcm_model,
        "model_args": (
            result["csd"],
            result["freqs"],
            a_mask,
            N,
        ),
        "N": N,
    }


@pytest.fixture()
def rdcm_data() -> dict:
    """Generate synthetic rDCM data for integration testing."""
    nr, nu = 3, 1
    A, a_mask = make_stable_A_rdcm(nr, density=0.5, seed=42)
    C = torch.tensor([[0.5], [0.0], [0.0]], dtype=torch.float64)
    c_mask = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)

    u_dt = 0.5
    y_dt = 2.0
    n_time = 400
    u = make_block_stimulus_rdcm(n_time, nu, u_dt, seed=42)

    bold_result = generate_bold(A, C, u, u_dt, y_dt, SNR=3.0)
    hrf = get_hrf(n_time, u_dt)
    X, Y, N_eff = create_regressors(hrf, bold_result["y"], u, u_dt, y_dt)

    return {
        "model": rdcm_model,
        "model_args": (Y, X, a_mask, c_mask, 1),
        "nr": nr,
        "nu": nu,
    }


# ---------------------------------------------------------------------------
# Guide factory tests
# ---------------------------------------------------------------------------


class TestCreateGuide:
    """Tests for the create_guide factory function."""

    def test_create_guide_returns_autonormal(
        self, task_data: dict,
    ) -> None:
        """create_guide must return an AutoNormal instance."""
        guide = create_guide(task_data["model"])
        assert isinstance(guide, AutoNormal)

    def test_create_guide_custom_init_scale(
        self, task_data: dict,
    ) -> None:
        """create_guide with custom init_scale uses that value.

        Verifies that two guides with different init_scale produce
        different initial guide parameters after running a single
        SVI step (different scale -> different initial distributions).
        """
        guide_small = create_guide(task_data["model"], init_scale=0.01)
        guide_large = create_guide(task_data["model"], init_scale=0.1)

        # Both should be AutoNormal
        assert isinstance(guide_small, AutoNormal)
        assert isinstance(guide_large, AutoNormal)

        # Run a single step with each to initialize guide params
        pyro.clear_param_store()
        from pyro.infer import SVI, Trace_ELBO
        svi_small = SVI(
            task_data["model"], guide_small,
            pyro.optim.ClippedAdam({"lr": 0.001}),
            loss=Trace_ELBO(),
        )
        svi_small.step(*task_data["model_args"])
        params_small = {
            k: v.detach().clone()
            for k, v in pyro.get_param_store().items()
            if "scale" in k
        }

        pyro.clear_param_store()
        svi_large = SVI(
            task_data["model"], guide_large,
            pyro.optim.ClippedAdam({"lr": 0.001}),
            loss=Trace_ELBO(),
        )
        svi_large.step(*task_data["model_args"])
        params_large = {
            k: v.detach().clone()
            for k, v in pyro.get_param_store().items()
            if "scale" in k
        }

        # Scales should differ -- larger init_scale produces larger scales
        # Check at least one common scale parameter differs
        common_keys = set(params_small.keys()) & set(params_large.keys())
        assert len(common_keys) > 0, "No common scale params found"
        any_differ = any(
            not torch.allclose(params_small[k], params_large[k])
            for k in common_keys
        )
        assert any_differ, "Init scales should produce different params"


# ---------------------------------------------------------------------------
# SVI runner tests
# ---------------------------------------------------------------------------


class TestRunSvi:
    """Tests for the run_svi runner function."""

    def test_run_svi_returns_expected_keys(
        self, rdcm_data: dict,
    ) -> None:
        """run_svi result must have 'losses', 'final_loss', 'num_steps'."""
        guide = create_guide(rdcm_data["model"])
        result = run_svi(
            rdcm_data["model"],
            guide,
            rdcm_data["model_args"],
            num_steps=10,
        )
        assert "losses" in result
        assert "final_loss" in result
        assert "num_steps" in result

    def test_run_svi_losses_length(
        self, rdcm_data: dict,
    ) -> None:
        """Losses list length must match num_steps."""
        guide = create_guide(rdcm_data["model"])
        result = run_svi(
            rdcm_data["model"],
            guide,
            rdcm_data["model_args"],
            num_steps=20,
        )
        assert len(result["losses"]) == 20

    def test_run_svi_nan_detection(self) -> None:
        """run_svi must raise RuntimeError on NaN ELBO.

        Uses unittest.mock to patch svi.step to return NaN, verifying
        that the NaN detection logic in run_svi works correctly.
        Triggering NaN through a real model is unreliable because Pyro
        validates distribution parameters and raises ValueError before
        a NaN ELBO can be computed.
        """
        from unittest.mock import patch, MagicMock
        import pyro.distributions as dist

        def simple_model(x: torch.Tensor) -> None:
            """Simple model for testing NaN detection."""
            val = pyro.sample(
                "val",
                dist.Normal(
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                ),
            )
            pyro.sample(
                "obs",
                dist.Normal(
                    val,
                    torch.tensor(1.0, dtype=torch.float64),
                ),
                obs=x,
            )

        x = torch.tensor(1.0, dtype=torch.float64)
        guide = create_guide(simple_model)

        # Patch run_svi to verify the NaN check works by calling
        # the function with a model whose SVI step returns NaN
        with pytest.raises(RuntimeError, match="NaN ELBO"):
            # We patch the SVI.step method to return NaN on first call
            original_run_svi = run_svi.__wrapped__ if hasattr(run_svi, '__wrapped__') else None

            pyro.clear_param_store()
            from pyro.infer import SVI as OrigSVI
            orig_step = OrigSVI.step

            call_count = [0]

            def nan_step(self, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] >= 2:
                    return float("nan")
                return orig_step(self, *args, **kwargs)

            with patch.object(OrigSVI, 'step', nan_step):
                run_svi(simple_model, guide, (x,), num_steps=10)


# ---------------------------------------------------------------------------
# All-models integration tests
# ---------------------------------------------------------------------------


class TestAllModelsIntegration:
    """End-to-end SVI tests for all three DCM model variants."""

    def test_task_dcm_svi_end_to_end(
        self, task_data: dict,
    ) -> None:
        """Task DCM: SVI runs, loss decreases, posterior extractable."""
        pyro.clear_param_store()
        model = task_data["model"]
        model_args = task_data["model_args"]

        guide = create_guide(model, init_scale=0.01)
        result = run_svi(
            model, guide, model_args, num_steps=100, lr=0.01,
        )

        # (a) No NaN in losses
        for i, loss in enumerate(result["losses"]):
            assert not math.isnan(loss), f"NaN loss at step {i}"

        # (b) Loss decreases
        first_20 = sum(result["losses"][:20]) / 20
        last_20 = sum(result["losses"][-20:]) / 20
        assert last_20 < first_20, (
            f"Task DCM loss did not decrease: "
            f"first 20 mean={first_20:.4f}, last 20 mean={last_20:.4f}"
        )

        # (c) Posterior extractable with A_free and C
        posterior = extract_posterior_params(guide, model_args)
        assert "median" in posterior
        assert "A_free" in posterior["median"]
        assert "C" in posterior["median"]

    def test_spectral_dcm_svi_end_to_end(
        self, spectral_data: dict,
    ) -> None:
        """Spectral DCM: SVI runs, loss decreases, posterior extractable."""
        pyro.clear_param_store()
        model = spectral_data["model"]
        model_args = spectral_data["model_args"]

        guide = create_guide(model, init_scale=0.01)
        result = run_svi(
            model, guide, model_args, num_steps=100, lr=0.01,
        )

        # (a) No NaN
        for i, loss in enumerate(result["losses"]):
            assert not math.isnan(loss), f"NaN loss at step {i}"

        # (b) Loss decreases
        first_20 = sum(result["losses"][:20]) / 20
        last_20 = sum(result["losses"][-20:]) / 20
        assert last_20 < first_20, (
            f"Spectral DCM loss did not decrease: "
            f"first 20 mean={first_20:.4f}, last 20 mean={last_20:.4f}"
        )

        # (c) Posterior extractable with A_free
        posterior = extract_posterior_params(guide, model_args)
        assert "median" in posterior
        assert "A_free" in posterior["median"]

    def test_rdcm_svi_end_to_end(
        self, rdcm_data: dict,
    ) -> None:
        """rDCM: SVI runs, loss decreases, posterior extractable."""
        pyro.clear_param_store()
        model = rdcm_data["model"]
        model_args = rdcm_data["model_args"]

        guide = create_guide(model, init_scale=0.01)
        result = run_svi(
            model, guide, model_args, num_steps=100, lr=0.01,
        )

        # (a) No NaN
        for i, loss in enumerate(result["losses"]):
            assert not math.isnan(loss), f"NaN loss at step {i}"

        # (b) Loss decreases
        first_20 = sum(result["losses"][:20]) / 20
        last_20 = sum(result["losses"][-20:]) / 20
        assert last_20 < first_20, (
            f"rDCM loss did not decrease: "
            f"first 20 mean={first_20:.4f}, last 20 mean={last_20:.4f}"
        )

        # (c) Posterior extractable with theta_0
        posterior = extract_posterior_params(guide, model_args)
        assert "median" in posterior
        assert "theta_0" in posterior["median"]

    def test_all_models_prior_samples_plausible(
        self,
        task_data: dict,
        spectral_data: dict,
        rdcm_data: dict,
    ) -> None:
        """Prior samples from all three models produce finite values.

        For ODE-based models (task DCM), conditions on small known-good
        parameters to avoid instability from random prior samples.
        """
        # --- Task DCM (conditioned for stability) ---
        N = task_data["N"]
        M = task_data["M"]
        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
            },
        )
        trace = pyro.poutine.trace(conditioned).get_trace(
            *task_data["model_args"]
        )
        pred_bold = trace.nodes["predicted_bold"]["value"]
        assert torch.isfinite(pred_bold).all(), (
            "Task DCM: NaN/Inf in predicted BOLD"
        )

        # --- Spectral DCM (prior samples fine) ---
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            *spectral_data["model_args"]
        )
        pred_csd = trace.nodes["predicted_csd"]["value"]
        assert torch.isfinite(pred_csd.real).all(), (
            "Spectral DCM: NaN/Inf in predicted CSD (real)"
        )
        assert torch.isfinite(pred_csd.imag).all(), (
            "Spectral DCM: NaN/Inf in predicted CSD (imag)"
        )

        # --- rDCM (prior samples fine, linear regression) ---
        trace = pyro.poutine.trace(rdcm_model).get_trace(
            *rdcm_data["model_args"]
        )
        # Check theta values are finite
        for r in range(rdcm_data["nr"]):
            theta_r = trace.nodes[f"theta_{r}"]["value"]
            assert torch.isfinite(theta_r).all(), (
                f"rDCM: NaN/Inf in theta_{r}"
            )
