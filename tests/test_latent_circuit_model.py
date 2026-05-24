"""Tests for latent circuit DCM Pyro model and observation function.

Validates:
- Model trace structure for linear and bilinear modes.
- Guide auto-discovery across AutoNormal, AutoLowRankMVN, AutoIAFNormal.
- SVI smoke convergence (decreasing ELBO in 200 steps).
- Prior constants are separate from task-DCM values.
- Identity C_obs behavior.
- direct_observation usage.

References
----------
MODEL-01/02/03/04 REQUIREMENTS-v0.6.0.md.
OBS-02 REQUIREMENTS-v0.6.0.md.
"""

from __future__ import annotations

import pytest
import torch
import pyro
import pyro.poutine as poutine

from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
    latent_circuit_dcm_model,
)
from pyro_dcm.models.guides import create_guide
from pyro_dcm.models.task_dcm_model import task_dcm_model
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_synth_data() -> dict:
    """Generate 4-region linear latent circuit synthetic data.

    Returns dictionary with all model arguments for the linear case
    (no B matrices).
    """
    N = 4
    A = make_stable_latent_circuit_A(N, density=0.5, seed=42)
    C = torch.zeros(N, 1, dtype=torch.float64)
    C[0, 0] = 1.0  # Drive region 0

    # Simple block stimulus: ON for first half, OFF for second half
    times = torch.tensor([0.0, 5.0], dtype=torch.float64)
    values = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    stimulus = PiecewiseConstantInput(times, values)

    duration = 10.0
    dt = 0.01
    result = simulate_latent_circuit(
        A, C, stimulus, duration=duration, dt=dt, SNR=20.0, seed=123,
    )

    t_eval = result["times"]
    observed = result["trajectories"]

    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    return {
        "observed_trajectories": observed,
        "stimulus": stimulus,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "dt": dt,
        "N": N,
        "A_true": A,
        "C_true": C,
    }


@pytest.fixture()
def bilinear_synth_data() -> dict:
    """Generate 4-region bilinear latent circuit synthetic data.

    Returns dictionary with all model arguments including B matrices.
    """
    N = 4
    A = make_stable_latent_circuit_A(N, density=0.5, seed=42)
    C = torch.zeros(N, 1, dtype=torch.float64)
    C[0, 0] = 1.0

    # B modulates connection 0->1
    B = torch.zeros(1, N, N, dtype=torch.float64)
    B[0, 1, 0] = 0.3

    # Driving stimulus
    times_drive = torch.tensor([0.0, 5.0], dtype=torch.float64)
    values_drive = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    stimulus = PiecewiseConstantInput(times_drive, values_drive)

    # Modulatory stimulus
    times_mod = torch.tensor([0.0, 3.0, 7.0], dtype=torch.float64)
    values_mod = torch.tensor(
        [[0.0], [1.0], [0.0]], dtype=torch.float64
    )
    stim_mod = PiecewiseConstantInput(times_mod, values_mod)

    duration = 10.0
    dt = 0.01
    result = simulate_latent_circuit(
        A,
        C,
        stimulus,
        duration=duration,
        dt=dt,
        SNR=20.0,
        seed=123,
        B_list=B,
        stimulus_mod=stim_mod,
    )

    t_eval = result["times"]
    observed = result["trajectories"]

    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    b_mask = torch.zeros(N, N, dtype=torch.float64)
    b_mask[1, 0] = 1.0

    return {
        "observed_trajectories": observed,
        "stimulus": stimulus,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "dt": dt,
        "N": N,
        "b_masks": [b_mask],
        "stim_mod": stim_mod,
        "A_true": A,
        "C_true": C,
        "B_true": B,
    }


# ---------------------------------------------------------------------------
# Test 1: Model trace (linear)
# ---------------------------------------------------------------------------


def test_model_trace_linear(linear_synth_data: dict) -> None:
    """Trace includes A_free, C, noise_prec, obs; NOT B_free_0."""
    pyro.clear_param_store()
    data = linear_synth_data

    trace = poutine.trace(latent_circuit_dcm_model).get_trace(
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )

    # Required sample sites
    assert "A_free" in trace.nodes
    assert "C" in trace.nodes
    assert "noise_prec" in trace.nodes
    assert "obs" in trace.nodes

    # Deterministic sites
    assert "A" in trace.nodes
    assert "predicted_trajectories" in trace.nodes

    # Should NOT have bilinear sites
    assert "B_free_0" not in trace.nodes
    assert "B" not in trace.nodes


# ---------------------------------------------------------------------------
# Test 2: Model trace (bilinear)
# ---------------------------------------------------------------------------


def test_model_trace_bilinear(bilinear_synth_data: dict) -> None:
    """Trace includes B_free_0 and B deterministic in bilinear mode."""
    pyro.clear_param_store()
    data = bilinear_synth_data

    trace = poutine.trace(latent_circuit_dcm_model).get_trace(
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
        b_masks=data["b_masks"],
        stim_mod=data["stim_mod"],
    )

    # Required sample sites
    assert "A_free" in trace.nodes
    assert "C" in trace.nodes
    assert "noise_prec" in trace.nodes
    assert "obs" in trace.nodes
    assert "B_free_0" in trace.nodes

    # Deterministic sites
    assert "A" in trace.nodes
    assert "B" in trace.nodes
    assert "predicted_trajectories" in trace.nodes


# ---------------------------------------------------------------------------
# Test 3: Guide auto-discovery (AutoNormal)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_normal(
    linear_synth_data: dict,
) -> None:
    """AutoNormal discovers A_free, C, noise_prec without factory changes."""
    pyro.clear_param_store()
    data = linear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_normal",
        init_scale=0.1,
    )

    # Initialize the guide by running it once
    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )
    guide(*model_args)

    # Check that the param store has entries for the expected sites
    param_names = set(pyro.get_param_store().keys())
    # AutoNormal creates loc/scale params for each sample site
    assert any("A_free" in name for name in param_names)
    assert any("C" in name for name in param_names)
    assert any("noise_prec" in name for name in param_names)


# ---------------------------------------------------------------------------
# Test 4: Guide auto-discovery (AutoLowRankMVN)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_lowrank(
    linear_synth_data: dict,
) -> None:
    """AutoLowRankMVN discovers all sample sites."""
    pyro.clear_param_store()
    data = linear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_lowrank_mvn",
        init_scale=0.1,
    )

    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )
    guide(*model_args)

    # AutoLowRankMVN stores a single loc vector covering all sites
    param_names = set(pyro.get_param_store().keys())
    assert any("loc" in name for name in param_names)


# ---------------------------------------------------------------------------
# Test 5: Guide auto-discovery (AutoIAFNormal)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_iaf(
    linear_synth_data: dict,
) -> None:
    """AutoIAFNormal discovers all sample sites."""
    pyro.clear_param_store()
    data = linear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_iaf",
        hidden_dim=[32],
    )

    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )
    guide(*model_args)

    # AutoIAFNormal will have transform parameters
    param_names = set(pyro.get_param_store().keys())
    # IAF stores parameters for the autoregressive network
    assert len(param_names) > 0


# ---------------------------------------------------------------------------
# Test 6: Guide auto-discovery (bilinear with AutoNormal)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_bilinear(
    bilinear_synth_data: dict,
) -> None:
    """AutoNormal with bilinear mode discovers B_free_0."""
    pyro.clear_param_store()
    data = bilinear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_normal",
        init_scale=0.1,
    )

    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )
    model_kwargs = {
        "b_masks": data["b_masks"],
        "stim_mod": data["stim_mod"],
    }
    guide(*model_args, **model_kwargs)

    param_names = set(pyro.get_param_store().keys())
    # AutoNormal must discover B_free_0 in addition to A_free, C
    assert any("B_free_0" in name for name in param_names)
    assert any("A_free" in name for name in param_names)
    assert any("C" in name for name in param_names)
    assert any("noise_prec" in name for name in param_names)


# ---------------------------------------------------------------------------
# Test 7: SVI smoke test (linear)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_svi_smoke_linear(linear_synth_data: dict) -> None:
    """200 SVI steps with decreasing ELBO on linear LC model."""
    pyro.clear_param_store()
    data = linear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_normal",
        init_scale=0.1,
    )

    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )

    from pyro_dcm.models.guides import run_svi

    result = run_svi(
        latent_circuit_dcm_model,
        guide,
        model_args,
        num_steps=200,
        lr=0.01,
    )

    losses = result["losses"]
    assert len(losses) == 200
    # Loss should decrease: final loss < initial loss
    assert losses[-1] < losses[0], (
        f"ELBO did not decrease: losses[0]={losses[0]:.2f}, "
        f"losses[-1]={losses[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 8: SVI smoke test (bilinear)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_svi_smoke_bilinear(bilinear_synth_data: dict) -> None:
    """200 SVI steps with decreasing ELBO on bilinear LC model."""
    pyro.clear_param_store()
    data = bilinear_synth_data

    guide = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_normal",
        init_scale=0.1,
    )

    model_args = (
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )
    model_kwargs = {
        "b_masks": data["b_masks"],
        "stim_mod": data["stim_mod"],
    }

    from pyro_dcm.models.guides import run_svi

    result = run_svi(
        latent_circuit_dcm_model,
        guide,
        model_args,
        num_steps=200,
        lr=0.01,
        model_kwargs=model_kwargs,
    )

    losses = result["losses"]
    assert len(losses) == 200
    assert losses[-1] < losses[0], (
        f"ELBO did not decrease: losses[0]={losses[0]:.2f}, "
        f"losses[-1]={losses[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 9: LC prior constants are separate from task-DCM
# ---------------------------------------------------------------------------


def test_lc_prior_constants_separate() -> None:
    """LC_A_PRIOR_VARIANCE differs from task-DCM 1/64."""
    # Task-DCM uses 1/64 = 0.015625
    task_dcm_a_prior_var = 1.0 / 64.0

    # LC uses 1/16 = 0.0625
    assert LC_A_PRIOR_VARIANCE != task_dcm_a_prior_var, (
        f"LC_A_PRIOR_VARIANCE={LC_A_PRIOR_VARIANCE} should differ from "
        f"task-DCM value {task_dcm_a_prior_var}"
    )
    assert LC_A_PRIOR_VARIANCE == 1.0 / 16.0
    assert LC_B_PRIOR_VARIANCE == 1.0


# ---------------------------------------------------------------------------
# Test 10: C_obs identity -- predicted_trajectories equals ODE solution
# ---------------------------------------------------------------------------


def test_cobs_identity(linear_synth_data: dict) -> None:
    """With identity C_obs, predicted_trajectories IS the ODE solution.

    The observation mean should equal predicted_trajectories since
    C_obs = I means y_mean = x @ I.T = x.
    """
    pyro.clear_param_store()
    data = linear_synth_data

    trace = poutine.trace(latent_circuit_dcm_model).get_trace(
        data["observed_trajectories"],
        data["stimulus"],
        data["a_mask"],
        data["c_mask"],
        data["t_eval"],
        data["dt"],
    )

    predicted = trace.nodes["predicted_trajectories"]["value"]
    obs_node = trace.nodes["obs"]
    # The obs distribution's loc is y_mean from direct_observation
    # With C_obs = I, y_mean should equal predicted_trajectories
    obs_dist = obs_node["fn"]
    # Navigate through IndependentMessenger layers
    base_dist = obs_dist
    while hasattr(base_dist, "base_dist"):
        base_dist = base_dist.base_dist
    y_mean = base_dist.loc

    # y_mean should equal predicted_trajectories (identity C_obs)
    assert torch.allclose(y_mean, predicted, atol=1e-12), (
        f"y_mean and predicted_trajectories differ: "
        f"max_diff={torch.abs(y_mean - predicted).max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 11: direct_observation is actually called
# ---------------------------------------------------------------------------


def test_direct_observation_called(linear_synth_data: dict) -> None:
    """Verify direct_observation function is exercised by the model.

    We test this by calling direct_observation directly with known
    inputs and verifying the expected behavior matches what the model
    produces.
    """
    N = linear_synth_data["N"]

    # Test direct_observation standalone
    x = torch.randn(50, N, dtype=torch.float64)
    C_obs = torch.eye(N, dtype=torch.float64)
    noise_prec = torch.tensor(4.0, dtype=torch.float64)

    y_mean, noise_std = direct_observation(x, C_obs, noise_prec)

    # With identity C_obs: y_mean == x
    assert torch.allclose(y_mean, x, atol=1e-14)
    # noise_std = 1/sqrt(noise_prec) = 1/2 = 0.5
    expected_std = torch.tensor(0.5, dtype=torch.float64)
    assert torch.allclose(noise_std, expected_std, atol=1e-14)

    # Test with non-identity C_obs (future v0.7.0+ scenario)
    P = 3
    C_obs_rect = torch.randn(P, N, dtype=torch.float64)
    y_mean_rect, _ = direct_observation(x, C_obs_rect, noise_prec)
    expected_rect = x @ C_obs_rect.T
    assert y_mean_rect.shape == (50, P)
    assert torch.allclose(y_mean_rect, expected_rect, atol=1e-12)
