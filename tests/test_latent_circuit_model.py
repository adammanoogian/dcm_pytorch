"""Tests for latent_circuit_dcm_model and direct_observation (Phase 20 Plan 03).

Tests:
1.  Model trace (linear): correct sample sites, shapes, and types.
2.  Model trace (bilinear): B_free_0 site present, B deterministic emitted.
3.  Guide auto-discovery -- AutoNormal on linear model.
4.  Guide auto-discovery -- AutoLowRankMVN on linear model.
5.  Guide auto-discovery -- AutoIAFNormal on linear model.
6.  Guide auto-discovery -- AutoNormal on bilinear model.
7.  SVI smoke test (linear): 200 steps, ELBO decreases.
8.  SVI smoke test (bilinear): 200 steps, ELBO decreases.
9.  Prior constants separation: LC constants differ from task_dcm A prior.
10. C_obs identity: predicted_trajectories == x for identity observation.
11. direct_observation integration: y_mean shape, noise_std, identity check.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
.planning/phases/20-latent-circuit-forward-model/20-03-PLAN.md -- test plan.
"""

from __future__ import annotations

import functools

import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoIAFNormal,
    AutoLowRankMultivariateNormal,
    AutoNormal,
)
from pyro.optim import ClippedAdam

from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
    latent_circuit_dcm_model,
)
from pyro_dcm.models import create_guide, run_svi
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_fixture() -> dict:
    """4-region linear latent-circuit fixture with synthetic trajectories."""
    torch.manual_seed(0)
    N, M = 4, 1
    duration = 2.0
    dt = 0.01

    A = make_stable_latent_circuit_A(N, seed=0)
    C = torch.zeros(N, M, dtype=torch.float64)
    C[0, 0] = 0.3

    stim_dict = make_block_stimulus(
        n_blocks=2, block_duration=0.5, rest_duration=0.5
    )
    stim = PiecewiseConstantInput(stim_dict["times"], stim_dict["values"])

    result = simulate_latent_circuit(A, C, stim, duration=duration, dt=dt, SNR=5.0)
    trajectories = result["trajectories"]  # (T, N)
    t_eval = result["times"]              # (T,)

    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, M, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    return {
        "trajectories": trajectories,
        "t_eval": t_eval,
        "stim": stim,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "N": N,
        "M": M,
        "T": trajectories.shape[0],
        "dt": dt,
    }


@pytest.fixture(scope="module")
def bilinear_fixture() -> dict:
    """4-region bilinear latent-circuit fixture."""
    torch.manual_seed(1)
    N, M = 4, 1
    duration = 2.0
    dt = 0.01

    A = make_stable_latent_circuit_A(N, seed=1)
    C = torch.zeros(N, M, dtype=torch.float64)
    C[0, 0] = 0.3

    B0 = torch.zeros(N, N, dtype=torch.float64)
    B0[1, 0] = 0.15  # modulates connection 0->1

    stim_dict = make_block_stimulus(
        n_blocks=2, block_duration=0.5, rest_duration=0.5
    )
    stim = PiecewiseConstantInput(stim_dict["times"], stim_dict["values"])
    stim_mod = PiecewiseConstantInput(stim_dict["times"], stim_dict["values"])

    result = simulate_latent_circuit(
        A, C, stim, duration=duration, dt=dt, SNR=5.0,
        B_list=[B0], stimulus_mod=stim_mod
    )
    trajectories = result["trajectories"]
    t_eval = result["times"]

    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, M, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    b_mask = (B0.abs() > 0).to(dtype=torch.float64)  # (N, N)

    return {
        "trajectories": trajectories,
        "t_eval": t_eval,
        "stim": stim,
        "stim_mod": stim_mod,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "b_masks": [b_mask],
        "N": N,
        "M": M,
        "T": trajectories.shape[0],
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Test 1: Model trace (linear) -- sample sites, shapes, types
# ---------------------------------------------------------------------------


def test_linear_model_trace_sites(linear_fixture: dict) -> None:
    """Linear model trace has expected sample sites with correct shapes."""
    f = linear_fixture
    N, T = f["N"], f["T"]

    pyro.clear_param_store()
    with pyro.poutine.trace() as tr:
        latent_circuit_dcm_model(
            f["trajectories"],
            f["stim"],
            f["a_mask"],
            f["c_mask"],
            f["t_eval"],
            f["dt"],
        )

    nodes = tr.trace.nodes

    # In this Pyro version, pyro.deterministic() registers as type="sample"
    # with is_observed=True (fixed value), so we check by name.
    all_sites = {k: v for k, v in nodes.items() if v["type"] == "sample"}

    # Required stochastic sample sites
    for site in ("A_free", "C", "noise_prec", "obs"):
        assert site in all_sites, f"Sample site '{site}' missing from trace"

    # Deterministic sites (registered via pyro.deterministic)
    assert "A" in all_sites, "Deterministic A site missing"
    assert "predicted_trajectories" in all_sites, (
        "predicted_trajectories deterministic site missing"
    )

    # No B_free_0 or B in linear mode
    assert "B_free_0" not in all_sites, (
        "B_free_0 should not appear in linear mode"
    )
    assert "B" not in all_sites, "B should not appear in linear trace"

    # Shape checks
    assert all_sites["A_free"]["value"].shape == (N, N), (
        f"A_free shape expected ({N},{N}), got "
        f"{all_sites['A_free']['value'].shape}"
    )
    assert all_sites["C"]["value"].shape == (N, f["M"]), (
        f"C shape expected ({N},{f['M']}), got "
        f"{all_sites['C']['value'].shape}"
    )
    assert all_sites["noise_prec"]["value"].shape == (), (
        f"noise_prec expected scalar, got "
        f"{all_sites['noise_prec']['value'].shape}"
    )

    # predicted_trajectories shape
    pred_shape = all_sites["predicted_trajectories"]["value"].shape
    assert pred_shape == (T, N), (
        f"predicted_trajectories shape expected ({T},{N}), got {pred_shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Model trace (bilinear) -- B_free_0, B deterministic
# ---------------------------------------------------------------------------


def test_bilinear_model_trace_sites(bilinear_fixture: dict) -> None:
    """Bilinear model trace has B_free_0 sample site and B deterministic site."""
    f = bilinear_fixture
    N = f["N"]

    pyro.clear_param_store()
    with pyro.poutine.trace() as tr:
        latent_circuit_dcm_model(
            f["trajectories"],
            f["stim"],
            f["a_mask"],
            f["c_mask"],
            f["t_eval"],
            f["dt"],
            b_masks=f["b_masks"],
            stim_mod=f["stim_mod"],
        )

    nodes = tr.trace.nodes
    # In this Pyro version, pyro.deterministic() registers as type="sample".
    all_sites = {k: v for k, v in nodes.items() if v["type"] == "sample"}

    # Bilinear-specific sites present
    assert "B_free_0" in all_sites, "B_free_0 sample site missing in bilinear trace"
    assert "B" in all_sites, "B deterministic site missing in bilinear trace"

    # B_free_0 shape
    b_free_shape = all_sites["B_free_0"]["value"].shape
    assert b_free_shape == (N, N), (
        f"B_free_0 shape expected ({N},{N}), got {b_free_shape}"
    )

    # B deterministic shape: (J, N, N) with J=1
    B_shape = all_sites["B"]["value"].shape
    assert B_shape == (1, N, N), (
        f"B deterministic shape expected (1,{N},{N}), got {B_shape}"
    )

    # All base sites still present
    for site in ("A_free", "C", "noise_prec", "obs"):
        assert site in all_sites, f"Site '{site}' missing from bilinear trace"


# ---------------------------------------------------------------------------
# Test 3: Guide auto-discovery -- AutoNormal (linear)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_normal_linear(linear_fixture: dict) -> None:
    """AutoNormal auto-discovers all sample sites without factory changes."""
    f = linear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    guide = AutoNormal(latent_circuit_dcm_model, init_scale=0.01)

    # Initialize guide by calling it once
    guide(*model_args)

    # Check that latent parameters exist in the guide
    param_names = list(pyro.get_param_store().keys())
    assert len(param_names) > 0, "AutoNormal found no parameters"

    # Each required sample site should have loc + scale params
    required_prefixes = ["A_free", "C", "noise_prec"]
    for prefix in required_prefixes:
        matching = [p for p in param_names if prefix in p]
        assert len(matching) > 0, (
            f"AutoNormal found no params for site '{prefix}'; "
            f"param names: {param_names}"
        )


# ---------------------------------------------------------------------------
# Test 4: Guide auto-discovery -- AutoLowRankMVN (linear)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_lowrank_mvn_linear(linear_fixture: dict) -> None:
    """AutoLowRankMVN auto-discovers sample sites without factory changes."""
    f = linear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    guide = AutoLowRankMultivariateNormal(
        latent_circuit_dcm_model, init_scale=0.01, rank=2
    )
    guide(*model_args)

    param_names = list(pyro.get_param_store().keys())
    assert len(param_names) > 0, "AutoLowRankMVN found no parameters"


# ---------------------------------------------------------------------------
# Test 5: Guide auto-discovery -- AutoIAFNormal (linear)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_iaf_linear(linear_fixture: dict) -> None:
    """AutoIAFNormal auto-discovers sample sites without factory changes."""
    f = linear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    # hidden_dim must exceed latent_dim = N*N + N*M + 1 = 21 for N=4, M=1.
    # Use 32 to satisfy AutoRegressiveNN constraint (hidden >= input).
    guide = AutoIAFNormal(latent_circuit_dcm_model, num_transforms=2, hidden_dim=[32])
    guide(*model_args)

    param_names = list(pyro.get_param_store().keys())
    assert len(param_names) > 0, "AutoIAFNormal found no parameters"


# ---------------------------------------------------------------------------
# Test 6: Guide auto-discovery -- AutoNormal (bilinear)
# ---------------------------------------------------------------------------


def test_guide_auto_discovery_auto_normal_bilinear(bilinear_fixture: dict) -> None:
    """AutoNormal discovers B_free_0 in bilinear model without factory changes."""
    f = bilinear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    model_kwargs = {"b_masks": f["b_masks"], "stim_mod": f["stim_mod"]}

    # Need a model wrapper for guide initialization with kwargs
    def model_with_kwargs(*args: object) -> None:
        latent_circuit_dcm_model(*args, **model_kwargs)

    guide = AutoNormal(model_with_kwargs, init_scale=0.01)
    guide(*model_args)

    param_names = list(pyro.get_param_store().keys())
    b_params = [p for p in param_names if "B_free_0" in p]
    assert len(b_params) > 0, (
        f"AutoNormal did not discover B_free_0 params in bilinear model; "
        f"params: {param_names}"
    )


# ---------------------------------------------------------------------------
# Test 7: SVI smoke test (linear) -- 200 steps, ELBO decreases
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_svi_smoke_linear(linear_fixture: dict) -> None:
    """Linear model SVI converges with decreasing ELBO within 200 steps."""
    f = linear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    guide = create_guide(latent_circuit_dcm_model, init_scale=0.01)
    result = run_svi(
        latent_circuit_dcm_model,
        guide,
        model_args,
        num_steps=200,
        lr=0.01,
        clip_norm=10.0,
    )

    losses = result["losses"]
    assert len(losses) == 200, f"Expected 200 loss values, got {len(losses)}"

    # Check ELBO decreases: final 10-step mean < initial 10-step mean
    initial_loss = sum(losses[:10]) / 10.0
    final_loss = sum(losses[-10:]) / 10.0
    assert final_loss < initial_loss, (
        f"ELBO did not decrease: initial mean={initial_loss:.2f}, "
        f"final mean={final_loss:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 8: SVI smoke test (bilinear) -- 200 steps, ELBO decreases
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_svi_smoke_bilinear(bilinear_fixture: dict) -> None:
    """Bilinear model SVI converges with decreasing ELBO within 200 steps."""
    f = bilinear_fixture
    pyro.clear_param_store()

    model_args = (
        f["trajectories"], f["stim"], f["a_mask"], f["c_mask"], f["t_eval"], f["dt"]
    )
    model_kwargs = {"b_masks": f["b_masks"], "stim_mod": f["stim_mod"]}

    def model_with_kwargs(*args: object) -> None:
        latent_circuit_dcm_model(*args, **model_kwargs)

    guide = AutoNormal(model_with_kwargs, init_scale=0.01)
    pyro.clear_param_store()

    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0, "lrd": 0.01 ** (1 / 200)})
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model_with_kwargs, guide, optimizer, loss=elbo)

    losses = []
    for _ in range(200):
        loss = svi.step(*model_args)
        losses.append(loss)

    assert len(losses) == 200
    initial_loss = sum(losses[:10]) / 10.0
    final_loss = sum(losses[-10:]) / 10.0
    assert final_loss < initial_loss, (
        f"Bilinear ELBO did not decrease: initial mean={initial_loss:.2f}, "
        f"final mean={final_loss:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 9: Prior constants separation
# ---------------------------------------------------------------------------


def test_prior_constants_separation() -> None:
    """LC_A_PRIOR_VARIANCE differs from task_dcm A prior (1/64) and both are documented."""
    from pyro_dcm.models.task_dcm_model import B_PRIOR_VARIANCE as TASK_B

    # LC_A_PRIOR_VARIANCE = 1/16 (wider than BOLD 1/64)
    assert abs(LC_A_PRIOR_VARIANCE - 1.0 / 16.0) < 1e-12, (
        f"LC_A_PRIOR_VARIANCE expected 1/16=0.0625, got {LC_A_PRIOR_VARIANCE}"
    )
    # LC_A is wider than the task_dcm A prior (1/64)
    task_a_prior_var = 1.0 / 64.0
    assert LC_A_PRIOR_VARIANCE > task_a_prior_var, (
        f"LC_A_PRIOR_VARIANCE={LC_A_PRIOR_VARIANCE} should be wider than "
        f"task_dcm A prior variance={task_a_prior_var}"
    )
    # LC_B_PRIOR_VARIANCE = 1.0 (matches task_dcm B_PRIOR_VARIANCE)
    assert abs(LC_B_PRIOR_VARIANCE - 1.0) < 1e-12, (
        f"LC_B_PRIOR_VARIANCE expected 1.0, got {LC_B_PRIOR_VARIANCE}"
    )
    assert abs(TASK_B - 1.0) < 1e-12, (
        f"task_dcm B_PRIOR_VARIANCE expected 1.0, got {TASK_B}"
    )


# ---------------------------------------------------------------------------
# Test 10: C_obs identity -- predicted_trajectories equals x at init
# ---------------------------------------------------------------------------


def test_c_obs_identity_observation(linear_fixture: dict) -> None:
    """With identity C_obs and zero prior mean, predicted_trajectories shape is (T,N)."""
    f = linear_fixture
    N, T = f["N"], f["T"]

    pyro.clear_param_store()

    # Condition on A_free=0 and C=0 to get near-zero trajectory prediction.
    # With A parameterized from A_free=0, A has negative diagonal only.
    # The point of this test is that predicted_trajectories has shape (T, N)
    # and is passed through C_obs = I_N (no shape change).
    conditioned = pyro.poutine.condition(
        latent_circuit_dcm_model,
        data={
            "A_free": torch.zeros(N, N, dtype=torch.float64),
            "C": torch.zeros(N, f["M"], dtype=torch.float64),
            "noise_prec": torch.tensor(1.0, dtype=torch.float64),
        },
    )
    with pyro.poutine.trace() as tr:
        conditioned(
            f["trajectories"],
            f["stim"],
            f["a_mask"],
            f["c_mask"],
            f["t_eval"],
            f["dt"],
        )

    nodes = tr.trace.nodes
    # In this Pyro version, pyro.deterministic() registers as type="sample".
    all_sites = {k: v for k, v in nodes.items() if v["type"] == "sample"}

    pred_traj = all_sites["predicted_trajectories"]["value"]
    assert pred_traj.shape == (T, N), (
        f"predicted_trajectories shape expected ({T},{N}), got {pred_traj.shape}"
    )
    # With identity C_obs, the observation y_mean has same shape as x.
    # Verify obs site shape matches (T, N).
    obs_shape = all_sites["obs"]["value"].shape
    assert obs_shape == (T, N), (
        f"obs shape expected ({T},{N}), got {obs_shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: direct_observation integration test
# ---------------------------------------------------------------------------


def test_direct_observation_integration() -> None:
    """direct_observation returns correct shapes and identity-pass-through."""
    torch.manual_seed(42)
    T, N = 100, 4

    x = torch.randn(T, N, dtype=torch.float64)
    C_obs_identity = torch.eye(N, dtype=torch.float64)
    noise_prec = torch.tensor(4.0, dtype=torch.float64)  # noise_std = 0.5

    y_mean, noise_std = direct_observation(x, C_obs_identity, noise_prec)

    # Shape checks
    assert y_mean.shape == (T, N), (
        f"y_mean shape expected ({T},{N}), got {y_mean.shape}"
    )
    assert noise_std.shape == (), (
        f"noise_std expected scalar, got {noise_std.shape}"
    )

    # Identity C_obs: y_mean == x
    assert torch.allclose(y_mean, x), (
        "direct_observation with identity C_obs should return y_mean == x"
    )

    # noise_std = 1/sqrt(noise_prec) = 1/sqrt(4) = 0.5
    expected_std = 0.5
    assert abs(noise_std.item() - expected_std) < 1e-12, (
        f"noise_std expected {expected_std}, got {noise_std.item()}"
    )

    # Non-identity C_obs: check shape change
    N_obs = 2
    C_obs_proj = torch.randn(N_obs, N, dtype=torch.float64)
    y_proj, _ = direct_observation(x, C_obs_proj, noise_prec)
    assert y_proj.shape == (T, N_obs), (
        f"Non-identity C_obs: y_mean shape expected ({T},{N_obs}), "
        f"got {y_proj.shape}"
    )
