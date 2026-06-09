"""Integration test: Variational Laplace on the latent-circuit DCM.

Exercises ``LatentCircuitForward`` -- the adapter that lets the model-agnostic
VL engine (``_run_vl_generic``) fit the direct-observation bilinear DCM that
``SpectralDCMForward``/``TaskDCMForward`` cannot. This is the inference path
chosen for the Phase 20-05 rework (Tier B): VL returns a FULL posterior
covariance, so it is the structured posterior (no AutoLowRankMVN needed).

The test is a small, fast smoke/recovery check (N=3, 8s, dt=0.05) proving the
adapter runs end-to-end and recovers ground truth at a sane level -- NOT the
full 20-05 acceptance gate (that is a multi-seed cluster run).
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.latent_circuit_metrics import compute_trajectory_r_squared
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.inference import (
    LatentCircuitForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# Kept deliberately small/coarse: the VL finite-difference Jacobian
# re-integrates the ODE once per parameter per iteration, so cost scales with
# (#timepoints x #params x #iters). This is a fast adapter smoke/recovery
# check, not the full 20-05 acceptance run (that is a multi-seed cluster job).
_DT = 0.1
_DURATION = 6.0
_N = 3
_MAX_ITER = 32


def _build_small_ground_truth() -> dict:
    """Build a 3-region chain with one bilinear modulator on the 0->1 edge."""
    A_true = make_stable_latent_circuit_A(
        _N, density=0.0, self_inhibition=0.5, seed=0,
    )
    A_true[1, 0] = 0.30  # region 0 -> 1 (strong, recoverable)
    A_true[2, 1] = 0.20  # region 1 -> 2

    B_true = torch.zeros(1, _N, _N, dtype=torch.float64)
    B_true[0, 1, 0] = 0.40  # modulate the 0->1 edge

    b_mask = torch.zeros(_N, _N, dtype=torch.float64)
    b_mask[1, 0] = 1.0

    C_true = torch.zeros(_N, 1, dtype=torch.float64)
    C_true[0, 0] = 1.0

    a_mask = torch.ones(_N, _N, dtype=torch.float64)
    c_mask = torch.zeros(_N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    # Driving: 2 blocks of 2s ON / 1s OFF over 6s.
    stim = make_block_stimulus(
        n_blocks=2, block_duration=2.0, rest_duration=1.0, n_inputs=1,
    )
    # Modulator: one window (t=1-4s), well inside the trajectory.
    stim_mod_dict = make_epoch_stimulus(
        event_times=[1.0], event_durations=[3.0], event_amplitudes=[1.0],
        duration=_DURATION, dt=_DT, n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )
    return {
        "A_true": A_true, "B_true": B_true, "C_true": C_true,
        "a_mask": a_mask, "c_mask": c_mask, "b_mask": b_mask,
        "stim": stim, "stim_mod": stim_mod,
    }


@pytest.mark.slow
def test_latent_circuit_forward_vl_recovery() -> None:
    """VL via LatentCircuitForward runs end-to-end and recovers the chain.

    ~80s on laptop CPU (finite-difference Jacobian re-integrates the ODE per
    parameter per iteration); marked ``slow`` to keep the default suite fast.
    """
    torch.manual_seed(0)
    gt = _build_small_ground_truth()

    sim = simulate_latent_circuit(
        gt["A_true"], gt["C_true"], gt["stim"],
        duration=_DURATION, dt=_DT, SNR=10.0,
        solver="rk4", seed=0,
        B_list=[gt["B_true"][0]],
        stimulus_mod=gt["stim_mod"],
    )
    trajs = sim["trajectories"].to(torch.float64)  # (T, N)
    t_eval = sim["times"].to(torch.float64)
    assert not torch.isnan(trajs).any(), "Ground-truth simulation diverged."

    driving = PiecewiseConstantInput(gt["stim"]["times"], gt["stim"]["values"])
    forward = LatentCircuitForward(
        stimulus=driving,
        c_mask=gt["c_mask"],
        t_eval=t_eval,
        dt=_DT,
        b_masks=[gt["b_mask"]],
        stim_mod=gt["stim_mod"],
        c_prior_variance=1.0,
        b_prior_variance=1.0,
    )

    result = run_variational_laplace_generic(
        forward,
        observed=trajs,
        a_mask=gt["a_mask"],
        n_regions=_N,
        max_iter=_MAX_ITER,
        prior_variance=1.0 / 16.0,  # LC_A_PRIOR_VARIANCE
        context={},
    )

    # --- 1. The engine produced a finite free-energy trajectory ---
    assert len(result.free_energy) > 0
    assert all(
        torch.isfinite(torch.tensor(f)) for f in result.free_energy
    )
    assert max(result.free_energy) >= result.free_energy[0]

    # --- 2. VL is a STRUCTURED posterior: full (non-diagonal) covariance ---
    sigma = result.sigma_post
    assert sigma is not None
    np_full = forward.param_count(_N)
    assert sigma.shape == (np_full, np_full)
    off_diag = sigma - torch.diag(torch.diag(sigma))
    assert off_diag.abs().max() > 1e-9, (
        "Posterior covariance is diagonal -- VL should be full-covariance."
    )

    # --- 3. Posterior extraction yields finite A/C/B ---
    posterior = extract_vl_posterior_generic(result, forward, _N)
    for name in ("A_free", "C_free", "B_free"):
        assert name in posterior
        assert torch.isfinite(posterior[name]["mean"]).all()

    # --- 4. Recovered A is stable (negative diagonal) and right-signed ---
    A_post = parameterize_A(
        result.theta_post["A_free"] * gt["a_mask"],
    )
    assert (A_post.diagonal() < 0).all(), "Recovered A is unstable."
    assert A_post[1, 0] > 0.0, "0->1 coupling sign not recovered."

    # --- 5. Bilinear B sign recovered on the modulated edge ---
    B_post = result.theta_post["B"][0]  # (N, N)
    assert B_post[1, 0] > 0.0, "B[1,0] sign not recovered."

    # --- 6. Posterior-mode trajectory reconstructs the data ---
    predicted = result.predicted_output[: trajs.shape[0]]
    r2 = compute_trajectory_r_squared(predicted, trajs)
    assert r2 > 0.7, f"Trajectory reconstruction R2={r2:.3f} too low."
