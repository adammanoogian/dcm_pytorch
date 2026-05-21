"""Tests for Variational Laplace DCM inversion.

Tests the VL optimizer (Gauss-Newton) on simulated task-DCM data
to verify that it can recover A, C, and B parameters.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.inference.variational_laplace import VLResult, variational_laplace
from pyro_dcm.inference.vl_dcm import (
    make_task_dcm_forward,
    unpack_theta,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)


@pytest.fixture()
def linear_fixture():
    """3-region linear DCM with known ground truth."""
    A_true = make_random_stable_A(3, density=0.5, seed=42)
    C_true = torch.zeros(3, 1, dtype=torch.float64)
    C_true[0, 0] = 0.5

    stim = make_block_stimulus(n_blocks=5, block_duration=20, rest_duration=15)

    sim = simulate_task_dcm(
        A_true, C_true, stim,
        duration=175.0, dt=0.5, TR=2.0, SNR=10.0, seed=42,
        solver="rk4",
    )

    a_mask = torch.ones(3, 3, dtype=torch.float64)
    c_mask = torch.zeros(3, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    t_eval = sim["times_fine"]
    stimulus_fn = sim["stimulus"]

    return {
        "A_true": A_true,
        "C_true": C_true,
        "bold": sim["bold"],
        "a_mask": a_mask,
        "c_mask": c_mask,
        "stimulus": stimulus_fn,
        "t_eval": t_eval,
        "TR": 2.0,
        "dt": 0.5,
    }


@pytest.fixture()
def bilinear_fixture():
    """3-region bilinear DCM with known B ground truth."""
    A_true = make_random_stable_A(3, density=0.5, seed=42)
    C_true = torch.zeros(3, 1, dtype=torch.float64)
    C_true[0, 0] = 0.5

    b_mask_0 = torch.zeros(3, 3, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0
    b_mask_0[2, 1] = 1.0

    B_true_0 = torch.zeros(3, 3, dtype=torch.float64)
    B_true_0[1, 0] = 0.4
    B_true_0[2, 1] = 0.3

    stim_drive = make_block_stimulus(
        n_blocks=5, block_duration=20, rest_duration=15,
    )

    stim_mod = make_epoch_stimulus(
        event_times=[20.0, 65.0, 110.0, 155.0],
        event_durations=[12.0, 12.0, 12.0, 12.0],
        event_amplitudes=[1.0, 1.0, 1.0, 1.0],
        duration=175.0, dt=0.5, n_inputs=1,
    )

    sim = simulate_task_dcm(
        A_true, C_true, stim_drive,
        duration=175.0, dt=0.5, TR=2.0, SNR=10.0, seed=42,
        solver="rk4",
        B_list=[B_true_0],
        stimulus_mod=stim_mod,
    )

    a_mask = torch.ones(3, 3, dtype=torch.float64)
    c_mask = torch.zeros(3, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    t_eval = sim["times_fine"]
    drive_fn = sim["stimulus"]
    mod_fn = sim["stimulus_mod"]

    return {
        "A_true": A_true,
        "C_true": C_true,
        "B_true_0": B_true_0,
        "b_mask_0": b_mask_0,
        "bold": sim["bold"],
        "a_mask": a_mask,
        "c_mask": c_mask,
        "stimulus": drive_fn,
        "stim_mod": mod_fn,
        "t_eval": t_eval,
        "TR": 2.0,
        "dt": 0.5,
    }


class TestMakeForward:
    """Structural tests for make_task_dcm_forward."""

    def test_linear_forward_returns_correct_shape(self, linear_fixture):
        """Linear forward function returns (T*N,) output."""
        f = linear_fixture
        forward_fn, prior_mean, prior_cov, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
        )
        y = forward_fn(prior_mean)
        T = f["bold"].shape[0]
        N = 3
        assert y.shape == (T * N,)

    def test_bilinear_forward_returns_correct_shape(self, bilinear_fixture):
        """Bilinear forward function returns (T*N,) output."""
        f = bilinear_fixture
        forward_fn, prior_mean, prior_cov, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
            b_masks=[f["b_mask_0"]],
            stim_mod=f["stim_mod"],
        )
        y = forward_fn(prior_mean)
        T = f["bold"].shape[0]
        N = 3
        assert y.shape == (T * N,)

    def test_layout_dimensions(self, bilinear_fixture):
        """Layout has correct total size and modulator count."""
        f = bilinear_fixture
        _, prior_mean, _, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
            b_masks=[f["b_mask_0"]],
            stim_mod=f["stim_mod"],
        )
        # A_free: 9, C: 3, log_noise_prec: 1, B_free_0: 9 = 22
        assert prior_mean.shape == (22,)
        assert layout.J == 1
        assert layout.N == 3
        assert layout.M == 1

    def test_prior_masked_elements_have_tiny_variance(self, linear_fixture):
        """Masked-out C elements get near-zero prior variance."""
        f = linear_fixture
        _, _, prior_cov, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
        )
        # c_mask has only [0,0] active; C[1,0] and C[2,0] should have tiny var
        c_slice = layout.slices["C"]
        c_vars = torch.diag(prior_cov)[c_slice]
        assert c_vars[0].item() == pytest.approx(1.0, abs=1e-6)
        assert c_vars[1].item() < 1e-8
        assert c_vars[2].item() < 1e-8


class TestUnpackTheta:
    """Test parameter unpacking from flat vector."""

    def test_unpack_recovers_identity(self, linear_fixture):
        """Unpacking prior mean recovers expected A diagonal."""
        f = linear_fixture
        _, prior_mean, _, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
        )
        params = unpack_theta(prior_mean, layout, f["a_mask"], f["c_mask"])
        assert params["A"].shape == (3, 3)
        assert params["C"].shape == (3, 1)
        # At prior mean (A_free=0), diagonal of A should be -0.5
        assert params["A"][0, 0].item() == pytest.approx(-0.5, abs=1e-6)


@pytest.mark.slow
class TestVLLinearRecovery:
    """VL recovers linear DCM parameters from simulated data."""

    def test_vl_recovers_A_and_C(self, linear_fixture):
        """VL recovers A and C within RMSE tolerances."""
        f = linear_fixture
        forward_fn, prior_mean, prior_cov, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
        )

        y_obs = f["bold"].reshape(-1)
        result = variational_laplace(
            forward_fn, y_obs, prior_mean, prior_cov,
            max_iter=64, tol=1e-6,
        )
        assert isinstance(result, VLResult)

        params = unpack_theta(
            result.theta_map, layout, f["a_mask"], f["c_mask"],
        )

        a_rmse = (params["A"] - f["A_true"]).pow(2).mean().sqrt().item()
        c_rmse = (params["C"] - f["C_true"]).pow(2).mean().sqrt().item()

        assert a_rmse < 0.15, f"A RMSE={a_rmse:.4f} > 0.15"
        assert c_rmse < 0.30, f"C RMSE={c_rmse:.4f} > 0.30"
        assert result.n_iterations < 64, "Did not converge within 64 iters"


@pytest.mark.slow
class TestVLBilinearRecovery:
    """VL recovers bilinear DCM parameters including B."""

    def test_vl_recovers_A_C_B(self, bilinear_fixture):
        """VL recovers A, C, and B within RMSE tolerances."""
        f = bilinear_fixture
        forward_fn, prior_mean, prior_cov, layout = make_task_dcm_forward(
            f["a_mask"], f["c_mask"], f["stimulus"],
            f["t_eval"], f["TR"], f["dt"],
            b_masks=[f["b_mask_0"]],
            stim_mod=f["stim_mod"],
        )

        y_obs = f["bold"].reshape(-1)
        result = variational_laplace(
            forward_fn, y_obs, prior_mean, prior_cov,
            max_iter=128, tol=1e-6,
        )

        params = unpack_theta(
            result.theta_map, layout,
            f["a_mask"], f["c_mask"],
            b_masks=[f["b_mask_0"]],
        )

        a_rmse = (params["A"] - f["A_true"]).pow(2).mean().sqrt().item()
        b_rmse_nonnull = 0.0
        B_est = params["B_0"]
        B_true = f["B_true_0"]
        nonnull = f["b_mask_0"] > 0
        if nonnull.any():
            b_rmse_nonnull = (
                (B_est[nonnull] - B_true[nonnull]).pow(2).mean().sqrt().item()
            )

        print("\nVL Bilinear Recovery:")
        print(f"  A RMSE:          {a_rmse:.4f}")
        print(f"  B RMSE (nonnull): {b_rmse_nonnull:.4f}")
        print(f"  B_est[1,0]:      {B_est[1,0].item():.4f} (true: 0.400)")
        print(f"  B_est[2,1]:      {B_est[2,1].item():.4f} (true: 0.300)")
        print(f"  Iterations:      {result.n_iterations}")
        print(f"  Converged:       {result.converged}")
        print(f"  Noise prec:      {result.noise_prec:.2f}")

        assert a_rmse < 0.15, f"A RMSE={a_rmse:.4f} > 0.15"
        assert b_rmse_nonnull < 0.20, (
            f"B RMSE (nonnull)={b_rmse_nonnull:.4f} > 0.20; "
            f"B_est[1,0]={B_est[1,0].item():.4f}, "
            f"B_est[2,1]={B_est[2,1].item():.4f}"
        )
