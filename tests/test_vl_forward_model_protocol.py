"""Tests for the ForwardModel protocol and VL backward compatibility.

Verifies that:
1. SpectralDCMForward satisfies the ForwardModel protocol
2. TaskDCMForward satisfies the ForwardModel protocol
3. run_variational_laplace (spectral) produces identical results
   via the generic path as the legacy implementation
4. Task DCM VL recovers A from synthetic BOLD data
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.inference.forward_models import (
    ForwardModel,
    SpectralDCMForward,
    TaskDCMForward,
)


def test_spectral_is_forward_model() -> None:
    """SpectralDCMForward satisfies ForwardModel protocol."""
    fm = SpectralDCMForward()
    assert isinstance(fm, ForwardModel)


def test_spectral_param_count() -> None:
    """SpectralDCMForward param_count matches N*N + 2N + 6."""
    fm = SpectralDCMForward()
    assert fm.param_count(3) == 3 * 3 + 2 * 3 + 6  # 21
    assert fm.param_count(4) == 4 * 4 + 2 * 4 + 6  # 30


def test_spectral_pack_unpack_roundtrip() -> None:
    """Pack then unpack recovers original tensors."""
    fm = SpectralDCMForward()
    N = 3
    params = {
        "A_free": torch.randn(N, N, dtype=torch.float64),
        "noise_a": torch.randn(2, 1, dtype=torch.float64),
        "noise_b": torch.randn(2, 1, dtype=torch.float64),
        "noise_c": torch.randn(1, N, dtype=torch.float64),
        "P_transit": torch.randn(N, dtype=torch.float64),
        "P_decay": torch.randn(1, dtype=torch.float64),
        "P_epsilon": torch.randn(1, dtype=torch.float64),
    }
    packed = fm.pack_params(**params)
    assert packed.shape == (fm.param_count(N),)
    unpacked = fm.unpack_params(packed, N)
    for key in params:
        torch.testing.assert_close(unpacked[key], params[key])


def test_spectral_prior_cov_masks_absent() -> None:
    """Prior covariance zeros out masked A_free entries."""
    fm = SpectralDCMForward()
    N = 3
    a_mask = torch.ones(N, N, dtype=torch.float64)
    a_mask[0, 1] = 0.0
    var_vec = fm.build_prior_cov(N, 1.0 / 64.0, a_mask)
    assert var_vec[0 * N + 1] == 0.0
    assert var_vec[0] > 0.0


def test_spectral_residual_is_complex() -> None:
    """Spectral forward model produces complex residuals."""
    fm = SpectralDCMForward()
    assert fm.residual_is_complex is True


def test_task_is_forward_model() -> None:
    """TaskDCMForward satisfies ForwardModel protocol."""
    from pyro_dcm.simulators.task_simulator import make_block_stimulus

    stim = make_block_stimulus(n_blocks=2, block_duration=5.0)
    c_mask = torch.ones(3, 1, dtype=torch.float64)
    t_eval = torch.linspace(0, 10, 21, dtype=torch.float64)
    fm = TaskDCMForward(stim, c_mask, t_eval)
    assert isinstance(fm, ForwardModel)


def test_task_param_count() -> None:
    """TaskDCMForward param_count matches N*N + N*M."""
    from pyro_dcm.simulators.task_simulator import make_block_stimulus

    stim = make_block_stimulus(n_blocks=2, block_duration=5.0)
    c_mask = torch.ones(3, 1, dtype=torch.float64)
    t_eval = torch.linspace(0, 10, 21, dtype=torch.float64)
    fm = TaskDCMForward(stim, c_mask, t_eval)
    assert fm.param_count(3) == 3 * 3 + 3 * 1  # 12


def test_task_pack_unpack_roundtrip() -> None:
    """TaskDCMForward pack/unpack roundtrip."""
    from pyro_dcm.simulators.task_simulator import make_block_stimulus

    stim = make_block_stimulus(n_blocks=2, block_duration=5.0)
    N, M = 3, 1
    c_mask = torch.ones(N, M, dtype=torch.float64)
    t_eval = torch.linspace(0, 10, 21, dtype=torch.float64)
    fm = TaskDCMForward(stim, c_mask, t_eval)

    params = {
        "A_free": torch.randn(N, N, dtype=torch.float64),
        "C_free": torch.randn(N, M, dtype=torch.float64),
    }
    packed = fm.pack_params(**params)
    assert packed.shape == (fm.param_count(N),)
    unpacked = fm.unpack_params(packed, N)
    torch.testing.assert_close(unpacked["A_free"], params["A_free"])
    torch.testing.assert_close(unpacked["C_free"], params["C_free"])


def test_task_residual_is_real() -> None:
    """Task forward model produces real residuals."""
    from pyro_dcm.simulators.task_simulator import make_block_stimulus

    stim = make_block_stimulus(n_blocks=2, block_duration=5.0)
    c_mask = torch.ones(3, 1, dtype=torch.float64)
    t_eval = torch.linspace(0, 10, 21, dtype=torch.float64)
    fm = TaskDCMForward(stim, c_mask, t_eval)
    assert fm.residual_is_complex is False


@pytest.mark.slow
def test_spectral_vl_backward_compat() -> None:
    """Spectral VL via generic path matches legacy API output.

    Fits the same synthetic CSD with both run_variational_laplace
    (backward-compat wrapper) and run_variational_laplace_generic
    (explicit SpectralDCMForward). Results should be identical since
    the wrapper delegates to the generic engine.
    """
    from pyro_dcm.inference import (
        run_variational_laplace,
        run_variational_laplace_generic,
    )
    from pyro_dcm.simulators.spectral_simulator import (
        make_stable_A_spectral,
        simulate_spectral_dcm,
    )

    torch.manual_seed(42)
    N = 3
    A_true = make_stable_A_spectral(N, seed=42)
    sim = simulate_spectral_dcm(A_true, n_freqs=16, seed=42)
    a_mask = torch.ones(N, N, dtype=torch.float64)

    result_legacy = run_variational_laplace(
        sim["csd"], sim["freqs"], a_mask, max_iter=20,
    )

    fm = SpectralDCMForward()
    result_generic = run_variational_laplace_generic(
        fm, sim["csd"], a_mask, max_iter=20,
        context={"freqs": sim["freqs"]},
    )

    torch.testing.assert_close(
        result_legacy.theta_post["A"],
        result_generic.theta_post["A"],
    )
    assert abs(
        result_legacy.free_energy[-1] - result_generic.free_energy[-1]
    ) < 1e-6


@pytest.mark.slow
def test_task_dcm_vl_recovery() -> None:
    """Task DCM VL recovers A from synthetic BOLD data.

    Simulates a 3-region task DCM, generates BOLD timeseries, and
    fits via VL with TaskDCMForward. Checks that the recovered A
    correlates with ground truth.
    """
    from pyro_dcm.inference import run_variational_laplace_generic
    from pyro_dcm.simulators.task_simulator import (
        make_block_stimulus,
        simulate_task_dcm,
    )

    torch.manual_seed(42)
    N, M = 3, 1

    A_true = torch.zeros(N, N, dtype=torch.float64)
    A_true[0, 0] = -0.5
    A_true[1, 1] = -0.5
    A_true[2, 2] = -0.5
    A_true[1, 0] = 0.3
    A_true[2, 1] = 0.2

    C_true = torch.zeros(N, M, dtype=torch.float64)
    C_true[0, 0] = 0.5

    stim = make_block_stimulus(
        n_blocks=5, block_duration=10.0, rest_duration=10.0,
    )

    sim = simulate_task_dcm(
        A_true, C_true, stim, duration=100.0, TR=2.0,
        dt_sim=0.01, dt_model=0.5, seed=42,
    )
    observed_bold = sim["bold"]
    t_eval = sim["t_eval"]

    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, M, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    fm = TaskDCMForward(stim, c_mask, t_eval, dt=0.5)
    result = run_variational_laplace_generic(
        fm, observed_bold, a_mask, max_iter=64,
        prior_variance=1.0 / 64.0,
        context={},
    )

    A_post = result.theta_post["A"]

    a_true_flat = A_true.flatten()
    a_post_flat = A_post.flatten().to(torch.float64)
    corr = torch.corrcoef(
        torch.stack([a_true_flat, a_post_flat]),
    )[0, 1].item()

    print(f"\nTask DCM VL Recovery:")
    print(f"  A_true:\n{A_true.numpy().round(4)}")
    print(f"  A_post:\n{A_post.detach().numpy().round(4)}")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Free energy: {result.free_energy[-1]:.2f}")

    assert corr > 0.5, (
        f"Task DCM VL A recovery correlation {corr:.3f} < 0.5"
    )
