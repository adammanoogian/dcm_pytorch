"""Tests for latent-circuit DCM forward model (Phase 20).

Covers:
1. CoupledDCMSystem(hemodynamic=False) returns N-dim derivatives.
2. CoupledDCMSystem(hemodynamic=True) preserves bit-exact existing behavior.
3. simulate_latent_circuit returns dict with 'trajectories' key of shape (T, N).
4. No hemodynamic or BOLD keys present in simulate_latent_circuit output.
5. Simulator trajectories match CoupledDCMSystem ODE integration (atol=1e-6).
6. Bilinear mode produces trajectories distinguishable from linear (B_list=None).
7. make_stable_latent_circuit_A produces stable matrices.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_region_system() -> dict:
    """Minimal 2-region stable system for forward-model tests."""
    A = torch.tensor([[-1.0, 0.2], [0.1, -1.0]], dtype=torch.float64)
    C = torch.tensor([[0.5], [0.0]], dtype=torch.float64)
    stim = make_block_stimulus(
        n_blocks=2, block_duration=0.5, rest_duration=0.5
    )
    return {"A": A, "C": C, "stim": stim}


@pytest.fixture
def three_region_system(seed: int = 7) -> dict:
    """3-region stable system for bilinear and integration tests."""
    A = make_stable_latent_circuit_A(3, seed=seed)
    C = torch.zeros(3, 1, dtype=torch.float64)
    C[0, 0] = 0.5
    stim = make_block_stimulus(
        n_blocks=2, block_duration=0.5, rest_duration=0.5
    )
    return {"A": A, "C": C, "stim": stim}


# ---------------------------------------------------------------------------
# Test 1: CoupledDCMSystem(hemodynamic=False) returns N-dim derivatives
# ---------------------------------------------------------------------------


def test_hemodynamic_false_returns_N_derivatives(two_region_system):
    """CoupledDCMSystem(hemodynamic=False) forward() shape is (N,), not (5N,).

    Validates the must-have truth: 'CoupledDCMSystem(hemodynamic=False)
    integrates an N-dim state vector and returns N derivatives (not 5N)'.
    """
    A = two_region_system["A"]
    C = two_region_system["C"]
    N = A.shape[0]

    input_fn = lambda t: torch.zeros(1, dtype=torch.float64)  # noqa: E731
    system = CoupledDCMSystem(A, C, input_fn, hemodynamic=False)

    # State is neural activity only: shape (N,)
    state = torch.zeros(N, dtype=torch.float64)
    t = torch.tensor(0.0, dtype=torch.float64)
    deriv = system.forward(t, state)

    assert deriv.shape == (N,), (
        f"Expected derivative shape ({N},) for hemodynamic=False; "
        f"got {deriv.shape}. State was shape (N,)={N}, "
        f"not (5N,)={5 * N}."
    )
    assert system.hemo is None, (
        "Expected hemo attribute to be None when hemodynamic=False"
    )


# ---------------------------------------------------------------------------
# Test 2: hemodynamic=True preserves bit-exact existing behavior
# ---------------------------------------------------------------------------


def test_hemodynamic_true_bit_exact(two_region_system):
    """CoupledDCMSystem(hemodynamic=True) matches no-kwarg construction.

    The pre-v0.6.0 callers that omit 'hemodynamic' must receive identical
    outputs to those that explicitly pass hemodynamic=True.
    """
    A = two_region_system["A"]
    C = two_region_system["C"]
    N = A.shape[0]

    input_fn = lambda t: torch.zeros(1, dtype=torch.float64)  # noqa: E731
    state = torch.zeros(5 * N, dtype=torch.float64)
    t = torch.tensor(0.0, dtype=torch.float64)

    system_default = CoupledDCMSystem(A, C, input_fn)
    system_explicit = CoupledDCMSystem(A, C, input_fn, hemodynamic=True)

    deriv_default = system_default.forward(t, state)
    deriv_explicit = system_explicit.forward(t, state)

    assert torch.equal(deriv_default, deriv_explicit), (
        "hemodynamic=True (explicit) must be bit-exact vs default (no kwarg). "
        f"Max abs diff: {(deriv_default - deriv_explicit).abs().max().item()}"
    )
    assert deriv_default.shape == (5 * N,), (
        f"Expected (5N,)=({5 * N},); got {deriv_default.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: simulate_latent_circuit returns 'trajectories' of shape (T, N)
# ---------------------------------------------------------------------------


def test_simulate_latent_circuit_trajectories_shape(two_region_system):
    """simulate_latent_circuit returns dict with 'trajectories' key of shape (T, N).

    Validates must-have truth: 'simulate_latent_circuit returns dict with
    trajectories key of shape (T, N) and no hemodynamic or BOLD keys'.
    """
    A = two_region_system["A"]
    C = two_region_system["C"]
    stim = two_region_system["stim"]
    N = A.shape[0]
    duration = 2.0
    dt = 0.01
    T_expected = int(duration / dt)

    result = simulate_latent_circuit(A, C, stim, duration=duration, dt=dt)

    assert "trajectories" in result, "Result must have 'trajectories' key"
    traj = result["trajectories"]
    assert traj.shape == (T_expected, N), (
        f"Expected trajectories shape ({T_expected}, {N}); got {traj.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: No hemodynamic or BOLD keys in result
# ---------------------------------------------------------------------------


def test_simulate_latent_circuit_no_hemodynamic_keys(two_region_system):
    """simulate_latent_circuit output has no hemodynamic or BOLD keys.

    The result dict must not contain 'bold', 'bold_clean', 'bold_fine',
    'hemodynamic', 'neural' -- these belong to task DCM only.
    """
    A = two_region_system["A"]
    C = two_region_system["C"]
    stim = two_region_system["stim"]

    result = simulate_latent_circuit(A, C, stim, duration=2.0, dt=0.01)

    forbidden_keys = {"bold", "bold_clean", "bold_fine", "hemodynamic", "neural"}
    present_forbidden = forbidden_keys & set(result.keys())
    assert not present_forbidden, (
        f"simulate_latent_circuit must not return hemodynamic/BOLD keys; "
        f"found: {present_forbidden}"
    )

    # Verify expected keys are present.
    required_keys = {
        "trajectories",
        "trajectories_clean",
        "times",
        "A",
        "C",
        "params",
        "simulation_diverged",
    }
    missing = required_keys - set(result.keys())
    assert not missing, (
        f"simulate_latent_circuit missing required keys: {missing}"
    )


# ---------------------------------------------------------------------------
# Test 5: Simulator output matches ODE integration at atol=1e-6
# ---------------------------------------------------------------------------


def test_simulator_matches_direct_ode_integration(two_region_system):
    """Simulator trajectories match direct CoupledDCMSystem ODE at atol=1e-6.

    Validates must-have truth: 'Simulator output matches CoupledDCMSystem
    ODE integration at atol=1e-6 for identical parameters'.

    Both use the same solver (rk4) and dt so results must agree tightly.
    """
    A = two_region_system["A"]
    C = two_region_system["C"]
    stim = two_region_system["stim"]
    N = A.shape[0]
    duration = 2.0
    dt = 0.01
    dtype = torch.float64
    device = "cpu"
    solver = "rk4"

    # Run simulator.
    result = simulate_latent_circuit(
        A, C, stim, duration=duration, dt=dt, SNR=0.0,
        solver=solver, dtype=dtype, device=device,
    )
    sim_traj = result["trajectories_clean"]  # no noise; (T, N)

    # Direct ODE integration.
    stim_times = stim["times"].to(device=device, dtype=dtype)
    stim_values = stim["values"].to(device=device, dtype=dtype)
    input_fn = PiecewiseConstantInput(stim_times, stim_values)
    A_dev = A.to(device=device, dtype=dtype)
    C_dev = C.to(device=device, dtype=dtype)
    system = CoupledDCMSystem(A_dev, C_dev, input_fn, hemodynamic=False)
    y0 = torch.zeros(N, dtype=dtype, device=device)
    t_eval = torch.arange(0, duration, dt, dtype=dtype, device=device)
    direct_solution = integrate_ode(
        system, y0, t_eval, method=solver, step_size=dt
    )  # (T, N)

    max_diff = (sim_traj - direct_solution).abs().max().item()
    assert max_diff < 1e-6, (
        f"Simulator and direct ODE mismatch: max abs diff = {max_diff:.2e}, "
        f"expected < 1e-6"
    )


# ---------------------------------------------------------------------------
# Test 6: Bilinear mode produces trajectories distinguishable from linear
# ---------------------------------------------------------------------------


def test_bilinear_trajectories_differ_from_linear(three_region_system):
    """Bilinear mode produces trajectories distinguishable from linear (B=None).

    Validates must-have truth: 'Bilinear mode works: simulate_latent_circuit
    with non-None B_list produces trajectories distinguishable from linear'.

    Uses a B matrix that creates a strong modulatory effect so that the
    bilinear and linear trajectories differ by more than 1e-4 in L-inf norm.
    """
    A = three_region_system["A"]
    C = three_region_system["C"]
    stim = three_region_system["stim"]
    N = A.shape[0]
    duration = 2.0
    dt = 0.01

    # Modulatory B matrix: region 0 drives region 1 when input is active.
    B0 = torch.zeros(N, N, dtype=torch.float64)
    B0[1, 0] = 0.5  # significant modulation
    B_list = [B0]

    # Build modulator stimulus matching driving stimulus shape.
    stim_mod = {
        "times": stim["times"].clone(),
        "values": stim["values"].clone(),  # same timing, 1 column = 1 modulator
    }

    # Linear simulation.
    result_linear = simulate_latent_circuit(
        A, C, stim, duration=duration, dt=dt, SNR=0.0
    )

    # Bilinear simulation.
    result_bilinear = simulate_latent_circuit(
        A, C, stim, duration=duration, dt=dt, SNR=0.0,
        B_list=B_list, stimulus_mod=stim_mod
    )

    traj_linear = result_linear["trajectories_clean"]
    traj_bilinear = result_bilinear["trajectories_clean"]

    diff = (traj_linear - traj_bilinear).abs().max().item()
    assert diff > 1e-4, (
        f"Expected bilinear trajectory to differ from linear by > 1e-4; "
        f"got max abs diff = {diff:.2e}. "
        f"Check B matrix is non-trivial and input is active during simulation."
    )


# ---------------------------------------------------------------------------
# Test 7: make_stable_latent_circuit_A produces stable matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_regions", [2, 3, 5])
def test_make_stable_latent_circuit_A_is_stable(n_regions: int):
    """make_stable_latent_circuit_A returns matrices with all negative eigenvalues.

    Tests N=2, 3, 5 to verify stability guarantee across typical latent
    circuit dimensions. Stability is required by [REF-001]: A must have
    all eigenvalues with strictly negative real parts for the ODE to have
    a finite fixed point.
    """
    A = make_stable_latent_circuit_A(n_regions, seed=42)

    assert A.shape == (n_regions, n_regions), (
        f"Expected shape ({n_regions}, {n_regions}); got {A.shape}"
    )

    eigs = torch.linalg.eigvals(A).real
    max_real = eigs.max().item()
    assert max_real < 0.0, (
        f"make_stable_latent_circuit_A(n_regions={n_regions}) produced "
        f"unstable matrix: max Re(eig)={max_real:.4f} >= 0. "
        f"Diagonal: {A.diagonal().tolist()}"
    )

    # Diagonal must be negative self-inhibition.
    diag = A.diagonal()
    assert (diag < 0).all(), (
        f"All diagonal entries must be negative; got {diag.tolist()}"
    )
