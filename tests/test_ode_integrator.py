"""Integration tests for coupled ODE system and ODE integration utilities.

Tests the coupled neural-hemodynamic ODE system (CoupledDCMSystem),
piecewise-constant input handling (PiecewiseConstantInput), ODE
integrator wrapper (integrate_ode), and end-to-end simulation
stability for block-design fMRI paradigms.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 — Neural state equation.
[REF-002] Stephan et al. (2007), Eq. 2-5 — Balloon-Windkessel model.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models import CoupledDCMSystem, bold_signal
from pyro_dcm.utils import PiecewiseConstantInput, integrate_ode, make_initial_state


# ---------------------------------------------------------------------------
# PiecewiseConstantInput tests
# ---------------------------------------------------------------------------


class TestPiecewiseConstantInput:
    """Tests for PiecewiseConstantInput stimulus function."""

    def test_piecewise_input_before_first_onset(self) -> None:
        """For t < times[0], returns values[0]."""
        times = torch.tensor([5.0, 10.0, 20.0], dtype=torch.float64)
        values = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)
        u = PiecewiseConstantInput(times, values)

        result = u(torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(result, torch.tensor([1.0], dtype=torch.float64))

    def test_piecewise_input_between_onsets(self) -> None:
        """Correct value returned between onset times via searchsorted."""
        times = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
        values = torch.tensor(
            [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
        )
        u = PiecewiseConstantInput(times, values)

        # Between 0 and 10 -> values[0] = [1.0]
        assert torch.allclose(
            u(torch.tensor(5.0, dtype=torch.float64)),
            torch.tensor([1.0], dtype=torch.float64),
        )
        # Between 10 and 20 -> values[1] = [0.0]
        assert torch.allclose(
            u(torch.tensor(15.0, dtype=torch.float64)),
            torch.tensor([0.0], dtype=torch.float64),
        )
        # Between 20 and 30 -> values[2] = [1.0]
        assert torch.allclose(
            u(torch.tensor(25.0, dtype=torch.float64)),
            torch.tensor([1.0], dtype=torch.float64),
        )

    def test_piecewise_input_after_last_onset(self) -> None:
        """For t > times[-1], returns values[-1]."""
        times = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
        values = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)
        u = PiecewiseConstantInput(times, values)

        result = u(torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(result, torch.tensor([1.0], dtype=torch.float64))

    def test_piecewise_input_at_onset(self) -> None:
        """At exact onset time, returns value for that onset."""
        times = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
        values = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)
        u = PiecewiseConstantInput(times, values)

        # At t=10.0 exactly, searchsorted(right=True)-1 = 1, so values[1]
        result = u(torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(result, torch.tensor([0.0], dtype=torch.float64))

    def test_piecewise_grid_points(self) -> None:
        """grid_points property returns the onset times tensor."""
        times = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
        values = torch.tensor(
            [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
        )
        u = PiecewiseConstantInput(times, values)

        assert torch.equal(u.grid_points, times)

    def test_piecewise_multi_input(self) -> None:
        """PiecewiseConstantInput works with multiple inputs (M > 1)."""
        times = torch.tensor([0.0, 10.0], dtype=torch.float64)
        values = torch.tensor(
            [[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64
        )
        u = PiecewiseConstantInput(times, values)

        result = u(torch.tensor(5.0, dtype=torch.float64))
        assert result.shape == (2,)
        assert torch.allclose(
            result, torch.tensor([1.0, 0.5], dtype=torch.float64)
        )


# ---------------------------------------------------------------------------
# make_initial_state tests
# ---------------------------------------------------------------------------


class TestMakeInitialState:
    """Tests for make_initial_state utility."""

    def test_initial_state_shape(self) -> None:
        """Initial state has shape (5*N,)."""
        y0 = make_initial_state(3)
        assert y0.shape == (15,)

    def test_initial_state_zeros(self) -> None:
        """Initial state is all zeros (steady state)."""
        y0 = make_initial_state(3)
        assert torch.all(y0 == 0.0)

    def test_initial_state_dtype(self) -> None:
        """Initial state respects dtype argument."""
        y0 = make_initial_state(2, dtype=torch.float32)
        assert y0.dtype == torch.float32

        y0_64 = make_initial_state(2, dtype=torch.float64)
        assert y0_64.dtype == torch.float64


# ---------------------------------------------------------------------------
# Coupled system steady-state test
# ---------------------------------------------------------------------------


class TestCoupledSteadyState:
    """Tests for CoupledDCMSystem at steady state."""

    def test_steady_state_no_input(self, test_A, test_C) -> None:
        """With u(t)=0 and zero initial conditions, states remain at zero.

        This verifies that the steady state is a fixed point of the
        coupled ODE system: when all states are zero and there is no
        external input, all derivatives should be zero.
        """
        N = test_A.shape[0]

        # Zero input function
        times = torch.tensor([0.0], dtype=torch.float64)
        values = torch.zeros(1, test_C.shape[1], dtype=torch.float64)
        input_fn = PiecewiseConstantInput(times, values)

        system = CoupledDCMSystem(test_A, test_C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        # Integrate for 100s
        t_eval = torch.linspace(0, 100, 500, dtype=torch.float64)
        sol = integrate_ode(system, y0, t_eval, method="dopri5")

        # All states should remain at zero within numerical tolerance
        assert sol.shape == (500, 5 * N)
        assert torch.max(torch.abs(sol)).item() < 1e-10, (
            f"Max deviation from steady state: {torch.max(torch.abs(sol)).item()}"
        )


# ---------------------------------------------------------------------------
# Block stimulus integration tests
# ---------------------------------------------------------------------------


def _make_block_design_3region(
    duration: float = 60.0,
    block_on: float = 30.0,
    block_off: float = 30.0,
    n_blocks: int = 1,
    c_strength: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, PiecewiseConstantInput, int]:
    """Helper: Create 3-region system with block stimulus design.

    Parameters
    ----------
    c_strength : float
        Driving input strength for region 0. Default 0.25 Hz, which
        is a moderate value producing BOLD peak ~3-4% typical of
        task-based fMRI experiments.

    Returns (A, C, input_fn, n_regions).
    """
    N = 3

    # Stable A matrix: diagonal dominant with known negative eigenvalues
    A = torch.tensor(
        [
            [-0.5, 0.1, 0.0],
            [0.2, -0.5, 0.1],
            [0.0, 0.3, -0.5],
        ],
        dtype=torch.float64,
    )

    # Drive region 0 only with moderate strength
    C = torch.tensor(
        [[c_strength], [0.0], [0.0]], dtype=torch.float64
    )

    # Build block design: ON/OFF alternating
    onset_times = []
    onset_values = []
    t = 0.0
    for _ in range(n_blocks):
        onset_times.append(t)
        onset_values.append([1.0])  # ON
        t += block_on
        onset_times.append(t)
        onset_values.append([0.0])  # OFF
        t += block_off

    times = torch.tensor(onset_times, dtype=torch.float64)
    values = torch.tensor(onset_values, dtype=torch.float64)
    input_fn = PiecewiseConstantInput(times, values)

    return A, C, input_fn, N


class TestBlockStimulus:
    """Tests for block-design stimulus integration."""

    def test_block_stimulus_3region(self) -> None:
        """3-region system with block stimulus: ON 30s, OFF 30s.

        Verifies:
        (a) No NaN in solution.
        (b) Neural states increase during stimulus ON period.
        (c) After stimulus OFF, neural states decay toward zero.
        (d) BOLD output is in realistic range [-5%, +5%].
        """
        A, C, input_fn, N = _make_block_design_3region(
            duration=60.0, block_on=30.0, block_off=30.0, n_blocks=1
        )

        system = CoupledDCMSystem(A, C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        t_eval = torch.linspace(0, 60, 600, dtype=torch.float64)
        sol = integrate_ode(
            system, y0, t_eval, method="dopri5",
            grid_points=input_fn.grid_points,
        )

        # (a) No NaN
        assert not torch.isnan(sol).any(), "Solution contains NaN"

        # (b) Neural states increase during ON period (t=0 to t=30)
        # Check the driven region (region 0) at t=15s (midpoint of ON)
        t15_idx = 150  # t=15s at 600 points over 60s -> index 150
        x_driven = sol[t15_idx, 0]  # neural state of region 0
        assert x_driven > 0.0, (
            f"Driven region neural state should be positive during ON: {x_driven}"
        )

        # (c) After OFF, neural states decay
        # Compare neural states at t=30s (end of ON) vs t=55s (deep into OFF)
        t30_idx = 300
        t55_idx = 550
        x_at_30 = sol[t30_idx, :N].abs().max()
        x_at_55 = sol[t55_idx, :N].abs().max()
        assert x_at_55 < x_at_30, (
            f"Neural states should decay after stimulus OFF: "
            f"|x(30)|={x_at_30:.4f}, |x(55)|={x_at_55:.4f}"
        )

        # (d) BOLD in realistic range
        lnv = sol[:, 3 * N:4 * N]
        lnq = sol[:, 4 * N:5 * N]
        v = torch.exp(lnv)
        q = torch.exp(lnq)
        bold = bold_signal(v, q)  # shape (T, N)

        # BOLD as percent signal change
        bold_pct = bold * 100.0  # V0=0.02 already in bold_signal
        max_bold_pct = bold_pct.abs().max().item()
        assert max_bold_pct < 5.0, (
            f"BOLD percent signal change should be < 5%: {max_bold_pct:.4f}%"
        )


# ---------------------------------------------------------------------------
# 500s stability test (Success Criterion #1)
# ---------------------------------------------------------------------------


class TestLongSimulationStability:
    """Tests for numerical stability over long simulations."""

    def test_500s_stability(self) -> None:
        """500s simulation with 10 ON/OFF blocks remains stable.

        This is SUCCESS CRITERION #1: 500s integration without NaN or Inf.
        Uses 10 blocks of 30s ON / 20s OFF.
        """
        A, C, input_fn, N = _make_block_design_3region(
            duration=500.0, block_on=30.0, block_off=20.0, n_blocks=10
        )

        system = CoupledDCMSystem(A, C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        t_eval = torch.linspace(0, 500, 2500, dtype=torch.float64)
        sol = integrate_ode(
            system, y0, t_eval, method="dopri5",
            grid_points=input_fn.grid_points,
        )

        # No NaN
        assert not torch.isnan(sol).any(), "500s solution contains NaN"

        # No Inf
        assert not torch.isinf(sol).any(), "500s solution contains Inf"

        # Solution remains bounded (states should not explode)
        max_val = sol.abs().max().item()
        assert max_val < 100.0, (
            f"Solution exploded: max |state| = {max_val:.2f}"
        )


# ---------------------------------------------------------------------------
# Solver selection tests
# ---------------------------------------------------------------------------


class TestSolverSelection:
    """Tests for different ODE solver methods."""

    def _short_simulation(
        self, method: str, step_size: float = 0.001
    ) -> torch.Tensor:
        """Run a short 10s simulation with given method."""
        A, C, input_fn, N = _make_block_design_3region(
            duration=10.0, block_on=5.0, block_off=5.0, n_blocks=1
        )

        system = CoupledDCMSystem(A, C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        t_eval = torch.linspace(0, 10, 100, dtype=torch.float64)

        kwargs: dict = {"method": method}
        if method == "dopri5":
            kwargs["grid_points"] = input_fn.grid_points
        else:
            kwargs["step_size"] = step_size

        return integrate_ode(system, y0, t_eval, **kwargs)

    def test_solver_selection_euler(self) -> None:
        """Euler solver produces solution close to dopri5 reference."""
        ref = self._short_simulation("dopri5")
        euler = self._short_simulation("euler", step_size=0.001)

        # Euler with small step should be within 1% of dopri5
        # Compare at final time point
        diff = (ref[-1] - euler[-1]).abs()
        ref_scale = ref[-1].abs().clamp(min=1e-10)
        rel_err = (diff / ref_scale).max().item()
        assert rel_err < 0.01, (
            f"Euler relative error too large: {rel_err:.4f}"
        )

    def test_solver_selection_rk4(self) -> None:
        """RK4 solver matches dopri5 within 0.1%."""
        ref = self._short_simulation("dopri5")
        rk4 = self._short_simulation("rk4", step_size=0.001)

        diff = (ref[-1] - rk4[-1]).abs()
        ref_scale = ref[-1].abs().clamp(min=1e-10)
        rel_err = (diff / ref_scale).max().item()
        assert rel_err < 0.001, (
            f"RK4 relative error too large: {rel_err:.6f}"
        )


# ---------------------------------------------------------------------------
# BOLD output range test (Success Criterion #3)
# ---------------------------------------------------------------------------


class TestBoldOutputRange:
    """Tests for realistic BOLD signal range."""

    def test_bold_output_realistic_range(self) -> None:
        """For 500s block design, peak BOLD is 0.5-5%.

        This is SUCCESS CRITERION #3: BOLD percent signal change in
        the directly driven region falls within 0.5-5% range.
        """
        A, C, input_fn, N = _make_block_design_3region(
            duration=500.0, block_on=30.0, block_off=20.0, n_blocks=10
        )

        system = CoupledDCMSystem(A, C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        t_eval = torch.linspace(0, 500, 2500, dtype=torch.float64)
        sol = integrate_ode(
            system, y0, t_eval, method="dopri5",
            grid_points=input_fn.grid_points,
        )

        # Extract hemodynamic states and compute BOLD
        lnv = sol[:, 3 * N:4 * N]
        lnq = sol[:, 4 * N:5 * N]
        v = torch.exp(lnv)
        q = torch.exp(lnq)
        bold = bold_signal(v, q)  # shape (T, N)

        # BOLD percent signal change = bold * 100 / V0... but V0 is
        # already multiplied inside bold_signal (V0=0.02), so bold
        # values are fractional signal change (e.g., 0.02 = 2%).
        # Convert to percent:
        bold_pct = bold * 100.0

        # Peak in driven region (region 0)
        peak_pct = bold_pct[:, 0].abs().max().item()
        assert 0.5 <= peak_pct <= 5.0, (
            f"Peak BOLD in driven region: {peak_pct:.4f}% "
            f"(expected 0.5-5%)"
        )


# ---------------------------------------------------------------------------
# Adjoint mode test
# ---------------------------------------------------------------------------


class TestAdjointMode:
    """Tests for adjoint method integration."""

    def test_adjoint_mode(self) -> None:
        """Adjoint mode produces same output as non-adjoint."""
        A, C, input_fn, N = _make_block_design_3region(
            duration=10.0, block_on=5.0, block_off=5.0, n_blocks=1
        )

        system = CoupledDCMSystem(A, C, input_fn)
        y0 = make_initial_state(N, dtype=torch.float64)

        t_eval = torch.linspace(0, 10, 50, dtype=torch.float64)

        sol_standard = integrate_ode(
            system, y0, t_eval, method="dopri5",
            grid_points=input_fn.grid_points, adjoint=False,
        )
        sol_adjoint = integrate_ode(
            system, y0, t_eval, method="dopri5",
            grid_points=input_fn.grid_points, adjoint=True,
        )

        # Should match within tight tolerance
        assert torch.allclose(sol_standard, sol_adjoint, rtol=1e-4, atol=1e-6), (
            f"Adjoint mismatch. Max diff: "
            f"{(sol_standard - sol_adjoint).abs().max().item():.2e}"
        )
