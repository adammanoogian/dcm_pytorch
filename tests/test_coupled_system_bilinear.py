"""Integration tests for CoupledDCMSystem bilinear extension (BILIN-04).

Verifies:
- ``B=None`` and no-kwarg paths are bit-exact through the full ODE integration.
- ``B`` is registered as a buffer (not a parameter).
- Device/dtype auto-alignment mitigates device drift (Research Section 10.3).
- Missing ``n_driving_inputs`` raises ``ValueError`` (explicit-split policy).
- Driving vs modulator input slicing is correct.
- Bilinear path produces numerically distinguishable BOLD from the linear path.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models import CoupledDCMSystem, bold_signal
from pyro_dcm.utils import PiecewiseConstantInput, integrate_ode, make_initial_state


@pytest.fixture
def linear_setup() -> dict:
    """Shared 2-region linear configuration for the bilinear integration tests."""
    A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    times = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
    values_linear = torch.tensor(
        [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
    )
    return {
        "A": A,
        "C": C,
        "times": times,
        "values": values_linear,
        "N": 2,
        "M_drive": 1,
    }


class TestCoupledDCMSystemBilinear:
    """BILIN-04 integration tests for the CoupledDCMSystem bilinear extension."""

    def test_linear_kwarg_none_matches_no_kwarg_bit_exact(
        self, linear_setup: dict
    ) -> None:
        """No-kwarg and explicit ``B=None`` paths are strictly bit-identical."""
        input_fn = PiecewiseConstantInput(
            linear_setup["times"], linear_setup["values"]
        )
        sys_nokwarg = CoupledDCMSystem(
            linear_setup["A"], linear_setup["C"], input_fn
        )
        sys_bnone = CoupledDCMSystem(
            linear_setup["A"], linear_setup["C"], input_fn, B=None
        )
        y0 = make_initial_state(linear_setup["N"])
        t_eval = torch.arange(0, 40.0, 0.1, dtype=torch.float64)
        sol_a = integrate_ode(
            sys_nokwarg, y0, t_eval, method="rk4", step_size=0.1
        )
        sol_b = integrate_ode(
            sys_bnone, y0, t_eval, method="rk4", step_size=0.1
        )
        assert torch.equal(sol_a, sol_b), (
            "no-kwarg and B=None paths diverged; expected bit-identical "
            f"solutions, got max abs diff = {(sol_a - sol_b).abs().max().item()}"
        )

    def test_b_registered_as_buffer(self, linear_setup: dict) -> None:
        """``B`` appears in named_buffers and is absent from named_parameters."""
        input_fn = PiecewiseConstantInput(
            linear_setup["times"],
            torch.tensor(
                [[1.0, 0.5], [0.0, 0.0], [1.0, 0.5], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        B = torch.zeros(1, 2, 2, dtype=torch.float64)
        B[0, 0, 1] = 0.3
        system = CoupledDCMSystem(
            linear_setup["A"],
            linear_setup["C"],
            input_fn,
            B=B,
            n_driving_inputs=1,
        )
        buffers = dict(system.named_buffers())
        params = dict(system.named_parameters())
        assert "B" in buffers, (
            f"B missing from named_buffers; expected 'B' in {list(buffers)}"
        )
        assert "B" not in params, (
            f"B unexpectedly registered as a Parameter; got {list(params)}"
        )
        assert buffers["B"].shape == torch.Size([1, 2, 2]), (
            f"Buffer B shape mismatch: expected (1, 2, 2), got {tuple(buffers['B'].shape)}"
        )

    def test_b_moved_to_A_device_dtype(self, linear_setup: dict) -> None:
        """B is auto-aligned to A.dtype (and device) at construction."""
        input_fn = PiecewiseConstantInput(
            linear_setup["times"],
            torch.tensor(
                [[1.0, 0.5], [0.0, 0.0], [1.0, 0.5], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        B_f32 = torch.zeros(1, 2, 2, dtype=torch.float32)
        B_f32[0, 0, 1] = 0.3
        system = CoupledDCMSystem(
            linear_setup["A"],
            linear_setup["C"],
            input_fn,
            B=B_f32,
            n_driving_inputs=1,
        )
        assert system.B.dtype == torch.float64, (
            "B dtype not aligned to A; expected torch.float64, "
            f"got {system.B.dtype}"
        )
        assert system.B.device == linear_setup["A"].device, (
            "B device not aligned to A; expected "
            f"{linear_setup['A'].device}, got {system.B.device}"
        )

    def test_missing_n_driving_inputs_raises(self, linear_setup: dict) -> None:
        """Non-empty B with no n_driving_inputs raises ValueError."""
        input_fn = PiecewiseConstantInput(
            linear_setup["times"],
            torch.tensor(
                [[1.0, 0.5], [0.0, 0.0], [1.0, 0.5], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        B = torch.zeros(1, 2, 2, dtype=torch.float64)
        B[0, 0, 1] = 0.3
        with pytest.raises(ValueError, match="n_driving_inputs"):
            CoupledDCMSystem(
                linear_setup["A"],
                linear_setup["C"],
                input_fn,
                B=B,  # no n_driving_inputs
            )

    def test_bilinear_output_differs_from_linear(
        self, linear_setup: dict
    ) -> None:
        """BOLD from bilinear path is numerically distinguishable from linear."""
        # Linear reference.
        input_fn_linear = PiecewiseConstantInput(
            linear_setup["times"], linear_setup["values"]
        )
        sys_linear = CoupledDCMSystem(
            linear_setup["A"], linear_setup["C"], input_fn_linear
        )
        # Bilinear with sustained u_mod.
        times = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
        values_bi = torch.tensor(
            [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        input_fn_bi = PiecewiseConstantInput(times, values_bi)
        B = torch.zeros(1, 2, 2, dtype=torch.float64)
        B[0, 0, 1] = 0.5
        sys_bi = CoupledDCMSystem(
            linear_setup["A"],
            linear_setup["C"],
            input_fn_bi,
            B=B,
            n_driving_inputs=1,
        )
        y0 = make_initial_state(linear_setup["N"])
        t_eval = torch.arange(0, 40.0, 0.1, dtype=torch.float64)
        sol_linear = integrate_ode(
            sys_linear, y0, t_eval, method="rk4", step_size=0.1
        )
        sol_bi = integrate_ode(
            sys_bi, y0, t_eval, method="rk4", step_size=0.1
        )
        # Extract BOLD from lnv, lnq (cols 3N:4N, 4N:5N of the 5N state).
        N = linear_setup["N"]
        bold_linear = bold_signal(
            torch.exp(sol_linear[:, 3 * N: 4 * N]),
            torch.exp(sol_linear[:, 4 * N: 5 * N]),
        )
        bold_bi = bold_signal(
            torch.exp(sol_bi[:, 3 * N: 4 * N]),
            torch.exp(sol_bi[:, 4 * N: 5 * N]),
        )
        rms_diff = (bold_linear - bold_bi).pow(2).mean().sqrt().item()
        assert rms_diff > 1e-6, (
            f"Bilinear and linear BOLD too similar (rms_diff={rms_diff:.3e}); "
            "expected > 1e-6 — bilinear path may not be exercised"
        )
