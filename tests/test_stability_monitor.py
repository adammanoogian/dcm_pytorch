"""Stability monitor tests (BILIN-05) and 3-sigma worst-case test (BILIN-06).

BILIN-05: ``CoupledDCMSystem`` emits a WARNING on the ``pyro_dcm.stability``
logger when ``max Re(eig(A_eff)) > 0`` at the configured cadence; never
raises (D4). Configurable via ``stability_check_every`` (cadence on RHS
evaluations; set to 0 to disable entirely).

BILIN-06: 3-sigma worst-case B (off-diagonal = 3.0, diagonal = 0),
sustained ``u_mod = 1.0``, 500s ``rk4`` integration at ``dt = 0.1``
produces no NaN / Inf in the full 5N state trajectory. Mitigates
Pitfall B1 (drawing unstable B and failing to integrate).
"""

from __future__ import annotations

import logging

import pytest
import torch

from pyro_dcm.forward_models import CoupledDCMSystem
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.utils import PiecewiseConstantInput, integrate_ode, make_initial_state


def _unstable_system(stability_check_every: int = 10) -> CoupledDCMSystem:
    """Build a small deterministic system with guaranteed ``max Re(eig(A_eff)) > 0``.

    ``A = diag(-0.5, -0.5)``, ``B[0] = [[0, 2], [2, 0]]``, ``u_mod = 1`` gives
    ``A_eff = [[-0.5, 2], [2, -0.5]]`` with eigenvalues ``-0.5 +/- 2 =
    {-2.5, 1.5}``. Max Re = 1.5 > 0, so the monitor must fire.
    """
    A = torch.tensor(
        [[-0.5, 0.0], [0.0, -0.5]], dtype=torch.float64
    )
    C = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
    B = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float64
    )
    times = torch.tensor([0.0], dtype=torch.float64)
    values = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    input_fn = PiecewiseConstantInput(times, values)
    return CoupledDCMSystem(
        A,
        C,
        input_fn,
        B=B,
        n_driving_inputs=1,
        stability_check_every=stability_check_every,
    )


def _stable_system(stability_check_every: int = 10) -> CoupledDCMSystem:
    """Build a small deterministic system with guaranteed ``max Re(eig(A_eff)) < 0``.

    ``A_eff = A + u_mod * B[0] = diag(-0.5, -0.5) + 0.1 * [[0, 0.05], [0.05, 0]]``
    which has eigenvalues close to ``-0.5 +/- 0.005``. Comfortably negative.
    """
    A = torch.tensor(
        [[-0.5, 0.0], [0.0, -0.5]], dtype=torch.float64
    )
    C = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
    B = torch.tensor(
        [[[0.0, 0.05], [0.05, 0.0]]], dtype=torch.float64
    )
    times = torch.tensor([0.0], dtype=torch.float64)
    values = torch.tensor([[0.0, 0.1]], dtype=torch.float64)
    input_fn = PiecewiseConstantInput(times, values)
    return CoupledDCMSystem(
        A,
        C,
        input_fn,
        B=B,
        n_driving_inputs=1,
        stability_check_every=stability_check_every,
    )


class TestStabilityMonitor:
    """BILIN-05: eigenvalue monitor log-only behavior."""

    def test_unstable_A_eff_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unstable ``A_eff`` emits >=1 WARNING on ``pyro_dcm.stability``."""
        caplog.set_level(logging.WARNING, logger="pyro_dcm.stability")
        system = _unstable_system(stability_check_every=10)
        y0 = make_initial_state(2)
        t_eval = torch.arange(0, 5.0, 0.1, dtype=torch.float64)
        integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)
        warn_records = [
            r
            for r in caplog.records
            if r.name == "pyro_dcm.stability" and r.levelno == logging.WARNING
        ]
        assert len(warn_records) >= 1, (
            "expected >=1 WARNING record on pyro_dcm.stability for "
            f"unstable A_eff; got {len(warn_records)}"
        )
        assert any(
            "Stability warning at t=" in r.getMessage() for r in warn_records
        ), (
            "WARNING message format drift; expected prefix "
            "'Stability warning at t='; got messages: "
            f"{[r.getMessage() for r in warn_records]}"
        )

    def test_stable_A_eff_emits_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stable ``A_eff`` emits zero warnings."""
        caplog.set_level(logging.WARNING, logger="pyro_dcm.stability")
        system = _stable_system(stability_check_every=10)
        y0 = make_initial_state(2)
        t_eval = torch.arange(0, 5.0, 0.1, dtype=torch.float64)
        integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)
        warn_records = [
            r
            for r in caplog.records
            if r.name == "pyro_dcm.stability" and r.levelno == logging.WARNING
        ]
        assert len(warn_records) == 0, (
            "stable A_eff should emit no warnings; expected 0, got "
            f"{len(warn_records)}: {[r.getMessage() for r in warn_records]}"
        )

    def test_stability_check_every_zero_disables(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``stability_check_every=0`` disables the monitor entirely."""
        caplog.set_level(logging.WARNING, logger="pyro_dcm.stability")
        system = _unstable_system(stability_check_every=0)
        y0 = make_initial_state(2)
        t_eval = torch.arange(0, 5.0, 0.1, dtype=torch.float64)
        integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)
        warn_records = [
            r
            for r in caplog.records
            if r.name == "pyro_dcm.stability" and r.levelno == logging.WARNING
        ]
        assert len(warn_records) == 0, (
            "stability_check_every=0 should disable the monitor even on "
            f"unstable A_eff; expected 0 warnings, got {len(warn_records)}"
        )

    def test_monitor_never_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Monitor never raises even when A_eff is unstable throughout integration."""
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")
        system = _unstable_system(stability_check_every=10)
        y0 = make_initial_state(2)
        t_eval = torch.arange(0, 5.0, 0.1, dtype=torch.float64)
        # Must complete without raising. NaN is acceptable here per D4;
        # BILIN-06 is the test that enforces finite trajectories.
        sol = integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)
        assert sol.shape == (
            t_eval.shape[0],
            10,
        ), f"unexpected solution shape {tuple(sol.shape)}; expected ({t_eval.shape[0]}, 10)"


class TestThreeSigmaWorstCase:
    """BILIN-06: 3-sigma worst-case stability test (500s, no NaN / Inf)."""

    def test_three_sigma_b_sustained_mod_no_nan_500s(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BILIN-06: 3-sigma worst-case B + sustained u_mod=1 for 500s, no NaN.

        Fixture (CONTEXT-locked):
          - N = 3 regions.
          - A = parameterize_A(zeros(3,3)) -> diagonal = -0.5, off-diagonal = 0.
          - B: (1, 3, 3) with off-diagonal = 3.0 = 3 * sqrt(prior_var=1.0)
            and diagonal = 0 (default safe-diag).
          - C = zeros(3, 1): driving input disabled.
          - u_drive = 0, u_mod = 1 sustained.
          - solver = rk4, dt = 0.1, duration = 500.0 s.

        Pass criterion: all entries of the full 5N state trajectory are
        finite in IEEE-754. States may grow very large but must not
        become NaN or +/-Inf.
        """
        # Silence stability warnings during CI (this test is expected to
        # fire the monitor many times; that is not the test's purpose).
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

        N = 3
        # Stable baseline A: parameterize_A on zeros -> diag = -0.5.
        A_free = torch.zeros(N, N, dtype=torch.float64)
        A = parameterize_A(A_free)

        # 3-sigma worst-case B: off-diagonal = 3.0 = 3 * sqrt(prior_var=1.0);
        # diagonal = 0 (CONTEXT-locked; default safe-diag).
        B = torch.tensor(
            [
                [
                    [0.0, 3.0, 3.0],
                    [3.0, 0.0, 3.0],
                    [3.0, 3.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        assert B.shape == (1, N, N), (
            f"BILIN-06 fixture shape drift; expected (1, {N}, {N}), "
            f"got {tuple(B.shape)}"
        )

        # C zeroed: adversarial scenario where the modulator fully
        # dominates the dynamics.
        C = torch.zeros(N, 1, dtype=torch.float64)

        # input_fn: 1 driving col held at 0 + 1 modulator col held at 1.
        times = torch.tensor([0.0], dtype=torch.float64)
        values = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        input_fn = PiecewiseConstantInput(times, values)

        system = CoupledDCMSystem(
            A,
            C,
            input_fn,
            B=B,
            n_driving_inputs=1,
            stability_check_every=10,
        )
        y0 = make_initial_state(N, dtype=torch.float64)
        t_eval = torch.arange(0.0, 500.0, 0.1, dtype=torch.float64)
        sol = integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)

        # Pass criterion: no NaN or Inf anywhere in the full 5N trajectory.
        finite_mask = torch.isfinite(sol)
        n_nonfinite = int((~finite_mask).sum().item())
        assert n_nonfinite == 0, (
            f"BILIN-06 failure: {n_nonfinite} non-finite entries in 5N "
            f"trajectory (expected 0); solution shape="
            f"{tuple(sol.shape)}"
        )
