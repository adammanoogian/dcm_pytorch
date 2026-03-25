"""Unit tests for Balloon-Windkessel hemodynamic model.

Tests BalloonWindkessel derivatives and DEFAULT_HEMO_PARAMS against
known values, verifying [REF-002] Eq. 2-5 implementation.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.balloon_model import (
    DEFAULT_HEMO_PARAMS,
    BalloonWindkessel,
)


class TestDefaultParams:
    """Tests for default hemodynamic parameter values."""

    def test_default_params_match_spm12(self) -> None:
        """Verify DEFAULT_HEMO_PARAMS matches SPM12 code values."""
        assert DEFAULT_HEMO_PARAMS["kappa"] == 0.64
        assert DEFAULT_HEMO_PARAMS["gamma"] == 0.32
        assert DEFAULT_HEMO_PARAMS["tau"] == 2.0
        assert DEFAULT_HEMO_PARAMS["alpha"] == 0.32
        assert DEFAULT_HEMO_PARAMS["E0"] == 0.40


class TestBalloonWindkessel:
    """Tests for BalloonWindkessel derivatives."""

    def test_steady_state_derivatives_zero(self, dtype: torch.dtype) -> None:
        """At steady state, ALL derivatives should be approximately zero.

        Steady state: x=0, s=0, lnf=0, lnv=0, lnq=0
        (i.e., f=1, v=1, q=1, no neural input, no vasodilatory signal).
        """
        bw = BalloonWindkessel()
        N = 3
        x = torch.zeros(N, dtype=dtype)
        s = torch.zeros(N, dtype=dtype)
        lnf = torch.zeros(N, dtype=dtype)
        lnv = torch.zeros(N, dtype=dtype)
        lnq = torch.zeros(N, dtype=dtype)

        ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

        tol = 1e-10
        assert torch.all(ds.abs() < tol), f"ds not zero: {ds}"
        assert torch.all(dlnf.abs() < tol), f"dlnf not zero: {dlnf}"
        assert torch.all(dlnv.abs() < tol), f"dlnv not zero: {dlnv}"
        assert torch.all(dlnq.abs() < tol), f"dlnq not zero: {dlnq}"

    def test_positive_input_increases_signal(
        self, dtype: torch.dtype
    ) -> None:
        """With x=1 (positive neural input at steady state), ds/dt > 0."""
        bw = BalloonWindkessel()
        N = 1
        x = torch.ones(N, dtype=dtype)
        s = torch.zeros(N, dtype=dtype)
        lnf = torch.zeros(N, dtype=dtype)
        lnv = torch.zeros(N, dtype=dtype)
        lnq = torch.zeros(N, dtype=dtype)

        ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

        assert ds.item() > 0, f"ds should be positive with x=1: {ds}"

    def test_log_space_positivity(self, dtype: torch.dtype) -> None:
        """Derivatives are finite even for extreme log-space states."""
        bw = BalloonWindkessel()
        N = 2
        x = torch.zeros(N, dtype=dtype)
        s = torch.zeros(N, dtype=dtype)
        lnf = torch.tensor([-10.0, 5.0], dtype=dtype)
        lnv = torch.tensor([5.0, -10.0], dtype=dtype)
        lnq = torch.tensor([-5.0, 3.0], dtype=dtype)

        ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

        for name, d in [("ds", ds), ("dlnf", dlnf), ("dlnv", dlnv), ("dlnq", dlnq)]:
            assert torch.all(torch.isfinite(d)), (
                f"{name} has non-finite values: {d}"
            )

    def test_oxygen_extraction_clamp(self, dtype: torch.dtype) -> None:
        """With very small f (lnf=-20), extraction produces finite values."""
        bw = BalloonWindkessel()
        N = 1
        x = torch.zeros(N, dtype=dtype)
        s = torch.zeros(N, dtype=dtype)
        lnf = torch.tensor([-20.0], dtype=dtype)
        lnv = torch.zeros(N, dtype=dtype)
        lnq = torch.zeros(N, dtype=dtype)

        ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

        for name, d in [("ds", ds), ("dlnf", dlnf), ("dlnv", dlnv), ("dlnq", dlnq)]:
            assert torch.all(torch.isfinite(d)), (
                f"{name} has NaN/Inf for lnf=-20: {d}"
            )

    def test_derivatives_shape(self, dtype: torch.dtype) -> None:
        """Output shapes match input shapes for various region counts."""
        bw = BalloonWindkessel()
        for N in [1, 3, 10]:
            x = torch.zeros(N, dtype=dtype)
            s = torch.zeros(N, dtype=dtype)
            lnf = torch.zeros(N, dtype=dtype)
            lnv = torch.zeros(N, dtype=dtype)
            lnq = torch.zeros(N, dtype=dtype)

            ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

            assert ds.shape == (N,), f"ds shape: {ds.shape}"
            assert dlnf.shape == (N,), f"dlnf shape: {dlnf.shape}"
            assert dlnv.shape == (N,), f"dlnv shape: {dlnv.shape}"
            assert dlnq.shape == (N,), f"dlnq shape: {dlnq.shape}"

    def test_known_derivative_values(self, dtype: torch.dtype) -> None:
        """Hand-compute derivatives for a specific state.

        With SPM12 defaults: kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40

        State: x=1, s=0.5, lnf=0.1, lnv=0.05, lnq=-0.02

        f = exp(0.1) = 1.10517...
        v = exp(0.05) = 1.05127...
        q = exp(-0.02) = 0.98020...

        fv = v^(1/0.32) = v^3.125
        E_f = 1 - (1 - 0.40)^(1/f) = 1 - 0.6^(1/1.10517)
        """
        bw = BalloonWindkessel()

        x = torch.tensor([1.0], dtype=dtype)
        s = torch.tensor([0.5], dtype=dtype)
        lnf = torch.tensor([0.1], dtype=dtype)
        lnv = torch.tensor([0.05], dtype=dtype)
        lnq = torch.tensor([-0.02], dtype=dtype)

        ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)

        # Hand-compute reference values
        f = torch.exp(lnf)
        v = torch.exp(lnv)
        q = torch.exp(lnq)
        fv = v.pow(1.0 / 0.32)
        E_f = 1.0 - (1.0 - 0.40) ** (1.0 / f)

        expected_ds = x - 0.64 * s - 0.32 * (f - 1.0)
        expected_dlnf = s / f
        expected_dlnv = (f - fv) / (2.0 * v)
        expected_dlnq = (f * E_f / 0.40 - fv * q / v) / (2.0 * q)

        torch.testing.assert_close(ds, expected_ds, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            dlnf, expected_dlnf, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            dlnv, expected_dlnv, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            dlnq, expected_dlnq, atol=1e-10, rtol=1e-10
        )
