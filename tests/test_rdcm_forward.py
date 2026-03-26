"""Unit tests for rDCM forward pipeline (rdcm_forward.py).

Tests cover:
- Euler integration (equilibrium, impulse, shapes, stability)
- HRF generation (shape, peak timing, positivity, baseline return)
- BOLD generation (shape, SNR, keys, no-NaN, zero-padding effect)
- Derivative coefficients (shape, DC, analytic formula)
- Real/imaginary splitting (even N, odd N, DC/Nyquist exclusion)
- reduce_zeros (no-zeros, with-zeros)
- create_regressors integration (shape, dtype, derivative in Y)

All tests use torch.float64 / torch.complex128 for precision.
"""

from __future__ import annotations

import torch
import pytest

from pyro_dcm.forward_models.rdcm_forward import (
    dcm_euler_step,
    euler_integrate_dcm,
    get_hrf,
    generate_bold,
    compute_derivative_coefficients,
    split_real_imag,
    reduce_zeros,
    create_regressors,
)


# -----------------------------------------------------------------------
# Euler integration tests
# -----------------------------------------------------------------------

class TestEulerStep:
    """Tests for dcm_euler_step."""

    def test_euler_step_equilibrium(self) -> None:
        """At equilibrium (x=0, s=0, f=1, v=1, q=1) with zero input,
        one Euler step should return the same state (within 1e-12)."""
        nr = 3
        A = -0.5 * torch.eye(nr, dtype=torch.float64)
        C = torch.zeros(nr, 1, dtype=torch.float64)
        u_t = torch.zeros(1, dtype=torch.float64)

        x = torch.zeros(nr, dtype=torch.float64)
        s = torch.zeros(nr, dtype=torch.float64)
        f = torch.ones(nr, dtype=torch.float64)
        v = torch.ones(nr, dtype=torch.float64)
        q = torch.ones(nr, dtype=torch.float64)

        dt = 0.01
        x2, s2, f2, v2, q2 = dcm_euler_step(
            x, s, f, v, q, A, C, u_t, dt
        )

        torch.testing.assert_close(x2, x, atol=1e-12, rtol=0)
        torch.testing.assert_close(s2, s, atol=1e-12, rtol=0)
        torch.testing.assert_close(f2, f, atol=1e-12, rtol=0)
        torch.testing.assert_close(v2, v, atol=1e-12, rtol=0)
        torch.testing.assert_close(q2, q, atol=1e-12, rtol=0)

    def test_euler_step_impulse_response(self) -> None:
        """Apply unit impulse to 1-region DCM (A=-1, C=1). After two
        steps, x and s should both be positive. The first step updates
        x from the input (C @ u_t), and s responds on the next step
        since it is driven by the *previous* x."""
        A = torch.tensor([[-1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        u_t = torch.tensor([1.0], dtype=torch.float64)
        u_zero = torch.tensor([0.0], dtype=torch.float64)

        x = torch.zeros(1, dtype=torch.float64)
        s = torch.zeros(1, dtype=torch.float64)
        f = torch.ones(1, dtype=torch.float64)
        v = torch.ones(1, dtype=torch.float64)
        q = torch.ones(1, dtype=torch.float64)

        dt = 0.01
        # Step 1: impulse drives x positive
        x2, s2, f2, v2, q2 = dcm_euler_step(
            x, s, f, v, q, A, C, u_t, dt
        )
        assert x2[0].item() > 0.0

        # Step 2: positive x drives s positive
        x3, s3, f3, v3, q3 = dcm_euler_step(
            x2, s2, f2, v2, q2, A, C, u_zero, dt
        )
        assert s3[0].item() > 0.0


class TestEulerIntegrate:
    """Tests for euler_integrate_dcm."""

    def test_euler_integrate_shape(self) -> None:
        """For nr=3, nu=2, N_t=1000, output shapes are correct."""
        nr, nu, N_t = 3, 2, 1000
        A = -0.5 * torch.eye(nr, dtype=torch.float64)
        C = torch.randn(nr, nu, dtype=torch.float64) * 0.1
        u = torch.randn(N_t, nu, dtype=torch.float64)
        dt = 0.01

        x_all, bold_all = euler_integrate_dcm(A, C, u, dt)

        assert x_all.shape == (N_t, nr)
        assert bold_all.shape == (N_t, nr)

    def test_euler_integrate_stability(self) -> None:
        """For a stable A, 500s integration produces no NaN or Inf."""
        nr, nu = 3, 1
        A = torch.tensor(
            [[-0.5, 0.1, 0.0],
             [0.2, -0.5, 0.1],
             [0.0, 0.3, -0.5]],
            dtype=torch.float64,
        )
        C = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
        dt = 0.01
        N_t = int(500.0 / dt)  # 500 seconds
        u = torch.zeros(N_t, nu, dtype=torch.float64)
        # Block stimulus: on for first 10s
        u[:int(10.0 / dt), 0] = 1.0

        x_all, bold_all = euler_integrate_dcm(A, C, u, dt)

        assert not torch.isnan(x_all).any()
        assert not torch.isinf(x_all).any()
        assert not torch.isnan(bold_all).any()
        assert not torch.isinf(bold_all).any()


# -----------------------------------------------------------------------
# HRF tests
# -----------------------------------------------------------------------

class TestGetHRF:
    """Tests for get_hrf."""

    def test_get_hrf_shape(self) -> None:
        """For N=1000, u_dt=0.01, HRF has shape (1000,)."""
        hrf = get_hrf(1000, 0.01)
        assert hrf.shape == (1000,)

    def test_get_hrf_peak(self) -> None:
        """HRF should peak at approximately 5-6 seconds (canonical).
        Check argmax * u_dt is in [4, 8] seconds."""
        u_dt = 0.01
        hrf = get_hrf(2000, u_dt)
        peak_time = hrf.argmax().item() * u_dt
        assert 4.0 <= peak_time <= 8.0, (
            f"HRF peak at {peak_time:.1f}s, expected [4, 8]s"
        )

    def test_get_hrf_positive_peak(self) -> None:
        """The HRF peak value should be positive."""
        hrf = get_hrf(1000, 0.01)
        assert hrf.max().item() > 0.0

    def test_get_hrf_returns_to_baseline(self) -> None:
        """HRF at the end (last 10%) should be near zero."""
        hrf = get_hrf(5000, 0.01)
        peak_val = hrf.max().item()
        tail = hrf[-500:]  # last 10%
        assert tail.abs().max().item() < 0.01 * abs(peak_val), (
            "HRF tail should return to near-zero baseline"
        )

    def test_get_hrf_no_nan(self) -> None:
        """HRF contains no NaN or Inf values."""
        hrf = get_hrf(1000, 0.01)
        assert not torch.isnan(hrf).any()
        assert not torch.isinf(hrf).any()


# -----------------------------------------------------------------------
# BOLD generation tests
# -----------------------------------------------------------------------

class TestGenerateBold:
    """Tests for generate_bold."""

    @pytest.fixture()
    def bold_params(self) -> dict:
        """Common parameters for BOLD generation tests."""
        nr, nu = 3, 2
        N_u = 5000
        u_dt = 0.01
        y_dt = 2.0
        A = -0.5 * torch.eye(nr, dtype=torch.float64)
        A[0, 1] = 0.1
        A[1, 0] = 0.2
        C = torch.zeros(nr, nu, dtype=torch.float64)
        C[0, 0] = 1.0
        C[1, 1] = 0.5
        u = torch.zeros(N_u, nu, dtype=torch.float64)
        # Block stimulus
        u[:int(5.0 / u_dt), 0] = 1.0
        u[int(15.0 / u_dt):int(20.0 / u_dt), 1] = 1.0
        return {
            "A": A, "C": C, "u": u,
            "u_dt": u_dt, "y_dt": y_dt,
            "nr": nr, "N_u": N_u,
        }

    def test_generate_bold_shape(self, bold_params: dict) -> None:
        """Output y has shape (N_u * u_dt / y_dt, nr)."""
        p = bold_params
        result = generate_bold(
            p["A"], p["C"], p["u"], p["u_dt"], p["y_dt"], SNR=1.0
        )
        N_y = int(p["N_u"] * p["u_dt"] / p["y_dt"])
        assert result["y"].shape == (N_y, p["nr"])
        assert result["y_clean"].shape == (N_y, p["nr"])

    def test_generate_bold_snr(self, bold_params: dict) -> None:
        """With SNR=1, noisy and clean differ. With very large SNR,
        they are very close."""
        p = bold_params
        torch.manual_seed(42)
        res_noisy = generate_bold(
            p["A"], p["C"], p["u"], p["u_dt"], p["y_dt"], SNR=1.0
        )
        res_clean = generate_bold(
            p["A"], p["C"], p["u"], p["u_dt"], p["y_dt"], SNR=1e12
        )

        # Noisy differs from clean
        diff_noisy = (res_noisy["y"] - res_noisy["y_clean"]).abs().max()
        assert diff_noisy.item() > 1e-10

        # Very high SNR is close
        diff_clean = (res_clean["y"] - res_clean["y_clean"]).abs().max()
        assert diff_clean.item() < 1e-6

    def test_generate_bold_keys(self, bold_params: dict) -> None:
        """Output dict contains expected keys."""
        p = bold_params
        result = generate_bold(
            p["A"], p["C"], p["u"], p["u_dt"], p["y_dt"]
        )
        assert set(result.keys()) == {"y", "y_clean", "x", "hrf"}

    def test_generate_bold_no_nan(self, bold_params: dict) -> None:
        """All output tensors contain no NaN or Inf."""
        p = bold_params
        result = generate_bold(
            p["A"], p["C"], p["u"], p["u_dt"], p["y_dt"]
        )
        for key, val in result.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"
            assert not torch.isinf(val).any(), f"Inf in {key}"

    def test_generate_bold_zero_padding(self) -> None:
        """Verify that convolution with zero-padding produces different
        results from direct (circular) FFT convolution."""
        nr, nu = 1, 1
        N_u = 500
        u_dt = 0.01
        A = torch.tensor([[-1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        u = torch.zeros(N_u, nu, dtype=torch.float64)
        u[:50, 0] = 1.0

        result = generate_bold(A, C, u, u_dt, u_dt, SNR=1e12)
        y_padded = result["y_clean"]

        # Direct circular convolution (no zero-padding)
        x, _ = euler_integrate_dcm(A, C, u, u_dt)
        hrf = get_hrf(N_u, u_dt)
        x_fft = torch.fft.fft(x[:, 0])
        hrf_fft = torch.fft.fft(hrf)
        y_circular = torch.fft.ifft(x_fft * hrf_fft).real

        # They should differ due to circular vs linear convolution
        diff = (y_padded[:, 0] - y_circular).abs().max()
        assert diff.item() > 1e-10, (
            "Zero-padded and circular convolution should differ"
        )


# -----------------------------------------------------------------------
# Derivative coefficient tests
# -----------------------------------------------------------------------

class TestDerivativeCoefficients:
    """Tests for compute_derivative_coefficients."""

    def test_derivative_coefficients_shape(self) -> None:
        """For N=100, output shape is (51,) complex128."""
        coef = compute_derivative_coefficients(100)
        assert coef.shape == (51,)
        assert coef.dtype == torch.complex128

    def test_derivative_coefficients_dc(self) -> None:
        """At k=0, coef = exp(0) - 1 = 0."""
        coef = compute_derivative_coefficients(100)
        assert abs(coef[0].item()) < 1e-15

    def test_derivative_coefficients_formula(self) -> None:
        """Verify coef[k] = exp(2*pi*i*k/N) - 1 for a few k values."""
        N = 64
        coef = compute_derivative_coefficients(N)
        for k in [1, 5, 10, 32]:
            angle = torch.tensor(
                2.0 * torch.pi * k / N, dtype=torch.float64
            )
            re = torch.cos(angle) - 1.0
            im = torch.sin(angle)
            expected = torch.complex(re, im)
            torch.testing.assert_close(
                coef[k], expected.to(torch.complex128),
                atol=1e-12, rtol=1e-12,
            )


# -----------------------------------------------------------------------
# Real/imaginary splitting tests
# -----------------------------------------------------------------------

class TestSplitRealImag:
    """Tests for split_real_imag."""

    def test_split_real_imag_even_N(self) -> None:
        """For even N_y, output length = N_y.

        N_rfft = N_y/2 + 1. Output = N_rfft (real) + (N_rfft - 2) (imag)
        = N_y/2 + 1 + N_y/2 - 1 = N_y.
        """
        N_y = 100
        z = torch.randn(N_y, 4, dtype=torch.float64)
        z_fft = torch.fft.rfft(z, dim=0)
        result = split_real_imag(z_fft, N_y)
        assert result.shape[0] == N_y
        assert result.shape[1] == 4
        assert result.dtype == torch.float64

    def test_split_real_imag_odd_N(self) -> None:
        """For odd N_y, output length = N_y.

        N_rfft = (N_y+1)/2. Output = N_rfft (real) + (N_rfft - 1) (imag)
        = (N_y+1)/2 + (N_y+1)/2 - 1 = N_y.
        """
        N_y = 101
        z = torch.randn(N_y, 3, dtype=torch.float64)
        z_fft = torch.fft.rfft(z, dim=0)
        result = split_real_imag(z_fft, N_y)
        assert result.shape[0] == N_y
        assert result.shape[1] == 3
        assert result.dtype == torch.float64

    def test_split_real_imag_excludes_dc_imag(self) -> None:
        """The DC imaginary component (always zero for real input)
        is not included in the output."""
        N_y = 50
        z = torch.randn(N_y, 2, dtype=torch.float64)
        z_fft = torch.fft.rfft(z, dim=0)
        result = split_real_imag(z_fft, N_y)

        # The real part occupies the first N_rfft rows
        N_rfft = N_y // 2 + 1
        # The imaginary part starts at index N_rfft and excludes DC
        # So first imag row corresponds to index 1 of z_fft.imag
        imag_start = N_rfft
        imag_section = result[imag_start:]

        # Verify DC imag (z_fft.imag[0]) is NOT in the output
        dc_imag = z_fft.imag[0]
        # DC imaginary should be ~0 for real input
        assert dc_imag.abs().max().item() < 1e-10
        # And the imag section should start from index 1
        torch.testing.assert_close(
            imag_section[0], z_fft.imag[1].to(torch.float64),
            atol=1e-14, rtol=0,
        )

    def test_split_real_imag_excludes_nyquist_imag_even(self) -> None:
        """For even N_y, the Nyquist imaginary component is excluded."""
        N_y = 50
        z = torch.randn(N_y, 2, dtype=torch.float64)
        z_fft = torch.fft.rfft(z, dim=0)
        result = split_real_imag(z_fft, N_y)

        N_rfft = N_y // 2 + 1
        # Imag section: indices [1:-1] of z_fft.imag -> length N_rfft-2
        imag_section = result[N_rfft:]
        assert imag_section.shape[0] == N_rfft - 2

        # Nyquist imag should be ~0 for real input
        nyquist_imag = z_fft.imag[-1]
        assert nyquist_imag.abs().max().item() < 1e-10

        # Last element of imag section should be z_fft.imag[-2]
        torch.testing.assert_close(
            imag_section[-1], z_fft.imag[-2].to(torch.float64),
            atol=1e-14, rtol=0,
        )


# -----------------------------------------------------------------------
# reduce_zeros tests
# -----------------------------------------------------------------------

class TestReduceZeros:
    """Tests for reduce_zeros."""

    def test_reduce_zeros_no_zeros(self) -> None:
        """If Y has no zero rows, output should be unchanged."""
        Y = torch.randn(50, 3, dtype=torch.float64) + 1.0
        X = torch.randn(50, 6, dtype=torch.float64)

        Y_out, X_out = reduce_zeros(Y, X)

        torch.testing.assert_close(Y_out, Y)
        torch.testing.assert_close(X_out, X)

    def test_reduce_zeros_with_zeros(self) -> None:
        """If Y has mostly zero rows for a region, some should be
        replaced with NaN to balance."""
        Y = torch.zeros(50, 2, dtype=torch.float64)
        X = torch.randn(50, 4, dtype=torch.float64)

        # Set only 10 rows to nonzero for region 0
        Y[:10, 0] = torch.randn(10, dtype=torch.float64) + 1.0
        # Region 1 all nonzero
        Y[:, 1] = torch.randn(50, dtype=torch.float64) + 1.0

        Y_out, X_out = reduce_zeros(Y, X)

        # Region 0: had 40 zeros, 10 nonzeros -> should remove 30 zeros
        nan_count_r0 = torch.isnan(Y_out[:, 0]).sum().item()
        assert nan_count_r0 == 30

        # Region 1: no zeros, so no NaN
        nan_count_r1 = torch.isnan(Y_out[:, 1]).sum().item()
        assert nan_count_r1 == 0


# -----------------------------------------------------------------------
# create_regressors integration tests
# -----------------------------------------------------------------------

class TestCreateRegressors:
    """Integration tests for create_regressors."""

    @pytest.fixture()
    def regressor_data(self) -> dict:
        """Prepare test data for create_regressors."""
        nr, nu = 3, 2
        N_u = 5000
        u_dt = 0.01
        y_dt = 2.0
        N_y = int(N_u * u_dt / y_dt)  # 25

        # Generate synthetic data
        A = -0.5 * torch.eye(nr, dtype=torch.float64)
        A[0, 1] = 0.1
        C = torch.zeros(nr, nu, dtype=torch.float64)
        C[0, 0] = 1.0
        C[1, 1] = 0.5

        u = torch.zeros(N_u, nu, dtype=torch.float64)
        u[:int(5.0 / u_dt), 0] = 1.0
        u[int(10.0 / u_dt):int(15.0 / u_dt), 1] = 1.0

        result = generate_bold(A, C, u, u_dt, y_dt, SNR=1e12)
        hrf = result["hrf"]
        y = result["y_clean"]

        return {
            "hrf": hrf, "y": y, "u": u,
            "u_dt": u_dt, "y_dt": y_dt,
            "nr": nr, "nu": nu, "N_y": N_y,
        }

    def test_create_regressors_shape(
        self, regressor_data: dict
    ) -> None:
        """X has shape (N_eff, nr+nu+1) and Y has shape (N_eff, nr)."""
        d = regressor_data
        X, Y, N_eff = create_regressors(
            d["hrf"], d["y"], d["u"], d["u_dt"], d["y_dt"]
        )
        # +1 for constant confound
        expected_cols = d["nr"] + d["nu"] + 1
        assert X.shape[1] == expected_cols
        assert Y.shape[1] == d["nr"]
        assert X.shape[0] == N_eff
        assert Y.shape[0] == N_eff

    def test_create_regressors_no_nan_in_valid_rows(
        self, regressor_data: dict
    ) -> None:
        """After filtering NaN rows, remaining X and Y are finite."""
        d = regressor_data
        X, Y, _ = create_regressors(
            d["hrf"], d["y"], d["u"], d["u_dt"], d["y_dt"]
        )

        # Get rows that are valid (not all NaN)
        # Check column 0 of Y for NaN
        valid_mask = ~torch.isnan(Y[:, 0])
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]

        assert torch.isfinite(X_valid).all(), "X has non-finite values"
        assert torch.isfinite(Y_valid).all(), "Y has non-finite values"

    def test_create_regressors_dtype(
        self, regressor_data: dict
    ) -> None:
        """X and Y should be float64 (not complex)."""
        d = regressor_data
        X, Y, _ = create_regressors(
            d["hrf"], d["y"], d["u"], d["u_dt"], d["y_dt"]
        )
        assert X.dtype == torch.float64
        assert Y.dtype == torch.float64

    def test_create_regressors_derivative_in_Y(
        self, regressor_data: dict
    ) -> None:
        """Y should represent the DFT of the temporal derivative of BOLD,
        not the raw BOLD DFT. Verify Y is not identical to rfft(y)."""
        d = regressor_data
        _, Y, _ = create_regressors(
            d["hrf"], d["y"], d["u"], d["u_dt"], d["y_dt"]
        )

        # Raw rfft of y, split into real/imag
        y_fft = torch.fft.rfft(d["y"], dim=0)
        y_split = split_real_imag(y_fft, d["N_y"])

        # Y should NOT be identical to y_split (derivative changes it)
        diff = (Y[:y_split.shape[0], :d["nr"]] - y_split).abs()
        # Filter NaN
        valid = ~torch.isnan(diff)
        if valid.any():
            assert diff[valid].max().item() > 1e-10, (
                "Y should differ from raw rfft(y) due to derivative"
            )
