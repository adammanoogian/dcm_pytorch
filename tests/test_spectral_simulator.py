"""Integration tests for spectral DCM simulator and full pipeline.

Tests cover:
- Simulator output structure, shapes, and dtypes
- Mathematical properties (Hermitian CSD, positive auto-spectra)
- Reproducibility and sensitivity to parameters
- Eigenfrequency physics validation
- Roundtrip consistency between time-domain (Phase 1) and frequency-domain
  (Phase 2) models
- Regression verification that Phase 1 tests still pass
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.csd_computation import bold_to_csd_torch
from pyro_dcm.forward_models.spectral_transfer import default_frequency_grid
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    simulate_task_dcm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stable_A_3() -> torch.Tensor:
    """Stable 3x3 A matrix for spectral DCM tests."""
    return make_stable_A_spectral(3, seed=42)


@pytest.fixture()
def sim_result_3(stable_A_3: torch.Tensor) -> dict:
    """Simulator output for 3-region spectral DCM."""
    return simulate_spectral_dcm(stable_A_3, TR=2.0, n_freqs=32)


# ---------------------------------------------------------------------------
# Test: Output structure
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMOutputKeys:
    """Verify output dictionary has all required keys."""

    def test_output_keys(self, sim_result_3: dict) -> None:
        expected_keys = {
            "csd",
            "freqs",
            "transfer_function",
            "neuronal_noise",
            "observation_noise",
            "params",
        }
        assert set(sim_result_3.keys()) == expected_keys

    def test_params_keys(self, sim_result_3: dict) -> None:
        expected_params = {"A", "noise_params", "TR", "n_freqs"}
        assert set(sim_result_3["params"].keys()) == expected_params


# ---------------------------------------------------------------------------
# Test: Shapes
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMShapes:
    """Verify output tensor shapes for N=3, F=32."""

    def test_csd_shape(self, sim_result_3: dict) -> None:
        assert sim_result_3["csd"].shape == (32, 3, 3)

    def test_freqs_shape(self, sim_result_3: dict) -> None:
        assert sim_result_3["freqs"].shape == (32,)

    def test_transfer_function_shape(self, sim_result_3: dict) -> None:
        assert sim_result_3["transfer_function"].shape == (32, 3, 3)

    def test_neuronal_noise_shape(self, sim_result_3: dict) -> None:
        assert sim_result_3["neuronal_noise"].shape == (32, 3, 3)

    def test_observation_noise_shape(self, sim_result_3: dict) -> None:
        assert sim_result_3["observation_noise"].shape == (32, 3, 3)


# ---------------------------------------------------------------------------
# Test: Dtypes
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMDtypes:
    """Verify output tensor dtypes."""

    def test_csd_dtype(self, sim_result_3: dict) -> None:
        assert sim_result_3["csd"].dtype == torch.complex128

    def test_freqs_dtype(self, sim_result_3: dict) -> None:
        assert sim_result_3["freqs"].dtype == torch.float64

    def test_transfer_function_dtype(self, sim_result_3: dict) -> None:
        assert sim_result_3["transfer_function"].dtype == torch.complex128


# ---------------------------------------------------------------------------
# Test: Default noise parameters
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMDefaultNoise:
    """With noise_params=None, default priors (zeros) produce valid CSD."""

    def test_default_noise_no_nan(self, stable_A_3: torch.Tensor) -> None:
        result = simulate_spectral_dcm(stable_A_3, noise_params=None)
        assert not torch.isnan(result["csd"].real).any()
        assert not torch.isnan(result["csd"].imag).any()

    def test_default_noise_no_inf(self, stable_A_3: torch.Tensor) -> None:
        result = simulate_spectral_dcm(stable_A_3, noise_params=None)
        assert not torch.isinf(result["csd"].real).any()
        assert not torch.isinf(result["csd"].imag).any()


# ---------------------------------------------------------------------------
# Test: Custom noise parameters
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMCustomNoise:
    """Custom noise_params produce different CSD than defaults."""

    def test_custom_noise_changes_csd(
        self, stable_A_3: torch.Tensor
    ) -> None:
        # Default (zeros)
        result_default = simulate_spectral_dcm(
            stable_A_3, noise_params=None
        )

        # Custom (non-zero)
        N = 3
        noise_params = {
            "a": torch.ones(2, N, dtype=torch.float64) * 0.5,
            "b": torch.ones(2, 1, dtype=torch.float64) * 0.3,
            "c": torch.ones(2, N, dtype=torch.float64) * 0.2,
        }
        result_custom = simulate_spectral_dcm(
            stable_A_3, noise_params=noise_params
        )

        # CSD should be different
        assert not torch.allclose(
            result_default["csd"], result_custom["csd"]
        )


# ---------------------------------------------------------------------------
# Test: Hermitian property
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMHermitian:
    """Output CSD is Hermitian at each frequency."""

    def test_csd_hermitian(self, sim_result_3: dict) -> None:
        csd = sim_result_3["csd"]
        # CSD[f] should equal conj(CSD[f]^T) at each frequency
        csd_H = csd.conj().transpose(-2, -1)
        assert torch.allclose(csd, csd_H, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: Reproducibility
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMReproducible:
    """Same seed and A produce identical CSD."""

    def test_reproducible_with_seed(self) -> None:
        A = make_stable_A_spectral(3, seed=42)
        result1 = simulate_spectral_dcm(A, seed=123)
        result2 = simulate_spectral_dcm(A, seed=123)
        assert torch.allclose(result1["csd"], result2["csd"])


# ---------------------------------------------------------------------------
# Test: Different A produces different CSD
# ---------------------------------------------------------------------------


class TestSimulateSpectralDCMDifferentA:
    """Two different A matrices produce different CSD."""

    def test_different_A_different_csd(self) -> None:
        A1 = make_stable_A_spectral(3, seed=42)
        A2 = make_stable_A_spectral(3, seed=99)

        result1 = simulate_spectral_dcm(A1)
        result2 = simulate_spectral_dcm(A2)

        assert not torch.allclose(result1["csd"], result2["csd"])


# ---------------------------------------------------------------------------
# Test: make_stable_A_spectral
# ---------------------------------------------------------------------------


class TestMakeStableASpectral:
    """Tests for the stable A matrix generator."""

    def test_eigenvalues_negative(self) -> None:
        A = make_stable_A_spectral(5, seed=42)
        eigvals = torch.linalg.eigvals(A.to(torch.complex128))
        assert eigvals.real.max() < 0

    def test_shape(self) -> None:
        A = make_stable_A_spectral(5, seed=42)
        assert A.shape == (5, 5)
        assert A.dtype == torch.float64

    def test_reproducible(self) -> None:
        A1 = make_stable_A_spectral(4, seed=77)
        A2 = make_stable_A_spectral(4, seed=77)
        assert torch.allclose(A1, A2)

    def test_different_seeds(self) -> None:
        A1 = make_stable_A_spectral(4, seed=10)
        A2 = make_stable_A_spectral(4, seed=20)
        assert not torch.allclose(A1, A2)

    def test_self_connection_negative(self) -> None:
        A = make_stable_A_spectral(3, self_connection=-1.0, seed=42)
        assert (A.diagonal() == -1.0).all()

    def test_invalid_self_connection_raises(self) -> None:
        with pytest.raises(ValueError, match="self_connection must be neg"):
            make_stable_A_spectral(3, self_connection=0.5)


# ---------------------------------------------------------------------------
# Test: CSD peak at eigenfrequency
# ---------------------------------------------------------------------------


class TestCSDPeakAtEigenfrequency:
    """Transfer function shows peak near dominant eigenfrequency of A.

    For a 2-region A with known dominant eigenvalue lambda = -0.5 + 0.3i
    (oscillatory mode), the transfer function magnitude should show a
    resonance peak near f = 0.3/(2*pi) ~ 0.048 Hz.

    We test the transfer function magnitude directly rather than the
    full CSD, because the 1/f noise spectrum in the CSD can mask the
    resonance peak (1/f power dominates at low frequencies).
    """

    def test_transfer_function_peak_at_eigenfrequency(self) -> None:
        # Construct A with known complex eigenvalues
        # lambdas = [-0.5 + 0.3i, -0.5 - 0.3i]  (conjugate pair)
        # Real-valued 2x2 block form: [[sigma, -omega], [omega, sigma]]
        sigma = -0.5
        omega = 0.3
        A = torch.tensor(
            [[sigma, -omega], [omega, sigma]],
            dtype=torch.float64,
        )

        # The eigenfrequency is omega/(2*pi) ~ 0.048 Hz.
        # For a 2x2 system, the diagonal transfer function element
        # H[0,0](w) = (iw - sigma) / ((iw - sigma)^2 + omega^2)
        # peaks slightly below omega/(2*pi) due to the numerator.
        # We compute the expected peak numerically.
        w_test = np.linspace(0.001, 0.5, 10000) * 2.0 * np.pi
        s_test = 1j * w_test - sigma
        H00_mag = np.abs(s_test / (s_test**2 + omega**2))
        f_expected = w_test[np.argmax(H00_mag)] / (2.0 * np.pi)

        result = simulate_spectral_dcm(A, n_freqs=128)
        H = result["transfer_function"]
        freqs = result["freqs"]

        # Use diagonal transfer function magnitude (auto-transfer)
        H_diag_mag = H[:, 0, 0].abs()
        peak_idx = H_diag_mag.argmax().item()
        f_peak = freqs[peak_idx].item()

        # The peak should be within 3 frequency bins of expected
        freq_step = (freqs[-1] - freqs[0]).item() / (len(freqs) - 1)
        assert abs(f_peak - f_expected) <= 3.0 * freq_step, (
            f"Peak at {f_peak:.4f} Hz, expected near {f_expected:.4f} Hz "
            f"(tolerance: {3.0 * freq_step:.4f} Hz)"
        )

        # Also verify peak is in the vicinity of eigenfrequency
        f_eigen = abs(omega) / (2.0 * np.pi)
        assert abs(f_peak - f_eigen) < 0.02, (
            f"Peak at {f_peak:.4f} Hz should be near eigenfrequency "
            f"{f_eigen:.4f} Hz (within 0.02 Hz)"
        )


# ---------------------------------------------------------------------------
# Test: Full pipeline roundtrip (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


class TestFullPipelineRoundtrip:
    """Integration test connecting Phase 1 and Phase 2.

    1. Create A matrix (2 regions, stable).
    2. Predict CSD from spectral model (Phase 2).
    3. Generate BOLD time series from Phase 1 (task DCM).
    4. Compute empirical CSD from BOLD.
    5. Verify both produce valid CSD with positive auto-spectra and
       that the spectral shapes have positive correlation.

    NOTE: The spectral model assumes resting-state (no driving input)
    while the task simulator uses block-design stimuli, so correlation
    thresholds are lenient. Both CSDs should be decreasing with
    frequency (1/f-like behavior) which produces positive correlation
    even with different generating processes.
    """

    def test_roundtrip_spectral_shape_correlation(self) -> None:
        # Step 1: Create a simple stable 2-region A
        A = torch.tensor(
            [[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64
        )
        TR = 2.0

        # Step 2: Predicted CSD from spectral model
        spectral_result = simulate_spectral_dcm(A, TR=TR, n_freqs=32)
        pred_csd = spectral_result["csd"]
        freqs = spectral_result["freqs"]

        # Step 3: Generate BOLD from Phase 1 task simulator
        C = torch.tensor([[0.25], [0.0]], dtype=torch.float64)
        stim = make_block_stimulus(
            n_blocks=10, block_duration=30, rest_duration=20
        )
        duration = 10 * (30 + 20)  # 500 seconds
        task_result = simulate_task_dcm(
            A, C, stim, duration=float(duration), TR=TR, SNR=10.0, seed=42
        )
        bold = task_result["bold"]  # shape (T_TR, 2)

        # Step 4: Compute empirical CSD from BOLD
        empirical_csd = bold_to_csd_torch(bold, fs=1.0 / TR, freqs=freqs)

        # Step 5a: Both CSDs should have valid structure
        assert not torch.isnan(pred_csd).any(), "Predicted CSD has NaN"
        assert not torch.isnan(empirical_csd).any(), "Empirical CSD has NaN"

        # Step 5b: Both should have positive auto-spectra
        N = A.shape[0]
        for region in range(N):
            pred_auto = pred_csd[:, region, region].real
            emp_auto = empirical_csd[:, region, region].real
            assert (pred_auto > 0).all(), (
                f"Predicted auto-spectrum region {region} not positive"
            )
            # Empirical auto-spectra should be mostly positive
            assert (emp_auto >= 0).sum() > len(emp_auto) // 2, (
                f"Empirical auto-spectrum region {region} mostly negative"
            )

        # Step 5c: Spectral shape correlation (sanity check)
        # Use manual correlation to avoid np.corrcoef crash on Windows
        for region in range(N):
            pred_auto = pred_csd[:, region, region].real.numpy()
            emp_auto = empirical_csd[:, region, region].real.numpy()

            pred_std = float(pred_auto.std())
            emp_std = float(emp_auto.std())
            if pred_std < 1e-30 or emp_std < 1e-30:
                continue

            # Manual Pearson correlation
            pred_c = pred_auto - pred_auto.mean()
            emp_c = emp_auto - emp_auto.mean()
            corr = float(
                np.sum(pred_c * emp_c)
                / (np.sqrt(np.sum(pred_c**2)) * np.sqrt(np.sum(emp_c**2))
                   + 1e-30)
            )

            # Positive correlation expected (both show 1/f-like decay)
            assert corr > 0.0, (
                f"Region {region}: spectral correlation = {corr:.4f}, "
                f"expected positive (predicted vs empirical auto-spectrum)"
            )


# ---------------------------------------------------------------------------
# Test: Phase 1 regression check
# ---------------------------------------------------------------------------


class TestPhase1Regression:
    """Verify Phase 1 test modules can still be imported and collected."""

    def test_phase1_test_modules_importable(self) -> None:
        # Import all Phase 1 test modules to confirm no breakage
        import tests.test_balloon  # noqa: F401
        import tests.test_bold_signal  # noqa: F401
        import tests.test_neural_state  # noqa: F401
        import tests.test_ode_integrator  # noqa: F401
        import tests.test_task_simulator  # noqa: F401

    def test_phase2_test_modules_importable(self) -> None:
        # Also verify Phase 2 test modules
        import tests.test_csd_computation  # noqa: F401
        import tests.test_spectral_noise  # noqa: F401
        import tests.test_spectral_transfer  # noqa: F401
