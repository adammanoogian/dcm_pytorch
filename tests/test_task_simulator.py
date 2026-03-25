"""Validation tests for the task-DCM data simulator.

Tests the full end-to-end simulation pipeline including ODE integration,
BOLD signal generation, downsampling, noise addition, and convenience
functions for block stimuli and random connectivity matrices.

These tests validate the Phase 1 success criteria:
- SC#1: 500s simulation without NaN for 3+ regions
- SC#2: Stable neural trajectories for stable A matrices
- SC#3: BOLD percent signal change in 0.5-5% range
- SC#4: Multiple solvers supported (tested in Plan 02)
- SC#5: Simulator generates N-region BOLD given (A, C, u(t), hemo_params, SNR)
"""

from __future__ import annotations

import torch
import pytest

from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_A_3() -> torch.Tensor:
    """3-region A matrix with moderate connectivity."""
    return torch.tensor(
        [
            [-0.5, 0.1, 0.0],
            [0.2, -0.5, 0.1],
            [0.0, 0.3, -0.5],
        ],
        dtype=torch.float64,
    )


@pytest.fixture()
def simple_C_3() -> torch.Tensor:
    """3-region C matrix, only region 0 driven."""
    return torch.tensor(
        [[0.25], [0.0], [0.0]],
        dtype=torch.float64,
    )


@pytest.fixture()
def block_stimulus() -> dict[str, torch.Tensor]:
    """Standard block-design stimulus: 5 blocks of 30s ON / 20s OFF."""
    return make_block_stimulus(n_blocks=5, block_duration=30, rest_duration=20)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------


class TestSimulatorOutputStructure:
    """Tests for simulator output dictionary keys and shapes."""

    def test_simulator_output_keys(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
        block_stimulus: dict[str, torch.Tensor],
    ) -> None:
        """Verify all expected keys are present in output dict."""
        result = simulate_task_dcm(
            simple_A_3, simple_C_3, block_stimulus,
            duration=50.0, seed=42,
        )

        expected_keys = {
            "bold", "bold_clean", "bold_fine", "neural",
            "hemodynamic", "times_fine", "times_TR",
            "params", "stimulus",
        }
        assert set(result.keys()) == expected_keys

        # Check hemodynamic sub-dict keys
        hemo_keys = {"s", "f", "v", "q"}
        assert set(result["hemodynamic"].keys()) == hemo_keys

        # Check params sub-dict keys
        params_keys = {"A", "C", "hemo_params", "SNR", "TR", "duration", "solver"}
        assert set(result["params"].keys()) == params_keys

    def test_simulator_output_shapes(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
    ) -> None:
        """For N=3, M=1, duration=100, TR=2.0, dt=0.01: verify shapes."""
        stim = make_block_stimulus(n_blocks=2, block_duration=30, rest_duration=20)
        result = simulate_task_dcm(
            simple_A_3, simple_C_3, stim,
            duration=100.0, dt=0.01, TR=2.0, seed=42,
        )

        N = 3
        T_fine = 10000  # duration / dt
        T_TR = 50       # duration / TR

        assert result["bold"].shape == (T_TR, N)
        assert result["bold_clean"].shape == (T_TR, N)
        assert result["bold_fine"].shape == (T_fine, N)
        assert result["neural"].shape == (T_fine, N)
        assert result["times_fine"].shape == (T_fine,)
        assert result["times_TR"].shape == (T_TR,)

        # Hemodynamic states
        for key in ("s", "f", "v", "q"):
            assert result["hemodynamic"][key].shape == (T_fine, N)


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------


class TestSimulatorNumerics:
    """Tests for numerical correctness and stability."""

    def test_simulator_no_nan(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
    ) -> None:
        """Run 300s simulation, verify no NaN in any output."""
        stim = make_block_stimulus(n_blocks=6, block_duration=30, rest_duration=20)
        result = simulate_task_dcm(
            simple_A_3, simple_C_3, stim,
            duration=300.0, SNR=5.0, seed=42,
        )

        assert not torch.isnan(result["bold"]).any(), "NaN in noisy BOLD"
        assert not torch.isnan(result["bold_clean"]).any(), "NaN in clean BOLD"
        assert not torch.isnan(result["bold_fine"]).any(), "NaN in fine BOLD"
        assert not torch.isnan(result["neural"]).any(), "NaN in neural states"
        for key, val in result["hemodynamic"].items():
            assert not torch.isnan(val).any(), f"NaN in hemodynamic {key}"

    def test_simulator_bold_range(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
    ) -> None:
        """Peak BOLD in driven region should be in 0.5-5% range.

        Validates SUCCESS CRITERION #3.
        """
        stim = make_block_stimulus(n_blocks=5, block_duration=30, rest_duration=20)
        result = simulate_task_dcm(
            simple_A_3, simple_C_3, stim,
            duration=250.0, SNR=0, seed=42,  # No noise for clean measurement
        )

        # Region 0 is driven (C[0,0]=0.25)
        peak_bold_region0 = result["bold_clean"][:, 0].abs().max().item()

        assert peak_bold_region0 >= 0.005, (
            f"Peak BOLD {peak_bold_region0:.6f} below 0.5% threshold"
        )
        assert peak_bold_region0 <= 0.05, (
            f"Peak BOLD {peak_bold_region0:.6f} above 5% threshold"
        )

    def test_simulator_snr(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
    ) -> None:
        """Empirical SNR should be within 20% of requested SNR=10.0."""
        stim = make_block_stimulus(n_blocks=5, block_duration=30, rest_duration=20)
        requested_snr = 10.0

        result = simulate_task_dcm(
            simple_A_3, simple_C_3, stim,
            duration=250.0, SNR=requested_snr, seed=42,
        )

        # Compute empirical SNR per region
        noise = result["bold"] - result["bold_clean"]
        signal_std = result["bold_clean"].std(dim=0)
        noise_std = noise.std(dim=0)

        # Avoid division by zero for undriven regions with near-zero signal
        driven_mask = signal_std > 1e-10
        empirical_snr = signal_std[driven_mask] / noise_std[driven_mask]

        for region_idx, snr_val in enumerate(empirical_snr):
            assert snr_val.item() > requested_snr * 0.8, (
                f"Region {region_idx}: SNR {snr_val:.2f} below 80% of {requested_snr}"
            )
            assert snr_val.item() < requested_snr * 1.2, (
                f"Region {region_idx}: SNR {snr_val:.2f} above 120% of {requested_snr}"
            )

    def test_simulator_no_noise_mode(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
        block_stimulus: dict[str, torch.Tensor],
    ) -> None:
        """With SNR<=0, noisy BOLD should equal clean BOLD exactly."""
        result = simulate_task_dcm(
            simple_A_3, simple_C_3, block_stimulus,
            duration=100.0, SNR=0, seed=42,
        )
        assert torch.allclose(result["bold"], result["bold_clean"]), (
            "SNR=0 should produce no noise"
        )

        # Also test negative SNR
        result_neg = simulate_task_dcm(
            simple_A_3, simple_C_3, block_stimulus,
            duration=100.0, SNR=-1.0, seed=42,
        )
        assert torch.allclose(result_neg["bold"], result_neg["bold_clean"]), (
            "SNR<0 should produce no noise"
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestSimulatorReproducibility:
    """Tests for seeded reproducibility."""

    def test_simulator_reproducibility(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
        block_stimulus: dict[str, torch.Tensor],
    ) -> None:
        """Same seed -> identical output; different seeds -> different output."""
        kwargs = dict(
            A=simple_A_3, C=simple_C_3, stimulus=block_stimulus,
            duration=100.0, SNR=5.0,
        )

        result1 = simulate_task_dcm(**kwargs, seed=42)
        result2 = simulate_task_dcm(**kwargs, seed=42)
        result3 = simulate_task_dcm(**kwargs, seed=99)

        # Same seed -> identical
        assert torch.allclose(result1["bold"], result2["bold"]), (
            "Same seed should produce identical noisy BOLD"
        )
        assert torch.allclose(result1["bold_clean"], result2["bold_clean"]), (
            "Same seed should produce identical clean BOLD"
        )

        # Different seed -> different noisy BOLD (clean should still match)
        assert torch.allclose(result1["bold_clean"], result3["bold_clean"]), (
            "Clean BOLD should be deterministic regardless of seed"
        )
        assert not torch.allclose(result1["bold"], result3["bold"]), (
            "Different seeds should produce different noisy BOLD"
        )


# ---------------------------------------------------------------------------
# Multi-region tests
# ---------------------------------------------------------------------------


class TestSimulatorMultiRegion:
    """Tests for multi-region network configurations."""

    def test_simulator_5region(self) -> None:
        """Run with 5-region network. Verify shapes and no NaN."""
        N = 5
        A = make_random_stable_A(N, density=0.4, seed=42)
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 0.25

        stim = make_block_stimulus(n_blocks=3, block_duration=30, rest_duration=20)
        result = simulate_task_dcm(
            A, C, stim, duration=150.0, SNR=5.0, seed=42,
        )

        T_fine = 15000  # 150 / 0.01
        T_TR = 75       # 150 / 2.0

        assert result["bold"].shape == (T_TR, N)
        assert result["neural"].shape == (T_fine, N)
        assert not torch.isnan(result["bold"]).any()
        assert not torch.isnan(result["neural"]).any()
        assert torch.isfinite(result["bold"]).all()

    def test_simulator_500s(self) -> None:
        """Full 500s simulation with 10 blocks of 30s ON / 20s OFF.

        Validates SUCCESS CRITERION #1 and #5: stable 500s simulation
        with 3 regions, realistic BOLD, no NaN.
        """
        A = torch.tensor(
            [
                [-0.5, 0.1, 0.0],
                [0.2, -0.5, 0.1],
                [0.0, 0.3, -0.5],
            ],
            dtype=torch.float64,
        )
        C = torch.tensor(
            [[0.25], [0.0], [0.0]],
            dtype=torch.float64,
        )
        stim = make_block_stimulus(
            n_blocks=10, block_duration=30, rest_duration=20,
        )

        result = simulate_task_dcm(
            A, C, stim, duration=500.0, SNR=5.0, seed=42,
        )

        # No NaN or Inf
        assert not torch.isnan(result["bold"]).any(), "NaN in 500s BOLD"
        assert torch.isfinite(result["bold"]).all(), "Inf in 500s BOLD"
        assert not torch.isnan(result["bold_clean"]).any(), "NaN in 500s clean BOLD"
        assert torch.isfinite(result["bold_clean"]).all(), "Inf in 500s clean BOLD"

        # Shape check
        T_TR = 250  # 500 / 2.0
        assert result["bold"].shape == (T_TR, 3)

        # BOLD in reasonable range (driven region should have 0.5-5%)
        peak = result["bold_clean"][:, 0].abs().max().item()
        assert peak >= 0.005, f"500s peak BOLD {peak:.6f} below 0.5%"
        assert peak <= 0.05, f"500s peak BOLD {peak:.6f} above 5%"

        # All hemodynamic states finite
        for key, val in result["hemodynamic"].items():
            assert torch.isfinite(val).all(), f"Non-finite hemodynamic {key} in 500s sim"


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestMakeBlockStimulus:
    """Tests for the block stimulus convenience function."""

    def test_make_block_stimulus(self) -> None:
        """Verify stimulus dict has correct structure and values."""
        stim = make_block_stimulus(
            n_blocks=10, block_duration=30, rest_duration=20,
        )

        # Correct number of transitions: 10 blocks * 2 (ON + OFF)
        assert stim["times"].shape == (20,)
        assert stim["values"].shape == (20, 1)

        # First onset at t=0, ON
        assert stim["times"][0].item() == 0.0
        assert stim["values"][0, 0].item() == 1.0

        # First offset at t=30, OFF
        assert stim["times"][1].item() == 30.0
        assert stim["values"][1, 0].item() == 0.0

        # Second onset at t=50 (30 + 20), ON
        assert stim["times"][2].item() == 50.0
        assert stim["values"][2, 0].item() == 1.0

        # Last onset at t=450 (9 * 50), ON
        assert stim["times"][-2].item() == 450.0
        assert stim["values"][-2, 0].item() == 1.0

        # Last offset at t=480 (450 + 30), OFF
        assert stim["times"][-1].item() == 480.0
        assert stim["values"][-1, 0].item() == 0.0

    def test_make_block_stimulus_multi_input(self) -> None:
        """Block stimulus with multiple inputs only activates input 0."""
        stim = make_block_stimulus(
            n_blocks=2, block_duration=10, rest_duration=10, n_inputs=3,
        )
        assert stim["values"].shape == (4, 3)

        # ON epochs: only input 0 is 1.0, others are 0.0
        assert stim["values"][0, 0].item() == 1.0
        assert stim["values"][0, 1].item() == 0.0
        assert stim["values"][0, 2].item() == 0.0


class TestMakeRandomStableA:
    """Tests for the random stable A matrix generator."""

    def test_make_random_stable_A(self) -> None:
        """A is stable: all eigenvalues have negative real part."""
        A = make_random_stable_A(5, density=0.5, seed=42)

        eigenvalues = torch.linalg.eigvals(A)
        max_real = eigenvalues.real.max().item()
        assert max_real < 0, (
            f"A matrix not stable: max eigenvalue real part = {max_real:.6f}"
        )

    def test_make_random_stable_A_diagonal(self) -> None:
        """Diagonal entries should be -self_inhibition."""
        A = make_random_stable_A(4, self_inhibition=0.5, seed=42)
        diag = torch.diagonal(A)
        expected = torch.full((4,), -0.5, dtype=torch.float64)
        assert torch.allclose(diag, expected), (
            f"Diagonal {diag} != expected {expected}"
        )

    def test_make_random_stable_A_density(self) -> None:
        """Approximate density of off-diagonal connections."""
        N = 10
        density = 0.3
        A = make_random_stable_A(N, density=density, seed=42)

        # Count non-zero off-diagonal entries
        mask = ~torch.eye(N, dtype=torch.bool)
        n_nonzero = (A[mask] != 0).sum().item()
        n_off_diag = N * (N - 1)
        actual_density = n_nonzero / n_off_diag

        # Should be approximately correct (within 10% due to rounding)
        assert abs(actual_density - density) < 0.15, (
            f"Actual density {actual_density:.3f} differs from {density}"
        )

    def test_make_random_stable_A_reproducibility(self) -> None:
        """Same seed produces same A matrix."""
        A1 = make_random_stable_A(5, seed=42)
        A2 = make_random_stable_A(5, seed=42)
        assert torch.allclose(A1, A2), "Same seed should produce same A"


# ---------------------------------------------------------------------------
# Neural dynamics tests
# ---------------------------------------------------------------------------


class TestNeuralDynamics:
    """Tests for neural state trajectory behavior."""

    def test_neural_state_stable_trajectory(self) -> None:
        """For stable A (all eigenvalues real part < 0), neural states should not diverge.

        Validates SUCCESS CRITERION #2.
        """
        A = torch.tensor(
            [
                [-0.5, 0.1, 0.0],
                [0.2, -0.5, 0.1],
                [0.0, 0.3, -0.5],
            ],
            dtype=torch.float64,
        )
        # Verify A is stable
        eigenvalues = torch.linalg.eigvals(A)
        assert eigenvalues.real.max().item() < 0, "A should be stable for this test"

        C = torch.tensor(
            [[0.25], [0.0], [0.0]],
            dtype=torch.float64,
        )
        stim = make_block_stimulus(n_blocks=10, block_duration=30, rest_duration=20)

        result = simulate_task_dcm(
            A, C, stim, duration=500.0, SNR=0, seed=42,
        )

        neural = result["neural"]

        # Neural states should remain bounded (not diverge)
        assert torch.isfinite(neural).all(), "Neural states should be finite"
        max_neural = neural.abs().max().item()
        assert max_neural < 10.0, (
            f"Neural states too large ({max_neural:.2f}), suggests instability"
        )

        # During rest periods, neural activity should decay toward zero
        # Check last 10s of final rest (if present)
        t_fine = result["times_fine"]
        last_rest_mask = t_fine > 490.0  # last 10s
        neural_last = neural[last_rest_mask]
        assert neural_last.abs().max().item() < 1.0, (
            "Neural states should decay during rest periods"
        )

    def test_driven_region_responds(self) -> None:
        """Driven region (region 0) should have larger BOLD than undriven regions.

        Region 0 receives direct input (C[0,0]=0.25), regions 1-2 only
        get input through A connections.
        """
        A = torch.tensor(
            [
                [-0.5, 0.0, 0.0],
                [0.1, -0.5, 0.0],
                [0.0, 0.1, -0.5],
            ],
            dtype=torch.float64,
        )
        C = torch.tensor(
            [[0.25], [0.0], [0.0]],
            dtype=torch.float64,
        )
        stim = make_block_stimulus(n_blocks=5, block_duration=30, rest_duration=20)

        result = simulate_task_dcm(
            A, C, stim, duration=250.0, SNR=0, seed=42,
        )

        # Region 0 (driven) should have largest BOLD variance
        bold_std = result["bold_clean"].std(dim=0)

        assert bold_std[0] > bold_std[1], (
            f"Driven region 0 BOLD std ({bold_std[0]:.6f}) should exceed "
            f"region 1 ({bold_std[1]:.6f})"
        )
        assert bold_std[0] > bold_std[2], (
            f"Driven region 0 BOLD std ({bold_std[0]:.6f}) should exceed "
            f"region 2 ({bold_std[2]:.6f})"
        )


# ---------------------------------------------------------------------------
# PiecewiseConstantInput passthrough test
# ---------------------------------------------------------------------------


class TestStimulusPassthrough:
    """Tests that simulator accepts PiecewiseConstantInput directly."""

    def test_piecewise_input_passthrough(
        self, simple_A_3: torch.Tensor, simple_C_3: torch.Tensor,
    ) -> None:
        """Simulator should accept PiecewiseConstantInput directly."""
        stim_dict = make_block_stimulus(n_blocks=2, block_duration=20, rest_duration=10)
        input_fn = PiecewiseConstantInput(stim_dict["times"], stim_dict["values"])

        result = simulate_task_dcm(
            simple_A_3, simple_C_3, input_fn,
            duration=60.0, SNR=5.0, seed=42,
        )
        assert result["bold"].shape[1] == 3
        assert not torch.isnan(result["bold"]).any()
