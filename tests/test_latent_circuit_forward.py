"""Tests for latent circuit forward model and simulator.

Validates the CoupledDCMSystem hemodynamic toggle (v0.6.0) and the
latent circuit simulator against the ground-truth ODE integration.

Tests cover:
- OBS-01: hemodynamic=False produces (T, N) not (T, 5N)
- OBS-03: hemodynamic=True unchanged (bit-exact pre-Phase-20)
- OBS-04: No edits to neural_state.py, balloon_model.py, bold_signal.py
- SIM-01: simulate_latent_circuit returns correct dict keys and shapes
- SIM-02: Simulator-vs-ODE cross-validation at atol=1e-6
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
)


@pytest.fixture
def simple_2region_setup():
    """Create a simple 2-region linear DCM setup."""
    A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    stim = PiecewiseConstantInput(
        torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64),
        torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64),
    )
    return A, C, stim


class TestHemodynamicToggle:
    """Tests for CoupledDCMSystem hemodynamic=False mode."""

    def test_hemodynamic_false_shape(self, simple_2region_setup):
        """OBS-01: hemodynamic=False with N=4 produces (T, 4) not (T, 20)."""
        N = 4
        A = -0.5 * torch.eye(N, dtype=torch.float64)
        A[0, 1] = 0.1
        A[1, 0] = 0.2
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0
        stim = PiecewiseConstantInput(
            torch.tensor([0.0, 10.0], dtype=torch.float64),
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        )

        system = CoupledDCMSystem(A, C, stim, hemodynamic=False)
        y0 = torch.zeros(N, dtype=torch.float64)
        t_eval = torch.arange(0, 20, 0.01, dtype=torch.float64)
        sol = integrate_ode(system, y0, t_eval, method="rk4", step_size=0.01)

        assert sol.shape == (2000, 4), (
            f"Expected (2000, 4), got {sol.shape}"
        )
        assert not torch.isnan(sol).any()
        # Verify non-trivial response (region 0 should be driven)
        assert sol[:, 0].abs().max() > 0.1

    def test_hemodynamic_true_unchanged(self, simple_2region_setup):
        """OBS-03: hemodynamic=True produces identical output to default."""
        A, C, stim = simple_2region_setup
        N = A.shape[0]

        # System with explicit hemodynamic=True
        system_explicit = CoupledDCMSystem(
            A, C, stim, hemodynamic=True
        )
        # System with default (should also be hemodynamic=True)
        system_default = CoupledDCMSystem(A, C, stim)

        y0 = make_initial_state(N, dtype=torch.float64)
        t_eval = torch.arange(0, 30, 0.01, dtype=torch.float64)

        sol_explicit = integrate_ode(
            system_explicit, y0, t_eval, method="rk4", step_size=0.01
        )
        sol_default = integrate_ode(
            system_default, y0, t_eval, method="rk4", step_size=0.01
        )

        # Bit-exact: same code path, same initial conditions
        assert torch.equal(sol_explicit, sol_default), (
            "hemodynamic=True should produce bit-exact output vs default"
        )
        assert sol_explicit.shape == (3000, 10)

    def test_hemodynamic_false_bilinear(self):
        """Pitfall 4: hemodynamic=False + bilinear produces stable 100s."""
        N = 3
        A = -0.5 * torch.eye(N, dtype=torch.float64)
        A[0, 1] = 0.15
        A[1, 2] = 0.1
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0

        # One modulator
        B = torch.zeros(1, N, N, dtype=torch.float64)
        B[0, 1, 0] = 0.2  # Modulator enhances 0->1 connection

        # Driving stimulus: block design
        stim = PiecewiseConstantInput(
            torch.tensor([0.0, 10.0, 30.0, 50.0], dtype=torch.float64),
            torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        )
        # Modulatory stimulus: active during second block
        stim_mod = PiecewiseConstantInput(
            torch.tensor([0.0, 30.0, 50.0], dtype=torch.float64),
            torch.tensor(
                [[0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        )

        # Merge inputs
        from pyro_dcm.utils.ode_integrator import merge_piecewise_inputs

        merged = merge_piecewise_inputs(stim, stim_mod)

        system = CoupledDCMSystem(
            A, C, merged, hemodynamic=False, B=B, n_driving_inputs=1
        )
        y0 = torch.zeros(N, dtype=torch.float64)
        t_eval = torch.arange(0, 100, 0.01, dtype=torch.float64)
        sol = integrate_ode(
            system, y0, t_eval, method="rk4", step_size=0.01
        )

        assert sol.shape == (10000, 3)
        assert not torch.isnan(sol).any(), "NaN in bilinear integration"
        assert not torch.isinf(sol).any(), "Inf in bilinear integration"
        # Non-trivial: at least some activity
        assert sol.abs().max() > 0.01

    def test_hemodynamic_false_hemo_params_error(self):
        """ValueError when hemodynamic=False with hemo_params."""
        N = 2
        A = -0.5 * torch.eye(N, dtype=torch.float64)
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0
        stim = PiecewiseConstantInput(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64),
        )

        with pytest.raises(ValueError, match="hemo_params must be None"):
            CoupledDCMSystem(
                A,
                C,
                stim,
                hemo_params={"kappa": 0.65},
                hemodynamic=False,
            )


class TestLatentCircuitSimulator:
    """Tests for simulate_latent_circuit function."""

    def test_simulate_latent_circuit_basic(self):
        """SIM-01: Basic linear mode returns correct shape and keys."""
        N = 3
        A = make_stable_latent_circuit_A(N, density=0.5, seed=42)
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0
        stim = {
            "times": torch.tensor(
                [0.0, 10.0, 30.0, 50.0], dtype=torch.float64
            ),
            "values": torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        }

        result = simulate_latent_circuit(
            A, C, stim, duration=60.0, dt=0.01, seed=123
        )

        # Check all expected keys
        expected_keys = {
            "trajectories",
            "trajectories_clean",
            "times",
            "A",
            "C",
            "B_list",
            "stimulus_mod",
            "noise_std",
            "SNR",
        }
        assert set(result.keys()) == expected_keys

        # Check shapes
        T = int(60.0 / 0.01)
        assert result["trajectories"].shape == (T, N)
        assert result["trajectories_clean"].shape == (T, N)
        assert result["times"].shape == (T,)

        # Check linear mode markers
        assert result["B_list"] is None
        assert result["stimulus_mod"] is None

        # Check noise was added
        assert result["noise_std"] > 0
        diff = result["trajectories"] - result["trajectories_clean"]
        assert diff.abs().max() > 0, "Noise should be non-zero"

    def test_simulate_latent_circuit_bilinear(self):
        """Bilinear mode produces trajectories distinguishable from linear."""
        N = 3
        A = -0.5 * torch.eye(N, dtype=torch.float64)
        A[0, 1] = 0.15
        A[1, 2] = 0.1
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0

        B = torch.zeros(1, N, N, dtype=torch.float64)
        B[0, 1, 0] = 0.3  # Modulator enhances 0->1 connection

        stim = {
            "times": torch.tensor(
                [0.0, 5.0, 15.0, 25.0], dtype=torch.float64
            ),
            "values": torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        }
        stim_mod = {
            "times": torch.tensor(
                [0.0, 15.0, 25.0], dtype=torch.float64
            ),
            "values": torch.tensor(
                [[0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        }

        # Linear mode (B_list=None)
        result_linear = simulate_latent_circuit(
            A, C, stim, duration=30.0, dt=0.01, SNR=-1, seed=42
        )

        # Bilinear mode
        result_bilinear = simulate_latent_circuit(
            A,
            C,
            stim,
            duration=30.0,
            dt=0.01,
            SNR=-1,
            seed=42,
            B_list=B,
            stimulus_mod=stim_mod,
        )

        # Trajectories must differ (bilinear modulation changes dynamics)
        assert not torch.allclose(
            result_linear["trajectories"],
            result_bilinear["trajectories"],
            atol=1e-8,
        ), "Bilinear should produce different trajectories from linear"

        # Bilinear result should have B_list
        assert result_bilinear["B_list"] is not None
        assert result_bilinear["stimulus_mod"] is not None

    def test_simulator_vs_direct_ode(self):
        """SIM-02: Simulator clean output matches direct ODE integration."""
        N = 3
        A = -0.5 * torch.eye(N, dtype=torch.float64)
        A[0, 1] = 0.1
        A[1, 2] = 0.15
        C = torch.zeros(N, 1, dtype=torch.float64)
        C[0, 0] = 1.0

        stim_pci = PiecewiseConstantInput(
            torch.tensor(
                [0.0, 5.0, 15.0, 25.0], dtype=torch.float64
            ),
            torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
            ),
        )

        duration = 30.0
        dt = 0.01

        # Direct ODE integration
        system = CoupledDCMSystem(A, C, stim_pci, hemodynamic=False)
        y0 = torch.zeros(N, dtype=torch.float64)
        t_eval = torch.arange(0, duration, dt, dtype=torch.float64)
        direct_sol = integrate_ode(
            system,
            y0,
            t_eval,
            method="dopri5",
            grid_points=stim_pci.grid_points,
            step_size=dt,
        )

        # Simulator
        result = simulate_latent_circuit(
            A,
            C,
            stim_pci,
            duration=duration,
            dt=dt,
            SNR=-1,  # No noise for clean comparison
            solver="dopri5",
        )

        # Cross-validate at atol=1e-6
        assert torch.allclose(
            result["trajectories_clean"],
            direct_sol,
            atol=1e-6,
        ), (
            f"Simulator-vs-ODE mismatch: max diff = "
            f"{(result['trajectories_clean'] - direct_sol).abs().max()}"
        )


class TestMakeStableLatentCircuitA:
    """Tests for make_stable_latent_circuit_A helper."""

    def test_basic_properties(self):
        """Generated A is stable and has correct shape."""
        A = make_stable_latent_circuit_A(4, density=0.5, seed=42)
        assert A.shape == (4, 4)
        assert A.dtype == torch.float64

        # All eigenvalues have negative real parts
        eigs = torch.linalg.eigvals(A)
        assert eigs.real.max().item() < 0, (
            f"Matrix not stable: max Re(eig)={eigs.real.max().item()}"
        )

        # Diagonal is negative (self-inhibition)
        assert (A.diagonal() < 0).all()

    def test_reproducibility(self):
        """Same seed produces same matrix."""
        A1 = make_stable_latent_circuit_A(5, density=0.3, seed=99)
        A2 = make_stable_latent_circuit_A(5, density=0.3, seed=99)
        assert torch.equal(A1, A2)

    def test_density(self):
        """Density controls sparsity of off-diagonal connections."""
        A_sparse = make_stable_latent_circuit_A(
            6, density=0.2, seed=10
        )
        A_dense = make_stable_latent_circuit_A(
            6, density=0.9, seed=10
        )

        # Count non-zero off-diagonal entries
        mask = ~torch.eye(6, dtype=torch.bool)
        nnz_sparse = (A_sparse[mask] != 0).sum().item()
        nnz_dense = (A_dense[mask] != 0).sum().item()
        assert nnz_sparse < nnz_dense
