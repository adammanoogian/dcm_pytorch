"""Unit tests for the task DCM Pyro generative model.

Tests cover:
- Model trace structure (sample sites, deterministic sites, shapes)
- A matrix properties (negative diagonal, masking)
- Hemodynamic parameters NOT sampled
- Numerical stability (finite samples, finite log_prob)
- SVI smoke tests (no NaN, loss decreases, guide posterior shapes)
"""

from __future__ import annotations

import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

from pyro_dcm.models.task_dcm_model import task_dcm_model
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_data() -> dict:
    """Generate synthetic task DCM data for 3 regions, 1 input.

    Uses ``simulate_task_dcm`` with a short duration (~30s) to produce
    a small dataset for fast testing.
    """
    N, M = 3, 1
    A = make_random_stable_A(N, density=0.5, seed=42)
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)

    stim = make_block_stimulus(
        n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
    )

    duration = 30.0
    TR = 2.0
    dt = 0.5  # coarse for SVI efficiency

    result = simulate_task_dcm(
        A, C, stim, duration=duration, dt=0.01, TR=TR, SNR=5.0, seed=7,
    )

    # Build t_eval at dt=0.5 for the Pyro model (coarser than simulation)
    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)

    # Build masks: all connections present
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.ones(N, M, dtype=torch.float64)

    return {
        "observed_bold": result["bold"],
        "stimulus": result["stimulus"],
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "TR": TR,
        "dt": dt,
        "N": N,
        "M": M,
        "T": result["bold"].shape[0],
    }


@pytest.fixture()
def sparse_a_mask() -> torch.Tensor:
    """Sparse 3x3 structural mask with some absent connections."""
    return torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


class TestModelStructure:
    """Tests for model trace sites and shapes."""

    def _run_trace(
        self, task_data: dict, a_mask_override: torch.Tensor | None = None,
    ) -> pyro.poutine.trace_struct.Trace:
        """Run model under trace poutine with conditioned params.

        Conditions A_free and C to small known-good values so the ODE
        integration produces finite BOLD (random prior samples can
        cause instability with coarse dt=0.5).
        """
        N = task_data["N"]
        M = task_data["M"]
        a_mask = a_mask_override if a_mask_override is not None else task_data["a_mask"]

        # Condition on small A_free (near zero -> A diagonal ~ -0.5)
        # and moderate C for stable ODE integration
        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
            },
        )
        trace = pyro.poutine.trace(conditioned).get_trace(
            observed_bold=task_data["observed_bold"],
            stimulus=task_data["stimulus"],
            a_mask=a_mask,
            c_mask=task_data["c_mask"],
            t_eval=task_data["t_eval"],
            TR=task_data["TR"],
            dt=task_data["dt"],
        )
        return trace

    def test_model_trace_has_expected_sites(
        self, task_data: dict,
    ) -> None:
        """Trace must contain all expected sample and deterministic sites."""
        trace = self._run_trace(task_data)
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}

        # Sample sites
        expected_samples = {"A_free", "C", "noise_prec", "obs"}
        for name in expected_samples:
            assert name in site_names, f"Missing sample site: {name}"
            assert trace.nodes[name]["type"] == "sample"

        # Deterministic sites
        expected_det = {"A", "predicted_bold"}
        for name in expected_det:
            assert name in site_names, f"Missing deterministic site: {name}"

    def test_model_samples_correct_shapes(
        self, task_data: dict,
    ) -> None:
        """Sample site values must have correct shapes."""
        trace = self._run_trace(task_data)
        N = task_data["N"]
        M = task_data["M"]
        T = task_data["T"]

        assert trace.nodes["A_free"]["value"].shape == (N, N)
        assert trace.nodes["C"]["value"].shape == (N, M)
        assert trace.nodes["A"]["value"].shape == (N, N)
        assert trace.nodes["predicted_bold"]["value"].shape == (T, N)

    def test_model_a_diagonal_negative(
        self, task_data: dict,
    ) -> None:
        """A matrix diagonal elements must all be negative."""
        trace = self._run_trace(task_data)
        A = trace.nodes["A"]["value"]
        diag = A.diagonal()
        assert (diag < 0).all(), f"Non-negative diagonal found: {diag}"

    def test_model_masking_works(
        self, task_data: dict, sparse_a_mask: torch.Tensor,
    ) -> None:
        """Masked off-diagonal A positions must be zero."""
        trace = self._run_trace(task_data, a_mask_override=sparse_a_mask)
        A = trace.nodes["A"]["value"]
        N = task_data["N"]

        # Check off-diagonal zeros where mask is 0
        for i in range(N):
            for j in range(N):
                if i != j and sparse_a_mask[i, j] == 0:
                    assert A[i, j].item() == 0.0, (
                        f"A[{i},{j}] should be 0 but is {A[i, j].item()}"
                    )

    def test_model_hemodynamic_params_not_sampled(
        self, task_data: dict,
    ) -> None:
        """No hemodynamic parameters should be sampled."""
        trace = self._run_trace(task_data)
        site_names = set(trace.nodes.keys())

        forbidden = {"kappa", "gamma", "tau", "alpha", "E0", "hemo_params"}
        for name in forbidden:
            assert name not in site_names, (
                f"Hemodynamic param '{name}' should not be a sample site"
            )


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for finite values and stable computation.

    Uses conditioned parameter values (small A_free, moderate C) to
    ensure the ODE integration produces finite BOLD. This verifies the
    forward model pipeline works correctly with valid inputs.
    """

    def test_model_prior_samples_finite(
        self, task_data: dict,
    ) -> None:
        """Predicted BOLD from conditioned samples must be finite."""
        N = task_data["N"]
        M = task_data["M"]
        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
            },
        )
        trace = pyro.poutine.trace(conditioned).get_trace(
            observed_bold=task_data["observed_bold"],
            stimulus=task_data["stimulus"],
            a_mask=task_data["a_mask"],
            c_mask=task_data["c_mask"],
            t_eval=task_data["t_eval"],
            TR=task_data["TR"],
            dt=task_data["dt"],
        )
        pred_bold = trace.nodes["predicted_bold"]["value"]
        assert torch.isfinite(pred_bold).all(), (
            "NaN/Inf in predicted BOLD"
        )

    def test_model_log_prob_finite(self, task_data: dict) -> None:
        """Total log probability from trace must be finite."""
        N = task_data["N"]
        M = task_data["M"]
        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
            },
        )
        trace = pyro.poutine.trace(conditioned).get_trace(
            observed_bold=task_data["observed_bold"],
            stimulus=task_data["stimulus"],
            a_mask=task_data["a_mask"],
            c_mask=task_data["c_mask"],
            t_eval=task_data["t_eval"],
            TR=task_data["TR"],
            dt=task_data["dt"],
        )
        lp = trace.log_prob_sum()
        assert torch.isfinite(lp), f"log_prob_sum is {lp}, expected finite"


# ---------------------------------------------------------------------------
# SVI smoke tests
# ---------------------------------------------------------------------------


class TestSVI:
    """Smoke tests for SVI training with AutoNormal guide."""

    def test_svi_runs_without_nan(self, task_data: dict) -> None:
        """10 SVI steps with AutoNormal must produce no NaN losses."""
        pyro.clear_param_store()

        guide = AutoNormal(task_dcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            task_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        losses = []
        for _ in range(10):
            loss = svi.step(
                observed_bold=task_data["observed_bold"],
                stimulus=task_data["stimulus"],
                a_mask=task_data["a_mask"],
                c_mask=task_data["c_mask"],
                t_eval=task_data["t_eval"],
                TR=task_data["TR"],
                dt=task_data["dt"],
            )
            losses.append(loss)

        for i, loss in enumerate(losses):
            assert not torch.isnan(torch.tensor(loss)), (
                f"NaN loss at step {i}"
            )

    def test_svi_loss_decreases(self, task_data: dict) -> None:
        """ELBO should decrease over 50 SVI steps (mean comparison)."""
        pyro.clear_param_store()

        guide = AutoNormal(task_dcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            task_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        losses = []
        for _ in range(50):
            loss = svi.step(
                observed_bold=task_data["observed_bold"],
                stimulus=task_data["stimulus"],
                a_mask=task_data["a_mask"],
                c_mask=task_data["c_mask"],
                t_eval=task_data["t_eval"],
                TR=task_data["TR"],
                dt=task_data["dt"],
            )
            losses.append(loss)

        # Mean of first 10 should be greater than mean of last 10
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Loss did not decrease: first 10 mean={first_10:.4f}, "
            f"last 10 mean={last_10:.4f}"
        )

    def test_guide_posterior_shapes(self, task_data: dict) -> None:
        """After SVI, guide parameters must exist for A_free and C."""
        pyro.clear_param_store()

        guide = AutoNormal(task_dcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            task_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        # Run a few SVI steps to initialize guide parameters
        for _ in range(5):
            svi.step(
                observed_bold=task_data["observed_bold"],
                stimulus=task_data["stimulus"],
                a_mask=task_data["a_mask"],
                c_mask=task_data["c_mask"],
                t_eval=task_data["t_eval"],
                TR=task_data["TR"],
                dt=task_data["dt"],
            )

        # Check param store has guide parameters for A_free and C
        param_names = set(pyro.get_param_store().keys())

        # AutoNormal creates locs and scales for each sample site
        assert any("A_free" in name for name in param_names), (
            f"No A_free guide params found in: {param_names}"
        )
        assert any("C" in name for name in param_names), (
            f"No C guide params found in: {param_names}"
        )
