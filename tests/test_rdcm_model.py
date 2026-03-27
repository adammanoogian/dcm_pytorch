"""Unit tests for the rDCM Pyro generative model.

Tests cover:
- Model trace structure (per-region sample sites, shapes)
- Per-region theta sizes match active connections from masks
- Different masks per region produce different theta sizes
- Numerical stability (finite log_prob)
- SVI smoke tests (no NaN, loss decreases)
"""

from __future__ import annotations

import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

from pyro_dcm.models.rdcm_model import rdcm_model
from pyro_dcm.simulators.rdcm_simulator import (
    make_stable_A_rdcm,
    make_block_stimulus_rdcm,
    simulate_rdcm,
)
from pyro_dcm.forward_models.rdcm_forward import (
    create_regressors,
    generate_bold,
    get_hrf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rdcm_data() -> dict:
    """Generate rDCM test data for 3 regions, 1 input.

    Uses ``simulate_rdcm`` pipeline to generate synthetic BOLD,
    then ``create_regressors`` for frequency-domain X, Y.
    """
    nr, nu = 3, 1
    A, a_mask = make_stable_A_rdcm(nr, density=0.5, seed=42)
    C = torch.tensor([[0.5], [0.0], [0.0]], dtype=torch.float64)
    c_mask = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)

    u_dt = 0.5
    y_dt = 2.0
    n_time = 400  # 200 seconds at u_dt=0.5
    u = make_block_stimulus_rdcm(n_time, nu, u_dt, seed=42)

    # Generate BOLD
    bold_result = generate_bold(A, C, u, u_dt, y_dt, SNR=3.0)

    # Create regressors
    hrf = get_hrf(n_time, u_dt)
    X, Y, N_eff = create_regressors(
        hrf, bold_result["y"], u, u_dt, y_dt,
    )

    return {
        "Y": Y,
        "X": X,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "nr": nr,
        "nu": nu,
        "N_eff": N_eff,
        "confound_cols": 1,
    }


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


class TestModelStructure:
    """Tests for model trace sites and shapes."""

    def _run_trace(
        self,
        rdcm_data: dict,
        a_mask_override: torch.Tensor | None = None,
        c_mask_override: torch.Tensor | None = None,
    ) -> pyro.poutine.trace_struct.Trace:
        """Run model under trace poutine and return the trace."""
        a_mask = (
            a_mask_override if a_mask_override is not None
            else rdcm_data["a_mask"]
        )
        c_mask = (
            c_mask_override if c_mask_override is not None
            else rdcm_data["c_mask"]
        )
        trace = pyro.poutine.trace(rdcm_model).get_trace(
            Y=rdcm_data["Y"],
            X=rdcm_data["X"],
            a_mask=a_mask,
            c_mask=c_mask,
            confound_cols=rdcm_data["confound_cols"],
        )
        return trace

    def test_model_trace_has_region_sites(
        self, rdcm_data: dict,
    ) -> None:
        """Trace must contain per-region theta, noise_prec, and obs sites."""
        trace = self._run_trace(rdcm_data)
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}
        nr = rdcm_data["nr"]

        for r in range(nr):
            assert f"theta_{r}" in site_names, (
                f"Missing sample site: theta_{r}"
            )
            assert f"noise_prec_{r}" in site_names, (
                f"Missing sample site: noise_prec_{r}"
            )
            assert f"obs_{r}" in site_names, (
                f"Missing sample site: obs_{r}"
            )

    def test_model_theta_shapes_match_active_columns(
        self, rdcm_data: dict,
    ) -> None:
        """Each theta_r shape must equal active connections + confounds."""
        trace = self._run_trace(rdcm_data)
        nr = rdcm_data["nr"]
        nu = rdcm_data["nu"]
        nc = rdcm_data["confound_cols"]
        a_mask = rdcm_data["a_mask"]
        c_mask = rdcm_data["c_mask"]

        for r in range(nr):
            # Expected D_r from masks
            n_a_active = int(a_mask[r, :].sum().item())
            n_c_active = int(c_mask[r, :].sum().item())
            expected_D_r = n_a_active + n_c_active + nc

            theta_r = trace.nodes[f"theta_{r}"]["value"]
            assert theta_r.shape == (expected_D_r,), (
                f"theta_{r} shape {theta_r.shape}, "
                f"expected ({expected_D_r},)"
            )

    def test_model_handles_different_masks_per_region(
        self, rdcm_data: dict,
    ) -> None:
        """Regions with different active connections get different theta sizes."""
        nr = rdcm_data["nr"]

        # Create a_mask where region 0 has 2 active A and region 1 has 3
        a_mask = torch.tensor(
            [
                [1, 1, 0],  # region 0: 2 active A
                [1, 1, 1],  # region 1: 3 active A
                [0, 1, 1],  # region 2: 2 active A
            ],
            dtype=torch.float64,
        )
        # c_mask: only region 0 has input
        c_mask = torch.tensor(
            [[1.0], [0.0], [0.0]],
            dtype=torch.float64,
        )

        trace = self._run_trace(rdcm_data, a_mask_override=a_mask,
                                c_mask_override=c_mask)

        nc = rdcm_data["confound_cols"]
        theta_0 = trace.nodes["theta_0"]["value"]
        theta_1 = trace.nodes["theta_1"]["value"]
        theta_2 = trace.nodes["theta_2"]["value"]

        # Region 0: 2 A + 1 C + 1 confound = 4
        assert theta_0.shape == (4,), (
            f"theta_0 shape {theta_0.shape}, expected (4,)"
        )
        # Region 1: 3 A + 0 C + 1 confound = 4
        assert theta_1.shape == (4,), (
            f"theta_1 shape {theta_1.shape}, expected (4,)"
        )
        # Region 2: 2 A + 0 C + 1 confound = 3
        assert theta_2.shape == (3,), (
            f"theta_2 shape {theta_2.shape}, expected (3,)"
        )

        # Key: theta_0 and theta_2 have different sizes
        assert theta_0.shape != theta_2.shape


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for finite log_prob values."""

    def test_model_log_prob_finite(self, rdcm_data: dict) -> None:
        """Total log probability from trace must be finite."""
        trace = pyro.poutine.trace(rdcm_model).get_trace(
            Y=rdcm_data["Y"],
            X=rdcm_data["X"],
            a_mask=rdcm_data["a_mask"],
            c_mask=rdcm_data["c_mask"],
            confound_cols=rdcm_data["confound_cols"],
        )
        lp = trace.log_prob_sum()
        assert torch.isfinite(lp), f"log_prob_sum is {lp}, expected finite"


# ---------------------------------------------------------------------------
# SVI smoke tests
# ---------------------------------------------------------------------------


class TestSVI:
    """Smoke tests for SVI training with AutoNormal guide."""

    def test_svi_runs_without_nan(self, rdcm_data: dict) -> None:
        """10 SVI steps with AutoNormal must produce no NaN losses."""
        pyro.clear_param_store()

        guide = AutoNormal(rdcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            rdcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        model_args = (
            rdcm_data["Y"],
            rdcm_data["X"],
            rdcm_data["a_mask"],
            rdcm_data["c_mask"],
            rdcm_data["confound_cols"],
        )

        losses = []
        for _ in range(10):
            loss = svi.step(*model_args)
            losses.append(loss)

        for i, loss in enumerate(losses):
            import math
            assert not math.isnan(loss), f"NaN loss at step {i}"

    def test_svi_loss_decreases(self, rdcm_data: dict) -> None:
        """ELBO should decrease over 50 SVI steps (mean comparison)."""
        pyro.clear_param_store()

        guide = AutoNormal(rdcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            rdcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        model_args = (
            rdcm_data["Y"],
            rdcm_data["X"],
            rdcm_data["a_mask"],
            rdcm_data["c_mask"],
            rdcm_data["confound_cols"],
        )

        losses = []
        for _ in range(50):
            loss = svi.step(*model_args)
            losses.append(loss)

        # Mean of first 10 should be greater than mean of last 10
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Loss did not decrease: first 10 mean={first_10:.4f}, "
            f"last 10 mean={last_10:.4f}"
        )
