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
def task_bilinear_data(task_data: dict) -> dict:
    """Extend ``task_data`` with bilinear b_masks and stim_mod for J=1 modulator.

    Mirrors ``task_data`` fixture parameters (3 regions, 1 driving input,
    dt=0.5, 30s duration, TR=2.0) and adds:

    - ``b_masks``: list with one ``(3, 3)`` mask (off-diagonal pattern,
      zero diagonal per Pitfall B5 recommendation).
    - ``stim_mod``: :class:`PiecewiseConstantInput` from
      :func:`make_epoch_stimulus` (single 10s epoch at t=10s,
      amplitude 1.0).

    The returned dict is a superset of ``task_data`` so it can be passed
    to :func:`task_dcm_model` via standard keyword unpacking.
    """
    from pyro_dcm.simulators.task_simulator import make_epoch_stimulus
    from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

    N = task_data["N"]
    # Single-modulator mask: off-diagonal 1 -> 2 connection modulated.
    b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0  # modulator gates the 1 <- 0 connection
    b_masks = [b_mask_0]

    # Single 10s epoch at t=10s, amplitude 1.0, over 30s total.
    stim_mod_dict = make_epoch_stimulus(
        event_times=[10.0],
        event_durations=[10.0],
        event_amplitudes=[1.0],
        duration=30.0,
        dt=0.01,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )

    return {
        **task_data,
        "b_masks": b_masks,
        "stim_mod": stim_mod,
        "J": 1,
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


# ---------------------------------------------------------------------------
# Bilinear structure tests (Phase 15 Plan 15-01)
# ---------------------------------------------------------------------------


class TestBilinearStructure:
    """Tests for the v0.3.0 bilinear branch of ``task_dcm_model`` (Phase 15).

    Covers MODEL-01 (per-modulator ``B_free_j`` loop), MODEL-02
    (``B_PRIOR_VARIANCE`` constant), MODEL-03 BOTH halves:

    - Phase 13 half (source-side, ``(N, N)`` call): see
      ``tests/test_bilinear_utils.py``.
    - Phase 15 half (stacked ``(J, N, N)`` call at the ``task_dcm_model``
      call-site):
      :meth:`test_bilinear_deprecation_warning_on_stacked_nonzero_diag` --
      closes the SC-4 coverage gap flagged by the Phase 15-01 checker.

    Also covers MODEL-04 edge cases (``b_masks=None``, ``b_masks=[]``,
    shape-mismatch errors).
    """

    def _run_bilinear_trace(
        self, task_bilinear_data: dict,
    ) -> pyro.poutine.trace_struct.Trace:
        """Run ``task_dcm_model`` in bilinear mode under ``condition + trace``.

        Conditions on small ``A_free``/``C``/``noise_prec`` and small
        ``B_free_0`` for finite BOLD (same pattern as
        :meth:`TestModelStructure._run_trace`).
        """
        N = task_bilinear_data["N"]
        M = task_bilinear_data["M"]
        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
                "B_free_0": 0.05 * torch.randn(N, N, dtype=torch.float64),
            },
        )
        trace = pyro.poutine.trace(conditioned).get_trace(
            observed_bold=task_bilinear_data["observed_bold"],
            stimulus=task_bilinear_data["stimulus"],
            a_mask=task_bilinear_data["a_mask"],
            c_mask=task_bilinear_data["c_mask"],
            t_eval=task_bilinear_data["t_eval"],
            TR=task_bilinear_data["TR"],
            dt=task_bilinear_data["dt"],
            b_masks=task_bilinear_data["b_masks"],
            stim_mod=task_bilinear_data["stim_mod"],
        )
        return trace

    def test_B_PRIOR_VARIANCE_constant(self) -> None:
        """``B_PRIOR_VARIANCE`` must be exactly 1.0 (D1, MODEL-02)."""
        from pyro_dcm.models.task_dcm_model import B_PRIOR_VARIANCE
        assert B_PRIOR_VARIANCE == 1.0, (
            f"B_PRIOR_VARIANCE must be 1.0 per D1 (SPM12 one-state match); "
            f"got {B_PRIOR_VARIANCE}. If you intentionally changed this, "
            f"update REQUIREMENTS.md MODEL-02 + STATE.md D1 first."
        )

    def test_linear_reduction_when_b_masks_none(
        self, task_data: dict,
    ) -> None:
        """``b_masks=None`` -> trace MUST match the pre-Phase-15 linear set.

        MODEL-04 acceptance: the set of sample + deterministic sites in
        the trace (excluding ``_INPUT``/``_RETURN``) must NOT contain
        ``'B'`` or any ``'B_free_*'`` site when ``b_masks`` is ``None``.
        """
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
            b_masks=None,  # explicit None
            stim_mod=None,
        )
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}
        expected = {"A_free", "C", "noise_prec", "obs", "A", "predicted_bold"}
        assert site_names == expected, (
            f"Linear-mode site set must equal {expected}; got {site_names}. "
            f"Extra sites: {site_names - expected}; "
            f"missing: {expected - site_names}."
        )
        # L3: no 'B' deterministic site in linear mode.
        assert "B" not in site_names
        assert not any(n.startswith("B_free_") for n in site_names)

    def test_linear_reduction_when_b_masks_empty_list(
        self, task_data: dict,
    ) -> None:
        """``b_masks=[]`` must normalize to None and take the linear path."""
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
            b_masks=[],  # empty list
            stim_mod=None,
        )
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}
        assert "B" not in site_names
        assert not any(n.startswith("B_free_") for n in site_names)

    def test_bilinear_trace_has_B_free_sites(
        self, task_bilinear_data: dict,
    ) -> None:
        """Bilinear J=1 trace must have ``B_free_0`` (not ``B_free``) and ``B``."""
        trace = self._run_bilinear_trace(task_bilinear_data)
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}
        N = task_bilinear_data["N"]

        assert "B_free_0" in site_names, (
            f"Missing bilinear sample site B_free_0; got {sorted(site_names)}"
        )
        # R1 (research note): bare 'B_free' must NOT exist (guard
        # against silent collision with per-modulator-indexed sites).
        assert "B_free" not in site_names

        # Shape check: sample value is (N, N) per L1.
        assert trace.nodes["B_free_0"]["value"].shape == (N, N)

        # L3: deterministic 'B' site exists ONLY in bilinear mode,
        # shape (J, N, N).
        assert "B" in site_names
        assert trace.nodes["B"]["value"].shape == (
            task_bilinear_data["J"], N, N,
        )

    def test_bilinear_masking_applied(
        self, task_bilinear_data: dict,
    ) -> None:
        """``B`` deterministic site must have ``b_mask`` applied (zeros preserved)."""
        trace = self._run_bilinear_trace(task_bilinear_data)
        B_det = trace.nodes["B"]["value"]  # shape (J, N, N)
        b_mask_0 = task_bilinear_data["b_masks"][0]
        # Zero-diagonal + off-diagonal pattern enforced: every index
        # where b_mask == 0, B_det must be exactly zero.
        for i in range(task_bilinear_data["N"]):
            for k in range(task_bilinear_data["N"]):
                if b_mask_0[i, k].item() == 0.0:
                    assert B_det[0, i, k].item() == 0.0, (
                        f"B[0,{i},{k}] must be 0 (b_mask=0) but is "
                        f"{B_det[0, i, k].item()}"
                    )

    def test_bilinear_stim_mod_required_error(
        self, task_bilinear_data: dict,
    ) -> None:
        """``b_masks`` non-empty + ``stim_mod=None`` -> ValueError."""
        with pytest.raises(ValueError, match="stim_mod is required"):
            # No pyro.sample conditioning needed -- validation happens
            # before sampling. Use trace to exercise the call path.
            pyro.poutine.trace(task_dcm_model).get_trace(
                observed_bold=task_bilinear_data["observed_bold"],
                stimulus=task_bilinear_data["stimulus"],
                a_mask=task_bilinear_data["a_mask"],
                c_mask=task_bilinear_data["c_mask"],
                t_eval=task_bilinear_data["t_eval"],
                TR=task_bilinear_data["TR"],
                dt=task_bilinear_data["dt"],
                b_masks=task_bilinear_data["b_masks"],
                stim_mod=None,  # the trigger
            )

    def test_bilinear_stim_mod_shape_mismatch_error(
        self, task_bilinear_data: dict,
    ) -> None:
        """``len(b_masks) != stim_mod.values.shape[1]`` -> ValueError."""
        from pyro_dcm.simulators.task_simulator import make_epoch_stimulus
        from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput
        # Build a 2-column mod but pass a 1-element b_masks (mismatch).
        stim_dict = make_epoch_stimulus(
            event_times=[5.0, 15.0],
            event_durations=[5.0, 5.0],
            event_amplitudes=[[1.0, 0.0], [0.0, 1.0]],  # 2 modulators
            duration=30.0,
            dt=0.01,
            n_inputs=2,
        )
        stim_mod_wrong = PiecewiseConstantInput(
            stim_dict["times"], stim_dict["values"],
        )
        with pytest.raises(ValueError, match=r"stim_mod\.values\.shape\[1\]=2"):
            pyro.poutine.trace(task_dcm_model).get_trace(
                observed_bold=task_bilinear_data["observed_bold"],
                stimulus=task_bilinear_data["stimulus"],
                a_mask=task_bilinear_data["a_mask"],
                c_mask=task_bilinear_data["c_mask"],
                t_eval=task_bilinear_data["t_eval"],
                TR=task_bilinear_data["TR"],
                dt=task_bilinear_data["dt"],
                b_masks=task_bilinear_data["b_masks"],  # J=1
                stim_mod=stim_mod_wrong,  # J=2
            )

    def test_bilinear_deprecation_warning_on_stacked_nonzero_diag(
        self, task_bilinear_data: dict,
    ) -> None:
        """Stacked-path DeprecationWarning closure for MODEL-03 / SC-4.

        Phase 13's ``tests/test_bilinear_utils.py`` exercises the
        ``(N, N)`` call-shape of ``parameterize_B`` directly. This test
        exercises the STACKED ``(J, N, N)`` call-path that Plan 15-01
        introduces inside ``task_dcm_model``: when ``task_dcm_model``
        is called with a ``b_masks`` list containing any non-zero
        diagonal entry, the stacked ``parameterize_B`` invocation at
        the end of the bilinear sampling loop must propagate the
        ``DeprecationWarning`` through the Pyro trace stack.

        Source of truth for the stacked path:
        ``src/pyro_dcm/forward_models/neural_state.py:137-151`` --
        after the ``ndim==3`` guard, the function constructs
        ``diag_entries`` via ``b_mask[:, diag_idx, diag_idx]`` (shape
        ``(J, N)``) and issues ONE ``DeprecationWarning`` if any entry
        is non-zero. The warning fires once per ``parameterize_B`` call
        regardless of ``J``, which is sufficient for the SC-4
        acceptance claim ("explicit non-zero diagonal triggers
        ``DeprecationWarning``") at the ``task_dcm_model`` call-site.
        """
        N = task_bilinear_data["N"]
        M = task_bilinear_data["M"]
        # Construct a b_masks with a non-zero diagonal entry (SC-4 trigger).
        bad_b_mask = torch.zeros(N, N, dtype=torch.float64)
        bad_b_mask[0, 0] = 1.0  # self-modulation on region 0
        bad_b_mask[1, 0] = 1.0  # plus off-diagonal from the fixture

        conditioned = pyro.poutine.condition(
            task_dcm_model,
            data={
                "A_free": 0.01 * torch.randn(N, N, dtype=torch.float64),
                "C": 0.25 * torch.ones(N, M, dtype=torch.float64),
                "noise_prec": torch.tensor(10.0, dtype=torch.float64),
                "B_free_0": 0.01 * torch.randn(N, N, dtype=torch.float64),
            },
        )
        with pytest.warns(DeprecationWarning, match="non-zero diagonal"):
            pyro.poutine.trace(conditioned).get_trace(
                observed_bold=task_bilinear_data["observed_bold"],
                stimulus=task_bilinear_data["stimulus"],
                a_mask=task_bilinear_data["a_mask"],
                c_mask=task_bilinear_data["c_mask"],
                t_eval=task_bilinear_data["t_eval"],
                TR=task_bilinear_data["TR"],
                dt=task_bilinear_data["dt"],
                b_masks=[bad_b_mask],
                stim_mod=task_bilinear_data["stim_mod"],
            )


# ---------------------------------------------------------------------------
# Bilinear SVI smoke test (Phase 15 Plan 15-01)
# ---------------------------------------------------------------------------


class TestBilinearSVI:
    """SVI smoke test for the bilinear branch (MODEL-04 convergence gate).

    Uses ``init_scale=0.005`` (L2 locked; half the linear default of 0.01)
    because bilinear B tails can push ``max Re(eig(A_eff))`` positive
    under ``N(0, 1.0)`` prior draws (Gershgorin analysis in
    ``15-RESEARCH.md`` Section 9). The NaN-safe ``predicted_bold`` guard
    in ``task_dcm_model`` + smaller ``init_scale`` together keep the SVI
    loop finite across 40 steps at N=3, J=1.

    Pyro stability logger is silenced via ``caplog`` autouse fixture
    because D4 = log-warn only; the monitor fires frequently in bilinear
    SVI early iterations but does NOT raise
    (``15-RESEARCH.md`` Section 9 R6).

    Runtime budget: target <75s at N=3, J=1, 40 SVI steps. Not a fail
    condition if exceeded -- test asserts convergence direction only.
    """

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog) -> None:
        """Silence ``pyro_dcm.stability`` WARNING spam during bilinear SVI."""
        import logging
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    def test_bilinear_svi_smoke_3region_converges(
        self, task_bilinear_data: dict,
    ) -> None:
        """40 SVI steps on 3-region J=1: finite losses + convergence direction.

        Closes MODEL-04 bilinear SVI acceptance gate. Uses AutoNormal
        with ``init_scale=0.005`` (L2). Does NOT assert a specific final
        loss value -- only the direction (end < start) and that no step
        produced NaN/Inf.
        """
        import time
        pyro.clear_param_store()

        guide = AutoNormal(task_dcm_model, init_scale=0.005)  # L2
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(task_dcm_model, guide, optimizer, loss=Trace_ELBO())

        model_kwargs = dict(
            observed_bold=task_bilinear_data["observed_bold"],
            stimulus=task_bilinear_data["stimulus"],
            a_mask=task_bilinear_data["a_mask"],
            c_mask=task_bilinear_data["c_mask"],
            t_eval=task_bilinear_data["t_eval"],
            TR=task_bilinear_data["TR"],
            dt=task_bilinear_data["dt"],
            b_masks=task_bilinear_data["b_masks"],
            stim_mod=task_bilinear_data["stim_mod"],
        )

        num_steps = 40
        losses: list[float] = []
        start = time.perf_counter()
        for _ in range(num_steps):
            loss = svi.step(**model_kwargs)
            losses.append(float(loss))
        elapsed = time.perf_counter() - start

        # Finite losses at every step (NaN-safe guard + L2 init_scale).
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)).item(), (
                f"Non-finite loss at step {i}: {loss}"
            )

        # Convergence direction: mean of last 10 < mean of first 10.
        first_mean = sum(losses[:10]) / 10
        last_mean = sum(losses[-10:]) / 10
        assert last_mean < first_mean, (
            f"SVI did not decrease: first_10_mean={first_mean:.2f}, "
            f"last_10_mean={last_mean:.2f} (losses={losses})"
        )

        # Runtime budget is soft -- issue a warning (not fail) if
        # exceeded. D4 + Pitfall B10 3-6x slowdown estimate -> 75s upper
        # bound at N=3, J=1.
        if elapsed > 75.0:
            import warnings
            warnings.warn(
                f"bilinear SVI smoke test exceeded 75s budget: "
                f"{elapsed:.1f}s (Pitfall B10 3-6x slowdown assumption "
                f"may be too optimistic; update 15-RESEARCH.md Section "
                f"8 budget if this persists).",
                UserWarning,
                stacklevel=1,
            )
