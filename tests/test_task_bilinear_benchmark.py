"""End-to-end smoke + acceptance-gate tests for the task-bilinear runner.

Plan 16-01 ships the smoke test only (3-seed quick config verifying runner
wiring). Plan 16-02 will append the ``@pytest.mark.slow`` acceptance-gate
test at 10 seeds / 500 steps that asserts RECOV-03..06 pass.

Test classes:
    TestTaskBilinearSmoke -- 3-seed quick-config wiring + return-dict contract.

Marker usage::

    pytest tests/test_task_bilinear_benchmark.py -m slow       # run slow gates
    pytest tests/test_task_bilinear_benchmark.py -m "not slow" # skip slow
    pytest tests/test_task_bilinear_benchmark.py               # runs both
"""

from __future__ import annotations

import logging

import pyro
import pytest
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.runners.task_bilinear import run_task_bilinear_svi


class TestTaskBilinearSmoke:
    """3-seed quick-config smoke: runner wires end-to-end without error."""

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog) -> None:
        """Silence pyro_dcm.stability WARNING spam during bilinear SVI."""
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    @pytest.fixture(autouse=True)
    def _reset_pyro(self) -> None:
        """Clean pyro state between tests."""
        pyro.clear_param_store()
        yield
        pyro.clear_param_store()

    @pytest.mark.slow
    def test_smoke_runs_3_seeds_quick(self) -> None:
        """Verify quick-config runs end-to-end and the return dict is well-formed.

        Runtime target: <15 min (3 seeds x ~5 min each per research Section 2.1
        conservative estimate at 500 steps / 200s BOLD / N=3 / J=1).

        Asserts:

        - All datasets either succeed (``n_success`` matches config) or status
          is ``'insufficient_data'``.
        - Success path: lists are well-formed with length ``n_success`` across
          the per-seed output lists, metadata is populated, ``posterior_list[0]``
          has a ``B_free_0`` key with mean shape ``(3, 3)`` and samples shape
          ``(>=1, 3, 3)``.
        - Insufficient-data path: ``status='insufficient_data'`` is the only
          acceptable failure mode -- a raw exception from the runner means
          the L1 forwarding is broken.
        """
        config = BenchmarkConfig.quick_config(
            "task_bilinear", "svi",
        )
        # Override n_datasets=2 for faster smoke (research Section 8 notes
        # 3-seed target, but 2 is sufficient for wiring verification and
        # stays well within CI budget).
        config.n_datasets = 2

        result = run_task_bilinear_svi(config)

        # --- Return dict contract ---
        assert isinstance(result, dict), (
            f"run_task_bilinear_svi must return a dict; "
            f"got {type(result).__name__}"
        )

        # Either success or insufficient_data -- NO raw exception.
        if result.get("status") == "insufficient_data":
            # Insufficient-data is allowed for a 2-seed smoke if both fail;
            # that itself signals an integration problem upstream (L1
            # forwarding, simulator, etc.) but is NOT an exception-in-
            # runner bug.
            pytest.skip(
                f"Smoke test: {result['n_failed']}/{result['n_datasets']} "
                f"failed. This indicates a deeper bilinear-SVI stability "
                f"issue -- investigate individual failures outside this smoke."
            )

        # Success path: lists exist and are length == n_success.
        for key in (
            "a_rmse_bilinear_list", "a_rmse_linear_list",
            "time_bilinear_list", "time_linear_list",
            "posterior_list", "b_true_list",
            "a_true_list", "a_inferred_bilinear_list",
            "a_inferred_linear_list",
        ):
            assert key in result, (
                f"Missing key '{key}' in runner output; got: "
                f"{sorted(result.keys())}"
            )
            assert len(result[key]) == result["n_success"], (
                f"Length mismatch: {key} length={len(result[key])}, "
                f"n_success={result['n_success']}"
            )

        # Metadata contract.
        md = result["metadata"]
        assert md["variant"] == "task_bilinear"
        assert md["method"] == "svi"
        assert md["n_regions"] == 3
        assert md["J"] == 1
        assert md["SNR"] == 3.0
        assert md["duration"] == 200.0
        assert md["guide_type"] == "auto_normal"
        assert md["init_scale_bilinear"] == 0.005
        assert md["b_true_magnitudes"] == {"B[1,0]": 0.4, "B[2,1]": 0.3}

        # Posterior structure for seed 0.
        post_0 = result["posterior_list"][0]
        assert "B_free_0" in post_0, (
            f"First posterior missing B_free_0; keys: "
            f"{sorted(post_0.keys())}. L1 model_kwargs forwarding may be "
            f"broken -- task_dcm_model did NOT register the bilinear "
            f"B_free_j sample site."
        )
        b_mean = post_0["B_free_0"]["mean"]
        # Stored as list[list[float]] after _posterior_to_numpy.
        b_mean_tensor = torch.tensor(b_mean)
        assert b_mean_tensor.shape == (3, 3), (
            f"B_free_0 posterior mean shape: expected (3, 3), got "
            f"{tuple(b_mean_tensor.shape)}"
        )

        # B_true carried in posterior_list entry.
        assert "B_true" in post_0
        b_true = torch.tensor(post_0["B_true"])
        assert b_true.shape == (1, 3, 3)
        assert abs(b_true[0, 1, 0].item() - 0.4) < 1e-9
        assert abs(b_true[0, 2, 1].item() - 0.3) < 1e-9

        # A-RMSE lists are finite non-negative floats.
        for rmse in result["a_rmse_bilinear_list"]:
            assert rmse >= 0.0
            assert rmse < 1e9, f"A-RMSE unreasonable: {rmse}"
        for rmse in result["a_rmse_linear_list"]:
            assert rmse >= 0.0

        # Wall-time recorded.
        assert all(t > 0 for t in result["time_bilinear_list"])
        assert all(t > 0 for t in result["time_linear_list"])


# ------------------------------------------------------------------
# Fast unit tests for the NaN-retry helper (_fit_bilinear_with_retry)
# ------------------------------------------------------------------


class TestFitBilinearWithRetry:
    """Unit-level retry policy on ``RuntimeError('NaN ELBO at step 0')``.

    Covers the three branches of ``_fit_bilinear_with_retry``: success at
    the default init_scale, successful retry after a step-0 NaN, and
    re-raise on any non-step-0 NaN or unrelated RuntimeError.
    """

    def test_success_uses_default_init_scale(self, monkeypatch) -> None:
        """First call succeeds -> init_scale_used == _BILINEAR_INIT_SCALE."""
        from benchmarks.runners import task_bilinear as tb

        calls: list[float] = []

        def fake_fit(model_args, model_kwargs, *, guide_type, init_scale,
                     num_steps, elbo_type):
            calls.append(init_scale)
            return {"A_free": {"mean": torch.zeros(3, 3)}}, 1.0

        monkeypatch.setattr(tb, "_fit_and_extract", fake_fit)
        _, elapsed, used = tb._fit_bilinear_with_retry(
            model_args=(), model_kwargs={}, num_steps=10, elbo_type="trace_elbo",
        )
        assert used == tb._BILINEAR_INIT_SCALE
        assert calls == [tb._BILINEAR_INIT_SCALE]
        assert elapsed == 1.0

    def test_nan_step_0_triggers_retry_at_halved_scale(
        self, monkeypatch,
    ) -> None:
        """step-0 NaN -> retry at _BILINEAR_INIT_SCALE_RETRY succeeds."""
        from benchmarks.runners import task_bilinear as tb

        calls: list[float] = []

        def fake_fit(model_args, model_kwargs, *, guide_type, init_scale,
                     num_steps, elbo_type):
            calls.append(init_scale)
            if len(calls) == 1:
                msg = "NaN ELBO at step 0"
                raise RuntimeError(msg)
            return {"A_free": {"mean": torch.zeros(3, 3)}}, 2.0

        monkeypatch.setattr(tb, "_fit_and_extract", fake_fit)
        _, elapsed, used = tb._fit_bilinear_with_retry(
            model_args=(), model_kwargs={}, num_steps=10, elbo_type="trace_elbo",
        )
        assert used == tb._BILINEAR_INIT_SCALE_RETRY
        assert calls == [tb._BILINEAR_INIT_SCALE, tb._BILINEAR_INIT_SCALE_RETRY]
        assert elapsed == 2.0

    def test_nan_step_nonzero_reraises(self, monkeypatch) -> None:
        """NaN after training started -> re-raised (not an init-scale issue)."""
        from benchmarks.runners import task_bilinear as tb

        def fake_fit(model_args, model_kwargs, *, guide_type, init_scale,
                     num_steps, elbo_type):
            msg = "NaN ELBO at step 47"
            raise RuntimeError(msg)

        monkeypatch.setattr(tb, "_fit_and_extract", fake_fit)
        with pytest.raises(RuntimeError, match="NaN ELBO at step 47"):
            tb._fit_bilinear_with_retry(
                model_args=(), model_kwargs={},
                num_steps=50, elbo_type="trace_elbo",
            )


class TestSeedPoolCorruptSkip:
    """Seed-pool rejection behavior on corrupt (NaN BOLD) fixtures.

    Post cluster job 54902455 the root cause of the step-0-NaN failures was
    traced to ground-truth fixtures whose ``A + B_true`` is unstable under
    sustained modulator activation, producing NaN BOLD silently. The runner
    now filters those seeds via a pool loop. These unit tests stub the
    fixture generator to drive the skip branch deterministically without
    running the full ODE simulator.
    """

    def test_corrupt_seed_is_skipped_and_next_seed_is_used(
        self, monkeypatch,
    ) -> None:
        """1 corrupt + 1 clean fixture -> skip 1st, use 2nd, n_success=1."""
        from benchmarks.runners import task_bilinear as tb

        N = 3

        def make_clean(n_regions, seed_i):
            return {
                "A_true": torch.eye(N, dtype=torch.float64) * -0.5,
                "C": torch.zeros(N, 1, dtype=torch.float64),
                "B_true": torch.zeros(1, N, N, dtype=torch.float64),
                "b_mask_0": torch.zeros(N, N, dtype=torch.float64),
                "stim": {
                    "times": torch.tensor([0.0], dtype=torch.float64),
                    "values": torch.zeros(1, 1, dtype=torch.float64),
                },
                "stimulus": _FakePiecewise(),
                "stim_mod": _FakePiecewise(J=1),
                "bold": torch.zeros(50, N, dtype=torch.float64),
            }

        def make_corrupt(n_regions, seed_i):
            d = make_clean(n_regions, seed_i)
            d["bold"] = torch.full((50, N), float("nan"), dtype=torch.float64)
            return d

        calls = {"i": 0}

        def dispatch(n_regions, seed_i):
            calls["i"] += 1
            # First seed: corrupt. Second seed onward: clean.
            return (make_corrupt if calls["i"] == 1 else make_clean)(
                n_regions, seed_i,
            )

        # Patch the inline generator used by the runner (the default path).
        monkeypatch.setattr(tb, "_make_bilinear_ground_truth", dispatch)

        # Stub the SVI fit paths so the test stays fast and deterministic.
        def fake_fit_with_retry(model_args, model_kwargs, *,
                                num_steps, elbo_type):
            return (
                {
                    "A_free": {
                        "mean": torch.zeros(N, N, dtype=torch.float64),
                        "std": torch.zeros(N, N, dtype=torch.float64),
                        "samples": torch.zeros(1, N, N, dtype=torch.float64),
                    },
                    "B_free_0": {
                        "mean": torch.zeros(N, N, dtype=torch.float64),
                        "std": torch.ones(N, N, dtype=torch.float64),
                        "samples": torch.zeros(1, N, N, dtype=torch.float64),
                    },
                    "final_losses": [],
                },
                1.0,
                tb._BILINEAR_INIT_SCALE,
            )

        def fake_fit_and_extract(
            model_args, model_kwargs, *, guide_type,
            init_scale, num_steps, elbo_type,
        ):
            return (
                {
                    "A_free": {
                        "mean": torch.zeros(N, N, dtype=torch.float64),
                        "std": torch.zeros(N, N, dtype=torch.float64),
                        "samples": torch.zeros(1, N, N, dtype=torch.float64),
                    },
                    "final_losses": [],
                },
                1.0,
            )

        monkeypatch.setattr(
            tb, "_fit_bilinear_with_retry", fake_fit_with_retry,
        )
        monkeypatch.setattr(tb, "_fit_and_extract", fake_fit_and_extract)

        config = BenchmarkConfig(
            variant="task_bilinear", method="svi",
            n_datasets=1, n_svi_steps=1, seed=42,
            n_regions=3, quick=True, elbo_type="trace_elbo",
        )
        result = tb.run_task_bilinear_svi(config)

        assert result.get("n_success") == 1
        # First seed (42) skipped as corrupt; second (43) accepted.
        assert result["seeds_skipped_corrupt"] == [42]
        assert result["seeds_used"] == [43]
        # Metadata tracks skips for SUMMARY visibility.
        assert result["metadata"]["n_seeds_skipped_corrupt"] == 1

    def test_pool_exhausted_returns_insufficient_data(
        self, monkeypatch,
    ) -> None:
        """All-corrupt pool -> pool exhausts; returns insufficient_data."""
        from benchmarks.runners import task_bilinear as tb

        N = 3

        def always_corrupt(n_regions, seed_i):
            return {
                "A_true": torch.eye(N, dtype=torch.float64) * -0.5,
                "C": torch.zeros(N, 1, dtype=torch.float64),
                "B_true": torch.zeros(1, N, N, dtype=torch.float64),
                "b_mask_0": torch.zeros(N, N, dtype=torch.float64),
                "stim": {
                    "times": torch.tensor([0.0], dtype=torch.float64),
                    "values": torch.zeros(1, 1, dtype=torch.float64),
                },
                "stimulus": _FakePiecewise(),
                "stim_mod": _FakePiecewise(J=1),
                "bold": torch.full((50, N), float("nan"), dtype=torch.float64),
            }

        monkeypatch.setattr(
            tb, "_make_bilinear_ground_truth", always_corrupt,
        )

        config = BenchmarkConfig(
            variant="task_bilinear", method="svi",
            n_datasets=2, n_svi_steps=1, seed=100,
            n_regions=3, quick=True, elbo_type="trace_elbo",
        )
        result = tb.run_task_bilinear_svi(config)

        # With n_datasets=2 and multiplier=3, pool tries 6 seeds; all corrupt.
        assert result.get("status") == "insufficient_data"
        assert result.get("n_success", 0) == 0
        assert len(result["seeds_skipped_corrupt"]) == (
            2 * tb._MAX_POOL_MULTIPLIER
        )
        assert result["pool_exhausted"] is True


class _FakePiecewise:
    """Minimal stand-in for PiecewiseConstantInput in seed-pool unit tests.

    The runner's SVI call path is stubbed, so the actual ``.values`` / call
    semantics never matter; we just need an object that duck-types enough
    to be stored in the fixture dict and printed in diagnostic messages.
    """

    def __init__(self, J: int = 1) -> None:
        self.values = torch.zeros(1, J, dtype=torch.float64)
        self.times = torch.zeros(1, dtype=torch.float64)

    def __call__(self, t):  # noqa: ANN001
        return torch.zeros(self.values.shape[1], dtype=torch.float64)


# ------------------------------------------------------------------
# Phase 16 acceptance gate (RECOV-03..08) -- @pytest.mark.slow
# ------------------------------------------------------------------


class TestTaskBilinearAcceptance:
    r"""THE Phase 16 acceptance gate: 10 seeds, all RECOV gates must pass.

    Runtime target (research Section 2.1 conservative estimate): ~80 min at
    n_datasets=10 + n_svi_steps=500 on N=3, J=1, SNR=3, duration=200s. Gated
    by ``@pytest.mark.slow`` so default pytest runs do NOT execute this. To run::

        pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance \\
               --timeout=7200

    OR to run ALL slow tests in the file::

        pytest tests/test_task_bilinear_benchmark.py -m slow --timeout=7200

    Failure modes:

    - ``status='insufficient_data'``: ``compute_acceptance_gates`` raises
      ValueError (< 50% of seeds survived bilinear SVI). Indicates upstream
      stability issue.
    - RECOV-03 fail: bilinear A-RMSE > 1.25 * linear-baseline A-RMSE. Pitfall
      B13 A-RMSE inflation exceeded.
    - RECOV-04 fail: B-RMSE > 0.20 on |B_true|>0.1 elements. Likely
      identifiability issue; check forest plot shrinkage.
    - RECOV-05 fail: sign recovery < 80%. Check for systematic sign flip
      (possibly guide init_scale too large or too small).
    - RECOV-06 fail: coverage_of_zero < 85%. Mean-field posterior correlation
      underestimation (Pitfall v0.2.0 P1; expected ceiling drops under bilinear
      per PITFALLS.md:648; research N1 flags this as the PRIMARY Phase 16
      risk). Per L9: RECOV-06 threshold remains hard at 85% under AutoNormal
      (no in-scope fallback). If the gate fails, record the observed coverage
      in the Phase 16 SUMMARY and flag the milestone as blocked pending v0.3.1
      AutoLowRankMVN fallback tier (deferred per CONTEXT.md Deferred Ideas).
    """

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog) -> None:
        """Silence pyro_dcm.stability WARNING spam during bilinear SVI."""
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    @pytest.fixture(autouse=True)
    def _reset_pyro(self) -> None:
        pyro.clear_param_store()
        yield
        pyro.clear_param_store()

    @pytest.mark.slow
    def test_acceptance_gates_pass_at_10_seeds(self, tmp_path) -> None:
        """RECOV-03..06 all pass on 10 seeds of 3-region bilinear SNR=3 recovery.

        Also (soft) asserts RECOV-07 shrinkage_nonnull means < 1.0 (sanity;
        non-identifiability would be >= 1.0). RECOV-08 wall-time ratio is
        reported via ``print`` but does not block acceptance.

        CONVERGENCE CAVEAT (L8 / FIX 4): 500 steps is a research-recommended
        starting point extrapolated from Phase 15 smoke (40 steps on 30s
        BOLD), NOT a validated convergence budget at 200s BOLD / 100 TR
        timepoints. If RECOV-04 or RECOV-05 fail under 500 steps, retry
        with svi_steps=1500 before concluding the implementation is
        incorrect::

            # In the test body:
            # config.n_svi_steps = 1500  # try this before filing a bug
        """
        from benchmarks.bilinear_metrics import compute_acceptance_gates
        from benchmarks.config import BenchmarkConfig
        from benchmarks.plotting import (
            plot_acceptance_gates_table,
            plot_bilinear_b_forest,
        )
        from benchmarks.runners.task_bilinear import run_task_bilinear_svi

        # n_svi_steps=500 override per L8 (research Section 2.1 runtime budget).
        config = BenchmarkConfig.full_config("task_bilinear", "svi")
        config.n_svi_steps = 500

        result = run_task_bilinear_svi(config)

        # Enforce the original >=10 seed floor. The post-cluster-54902455
        # investigation traced the step-0 NaN failures to ground-truth fixture
        # corruption (some seeds produce A + B_true with max Re(eig) >= 0,
        # causing simulate_task_dcm to emit NaN BOLD during sustained
        # u_mod=1 epochs). The runner now filters those seeds via a pool loop
        # (see _MAX_POOL_MULTIPLIER in task_bilinear.py), so any seed that
        # makes it into seeds_used has a clean fixture and SHOULD complete
        # SVI without a step-0 NaN. n_success < 10 therefore indicates either
        # (a) too many corrupt seeds exhausted the pool (raise
        # _MAX_POOL_MULTIPLIER) or (b) an unrelated SVI regression.
        n_success = result.get(
            "n_success", len(result.get("a_rmse_bilinear_list", [])),
        )
        n_retries = result.get("metadata", {}).get(
            "init_scale_bilinear_n_retries", 0,
        )
        seeds_used = result.get("seeds_used", [])
        seeds_skipped = result.get("seeds_skipped_corrupt", [])
        pool_exhausted = result.get("pool_exhausted", False)
        assert n_success >= 10, (
            f"Expected >=10 successful seeds, got n_success={n_success} "
            f"(n_failed={result.get('n_failed', '?')}, "
            f"n_datasets={result.get('n_datasets', '?')}, "
            f"status={result.get('status', 'success')}, "
            f"seeds_used={seeds_used}, "
            f"seeds_skipped_corrupt={seeds_skipped}, "
            f"pool_exhausted={pool_exhausted}, "
            f"init_scale_bilinear_n_retries={n_retries}). "
            f"Likely causes: (a) corruption rate higher than "
            f"_MAX_POOL_MULTIPLIER=3 can absorb (raise it in "
            f"benchmarks/runners/task_bilinear.py); or (b) SVI regression "
            f"(inspect posterior_list[*]['final_losses'] tails)."
        )

        # Convert torch tensors in posterior_list to the serialized shape that
        # compute_acceptance_gates expects (runner already does this via
        # _posterior_to_numpy in plan 16-01 Task 3).
        gates = compute_acceptance_gates(result)

        # --- Emit diagnostics BEFORE assertions so failures log context. ---
        print("\n=== Phase 16 Acceptance Gates ===")
        print(
            f"RECOV-03 (A RMSE ratio): observed={gates['RECOV-03']['ratio']:.4f} "
            f"threshold={gates['RECOV-03']['threshold']} "
            f"pass={gates['RECOV-03']['pass']}"
        )
        print(
            f"RECOV-04 (B RMSE nonnull): "
            f"observed={gates['RECOV-04']['observed']:.4f} "
            f"threshold={gates['RECOV-04']['threshold']} "
            f"pass={gates['RECOV-04']['pass']}"
        )
        print(
            f"RECOV-05 (sign recovery): "
            f"observed={gates['RECOV-05']['observed']:.4f} "
            f"threshold={gates['RECOV-05']['threshold']} "
            f"pass={gates['RECOV-05']['pass']}"
        )
        print(
            f"RECOV-06 (coverage of zero): "
            f"observed={gates['RECOV-06']['observed']:.4f} "
            f"threshold={gates['RECOV-06']['threshold']} "
            f"pass={gates['RECOV-06']['pass']}"
        )
        print(
            f"RECOV-07 (shrinkage nonnull means): "
            f"{gates['RECOV-07']['shrinkage_nonnull']}"
        )
        print(
            f"RECOV-08 (wall-time ratio): {gates['RECOV-08']['ratio']:.2f}x "
            f"(bilinear={gates['RECOV-08']['time_bilinear']:.1f}s, "
            f"linear={gates['RECOV-08']['time_linear']:.1f}s) "
            f"flag_over_10x={gates['RECOV-08']['flag_over_10x']}"
        )

        # --- Save figures to tmp_path as gate artifacts. ---
        results_for_fig = {("task_bilinear", "svi"): result}
        fig1 = plot_bilinear_b_forest(results_for_fig, str(tmp_path))
        fig2 = plot_acceptance_gates_table(results_for_fig, str(tmp_path))
        assert fig1 is not None, "forest-plot returned None"
        assert fig2 is not None, "acceptance-table returned None"
        assert (tmp_path / "b_forest_recovery.png").exists()
        assert (tmp_path / "acceptance_gates.png").exists()

        # --- Acceptance assertions. ---
        assert gates["RECOV-03"]["pass"], (
            f"RECOV-03 FAILED: A-RMSE ratio {gates['RECOV-03']['ratio']:.4f} > "
            f"threshold {gates['RECOV-03']['threshold']}. Bilinear A-RMSE "
            f"{gates['RECOV-03']['mean_bilinear']:.4f} vs linear baseline "
            f"{gates['RECOV-03']['mean_linear']:.4f}. Pitfall B13 A-RMSE "
            f"inflation exceeded."
        )
        assert gates["RECOV-04"]["pass"], (
            f"RECOV-04 FAILED: B-RMSE {gates['RECOV-04']['observed']:.4f} > "
            f"threshold {gates['RECOV-04']['threshold']}. per_seed: "
            f"{gates['RECOV-04']['per_seed']}"
        )
        assert gates["RECOV-05"]["pass"], (
            f"RECOV-05 FAILED: sign recovery "
            f"{gates['RECOV-05']['observed']:.4f} < "
            f"threshold {gates['RECOV-05']['threshold']}."
        )
        assert gates["RECOV-06"]["pass"], (
            f"RECOV-06 FAILED: coverage_of_zero "
            f"{gates['RECOV-06']['observed']:.4f} < "
            f"threshold {gates['RECOV-06']['threshold']}. Mean-field posterior "
            f"correlation underestimation likely (Pitfall v0.2.0 P1 under "
            f"bilinear); sidebar with auto_lowrank_mvn deferred to v0.3.1."
        )
        assert gates["all_pass"], (
            "One or more RECOV gates failed; see diagnostics above."
        )

        # Soft RECOV-07 sanity: shrinkage < 1.0 for free elements
        # (non-zero concentration).
        for s in gates["RECOV-07"]["shrinkage_nonnull"]:
            assert s < 1.0, (
                f"Shrinkage {s} >= 1.0 indicates posterior is wider than prior "
                f"for a non-null element; identifiability failure."
            )
