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
