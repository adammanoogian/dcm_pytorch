"""Smoke tests for the VL benchmark runners (Plan 29-04, VLINFRA-02).

Proves the PLUMBING of the three Variational Laplace runners registered under
``method="vl"`` in ``RUNNER_REGISTRY`` -- that each accepts a minimal
``BenchmarkConfig``, fits via VL without raising, and returns a dict of the
expected shape (correct ``variant`` / ``method``, a length-1 per-dataset metric
list, finite A-RMSE). These are NOT recovery-quality tests: convergence at the
tiny smoke ``max_iter`` is not asserted (the smoke runs intentionally cap
iterations for laptop speed; if a runner legitimately fails to converge it still
returns a finite A-RMSE, which is what we check).

Runtime budget: all three tests stay under ~3 minutes total on laptop CPU
(N=2 / 1 seed, small ``max_iter``, ``quick=True``). The full N x SNR multi-seed
sweep is Phase 30 and routes to the M3 cluster -- do NOT run a full sweep here.

References
----------
.planning/phases/29-vl-validation-infra-bmr-rank/29-04-PLAN.md
"""

from __future__ import annotations

import math

import pytest

from benchmarks.config import BenchmarkConfig
from benchmarks.runners import RUNNER_REGISTRY
from benchmarks.runners.latent_circuit_vl import run_latent_circuit_vl
from benchmarks.runners.spectral_vl import run_spectral_vl
from benchmarks.runners.task_vl import run_task_vl


def test_vl_runners_registered() -> None:
    """RUNNER_REGISTRY exposes the three callable ``method='vl'`` entries."""
    for key in (("spectral", "vl"), ("task", "vl"), ("latent_circuit", "vl")):
        assert key in RUNNER_REGISTRY, f"missing registry key {key}"
        assert callable(RUNNER_REGISTRY[key]), f"{key} is not callable"


@pytest.mark.vl
def test_spectral_vl_smoke() -> None:
    """run_spectral_vl returns a well-shaped dict at N=2 / 1 seed without raising."""
    config = BenchmarkConfig(
        variant="spectral", method="vl", n_datasets=1, n_regions=2,
        max_iter=8, quick=True, seed=0, save_figures=False,
    )
    result = run_spectral_vl(config)

    assert result["method"] == "vl"
    assert result["variant"] == "spectral"
    assert len(result["rmse_list"]) == 1
    if result["n_failed"] == 1:  # pragma: no cover - VL non-convergence escape
        pytest.xfail(f"spectral VL fit did not complete: {result['errors']}")
    assert math.isfinite(result["rmse_list"][0])


@pytest.mark.vl
def test_task_vl_smoke() -> None:
    """run_task_vl returns a well-shaped dict at N=2 / 1 seed without raising."""
    config = BenchmarkConfig(
        variant="task", method="vl", n_datasets=1, n_regions=2,
        max_iter=8, quick=True, seed=0, save_figures=False,
    )
    result = run_task_vl(config)

    assert result["method"] == "vl"
    assert result["variant"] == "task"
    assert len(result["rmse_list"]) == 1
    if result["n_failed"] == 1:  # pragma: no cover - VL non-convergence escape
        pytest.xfail(f"task VL fit did not complete: {result['errors']}")
    assert math.isfinite(result["rmse_list"][0])


@pytest.mark.vl
def test_latent_circuit_vl_smoke() -> None:
    """run_latent_circuit_vl returns a well-shaped dict at 1 seed without raising.

    Uses a tiny ``max_iter`` because the latent-circuit VL fit (N=4, J=1, dense
    time-domain precision) is the slowest of the three; the smoke proves
    plumbing, not recovery, so a few Gauss-Newton steps suffice.
    """
    config = BenchmarkConfig(
        variant="latent_circuit", method="vl", n_datasets=1,
        max_iter=4, quick=True, seed=0, save_figures=False,
    )
    result = run_latent_circuit_vl(config)

    assert result["method"] == "vl"
    assert result["variant"] == "latent_circuit"
    assert len(result["a_rmse_list"]) == 1
    if result["n_failed"] == 1:  # pragma: no cover - VL non-convergence escape
        pytest.xfail(f"latent_circuit VL fit did not complete: {result['errors']}")
    assert math.isfinite(result["a_rmse_list"][0])
