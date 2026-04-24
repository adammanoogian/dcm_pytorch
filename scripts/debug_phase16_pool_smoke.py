"""Smoke-test the seed-pool corrupt-fixture skip WITHOUT running SVI.

Monkeypatches the SVI fit calls so we can verify in seconds that the pool
loop correctly skips seed 44 (known corrupt per cluster job 54902455) and
finds 10 clean seeds within the pool cap.

Usage::

    python scripts/debug_phase16_pool_smoke.py
"""

from __future__ import annotations

import logging
import sys

import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.runners import task_bilinear as tb


def fake_fit_with_retry(
    model_args, model_kwargs, *, num_steps, elbo_type,
):
    """Instant SVI-free stub; returns a trivial posterior dict."""
    N = 3
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
    model_args, model_kwargs, *, guide_type, init_scale, num_steps, elbo_type,
):
    """Instant linear-baseline stub."""
    N = 3
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


def main() -> int:
    logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)

    # Swap SVI fits for instant stubs so we're only testing pool logic.
    tb._fit_bilinear_with_retry = fake_fit_with_retry  # type: ignore[assignment]
    tb._fit_and_extract = fake_fit_and_extract  # type: ignore[assignment]

    config = BenchmarkConfig.full_config("task_bilinear", "svi")
    config.n_svi_steps = 1  # stubbed, but keep it minimal

    print(
        f"Pool smoke: n_datasets={config.n_datasets}, "
        f"seed_pool_max={config.n_datasets * tb._MAX_POOL_MULTIPLIER}"
    )
    print(
        "(runs fixture generation for up to 30 seeds; SVI is stubbed — "
        "this is purely a pool-plumbing + fixture-finiteness check)"
    )
    result = tb.run_task_bilinear_svi(config)

    status = result.get("status", "success")
    print(f"\nResult status: {status}")
    print(f"n_success      = {result.get('n_success', 0)}")
    print(f"n_failed       = {result.get('n_failed', 0)}")
    print(f"seeds_used     = {result.get('seeds_used', [])}")
    print(f"seeds_skipped  = {result.get('seeds_skipped_corrupt', [])}")
    if status == "insufficient_data":
        print(f"pool_exhausted = {result.get('pool_exhausted', False)}")
    else:
        print(
            f"n_seeds_skipped_corrupt (metadata) = "
            f"{result['metadata']['n_seeds_skipped_corrupt']}"
        )

    ok = (
        status != "insufficient_data"
        and result.get("n_success", 0) == config.n_datasets
        and 44 in result.get("seeds_skipped_corrupt", [])
    )
    if ok:
        print("\nPASS: 10 seeds collected; seed 44 was correctly skipped.")
        return 0
    print("\nFAIL: pool smoke did not produce 10 clean seeds with 44 skipped.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
