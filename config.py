"""Repo-wide path constants.

Single source of truth for paths used across `benchmarks/`, `validation/`, and
`scripts/`. Intentionally minimal in this revision: only the cluster-blocking
absolute path (`TAPAS_RDCM_PATH`) is centralized.

The broader migration of `benchmarks/results`, `benchmarks/figures`, and
`benchmarks/fixtures` literals (~15 call sites across `benchmarks/`) is
deferred -- see SUMMARY "Deferred future work."

Constants
---------
PROJECT_ROOT : pathlib.Path
    Absolute path to the repository root (parent of this file).
BENCHMARK_RESULTS_DIR : pathlib.Path
    Default directory for benchmark JSON / CSV outputs.
    Note: most `benchmarks/` callers still hardcode this literal; this
    constant is the migration target, not yet the call-site source of truth.
BENCHMARK_FIGURES_DIR : pathlib.Path
    Default directory for benchmark figure outputs.
BENCHMARK_FIXTURES_DIR : pathlib.Path
    Default directory for benchmark `.npz` fixture caches.
TAPAS_RDCM_PATH : pathlib.Path
    Path to the local clone of `tapas/rDCM` (MATLAB rDCM toolbox), used by
    `validation/run_rdcm_validation.py` and `validation/run_validation.py`.
    Override via the ``TAPAS_RDCM_PATH`` environment variable when running
    on a different machine (e.g., Monash M3 cluster).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent

BENCHMARK_RESULTS_DIR: Path = PROJECT_ROOT / "benchmarks" / "results"
BENCHMARK_FIGURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "figures"
BENCHMARK_FIXTURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "fixtures"

TAPAS_RDCM_PATH: Path = Path(
    os.environ.get(
        "TAPAS_RDCM_PATH",
        "C:/Users/aman0087/Documents/Github/tapas/rDCM",
    )
)
