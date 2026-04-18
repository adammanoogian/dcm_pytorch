"""Runner registry for benchmark execution.

Maps ``(variant, method)`` tuples to runner functions. Each runner
accepts a ``BenchmarkConfig`` and returns a results dictionary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmarks.config import BenchmarkConfig
from benchmarks.runners.rdcm_vb import (
    run_rdcm_rigid_vb,
    run_rdcm_sparse_vb,
)
from benchmarks.runners.spectral_amortized import run_spectral_amortized
from benchmarks.runners.spectral_svi import run_spectral_svi
from benchmarks.runners.spm_reference import run_spm_reference
from benchmarks.runners.task_amortized import run_task_amortized
from benchmarks.runners.task_bilinear import run_task_bilinear_svi
from benchmarks.runners.task_svi import run_task_svi

RUNNER_REGISTRY: dict[tuple[str, str], Callable[..., dict[str, Any]]] = {
    ("task", "svi"): run_task_svi,
    ("task", "amortized"): run_task_amortized,
    ("task_bilinear", "svi"): run_task_bilinear_svi,
    ("spectral", "svi"): run_spectral_svi,
    ("spectral", "amortized"): run_spectral_amortized,
    ("rdcm_rigid", "vb"): run_rdcm_rigid_vb,
    ("rdcm_sparse", "vb"): run_rdcm_sparse_vb,
    ("spm", "reference"): run_spm_reference,
}
