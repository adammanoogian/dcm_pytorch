"""Runner registry for benchmark execution.

Maps ``(variant, method)`` tuples to runner functions. Each runner
accepts a ``BenchmarkConfig`` and returns a results dictionary.

Runners are registered in Plan 08-03. Initial entries raise
``NotImplementedError`` as placeholders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmarks.config import BenchmarkConfig


def _not_implemented(config: BenchmarkConfig) -> dict[str, Any]:
    """Placeholder runner that raises NotImplementedError.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration (unused).

    Raises
    ------
    NotImplementedError
        Always raised with message indicating Plan 08-03.
    """
    raise NotImplementedError(
        f"Runner ({config.variant}, {config.method}) "
        f"will be implemented in Plan 08-03"
    )


RUNNER_REGISTRY: dict[tuple[str, str], Callable[..., dict[str, Any]]] = {
    ("task", "svi"): _not_implemented,
    ("task", "amortized"): _not_implemented,
    ("spectral", "svi"): _not_implemented,
    ("spectral", "amortized"): _not_implemented,
    ("rdcm_rigid", "vb"): _not_implemented,
    ("rdcm_sparse", "vb"): _not_implemented,
    ("spm", "reference"): _not_implemented,
}
