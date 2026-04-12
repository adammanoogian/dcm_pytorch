"""Benchmark configuration for DCM parameter recovery runs.

Provides a ``BenchmarkConfig`` dataclass for configuring benchmark
parameters including variant selection, inference method, dataset
counts, and output settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Parameters
    ----------
    variant : str
        DCM variant: ``"task"``, ``"spectral"``, or ``"rdcm"``.
    method : str
        Inference method: ``"svi"``, ``"amortized"``, or ``"vb"``.
    n_datasets : int
        Number of synthetic datasets to generate.
    n_regions : int
        Number of brain regions per dataset.
    n_svi_steps : int
        Number of SVI optimization steps.
    seed : int
        Random seed for reproducibility.
    quick : bool
        If True, use reduced parameters for CI/development.
    output_dir : str
        Directory for saving benchmark results.
    save_figures : bool
        If True, generate and save figures.
    figure_dir : str
        Directory for saving figures.
    guide_type : str
        Guide type for inference. Uses string keys matching
        ``GUIDE_REGISTRY`` in ``guides.py``: ``"auto_normal"``
        (default), ``"auto_delta"``, ``"auto_lowrank_mvn"``,
        ``"auto_mvn"``, ``"auto_iaf"``, ``"auto_laplace"``.
    n_regions_list : list[int]
        List of network sizes to benchmark.
    elbo_type : str
        ELBO objective type. Uses string keys matching
        ``ELBO_REGISTRY`` in ``guides.py``: ``"trace_elbo"``
        (default), ``"tracemeanfield_elbo"``, ``"renyi_elbo"``.
    fixtures_dir : str or None
        Path to shared fixtures directory. ``None`` means inline
        generation (v0.1.0 behavior).
    """

    variant: str
    method: str
    n_datasets: int = 20
    n_regions: int = 3
    n_svi_steps: int = 3000
    seed: int = 42
    quick: bool = False
    output_dir: str = "benchmarks/results"
    save_figures: bool = True
    figure_dir: str = "figures"
    guide_type: str = "auto_normal"
    n_regions_list: list[int] = field(default_factory=lambda: [3])
    elbo_type: str = "trace_elbo"
    fixtures_dir: str | None = None

    @classmethod
    def quick_config(
        cls, variant: str, method: str, **kwargs: Any,
    ) -> BenchmarkConfig:
        """Create a reduced-parameter config for CI/development runs.

        Parameters
        ----------
        variant : str
            DCM variant: ``"task"``, ``"spectral"``, or ``"rdcm"``.
        method : str
            Inference method.
        **kwargs : Any
            Additional keyword arguments forwarded to the constructor
            (e.g., ``fixtures_dir``, ``guide_type``).

        Returns
        -------
        BenchmarkConfig
            Config with reduced dataset counts and SVI steps.
        """
        defaults: dict[str, dict[str, int]] = {
            "task": {"n_datasets": 3, "n_svi_steps": 500},
            "spectral": {"n_datasets": 5, "n_svi_steps": 500},
            "rdcm": {"n_datasets": 5, "n_svi_steps": 500},
            "rdcm_rigid": {"n_datasets": 5, "n_svi_steps": 500},
            "rdcm_sparse": {"n_datasets": 5, "n_svi_steps": 500},
        }
        params = defaults.get(variant, {"n_datasets": 5, "n_svi_steps": 500})
        return cls(
            variant=variant,
            method=method,
            n_datasets=params["n_datasets"],
            n_svi_steps=params["n_svi_steps"],
            quick=True,
            **kwargs,
        )

    @classmethod
    def full_config(
        cls, variant: str, method: str, **kwargs: Any,
    ) -> BenchmarkConfig:
        """Create a paper-quality config for full benchmark runs.

        Parameters
        ----------
        variant : str
            DCM variant: ``"task"``, ``"spectral"``, or ``"rdcm"``.
        method : str
            Inference method.
        **kwargs : Any
            Additional keyword arguments forwarded to the constructor
            (e.g., ``fixtures_dir``, ``guide_type``).

        Returns
        -------
        BenchmarkConfig
            Config with full dataset counts and SVI steps.
        """
        defaults: dict[str, dict[str, int]] = {
            "task": {"n_datasets": 20, "n_svi_steps": 3000},
            "spectral": {"n_datasets": 50, "n_svi_steps": 500},
            "rdcm": {"n_datasets": 50, "n_svi_steps": 500},
            "rdcm_rigid": {"n_datasets": 50, "n_svi_steps": 500},
            "rdcm_sparse": {"n_datasets": 50, "n_svi_steps": 500},
        }
        params = defaults.get(variant, {"n_datasets": 20, "n_svi_steps": 3000})
        return cls(
            variant=variant,
            method=method,
            n_datasets=params["n_datasets"],
            n_svi_steps=params["n_svi_steps"],
            quick=False,
            **kwargs,
        )
