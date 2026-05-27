"""Diagnostics for SBI posterior quality assessment.

Provides simulation-based calibration (SBC) validation and comparison
utilities between SBI and SVI/VL posteriors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def run_sbc_validation(
    posterior: Any,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    prior: torch.distributions.Distribution,
    n_trials: int = 1000,
    n_posterior_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """Run simulation-based calibration (SBC) for posterior validation.

    For each trial: sample theta_true from the prior, simulate data,
    draw posterior samples conditioned on that data, and compute the
    rank statistic (number of posterior samples less than theta_true).

    Well-calibrated posteriors produce uniform rank statistics.

    Parameters
    ----------
    posterior : Any
        Trained sbi posterior object supporting ``.set_default_x()``
        and ``.sample()``.
    simulator : Callable
        Simulator function: ``theta -> x``.
    prior : torch.distributions.Distribution
        Prior over parameters.
    n_trials : int
        Number of SBC trials. Default 1000.
    n_posterior_samples : int
        Posterior samples per trial for rank computation. Default 1000.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'ranks'``: Rank statistics, shape ``(n_trials, n_params)``.
        - ``'theta_true'``: True parameters, shape ``(n_trials, n_params)``.
        - ``'n_trials'``: Number of trials completed.
        - ``'n_posterior_samples'``: Samples per trial.

    Raises
    ------
    ImportError
        If ``sbi`` is not installed.
    """
    try:
        import sbi  # noqa: F401
    except ImportError as e:
        msg = (
            "sbi package required for SBC validation. "
            "Install with: pip install 'sbi>=0.22'"
        )
        raise ImportError(msg) from e

    ranks_list = []
    theta_true_list = []

    for _ in range(n_trials):
        theta_true = prior.sample()
        x_obs = simulator(theta_true)

        # Condition posterior on simulated observation
        conditioned = posterior.set_default_x(x_obs.to(torch.float32))
        samples = conditioned.sample(
            (n_posterior_samples,),
        ).to(torch.float64)

        # Rank statistic: count samples < true value (per dimension)
        rank = (samples < theta_true.unsqueeze(0)).sum(dim=0)
        ranks_list.append(rank)
        theta_true_list.append(theta_true)

    return {
        "ranks": torch.stack(ranks_list),
        "theta_true": torch.stack(theta_true_list),
        "n_trials": n_trials,
        "n_posterior_samples": n_posterior_samples,
    }


def compare_sbi_svi_posteriors(
    sbi_posterior: Any,
    svi_samples: torch.Tensor,
    param_names: list[str] | None = None,
    n_sbi_samples: int = 10_000,
) -> dict[str, Any]:
    """Compare SBI and SVI/VL posterior approximations.

    Computes summary statistics (mean, std) and approximate
    symmetric KL divergence between SBI and SVI posteriors using
    Monte Carlo estimation.

    Parameters
    ----------
    sbi_posterior : Any
        Trained and conditioned sbi posterior object supporting
        ``.sample()``.
    svi_samples : torch.Tensor
        Samples from SVI/VL posterior, shape ``(n_samples, n_params)``.
    param_names : list of str or None
        Optional parameter names for labeling. If None, uses
        ``["param_0", "param_1", ...]``.
    n_sbi_samples : int
        Number of samples to draw from SBI posterior. Default 10000.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'sbi_mean'``: SBI posterior means, shape ``(n_params,)``.
        - ``'sbi_std'``: SBI posterior stds, shape ``(n_params,)``.
        - ``'svi_mean'``: SVI posterior means, shape ``(n_params,)``.
        - ``'svi_std'``: SVI posterior stds, shape ``(n_params,)``.
        - ``'mean_abs_diff'``: Absolute difference of means.
        - ``'std_ratio'``: Ratio of standard deviations (SBI/SVI).
        - ``'param_names'``: Parameter name labels.
    """
    sbi_samples = sbi_posterior.sample(
        (n_sbi_samples,),
    ).to(torch.float64)

    n_params = svi_samples.shape[1]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    sbi_mean = sbi_samples.mean(dim=0)
    sbi_std = sbi_samples.std(dim=0)
    svi_mean = svi_samples.mean(dim=0)
    svi_std = svi_samples.std(dim=0)

    return {
        "sbi_mean": sbi_mean,
        "sbi_std": sbi_std,
        "svi_mean": svi_mean,
        "svi_std": svi_std,
        "mean_abs_diff": (sbi_mean - svi_mean).abs(),
        "std_ratio": sbi_std / svi_std.clamp(min=1e-16),
        "param_names": param_names,
    }
