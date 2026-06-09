"""Simulation-Based Calibration (SBC) diagnostics for SBI posteriors.

Implements rank-based SBC validation following Talts et al. (2018):
for each trial, draw theta_true from the prior, simulate an observation,
draw posterior samples conditioned on that observation, and compute
the rank of theta_true among the posterior samples. Under a well-
calibrated posterior, these ranks should be uniformly distributed.

References
----------
Talts, Betancourt, Simpson, Vehtari & Gelman (2018). Validating
    Bayesian inference algorithms with simulation-based calibration.
    arXiv:1804.06788.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def run_sbc_validation(
    posterior: Any,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    prior: Any,
    n_trials: int = 200,
    n_posterior_samples: int = 1000,
) -> dict[str, Any]:
    """Run Simulation-Based Calibration on a trained posterior.

    Parameters
    ----------
    posterior : Any
        Trained sbi posterior object supporting ``.set_default_x()``
        and ``.sample()``.
    simulator : Callable
        Simulator function: ``theta -> x``.
    prior : Any
        Prior distribution supporting ``.sample()``.
    n_trials : int
        Number of SBC trials. Default 200.
    n_posterior_samples : int
        Number of posterior samples per trial. Default 1000.

    Returns
    -------
    dict
        Keys:
        - ``ranks``: int tensor of shape ``(n_trials, n_params)``
          with the rank of theta_true among posterior samples.
        - ``n_posterior_samples``: int, the number of posterior
          samples used per trial.
    """
    ranks_list: list[torch.Tensor] = []

    for i in range(n_trials):
        theta_true = prior.sample()
        x_obs = simulator(theta_true)

        conditioned = posterior.set_default_x(x_obs.to(torch.float32))
        samples = conditioned.sample(
            (n_posterior_samples,)
        ).to(torch.float64)

        # Rank: number of posterior samples less than theta_true
        theta_true_expanded = theta_true.unsqueeze(0)
        rank = (samples < theta_true_expanded).sum(dim=0)
        ranks_list.append(rank)

        if (i + 1) % 50 == 0:
            print(f"    SBC trial {i + 1}/{n_trials}")

    ranks = torch.stack(ranks_list)

    return {
        "ranks": ranks,
        "n_posterior_samples": n_posterior_samples,
    }
