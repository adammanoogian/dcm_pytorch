"""Latent circuit DCM metric helpers for Phase 20 acceptance gates.

Extends the v0.3.0 bilinear benchmark pattern (``bilinear_metrics.py``) with
trajectory R-squared, ELBO model selection, and multi-level CI coverage
metrics. Reuses ``compute_b_rmse_magnitude``, ``compute_sign_recovery_nonzero``,
``compute_coverage_of_zero``, and ``compute_shrinkage`` from the bilinear
metrics module.

Entry point for end-to-end pass/fail computation:
``compute_latent_circuit_acceptance_gates``.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md (spec)
.planning/REQUIREMENTS-v0.6.0.md SYNTH-01..03 (acceptance criteria)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from benchmarks.bilinear_metrics import (
    compute_b_rmse_magnitude,
    compute_coverage_of_zero,
    compute_shrinkage,
    compute_sign_recovery_nonzero,
)

# Default acceptance thresholds (provisional; calibrated in Plan 20-05).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "a_rmse_threshold": 0.15,
    "b_rmse_threshold": 0.20,
    "sign_recovery_threshold": 0.80,
    "ci_coverage_threshold": 0.85,
    "trajectory_r_squared_threshold": 0.95,
}


def compute_trajectory_r_squared(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> float:
    """Per-region R-squared averaged across regions.

    Computes the coefficient of determination (R-squared) for each region
    independently, then returns the mean across all regions. R-squared is
    defined as ``1 - SS_res / SS_tot``, where ``SS_tot`` is clamped to a
    minimum of ``1e-12`` to avoid division by zero for constant signals.

    Parameters
    ----------
    predicted : torch.Tensor
        Model-predicted trajectories, shape ``(T, N)`` where T is time
        points and N is number of regions.
    observed : torch.Tensor
        Ground-truth observed trajectories, same shape ``(T, N)``.

    Returns
    -------
    float
        Mean R-squared across all N regions. Values close to 1.0
        indicate excellent trajectory reconstruction.

    Notes
    -----
    Negative R-squared values are possible (and meaningful) when the
    model predictions are worse than a constant mean predictor. These
    are NOT clamped to zero -- they indicate genuine fitting failure.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md Task 1.
    """
    # Residuals: (T, N)
    residuals = observed - predicted
    ss_res = (residuals**2).sum(dim=0)  # shape (N,)

    # Total variance: (T, N) centered
    obs_mean = observed.mean(dim=0, keepdim=True)  # (1, N)
    ss_tot = ((observed - obs_mean) ** 2).sum(dim=0)  # shape (N,)

    # Clamp SS_tot to avoid division by zero for constant signals.
    ss_tot_safe = torch.clamp(ss_tot, min=1e-12)

    # Per-region R-squared
    r_squared_per_region = 1.0 - ss_res / ss_tot_safe  # shape (N,)

    return float(r_squared_per_region.mean().item())


def compute_elbo_model_selection(
    elbo_dict: dict[int, float],
    *,
    true_n: int | None = None,
) -> dict[str, Any]:
    """Select the best model dimension via ELBO comparison.

    Given a dictionary mapping candidate latent dimension N to the final
    ELBO loss (lower = better fit), selects the model with the lowest
    loss. Optionally checks whether the selection matches the true
    dimension.

    Parameters
    ----------
    elbo_dict : dict[int, float]
        Mapping ``{N: loss_float}`` where lower loss indicates better
        model fit. Entries with ``NaN`` or ``Inf`` loss are treated as
        failed fits and excluded.
    true_n : int or None, optional
        True number of latent dimensions. If provided, the result
        includes whether the selection was correct.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``'selected_n'``: int, the N with lowest loss.
        - ``'elbos'``: dict[int, float], cleaned input dict (NaN/Inf
          removed).
        - ``'correct'``: bool or None. True if ``selected_n == true_n``,
          None if ``true_n`` was not provided.

    Raises
    ------
    ValueError
        If ``elbo_dict`` is empty or all entries are NaN/Inf.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md Task 1.
    """
    # Filter out NaN/Inf entries.
    valid = {
        n: loss
        for n, loss in elbo_dict.items()
        if np.isfinite(loss)
    }
    if not valid:
        raise ValueError(
            "compute_elbo_model_selection: all entries in elbo_dict "
            f"are NaN/Inf. Got: {elbo_dict}"
        )

    selected_n = min(valid, key=lambda n: valid[n])

    correct: bool | None = None
    if true_n is not None:
        correct = selected_n == true_n

    return {
        "selected_n": selected_n,
        "elbos": valid,
        "correct": correct,
    }


def compute_latent_circuit_acceptance_gates(
    runner_results: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute all latent circuit DCM acceptance gates.

    Evaluates whether parameter recovery meets provisional thresholds
    across multiple seeds. Gate passes if the MEDIAN across seeds meets
    the threshold.

    Parameters
    ----------
    runner_results : dict[str, Any]
        Output of ``run_latent_circuit_recovery``. Required keys:

        - ``'per_seed_results'``: list of per-seed dicts, each with keys:
          ``'a_rmse'``, ``'b_rmse'``, ``'sign_recovery'``,
          ``'ci_coverage'``, ``'trajectory_r_squared'``.
        - ``'ground_truth'``: dict with ``'B_true'`` tensor.

    thresholds : dict[str, float] or None, optional
        Override thresholds. Uses ``DEFAULT_THRESHOLDS`` when None.
        Keys: ``a_rmse_threshold``, ``b_rmse_threshold``,
        ``sign_recovery_threshold``, ``ci_coverage_threshold``,
        ``trajectory_r_squared_threshold``.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``'gates'``: dict per gate name with ``pass``, ``median``,
          ``threshold``, ``per_seed`` values.
        - ``'medians'``: dict of median values per metric.
        - ``'overall_pass'``: bool, True if ALL gates pass.
        - ``'n_seeds'``: int, number of valid seeds.

    Raises
    ------
    ValueError
        If ``runner_results`` has no valid seed results.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md Task 1.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        thr.update(thresholds)

    per_seed = runner_results["per_seed_results"]
    if not per_seed:
        raise ValueError(
            "compute_latent_circuit_acceptance_gates: "
            "runner_results['per_seed_results'] is empty."
        )

    # Collect per-seed metric values.
    a_rmse_vals = [s["a_rmse"] for s in per_seed]
    b_rmse_vals = [s["b_rmse"] for s in per_seed]
    sign_rec_vals = [s["sign_recovery"] for s in per_seed]
    ci_cov_vals = [s["ci_coverage"] for s in per_seed]
    traj_r2_vals = [s["trajectory_r_squared"] for s in per_seed]

    # Compute medians.
    med_a_rmse = float(np.median(a_rmse_vals))
    med_b_rmse = float(np.median(b_rmse_vals))
    med_sign_rec = float(np.median(sign_rec_vals))
    med_ci_cov = float(np.median(ci_cov_vals))
    med_traj_r2 = float(np.median(traj_r2_vals))

    # Evaluate gates (lower-is-better for RMSE, higher for others).
    gate_a_rmse = {
        "pass": med_a_rmse <= thr["a_rmse_threshold"],
        "median": med_a_rmse,
        "threshold": thr["a_rmse_threshold"],
        "per_seed": a_rmse_vals,
    }
    gate_b_rmse = {
        "pass": med_b_rmse <= thr["b_rmse_threshold"],
        "median": med_b_rmse,
        "threshold": thr["b_rmse_threshold"],
        "per_seed": b_rmse_vals,
    }
    gate_sign_rec = {
        "pass": med_sign_rec >= thr["sign_recovery_threshold"],
        "median": med_sign_rec,
        "threshold": thr["sign_recovery_threshold"],
        "per_seed": sign_rec_vals,
    }
    gate_ci_cov = {
        "pass": med_ci_cov >= thr["ci_coverage_threshold"],
        "median": med_ci_cov,
        "threshold": thr["ci_coverage_threshold"],
        "per_seed": ci_cov_vals,
    }
    gate_traj_r2 = {
        "pass": med_traj_r2 >= thr["trajectory_r_squared_threshold"],
        "median": med_traj_r2,
        "threshold": thr["trajectory_r_squared_threshold"],
        "per_seed": traj_r2_vals,
    }

    overall_pass = (
        gate_a_rmse["pass"]
        and gate_b_rmse["pass"]
        and gate_sign_rec["pass"]
        and gate_ci_cov["pass"]
        and gate_traj_r2["pass"]
    )

    return {
        "gates": {
            "a_rmse": gate_a_rmse,
            "b_rmse": gate_b_rmse,
            "sign_recovery": gate_sign_rec,
            "ci_coverage": gate_ci_cov,
            "trajectory_r_squared": gate_traj_r2,
        },
        "medians": {
            "a_rmse": med_a_rmse,
            "b_rmse": med_b_rmse,
            "sign_recovery": med_sign_rec,
            "ci_coverage": med_ci_cov,
            "trajectory_r_squared": med_traj_r2,
        },
        "overall_pass": overall_pass,
        "n_seeds": len(per_seed),
    }


def compute_coverage_multi_level(
    samples: torch.Tensor,
    true_value: torch.Tensor,
    levels: list[float] | None = None,
) -> dict[str, float]:
    """CI coverage at multiple credible interval levels.

    For each confidence level, computes the fraction of elements whose
    true value falls within the corresponding symmetric credible
    interval derived from the posterior samples.

    Parameters
    ----------
    samples : torch.Tensor
        Posterior samples, shape ``(S, *param_shape)`` where S is the
        number of Monte Carlo draws and ``*param_shape`` is the shape
        of the parameter (e.g., ``(N, N)`` for A matrix).
    true_value : torch.Tensor
        Ground-truth parameter, shape ``(*param_shape)``. Must be
        broadcastable against ``samples[0]``.
    levels : list of float or None, optional
        Confidence levels to evaluate. Default ``[0.50, 0.75, 0.90,
        0.95]``.

    Returns
    -------
    dict[str, float]
        Keys are formatted as ``'coverage_{level}'`` (e.g.,
        ``'coverage_0.95'``), values are the fraction of elements
        covered at that level.

    Notes
    -----
    This function operates element-wise: for each element in
    ``param_shape``, it computes the quantile band from the ``S``
    posterior samples and checks whether the true value falls within.
    The returned coverage is the fraction of ALL elements that are
    covered.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md Task 1.
    """
    if levels is None:
        levels = [0.50, 0.75, 0.90, 0.95]

    result: dict[str, float] = {}
    total_elements = true_value.numel()

    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = torch.quantile(samples.float(), alpha, dim=0)
        hi = torch.quantile(samples.float(), 1.0 - alpha, dim=0)

        covered = (lo <= true_value.float()) & (true_value.float() <= hi)
        coverage_frac = float(covered.sum().item()) / total_elements
        result[f"coverage_{level}"] = coverage_frac

    return result
