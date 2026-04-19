"""Bilinear-specific metric helpers for Phase 16 RECOV-03..08 acceptance gates.

Consumes the ``run_task_bilinear_svi`` runner output (plan 16-01) and produces
per-seed + aggregate metrics for RECOV-03 (relative A-RMSE), RECOV-04 (magnitude-
masked B-RMSE), RECOV-05 (sign recovery on non-null B), RECOV-06 (coverage-of-zero
on null B), RECOV-07 (identifiability shrinkage std_post/sigma_prior), and RECOV-08
(wall-time ratio vs inline linear baseline).

All functions are pure and operate on torch tensors / python lists. The single
entry point for end-to-end pass/fail computation is ``compute_acceptance_gates``.

References
----------
.planning/REQUIREMENTS.md RECOV-03..08 (acceptance criteria specification)
.planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md Section 4 (formulas)
.planning/phases/16-bilinear-recovery-benchmark/16-CONTEXT.md (forest plot, pass-fail
format)
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np
import torch

# Module constants (from Phase 15 task_dcm_model.B_PRIOR_VARIANCE = 1.0; D1 SPM
# one-state).
SIGMA_PRIOR: float = 1.0
"""Prior standard deviation on B_free elements (sqrt(B_PRIOR_VARIANCE))."""


# RECOV thresholds (from .planning/REQUIREMENTS.md).
RECOV_03_A_RMSE_RATIO_THRESHOLD: float = 1.25
RECOV_04_B_RMSE_THRESHOLD: float = 0.20
RECOV_04_MAGNITUDE_MASK: float = 0.1
RECOV_05_SIGN_RECOVERY_THRESHOLD: float = 0.80
RECOV_06_COVERAGE_THRESHOLD: float = 0.85
# NOTE: The planner spec gave ``0.5 * SIGMA_PRIOR = 0.5`` per D3, but the
# canonical topology has non-zero B magnitudes 0.3 and 0.4 -- both BELOW
# 0.5 -- which means a 0.5 threshold would mis-classify the 2 free elements
# as "null", contradicting the documented intent ("selects the 7 nulls per
# seed", 16-02-PLAN.md R-topology). Use ``RECOV_04_MAGNITUDE_MASK = 0.1``
# for complementary masks: |B|>0.1 -> non-null (RECOV-04/05), |B|<=0.1 ->
# null (RECOV-06). Auto-fix per Deviation Rule 1 (bug in planner-supplied
# threshold value vs documented intent); see SUMMARY.md.
RECOV_06_NULL_MASK: float = 0.1
RECOV_07_SHRINKAGE_SOFT_TARGET: float = 0.7
RECOV_08_WALL_TIME_FLAG_RATIO: float = 10.0


def compute_b_rmse_magnitude(
    B_true: torch.Tensor,
    B_inferred: torch.Tensor,
    *,
    magnitude_threshold: float = RECOV_04_MAGNITUDE_MASK,
) -> float:
    """Magnitude-masked B-RMSE (RECOV-04).

    Root-mean-squared error between ``B_true`` and ``B_inferred`` restricted
    to elements where ``|B_true| > magnitude_threshold``. Returns 0.0 if the
    mask selects zero elements (vacuous).

    Parameters
    ----------
    B_true : torch.Tensor
        Ground-truth B tensor, shape ``(J, N, N)``.
    B_inferred : torch.Tensor
        Inferred posterior-mean B tensor, same shape.
    magnitude_threshold : float, optional
        Mask threshold. Default 0.1 (RECOV-04 specification).

    Returns
    -------
    float
        RMSE restricted to ``|B_true| > threshold`` elements, or 0.0 if vacuous.

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-04.
    """
    mask = torch.abs(B_true) > magnitude_threshold
    if not mask.any():
        return 0.0
    return torch.sqrt(((B_true - B_inferred)[mask] ** 2).mean()).item()


def compute_sign_recovery_nonzero(
    B_true_list: list[torch.Tensor],
    B_inferred_list: list[torch.Tensor],
    *,
    magnitude_threshold: float = RECOV_04_MAGNITUDE_MASK,
) -> float:
    """Pooled sign recovery on |B_true| > threshold elements (RECOV-05).

    For each seed, gate by ``|B_true| > magnitude_threshold`` and count the
    fraction of (seed, element) pairs where ``sign(B_posterior) == sign(B_true)``.
    Aggregation is POOLED per L5: sum of matches over all eligible pairs,
    divided by the total number of eligible pairs.

    Parameters
    ----------
    B_true_list : list of torch.Tensor
        Per-seed ground-truth B tensors, each shape ``(J, N, N)``. All must
        share shape.
    B_inferred_list : list of torch.Tensor
        Per-seed posterior-mean B tensors, same shape as ``B_true_list[i]``.
    magnitude_threshold : float, optional
        Non-null mask threshold. Default 0.1 (RECOV-05).

    Returns
    -------
    float
        Pooled sign recovery fraction in ``[0, 1]``, or 0.0 if no eligible pairs.

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-05; L5 pooled aggregation rationale
    in 16-02-PLAN.md.
    """
    assert len(B_true_list) == len(B_inferred_list), (
        f"Length mismatch: B_true_list={len(B_true_list)} vs "
        f"B_inferred_list={len(B_inferred_list)}"
    )
    total_matches = 0
    total_eligible = 0
    for B_true, B_inferred in zip(B_true_list, B_inferred_list, strict=True):
        mask = torch.abs(B_true) > magnitude_threshold
        if not mask.any():
            continue
        match = (torch.sign(B_inferred) == torch.sign(B_true))[mask]
        total_matches += int(match.sum().item())
        total_eligible += int(mask.sum().item())
    if total_eligible == 0:
        return 0.0
    return total_matches / total_eligible


def compute_coverage_of_zero(
    B_true_list: list[torch.Tensor],
    B_samples_list: list[torch.Tensor],
    *,
    null_threshold: float = RECOV_06_NULL_MASK,
    ci_level: float = 0.95,
) -> float:
    """Pooled coverage-of-zero on |B_true| < null_threshold elements (RECOV-06).

    For each seed: gate by ``|B_true| < null_threshold`` (the null-element mask;
    default 0.5 per D3 convention). Compute the ``ci_level`` quantile band of
    the per-element posterior samples and count the fraction of null (seed,
    element) pairs whose CI contains zero. Pooled aggregation per L5.

    Parameters
    ----------
    B_true_list : list of torch.Tensor
        Per-seed ground-truth B tensors, each shape ``(J, N, N)``.
    B_samples_list : list of torch.Tensor
        Per-seed posterior sample tensors. Each shape ``(S, J, N, N)`` where
        S is the number of draws from the Predictive call. Computes quantile
        along dim 0.
    null_threshold : float, optional
        Null-element mask threshold. Default 0.5 (0.5 * sigma_prior).
    ci_level : float, optional
        Credible-interval level. Default 0.95 (L7).

    Returns
    -------
    float
        Pooled coverage-of-zero fraction in ``[0, 1]``, or 0.0 if no eligible pairs.

    Notes
    -----
    **Mean-field coverage limitation (RECOV-06 primary Phase 16 risk, per
    research N1).** Under ``AutoNormal``, the guide factorizes across sample
    sites, which systematically underestimates posterior correlations
    between ``A`` and ``B`` elements under bilinear coupling (the A-B
    correlation is stronger than under pure linear DCM per
    PITFALLS.md:648). This can depress the pooled coverage estimate below
    the RECOV-06 85% threshold even when the posterior mean is well
    calibrated.

    Per L9: the 85% threshold stays HARD under AutoNormal for v0.3.0. If
    this function reports coverage < 0.85 in the Phase 16 acceptance gate,
    the Phase 16 SUMMARY must record the observed value and flag the
    milestone as blocked pending a v0.3.1 ``AutoLowRankMVN`` fallback tier
    (deferred per CONTEXT.md Deferred Ideas). Do NOT silently lower the
    threshold; the failure is a surfacing gate for v0.3.1 planning, not an
    implementation bug.

    The ``samples`` consumed by this helper are RAW ``B_free_j`` draws
    (shape ``(S, N, N)`` per-modulator from ``task_dcm_model``'s
    ``pyro.sample(f"B_free_{j}", ...)`` site at ``task_dcm_model.py:304``),
    unsqueezed to ``(S, J=1, N, N)`` at the call site. For masked-out
    elements the posterior equals the ``N(0, 1)`` prior, so the CI covers
    zero by construction -- which is precisely what this metric measures
    (the guide's ability to PRESERVE prior uncertainty on unconstrained
    elements).

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-06; L5 pooled aggregation; L7 95% CI level;
    L9 mean-field fallback decision.
    """
    assert len(B_true_list) == len(B_samples_list), (
        f"Length mismatch: {len(B_true_list)} vs {len(B_samples_list)}"
    )
    alpha = (1.0 - ci_level) / 2.0
    total_contains = 0
    total_eligible = 0
    for B_true, B_samples in zip(B_true_list, B_samples_list, strict=True):
        mask = torch.abs(B_true) < null_threshold
        if not mask.any():
            continue
        lo = torch.quantile(B_samples.float(), alpha, dim=0)
        hi = torch.quantile(B_samples.float(), 1.0 - alpha, dim=0)
        contains_zero = (lo <= 0) & (0 <= hi)
        total_contains += int(contains_zero[mask].sum().item())
        total_eligible += int(mask.sum().item())
    if total_eligible == 0:
        return 0.0
    return total_contains / total_eligible


def compute_shrinkage(
    B_std_post: torch.Tensor,
    *,
    sigma_prior: float = SIGMA_PRIOR,
) -> torch.Tensor:
    """Element-wise identifiability shrinkage (RECOV-07, soft target).

    Returns ``B_std_post / sigma_prior``. Lower values indicate stronger
    posterior concentration (better identifiability). Reported per element;
    does NOT block acceptance per RECOV-07.

    Parameters
    ----------
    B_std_post : torch.Tensor
        Posterior standard deviation per B element, shape ``(J, N, N)``.
    sigma_prior : float, optional
        Prior std (constant). Default 1.0 (from ``B_PRIOR_VARIANCE=1.0``, D1).

    Returns
    -------
    torch.Tensor
        Element-wise shrinkage ratio, same shape as input.

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-07 (soft target <= 0.7).
    """
    return B_std_post / sigma_prior


def compute_a_rmse_relative(
    a_rmse_bilinear_list: list[float],
    a_rmse_linear_list: list[float],
) -> dict[str, Any]:
    """Relative A-RMSE threshold check (RECOV-03).

    Compares mean bilinear A-RMSE to mean linear-baseline A-RMSE. The RECOV-03
    threshold is 1.25x: bilinear A-RMSE must NOT exceed 1.25x the linear
    baseline (mitigates Pitfall B13 A-RMSE inflation under Bayesian parameter
    pricing).

    Parameters
    ----------
    a_rmse_bilinear_list : list of float
        Per-seed bilinear A-RMSE values.
    a_rmse_linear_list : list of float
        Per-seed linear-baseline A-RMSE values on the SAME seeds (plan 16-01 L3).

    Returns
    -------
    dict
        Keys: ``mean_bilinear``, ``mean_linear``, ``ratio`` (= bilinear / linear),
        ``threshold`` (= 1.25), ``pass`` (bool).

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-03; .planning/research/v0.3.0/PITFALLS.md B13.
    """
    assert len(a_rmse_bilinear_list) == len(a_rmse_linear_list), (
        f"Length mismatch: bilinear={len(a_rmse_bilinear_list)} vs "
        f"linear={len(a_rmse_linear_list)}"
    )
    mean_bi = float(np.mean(a_rmse_bilinear_list))
    mean_lin = float(np.mean(a_rmse_linear_list))
    ratio = mean_bi / mean_lin if mean_lin > 0 else float("inf")
    return {
        "mean_bilinear": mean_bi,
        "mean_linear": mean_lin,
        "ratio": ratio,
        "threshold": RECOV_03_A_RMSE_RATIO_THRESHOLD,
        "pass": ratio <= RECOV_03_A_RMSE_RATIO_THRESHOLD,
    }


def compute_acceptance_gates(runner_result: dict[str, Any]) -> dict[str, Any]:
    """Compute all 4 RECOV acceptance gates + RECOV-07/08 info rows.

    Single-source-of-truth acceptance computation consumed by (a) the slow
    acceptance-gate test and (b) the plot_acceptance_gates_table figure.

    Parameters
    ----------
    runner_result : dict
        Output of ``run_task_bilinear_svi(config)`` (plan 16-01 contract).
        Required keys: ``a_rmse_bilinear_list``, ``a_rmse_linear_list``,
        ``time_bilinear_list``, ``time_linear_list``, ``posterior_list``
        (per-seed dict with ``B_free_0`` key containing ``mean``, ``std``,
        ``samples`` as nested Python lists / numpy arrays), ``b_true_list``.

    Returns
    -------
    dict
        Keys: ``RECOV-03`` (A-RMSE ratio), ``RECOV-04`` (B-RMSE magnitude),
        ``RECOV-05`` (sign recovery), ``RECOV-06`` (coverage of zero),
        ``RECOV-07`` (shrinkage per non-null element), ``RECOV-08`` (wall-time
        ratio + flag). Each gate value is a dict with at least ``observed``,
        ``threshold``, and ``pass`` keys (RECOV-07/08 have ``flag`` instead
        of ``pass``).

    Raises
    ------
    ValueError
        If ``runner_result`` has ``status='insufficient_data'`` or if any
        required key is missing.

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-03..08.
    """
    if runner_result.get("status") == "insufficient_data":
        raise ValueError(
            f"Cannot compute acceptance gates: runner returned "
            f"insufficient_data (n_success={runner_result.get('n_success')}, "
            f"n_failed={runner_result.get('n_failed')}, "
            f"n_datasets={runner_result.get('n_datasets')})"
        )
    required = [
        "a_rmse_bilinear_list", "a_rmse_linear_list",
        "time_bilinear_list", "time_linear_list",
        "posterior_list", "b_true_list",
    ]
    missing = [k for k in required if k not in runner_result]
    if missing:
        raise ValueError(
            f"runner_result missing required keys: {missing}. "
            f"Got: {sorted(runner_result.keys())}"
        )

    # Reconstruct per-seed tensors from the JSON-ified lists.
    # Per-seed B_true is stored on the per-seed posterior dict
    # (``post['B_true']`` shape (J, N, N)); the runner's parallel
    # ``b_true_list`` is the flattened version.
    B_true_list: list[torch.Tensor] = [
        torch.tensor(post["B_true"], dtype=torch.float64)
        for post in runner_result["posterior_list"]
    ]
    # Per-seed posterior-mean B. posterior["B_free_0"]["mean"] shape (N, N)
    # for J=1 -> wrap in leading dim to shape (J, N, N) = (1, N, N).
    B_inferred_list: list[torch.Tensor] = []
    B_samples_list: list[torch.Tensor] = []
    B_std_list: list[torch.Tensor] = []
    for post in runner_result["posterior_list"]:
        b_mean_jnn = torch.tensor(
            post["B_free_0"]["mean"], dtype=torch.float64,
        ).unsqueeze(0)
        # Samples (S, N, N) -> (S, J=1, N, N) along new dim 1.
        b_samples_sjnn = torch.tensor(
            post["B_free_0"]["samples"], dtype=torch.float64,
        ).unsqueeze(1)
        B_inferred_list.append(b_mean_jnn)
        B_samples_list.append(b_samples_sjnn)
        B_std_list.append(torch.tensor(
            post["B_free_0"]["std"], dtype=torch.float64,
        ).unsqueeze(0))

    # RECOV-03: relative A-RMSE.
    recov_03 = compute_a_rmse_relative(
        runner_result["a_rmse_bilinear_list"],
        runner_result["a_rmse_linear_list"],
    )

    # RECOV-04: magnitude-masked B-RMSE, mean across seeds.
    b_rmse_list = [
        compute_b_rmse_magnitude(b_t, b_i)
        for b_t, b_i in zip(B_true_list, B_inferred_list, strict=True)
    ]
    mean_b_rmse = float(np.mean(b_rmse_list))
    recov_04 = {
        "observed": mean_b_rmse,
        "threshold": RECOV_04_B_RMSE_THRESHOLD,
        "pass": mean_b_rmse <= RECOV_04_B_RMSE_THRESHOLD,
        "per_seed": b_rmse_list,
    }

    # RECOV-05: pooled sign recovery.
    sign_rec = compute_sign_recovery_nonzero(B_true_list, B_inferred_list)
    recov_05 = {
        "observed": sign_rec,
        "threshold": RECOV_05_SIGN_RECOVERY_THRESHOLD,
        "pass": sign_rec >= RECOV_05_SIGN_RECOVERY_THRESHOLD,
    }

    # RECOV-06: pooled coverage of zero (95% CI per L7).
    cov_zero = compute_coverage_of_zero(B_true_list, B_samples_list)
    recov_06 = {
        "observed": cov_zero,
        "threshold": RECOV_06_COVERAGE_THRESHOLD,
        "pass": cov_zero >= RECOV_06_COVERAGE_THRESHOLD,
    }

    # RECOV-07: shrinkage per element (info only; soft target 0.7 non-blocking).
    # Aggregate across seeds: per-element mean of per-seed shrinkage.
    shrinkage_stack = torch.stack([
        compute_shrinkage(b_std) for b_std in B_std_list
    ])  # (n_seeds, J, N, N)
    shrinkage_mean = shrinkage_stack.mean(dim=0)  # (J, N, N)
    # Non-null mask (|B_true| > 0.1) -> the 2 free elements.
    B_true_sample = B_true_list[0]  # all seeds share the same B_true structure
    nonnull_mask = torch.abs(B_true_sample) > RECOV_04_MAGNITUDE_MASK
    recov_07 = {
        "shrinkage_mean": shrinkage_mean.numpy().tolist(),
        "shrinkage_nonnull": shrinkage_mean[nonnull_mask].numpy().tolist(),
        "soft_target": RECOV_07_SHRINKAGE_SOFT_TARGET,
        "all_below_soft_target": bool(
            (shrinkage_mean[nonnull_mask] <= RECOV_07_SHRINKAGE_SOFT_TARGET).all()
        ),
    }

    # RECOV-08: wall-time ratio + 10x flag.
    t_bi = float(mean(runner_result["time_bilinear_list"]))
    t_lin = float(mean(runner_result["time_linear_list"]))
    ratio = t_bi / t_lin if t_lin > 0 else float("inf")
    recov_08 = {
        "time_bilinear": t_bi,
        "time_linear": t_lin,
        "ratio": ratio,
        "flag_over_10x": ratio > RECOV_08_WALL_TIME_FLAG_RATIO,
    }

    return {
        "RECOV-03": recov_03,
        "RECOV-04": recov_04,
        "RECOV-05": recov_05,
        "RECOV-06": recov_06,
        "RECOV-07": recov_07,
        "RECOV-08": recov_08,
        "all_pass": (
            recov_03["pass"]
            and recov_04["pass"]
            and recov_05["pass"]
            and recov_06["pass"]
        ),
    }
