"""Latent-circuit DCM metrics and acceptance gates (v0.6.0 Phase 20).

Provides latent-circuit-specific metrics that extend the Phase 16 bilinear
recovery infrastructure for the latent-circuit model (Plan 20-04):

- ``compute_trajectory_r_squared``: Held-out R-squared for SYNTH-02
  trajectory reconstruction gate.
- ``compute_elbo_model_selection``: ELBO-based model order selection
  across candidate latent dimensionalities (for future ELBO selection
  experiments).
- ``compute_latent_circuit_acceptance_gates``: Aggregates per-seed
  metrics from the runner into pass/fail acceptance gates with median
  aggregation.
- ``compute_coverage_multi_level``: Multi-level CI coverage (re-exported
  wrapper with tensor-input interface).

Phase 16 bilinear metric functions are reused directly:
``compute_b_rmse_magnitude``, ``compute_sign_recovery_nonzero``,
``compute_coverage_of_zero``, ``compute_shrinkage``.

**Provisional thresholds.** All gate thresholds in this module are
provisional values set for Plan 20-04. They will be empirically
recalibrated in Plan 20-05 using a prior-variance sweep on synthetic
data. The constants are named ``LC_*_THRESHOLD`` and carry docstrings
explaining their provenance. Do NOT use these thresholds as hard gates
before Plan 20-05 calibration.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
    SYNTH-01 (A-RMSE, B-RMSE, sign recovery), SYNTH-02 (trajectory R2).
.planning/phases/20-latent-circuit-forward-model/20-RESEARCH.md
    Prior recalibration rationale (Sections 3, 4).
.planning/REQUIREMENTS.md -- RECOV-03..08 (bilinear acceptance patterns).
"""

from __future__ import annotations

from statistics import median
from typing import Any

import torch

from benchmarks.bilinear_metrics import (
    compute_b_rmse_magnitude,
    compute_coverage_of_zero,
    compute_shrinkage,
    compute_sign_recovery_nonzero,
)

__all__ = [
    "compute_trajectory_r_squared",
    "compute_elbo_model_selection",
    "compute_latent_circuit_acceptance_gates",
    "compute_coverage_multi_level",
    "compute_b_rmse_magnitude",
    "compute_sign_recovery_nonzero",
    "compute_coverage_of_zero",
    "compute_shrinkage",
    "LC_A_RMSE_THRESHOLD",
    "LC_B_RMSE_THRESHOLD",
    "LC_SIGN_RECOVERY_THRESHOLD",
    "LC_CI_COVERAGE_THRESHOLD",
    "LC_TRAJECTORY_R2_THRESHOLD",
]


# ---------------------------------------------------------------------------
# Provisional acceptance thresholds
# ---------------------------------------------------------------------------

LC_A_RMSE_THRESHOLD: float = 0.15
"""Provisional A-RMSE gate for latent-circuit DCM.

0.15 (provisional -- Plan 20-05 calibration pending).

Tighter than the RECOV-03 bilinear BOLD threshold (ratio-based) because
the latent-circuit forward model uses direct observation with identity
C_obs (pitfall LC5 avoided): no hemodynamic smearing means A should be
recovered more precisely than from BOLD data. The absolute value 0.15
corresponds roughly to half the prior std (LC_A_PRIOR_VARIANCE^0.5 = 0.25)
at 2 sigma, appropriate for N=4 chains with off-diagonal strength 0.2.

This value is PROVISIONAL. Plan 20-05 will run a prior-variance sweep on
5+ synthetic RNNs and tighten/loosen each gate based on observed recovery
distributions. Do not treat 0.15 as a hard requirement before calibration.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md (SYNTH-01)
.planning/phases/20-latent-circuit-forward-model/20-05-PLAN.md
"""

LC_B_RMSE_THRESHOLD: float = 0.20
"""Provisional magnitude-masked B-RMSE gate for latent-circuit DCM.

0.20 (same as RECOV-04 for bilinear BOLD). Unchanged because the B prior
variance (LC_B_PRIOR_VARIANCE = 1.0) matches the task-DCM B prior, so the
same absolute threshold applies until Plan 20-05 calibration.

References
----------
.planning/REQUIREMENTS.md -- RECOV-04 (B-RMSE threshold, bilinear BOLD).
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md (SYNTH-01)
"""

LC_SIGN_RECOVERY_THRESHOLD: float = 0.80
"""Provisional pooled sign-recovery gate for non-zero B elements.

0.80 (same as RECOV-05). Same reasoning as LC_B_RMSE_THRESHOLD -- B prior
matches, so the RECOV-05 80% floor is a reasonable provisional gate.

References
----------
.planning/REQUIREMENTS.md -- RECOV-05 (sign recovery, bilinear BOLD).
"""

LC_CI_COVERAGE_THRESHOLD: float = 0.85
"""Provisional 95% CI coverage gate for null B elements.

0.85 (same as RECOV-06). The AutoNormal mean-field limitation (N1 in
bilinear research) applies equally here; 85% is the right provisional
floor before calibration.

References
----------
.planning/REQUIREMENTS.md -- RECOV-06 (coverage-of-zero, bilinear BOLD).
"""

LC_TRAJECTORY_R2_THRESHOLD: float = 0.90
"""Held-out trajectory R-squared gate (SYNTH-02), variance-pooled.

0.90 against the **variance-pooled** R-squared (the default of
``compute_trajectory_r_squared``; see its docstring for why mean-of-per-region
is wrong here).

Calibrated 2026-06-09 on the VL acceptance run (job 56267222, 10 seeds). The
N=4 chain attenuates signal down its length, so the held-out test segment has
~100x more variance in region 0 than in regions 2-3. At SNR=10 the noise floor
caps reconstruction: the **clean (noise-free) trajectory vs the noisy test
segment** scores pooled-R2 = 0.957 -- that is the best any model can achieve.
The VL fit recovers exactly that ceiling (pooled-R2 = 0.957, identical to the
oracle), so 0.90 is set ~0.057 below the achievable ceiling as documented
margin. (The provisional 0.95 was against the mean-of-per-region metric, which
caps at 0.70 even for the *true* parameters -- it was unachievable by
construction; see 20-05-SUMMARY.)

References
----------
.planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md (R2 diagnosis)
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md (SYNTH-02)
"""


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_trajectory_r_squared(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    *,
    pooled: bool = True,
) -> float:
    """R-squared between predicted and observed trajectories (SYNTH-02).

    Two reductions over the ``N`` regions are available; **variance-pooled is
    the default and the correct choice** for latent-circuit trajectories.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted latent-state trajectories, shape ``(T, N)``.
    observed : torch.Tensor
        Observed latent-state trajectories, shape ``(T, N)``.
    pooled : bool, optional
        If ``True`` (default), return the **variance-pooled** R-squared
        (sklearn ``multioutput='variance_weighted'``)::

            R2 = 1 - sum_n SS_res_n / sum_n SS_tot_n

        i.e. residual and total sums of squares are pooled across regions
        *before* the ratio, so each region contributes in proportion to its
        signal variance. If ``False``, return the mean of per-region R-squared
        (``multioutput='uniform_average'``).

    Returns
    -------
    float
        Pooled (or mean) R-squared across regions. ``nan`` if ``observed`` has
        fewer than 2 time points (undefined variance).

    Notes
    -----
    **Why pooled is the default (calibration finding, 2026-06-09).** The N=4
    chain attenuates signal down its length, so the held-out test segment can
    have ~100x more variance in region 0 than in regions 2-3. Mean-of-
    per-region R-squared gives a near-silent region (variance dominated by
    measurement noise) the same weight as an informative one, so two noisy
    tail regions dragged the mean to 0.70 *even for the true parameters* --
    the metric, not the recovery, was failing the gate. Pooling by variance
    judges the model on the regions that carry signal: on the VL acceptance
    run the recovered model reaches the noise-floor ceiling (pooled-R2 0.957)
    while mean-R2 sat at 0.70. See 20-05-SUMMARY.

    ``SS_tot`` is clamped to at least ``1e-12`` to avoid division by zero.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md
        Trajectory-R2 diagnosis (oracle == recovered; metric correction).
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
        SYNTH-02 trajectory R-squared acceptance gate.
    """
    if observed.shape[0] < 2:
        return float("nan")

    predicted = predicted.to(dtype=torch.float64)
    observed = observed.to(dtype=torch.float64)

    obs_mean = observed.mean(dim=0)  # (N,)
    ss_res = ((observed - predicted) ** 2).sum(dim=0)  # (N,)
    ss_tot = ((observed - obs_mean) ** 2).sum(dim=0)   # (N,)

    if pooled:
        total = ss_tot.sum().clamp(min=1e-12)
        return (1.0 - ss_res.sum() / total).item()

    ss_tot = ss_tot.clamp(min=1e-12)
    r2_per_region = 1.0 - ss_res / ss_tot  # (N,)
    return r2_per_region.mean().item()


def compute_elbo_model_selection(
    elbo_dict: dict[int, float],
    *,
    true_n: int | None = None,
    observed_element_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """ELBO-based model selection across candidate models (same data only).

    Selects the candidate with the lowest ELBO loss (= best model fit, since
    Pyro ``run_svi`` returns the negative ELBO as a positive loss -- lower is
    better).

    .. warning::

        **Valid only when every candidate ELBO is computed on IDENTICAL
        observed data.** The ELBO/free-energy bound is *not* comparable across
        models fit to datasets of different size or dimensionality: the
        likelihood sums over all observed elements, so a model with fewer
        observed elements has a systematically smaller loss and would always
        "win". This invalidates the original Phase 20-05 SYNTH-03 design, which
        compared candidates ``N in {2..6}`` by fitting each ``N`` to data of a
        *different* observed dimensionality (``(T, N)``) -- ``min(loss)``
        trivially selected ``N=2``. See decision 20-05-D2 in
        ``20-05-SUMMARY.md``.

        For **latent-dimensionality** selection under ``C_obs = I`` (observed
        dim == latent dim), the comparison is ill-posed by construction and
        cannot be salvaged by rescaling. Use the SPM-aligned approach instead:
        compare the **Variational Laplace free energy** of nested models on the
        same data, and/or **Bayesian Model Reduction** (Phase 23,
        ``pyro_dcm.model_selection.bmr_circuit_selection``; SPM12
        ``spm_dcm_bmr``) to score connectivity structure at fixed ``N``.

        Pass ``observed_element_counts`` to make this function REFUSE an
        invalid cross-data comparison rather than return a silently wrong
        answer.

    Parameters
    ----------
    elbo_dict : dict[int, float]
        Mapping from candidate id (e.g. region count ``N``) to best final ELBO
        loss (``run_svi``'s ``'final_loss'``). Lower is better.
    true_n : int or None, optional
        Ground-truth candidate id (for accuracy evaluation). When provided,
        ``correct`` is ``True`` iff ``selected_n == true_n``.
    observed_element_counts : dict[int, int] or None, optional
        Number of observed scalar elements (``T * N_obs``) each candidate's
        ELBO was computed on. When provided, this function raises if the counts
        differ across candidates -- a fail-loud guard against the
        cross-dimensional comparison bug (20-05-D2). Keys must match
        ``elbo_dict``. When ``None`` (default), no guard is applied and the
        caller is trusted to have used identical data.

    Returns
    -------
    dict
        Keys:

        - ``selected_n`` (int): candidate with the lowest ELBO loss.
        - ``elbos`` (dict): copy of input ``elbo_dict``.
        - ``correct`` (bool or None): whether ``selected_n == true_n``;
          ``None`` when ``true_n`` is not provided.
        - ``best_loss`` (float): minimum loss value.

    Raises
    ------
    ValueError
        If ``elbo_dict`` is empty; if ``observed_element_counts`` is provided
        and its keys do not match ``elbo_dict``; or if the provided element
        counts are not all equal (invalid cross-data comparison).

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md
        Decision 20-05-D2 (why cross-dimensional ELBO selection is invalid).
    .planning/REFERENCES.md -- REF-070 (Friston & Penny 2011, BMR).
    """
    if not elbo_dict:
        raise ValueError(
            "elbo_dict must be non-empty; got empty dict. "
            "Provide at least one {N: loss} entry."
        )

    if observed_element_counts is not None:
        if set(observed_element_counts) != set(elbo_dict):
            raise ValueError(
                "observed_element_counts keys must match elbo_dict keys; got "
                f"{sorted(observed_element_counts)} vs {sorted(elbo_dict)}."
            )
        unique_counts = set(observed_element_counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                "ELBO model selection requires identical observed data across "
                "candidates, but observed_element_counts differ: "
                f"{dict(sorted(observed_element_counts.items()))}. The ELBO "
                "scales with the number of observed elements, so comparing "
                "these losses is invalid (decision 20-05-D2) and would favour "
                "the lowest-dimensional candidate regardless of fit. Use "
                "Variational Laplace free energy on identical data and/or "
                "Bayesian Model Reduction (bmr_circuit_selection) instead."
            )

    selected_n = min(elbo_dict, key=lambda k: elbo_dict[k])
    best_loss = elbo_dict[selected_n]
    correct: bool | None = None
    if true_n is not None:
        correct = selected_n == true_n

    return {
        "selected_n": selected_n,
        "elbos": dict(elbo_dict),
        "correct": correct,
        "best_loss": best_loss,
    }


def compute_coverage_multi_level(
    samples: torch.Tensor,
    true_value: torch.Tensor,
    levels: list[float] | None = None,
) -> dict[float, float]:
    """Empirical CI coverage at multiple credible interval levels.

    For each CI level, computes the lower and upper quantile bounds of
    the sample distribution via ``torch.quantile`` and checks what
    fraction of ``true_value`` elements fall within the interval.

    Parameters
    ----------
    samples : torch.Tensor
        Posterior samples, shape ``(S, ...)`` where S is the number of
        samples. Flattened internally to ``(S, D)`` for computation.
    true_value : torch.Tensor
        Ground-truth values, shape ``(...)`` matching ``samples[0]``.
        Flattened internally to ``(D,)``.
    levels : list of float or None, optional
        CI levels to evaluate. Default ``[0.50, 0.75, 0.90, 0.95]``.

    Returns
    -------
    dict[float, float]
        Mapping from CI level to coverage fraction in ``[0.0, 1.0]``.

    Notes
    -----
    Uses empirical quantiles (not z-scores) so the result is accurate
    for non-Gaussian posteriors (e.g., AutoIAF). For AutoNormal, z-score
    and quantile methods agree closely.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
        SYNTH-01 coverage gate.
    """
    if levels is None:
        levels = [0.50, 0.75, 0.90, 0.95]

    # Flatten to 2D for quantile computation.
    S = samples.shape[0]
    samples_2d = samples.reshape(S, -1).float()
    true_flat = true_value.reshape(-1).float()

    result: dict[float, float] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = torch.quantile(samples_2d, alpha, dim=0)
        hi = torch.quantile(samples_2d, 1.0 - alpha, dim=0)
        in_ci = (true_flat >= lo) & (true_flat <= hi)
        result[level] = in_ci.float().mean().item()
    return result


def compute_latent_circuit_acceptance_gates(
    runner_results: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed metrics into latent-circuit acceptance gates.

    Each gate passes if the MEDIAN across seeds meets its threshold.
    Median aggregation is more robust than mean for small seed counts
    (N=3 quick, N=10+ full) and matches the bilinear benchmark pattern.

    Parameters
    ----------
    runner_results : list of dict
        Per-seed dicts from ``run_latent_circuit_recovery``. Each must
        contain the following float keys:

        - ``a_rmse``: A-matrix RMSE (SYNTH-01).
        - ``b_rmse``: Magnitude-masked B-RMSE (SYNTH-01).
        - ``sign_recovery``: Fraction of non-zero B elements with
          correct sign (SYNTH-01).
        - ``ci_coverage_95``: 95% CI coverage on null B elements
          (SYNTH-01).
        - ``trajectory_r_squared``: Held-out trajectory R-squared
          (SYNTH-02).

        Optional but logged if present:

        - ``shrinkage_A``: Mean shrinkage ratio for A posterior.
        - ``shrinkage_B``: Mean shrinkage ratio for B posterior.

    thresholds : dict of str to float, or None, optional
        Override any subset of the default provisional thresholds::

            {
                "a_rmse": LC_A_RMSE_THRESHOLD,
                "b_rmse": LC_B_RMSE_THRESHOLD,
                "sign_recovery": LC_SIGN_RECOVERY_THRESHOLD,
                "ci_coverage_95": LC_CI_COVERAGE_THRESHOLD,
                "trajectory_r_squared": LC_TRAJECTORY_R2_THRESHOLD,
            }

    Returns
    -------
    dict
        Keys:

        - ``gates`` (dict): per-gate dicts with ``observed_median``,
          ``threshold``, and ``pass`` (bool).
        - ``per_seed`` (dict): per-gate lists of per-seed values.
        - ``all_pass`` (bool): True iff all 5 gates pass.
        - ``n_seeds`` (int): number of seed results aggregated.
        - ``thresholds_used`` (dict): actual thresholds applied.

    Raises
    ------
    ValueError
        If ``runner_results`` is empty or any required key is missing
        from all seed dicts.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
        SYNTH-01, SYNTH-02 acceptance criteria.
    """
    if not runner_results:
        raise ValueError(
            "runner_results must be non-empty; cannot compute gates from "
            "an empty list."
        )

    # Required per-seed keys and their gate sense (lower-is-better vs
    # higher-is-better).
    gate_config: dict[str, tuple[str, float, bool]] = {
        # key: (default_threshold, higher_is_better)
        "a_rmse": (
            "a_rmse",
            LC_A_RMSE_THRESHOLD,
            False,  # lower is better
        ),
        "b_rmse": (
            "b_rmse",
            LC_B_RMSE_THRESHOLD,
            False,
        ),
        "sign_recovery": (
            "sign_recovery",
            LC_SIGN_RECOVERY_THRESHOLD,
            True,  # higher is better
        ),
        "ci_coverage_95": (
            "ci_coverage_95",
            LC_CI_COVERAGE_THRESHOLD,
            True,
        ),
        "trajectory_r_squared": (
            "trajectory_r_squared",
            LC_TRAJECTORY_R2_THRESHOLD,
            True,
        ),
    }

    # Merge with caller overrides.
    effective_thresholds: dict[str, float] = {
        name: cfg[1] for name, cfg in gate_config.items()
    }
    if thresholds:
        for k, v in thresholds.items():
            if k in effective_thresholds:
                effective_thresholds[k] = v

    # Collect per-seed values for each gate.
    per_seed: dict[str, list[float]] = {name: [] for name in gate_config}
    for i, seed_dict in enumerate(runner_results):
        for name in gate_config:
            if name not in seed_dict:
                raise ValueError(
                    f"runner_results[{i}] is missing required key {name!r}. "
                    f"Got keys: {sorted(seed_dict.keys())}."
                )
            per_seed[name].append(float(seed_dict[name]))

    # Compute gates: median across seeds.
    gates: dict[str, dict[str, Any]] = {}
    all_pass = True
    for name, (_, _, higher_is_better) in gate_config.items():
        vals = per_seed[name]
        obs_median = median(vals)
        thresh = effective_thresholds[name]
        passed = (
            obs_median >= thresh if higher_is_better
            else obs_median <= thresh
        )
        gates[name] = {
            "observed_median": obs_median,
            "threshold": thresh,
            "pass": bool(passed),
            "higher_is_better": higher_is_better,
        }
        if not passed:
            all_pass = False

    return {
        "gates": gates,
        "per_seed": per_seed,
        "all_pass": all_pass,
        "n_seeds": len(runner_results),
        "thresholds_used": dict(effective_thresholds),
    }
