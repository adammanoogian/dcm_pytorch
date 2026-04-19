"""Consolidated benchmark metrics for DCM parameter recovery evaluation.

Provides authoritative implementations of RMSE, Pearson correlation,
coverage (from CI bounds and from posterior samples), and amortization
gap. These replace duplicate helper functions previously scattered across
test_task_dcm_recovery.py, test_spectral_dcm_recovery.py, and
test_amortized_benchmark.py.

All functions use torch tensors (no numpy dependency) and return Python
floats for easy serialization.
"""

from __future__ import annotations

import torch


def compute_rmse(
    A_true: torch.Tensor,
    A_inferred: torch.Tensor,
) -> float:
    """Root mean squared error between two tensors.

    Parameters
    ----------
    A_true : torch.Tensor
        Ground-truth tensor of any shape.
    A_inferred : torch.Tensor
        Inferred tensor of matching shape.

    Returns
    -------
    float
        Root mean squared error.
    """
    return torch.sqrt(torch.mean((A_true - A_inferred) ** 2)).item()


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1D tensors.

    Manual implementation to avoid numpy.corrcoef process abort on
    Windows (see STATE.md decision: "Manual Pearson correlation over
    np.corrcoef").

    Parameters
    ----------
    x : torch.Tensor
        First 1D tensor.
    y : torch.Tensor
        Second 1D tensor of same length.

    Returns
    -------
    float
        Pearson correlation coefficient. Returns 0.0 if either input
        has zero variance.
    """
    x_mean = x.mean()
    y_mean = y.mean()
    xd = x - x_mean
    yd = y - y_mean
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()


def compute_coverage_from_ci(
    A_true: torch.Tensor,
    A_lo: torch.Tensor,
    A_hi: torch.Tensor,
) -> float:
    """Coverage fraction from credible interval bounds.

    Computes the fraction of elements where the true value falls within
    the interval [A_lo, A_hi].

    Parameters
    ----------
    A_true : torch.Tensor
        Ground-truth tensor of any shape.
    A_lo : torch.Tensor
        Lower bound tensor of matching shape.
    A_hi : torch.Tensor
        Upper bound tensor of matching shape.

    Returns
    -------
    float
        Fraction of elements covered, in [0.0, 1.0].
    """
    in_ci = (A_true >= A_lo) & (A_true <= A_hi)
    return in_ci.float().mean().item()


def compute_coverage_from_samples(
    true_vals: torch.Tensor,
    samples: torch.Tensor,
    ci_level: float = 0.90,
) -> float:
    """Coverage from posterior samples using z-score CI.

    Constructs a symmetric credible interval from the sample mean and
    standard deviation, then computes what fraction of true values fall
    within the interval.

    Parameters
    ----------
    true_vals : torch.Tensor
        Ground-truth values, shape ``(D,)`` or ``(N, N)``.
    samples : torch.Tensor
        Posterior samples, shape ``(S, D)`` or ``(S, N, N)`` where S is
        the number of samples.
    ci_level : float, optional
        Credible interval level. Default 0.90 (90% CI uses z=1.645).

    Returns
    -------
    float
        Fraction of elements covered, in [0.0, 1.0].

    Notes
    -----
    Uses z-score multipliers for common CI levels:
    - 0.90 -> z = 1.645
    - 0.95 -> z = 1.960
    - 0.99 -> z = 2.576
    For other levels, uses the normal inverse CDF approximation.
    """
    # Common z-scores to avoid scipy dependency
    z_table: dict[float, float] = {
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }

    if ci_level in z_table:
        z = z_table[ci_level]
    else:
        # Rational approximation of normal inverse CDF
        # Abramowitz and Stegun formula 26.2.23
        p = (1.0 + ci_level) / 2.0
        t = (-2.0 * torch.tensor(1.0 - p).log()).sqrt().item()
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
            1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
        )

    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)

    lower = sample_mean - z * sample_std
    upper = sample_mean + z * sample_std

    in_ci = (true_vals >= lower) & (true_vals <= upper)
    return in_ci.float().mean().item()


def compute_coverage_multi_level(
    true_vals: torch.Tensor,
    samples: torch.Tensor,
    ci_levels: list[float] | None = None,
) -> dict[float, float]:
    """Coverage at multiple credible interval levels using empirical quantiles.

    For each CI level, computes lower and upper bounds from the sample
    distribution via ``torch.quantile`` (not z-scores), which gives
    accurate intervals for non-Gaussian posteriors (e.g., AutoIAF).

    Parameters
    ----------
    true_vals : torch.Tensor
        Ground-truth values, shape ``(D,)`` where D is the number of
        parameters.
    samples : torch.Tensor
        Posterior samples, shape ``(S, D)`` where S is the number of
        samples.
    ci_levels : list[float] or None, optional
        CI levels to evaluate. Default ``[0.50, 0.75, 0.90, 0.95]``.

    Returns
    -------
    dict[float, float]
        Mapping from CI level to coverage fraction in [0, 1].

    Notes
    -----
    Uses empirical quantiles rather than z-scores for accuracy with
    non-Gaussian guide families (research Section 3.2).
    """
    if ci_levels is None:
        ci_levels = [0.50, 0.75, 0.90, 0.95]

    result: dict[float, float] = {}
    for level in ci_levels:
        alpha = (1.0 - level) / 2.0
        lo = torch.quantile(samples.float(), alpha, dim=0)
        hi = torch.quantile(samples.float(), 1.0 - alpha, dim=0)
        in_ci = (true_vals.float() >= lo) & (true_vals.float() <= hi)
        result[level] = in_ci.float().mean().item()
    return result


def compute_coverage_by_param_type(
    A_true: torch.Tensor,
    samples: torch.Tensor,
    ci_levels: list[float] | None = None,
) -> dict[str, dict[float, float]]:
    """Coverage split by diagonal vs off-diagonal A elements.

    Parameters
    ----------
    A_true : torch.Tensor
        Ground-truth connectivity matrix, shape ``(N, N)``.
    samples : torch.Tensor
        Posterior samples, shape ``(S, N, N)`` where S is the number
        of samples.
    ci_levels : list[float] or None, optional
        CI levels to evaluate. Default ``[0.50, 0.75, 0.90, 0.95]``.

    Returns
    -------
    dict[str, dict[float, float]]
        Keys ``"all"``, ``"diagonal"``, ``"off_diagonal"``, each
        mapping CI level to coverage fraction.
    """
    N = A_true.shape[0]
    S = samples.shape[0]

    diag_mask = torch.eye(N, dtype=torch.bool)
    offdiag_mask = ~diag_mask

    # Flatten for compute_coverage_multi_level
    all_cov = compute_coverage_multi_level(
        A_true.flatten(),
        samples.reshape(S, -1),
        ci_levels=ci_levels,
    )

    # Diagonal elements
    diag_true = A_true[diag_mask]  # shape (N,)
    diag_samples = samples[:, diag_mask]  # shape (S, N)
    diag_cov = compute_coverage_multi_level(
        diag_true, diag_samples, ci_levels=ci_levels,
    )

    # Off-diagonal elements
    offdiag_true = A_true[offdiag_mask]  # shape (N*(N-1),)
    offdiag_samples = samples[:, offdiag_mask]  # shape (S, N*(N-1))
    offdiag_cov = compute_coverage_multi_level(
        offdiag_true, offdiag_samples, ci_levels=ci_levels,
    )

    return {
        "all": all_cov,
        "diagonal": diag_cov,
        "off_diagonal": offdiag_cov,
    }


def compute_summary_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics with median and IQR.

    Parameters
    ----------
    values : list[float]
        List of scalar values.

    Returns
    -------
    dict[str, float]
        Keys: ``"median"``, ``"q25"``, ``"q75"``, ``"mean"``,
        ``"std"``.

    Notes
    -----
    Reports median + IQR per STATE.md risk P12 (avoid mean-only
    summaries that hide distributional shape).
    """
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "median": torch.median(t).item(),
        "q25": torch.quantile(t, 0.25).item(),
        "q75": torch.quantile(t, 0.75).item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
    }


def compute_amortization_gap(
    elbo_svi: float,
    elbo_amortized: float,
) -> dict[str, float]:
    """Compute amortization gap between per-subject SVI and amortized ELBO.

    The amortization gap measures how much inference quality is lost by
    using an amortized (shared) guide instead of per-subject optimization.

    Parameters
    ----------
    elbo_svi : float
        ELBO from per-subject SVI (typically tighter / lower loss).
    elbo_amortized : float
        ELBO from amortized guide (typically looser / higher loss).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``"absolute_gap"``: elbo_amortized - elbo_svi
        - ``"relative_gap"``: absolute_gap / |elbo_svi|

    Notes
    -----
    Positive gap means amortized is worse (higher loss). In Pyro,
    ELBO loss is the negative ELBO, so lower is better.
    """
    absolute_gap = elbo_amortized - elbo_svi
    denom = abs(elbo_svi) if abs(elbo_svi) > 1e-15 else 1.0
    relative_gap = absolute_gap / denom

    return {
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
    }
