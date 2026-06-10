"""Bayesian Model Reduction for DCM.

Implements post hoc Bayesian model selection from a single full-model
inversion, following [REF-070] Friston & Penny (2011).
"""

from __future__ import annotations

import itertools
import logging
import warnings

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "bayesian_model_reduction",
    "bmr_circuit_selection",
    "enumerate_reduced_models",
    "make_reduced_prior_zero_connection",
    "rank_connections",
    "temper_vl_posterior",
]


def bayesian_model_reduction(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    reduced_prior_mean: torch.Tensor,
    reduced_prior_cov: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Analytically compute reduced posterior and change in log evidence.

    Given the posterior from a full-model inversion and a reduced prior,
    compute the reduced posterior and the change in log model evidence
    (delta F) without re-inverting the model.

    Implements [REF-070] Eq. 4-8 (Friston & Penny, 2011).

    Parameters
    ----------
    posterior_mean : torch.Tensor, shape (D,)
        Mean of the full-model posterior, mu_f.
    posterior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model posterior, Sigma_f.
    prior_mean : torch.Tensor, shape (D,)
        Mean of the full-model prior, mu_0.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model prior, Sigma_0.
    reduced_prior_mean : torch.Tensor, shape (D,)
        Mean of the reduced-model prior, mu_r0.
    reduced_prior_cov : torch.Tensor, shape (D, D)
        Covariance of the reduced-model prior, Sigma_r0.

    Returns
    -------
    delta_f : float
        Change in log model evidence (positive favours reduced model).
    reduced_posterior_mean : torch.Tensor, shape (D,)
        Mean of the reduced posterior, mu_r.
    reduced_posterior_cov : torch.Tensor, shape (D, D)
        Covariance of the reduced posterior, Sigma_r.

    Notes
    -----
    All input tensors are cast to float64 internally for numerical
    stability. If the reduced posterior covariance is not positive
    definite, the function returns ``delta_f = -inf`` with a warning.

    References
    ----------
    Friston, K. J. & Penny, W. D. (2011). Post hoc Bayesian model
    selection. NeuroImage, 56(4), 2089-2099.
    """
    # Cast to float64 for numerical stability
    mu_f = posterior_mean.to(torch.float64)
    sigma_f = posterior_cov.to(torch.float64)
    mu_0 = prior_mean.to(torch.float64)
    sigma_0 = prior_cov.to(torch.float64)
    mu_r0 = reduced_prior_mean.to(torch.float64)
    sigma_r0 = reduced_prior_cov.to(torch.float64)

    # Precision matrices via solve (avoid inverse)
    eye = torch.eye(mu_f.shape[0], dtype=torch.float64)
    sigma_f_inv = torch.linalg.solve(sigma_f, eye)
    sigma_0_inv = torch.linalg.solve(sigma_0, eye)
    sigma_r0_inv = torch.linalg.solve(sigma_r0, eye)

    # -----------------------------------------------------------------
    # Step 1: Reduced posterior via Bayes rule  [REF-070] Eq. 4-5
    # -----------------------------------------------------------------
    sigma_r_post_inv = sigma_f_inv + sigma_r0_inv - sigma_0_inv

    # Check positive definiteness of reduced posterior precision
    try:
        sigma_r_post = torch.linalg.solve(sigma_r_post_inv, eye)
        # Verify symmetry and positive definiteness
        sigma_r_post = 0.5 * (sigma_r_post + sigma_r_post.T)
        torch.linalg.cholesky(sigma_r_post)
    except torch.linalg.LinAlgError:
        warnings.warn(
            "Reduced posterior covariance is not positive definite. "
            "Returning delta_F = -inf.",
            stacklevel=2,
        )
        d = mu_f.shape[0]
        return (
            float("-inf"),
            torch.full_like(mu_f, float("nan")),
            torch.full((d, d), float("nan"), dtype=torch.float64),
        )

    info_vec = sigma_f_inv @ mu_f + sigma_r0_inv @ mu_r0 - sigma_0_inv @ mu_0
    mu_r_post = sigma_r_post @ info_vec

    # -----------------------------------------------------------------
    # Step 2: Change in log evidence  [REF-070] Eq. 6-8
    #
    # Laplace approximation at the full posterior mean:
    #   delta_F = log p(mu_f | m_r) - log p(mu_f | m_f)
    #           + 0.5 * [log|Sigma_r_post| - log|Sigma_f|]
    #
    # Expanding the Gaussian log-prior evaluations and cancelling
    # the common D/2 * log(2*pi) terms:
    #   delta_F = 0.5 * [log|Sigma_r_post| - log|Sigma_f|
    #                  + log|Sigma_0| - log|Sigma_r0|
    #                  - (mu_f - mu_r0)' P_r0 (mu_f - mu_r0)
    #                  + (mu_f - mu_0)' P_0 (mu_f - mu_0)]
    # -----------------------------------------------------------------
    # Log-determinants via slogdet (covariance matrices)
    _, logdet_sigma_r_post = torch.linalg.slogdet(sigma_r_post)
    _, logdet_sigma_f = torch.linalg.slogdet(sigma_f)
    _, logdet_sigma_0 = torch.linalg.slogdet(sigma_0)
    _, logdet_sigma_r0 = torch.linalg.slogdet(sigma_r0)

    # Quadratic terms (prior mismatch penalty)
    diff_reduced = mu_f - mu_r0
    diff_full = mu_f - mu_0
    quad_reduced = diff_reduced @ (sigma_r0_inv @ diff_reduced)
    quad_full = diff_full @ (sigma_0_inv @ diff_full)

    delta_f = 0.5 * (
        logdet_sigma_r_post
        - logdet_sigma_f
        + logdet_sigma_0
        - logdet_sigma_r0
        - quad_reduced
        + quad_full
    )

    return float(delta_f.item()), mu_r_post, sigma_r_post


def make_reduced_prior_zero_connection(
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    indices: list[int],
    shrinkage_variance: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create reduced priors that shrink specified parameters to zero.

    Generates a reduced prior where the parameters at the given indices
    have mean zero and very small variance (effectively fixing them at
    zero), while all other parameters retain the original prior.

    Parameters
    ----------
    prior_mean : torch.Tensor, shape (D,)
        Mean of the original prior.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the original prior.
    indices : list[int]
        Indices of parameters to shrink to zero.
    shrinkage_variance : float, optional
        Variance for the shrunk parameters. Default is 1e-8.

    Returns
    -------
    reduced_mean : torch.Tensor, shape (D,)
        Reduced prior mean (zeroed at specified indices).
    reduced_cov : torch.Tensor, shape (D, D)
        Reduced prior covariance (shrunk at specified indices).
    """
    reduced_mean = prior_mean.clone().to(torch.float64)
    reduced_cov = prior_cov.clone().to(torch.float64)

    for idx in indices:
        reduced_mean[idx] = 0.0
        # Zero out cross-covariances for this parameter
        reduced_cov[idx, :] = 0.0
        reduced_cov[:, idx] = 0.0
        reduced_cov[idx, idx] = shrinkage_variance

    return reduced_mean, reduced_cov


def enumerate_reduced_models(
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    prunable_indices: list[int],
    shrinkage_variance: float = 1e-8,
) -> list[dict]:
    """Enumerate all non-trivial reduced models from prunable parameters.

    Generates all 2^k - 1 non-trivial subsets of ``prunable_indices``
    (excluding the empty set, which corresponds to the full model),
    and creates reduced priors for each via
    :func:`make_reduced_prior_zero_connection`.

    Parameters
    ----------
    prior_mean : torch.Tensor, shape (D,)
        Mean of the full-model prior.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model prior.
    prunable_indices : list[int]
        Indices of parameters eligible for pruning. Must have
        length k <= 20 (raises ``ValueError`` if k > 20).
    shrinkage_variance : float, optional
        Variance for shrunk parameters. Default is 1e-8.

    Returns
    -------
    list[dict]
        Each dict contains:

        - ``pruned_indices`` : tuple[int, ...] -- indices pruned
        - ``reduced_prior_mean`` : torch.Tensor, shape (D,)
        - ``reduced_prior_cov`` : torch.Tensor, shape (D, D)
        - ``n_pruned`` : int -- number of parameters pruned
        - ``label`` : str -- human-readable label

    Raises
    ------
    ValueError
        If ``len(prunable_indices) > 20``.
    """
    k = len(prunable_indices)
    if k > 20:
        msg = (
            f"enumerate_reduced_models received {k} prunable indices. "
            f"This would generate 2^{k} - 1 = {2**k - 1} candidates, "
            f"which is computationally prohibitive. Maximum is 20."
        )
        raise ValueError(msg)
    if k > 15:
        warnings.warn(
            f"enumerate_reduced_models: {k} prunable indices will "
            f"generate {2**k - 1} candidate models. This may be slow.",
            stacklevel=2,
        )

    candidates: list[dict] = []

    # Enumerate subsets grouped by pruning level (1, 2, ..., k)
    for n_pruned in range(1, k + 1):
        for combo in itertools.combinations(prunable_indices, n_pruned):
            indices = list(combo)
            r_mean, r_cov = make_reduced_prior_zero_connection(
                prior_mean,
                prior_cov,
                indices,
                shrinkage_variance=shrinkage_variance,
            )
            label = f"prune({','.join(str(i) for i in combo)})"
            candidates.append(
                {
                    "pruned_indices": tuple(combo),
                    "reduced_prior_mean": r_mean,
                    "reduced_prior_cov": r_cov,
                    "n_pruned": n_pruned,
                    "label": label,
                }
            )

    return candidates


def bmr_circuit_selection(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    prunable_indices: list[int],
    shrinkage_variance: float = 1e-8,
) -> dict:
    """Select the best circuit topology via Bayesian Model Reduction.

    Enumerates all reduced models formed by pruning subsets of
    ``prunable_indices``, scores each against the full model using
    :func:`bayesian_model_reduction`, and returns a ranked list of
    candidates including the full model as baseline.

    Implements exhaustive post hoc model comparison following
    [REF-070] Friston & Penny (2011).

    Parameters
    ----------
    posterior_mean : torch.Tensor, shape (D,)
        Mean of the full-model posterior.
    posterior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model posterior.
    prior_mean : torch.Tensor, shape (D,)
        Mean of the full-model prior.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model prior.
    prunable_indices : list[int]
        Indices of parameters eligible for pruning.
    shrinkage_variance : float, optional
        Variance for shrunk parameters. Default is 1e-8.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``results`` : list[dict] -- all candidates sorted by
          ``delta_log_evidence`` descending. Each entry has keys:
          ``pruned_indices``, ``delta_log_evidence``, ``n_pruned``,
          ``label``, ``reduced_posterior_mean``,
          ``reduced_posterior_cov``.
        - ``best`` : dict -- the top-ranked result.
        - ``full_model_rank`` : int -- 1-based rank of the full model.
        - ``n_candidates`` : int -- total number of candidates
          (including full model).
        - ``prunable_indices`` : list[int] -- echo of input for
          provenance.

    References
    ----------
    Friston, K. J. & Penny, W. D. (2011). Post hoc Bayesian model
    selection. NeuroImage, 56(4), 2089-2099.
    """
    candidates = enumerate_reduced_models(
        prior_mean,
        prior_cov,
        prunable_indices,
        shrinkage_variance=shrinkage_variance,
    )

    results: list[dict] = []

    # Score each reduced model
    for cand in candidates:
        delta_f, mu_r, sigma_r = bayesian_model_reduction(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            cand["reduced_prior_mean"],
            cand["reduced_prior_cov"],
        )
        results.append(
            {
                "pruned_indices": cand["pruned_indices"],
                "delta_log_evidence": delta_f,
                "n_pruned": cand["n_pruned"],
                "label": cand["label"],
                "reduced_posterior_mean": mu_r,
                "reduced_posterior_cov": sigma_r,
            }
        )

    # Add full model as baseline
    results.append(
        {
            "pruned_indices": (),
            "delta_log_evidence": 0.0,
            "n_pruned": 0,
            "label": "full_model",
            "reduced_posterior_mean": posterior_mean.to(torch.float64),
            "reduced_posterior_cov": posterior_cov.to(torch.float64),
        }
    )

    # Sort by delta_log_evidence descending
    results.sort(key=lambda r: r["delta_log_evidence"], reverse=True)

    # Find full model rank (1-based)
    full_model_rank = next(
        i + 1
        for i, r in enumerate(results)
        if r["pruned_indices"] == ()
    )

    return {
        "results": results,
        "best": results[0],
        "full_model_rank": full_model_rank,
        "n_candidates": len(results),
        "prunable_indices": prunable_indices,
    }


def _single_prune_costs(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    prunable_indices: list[int],
    shrinkage_variance: float,
) -> list[dict]:
    """Score each single-connection reduction (K BMR calls).

    Returns one dict ``{"index": k, "prune_delta_f": delta_f}`` per index
    in ``prunable_indices``, in input order.
    """
    costs: list[dict] = []
    for k in prunable_indices:
        r_mean, r_cov = make_reduced_prior_zero_connection(
            prior_mean,
            prior_cov,
            [k],
            shrinkage_variance=shrinkage_variance,
        )
        delta_f, _, _ = bayesian_model_reduction(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            r_mean,
            r_cov,
        )
        costs.append({"index": k, "prune_delta_f": delta_f})
    return costs


def rank_connections(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    prunable_indices: list[int],
    shrinkage_variance: float = 1e-8,
) -> dict:
    """Rank connections by single-prune cost (relative BMR, never absolute).

    For each prunable index ``k``, builds a single-connection reduced prior
    via :func:`make_reduced_prior_zero_connection` and scores it with
    :func:`bayesian_model_reduction`. This is exactly ``K`` BMR calls (one
    per index), not the ``2^K`` of :func:`enumerate_reduced_models`. The
    single-prune ``delta_f`` is the *prune cost*: a more negative value means
    pruning that connection costs more evidence, i.e. the connection is more
    essential. Connections are returned ordered most-essential-first.

    Implements the relative-ranking mode of [REF-070] Friston & Penny (2011)
    Eq. 4-8 BMR delta-F.

    Parameters
    ----------
    posterior_mean : torch.Tensor, shape (D,)
        Mean of the full-model posterior.
    posterior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model posterior.
    prior_mean : torch.Tensor, shape (D,)
        Mean of the full-model prior.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model prior.
    prunable_indices : list[int]
        Indices of parameters eligible for pruning (length ``K >= 1``).
    shrinkage_variance : float, optional
        Variance for the shrunk parameter in each reduced prior.
        Default is 1e-8.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``ranked`` : list[dict] -- the per-connection dicts sorted ascending
          by ``prune_delta_f`` (most-essential / most-negative first). Each
          entry has keys ``index``, ``prune_delta_f``, ``rank`` (1-based), and
          ``gap_to_next`` (``prune_delta_f`` of the next entry minus this one,
          a non-negative number, or ``None`` for the last entry).
        - ``separation_gap`` : float -- the maximum ``gap_to_next`` across the
          list, the largest consecutive drop in essentiality.
        - ``separation_after_rank`` : int -- the 1-based rank after which
          ``separation_gap`` occurs (the cut between essential and
          non-essential edges).
        - ``prunable_indices`` : list[int] -- echo of input for provenance.

    Raises
    ------
    ValueError
        If ``prunable_indices`` is empty, or any index is out of range for
        ``posterior_mean.shape[0]``.

    Notes
    -----
    Absolute delta-F is deliberately NOT used as a pass/fail pruning
    criterion. Under VL the Laplace posterior is sharply overconfident at high
    SNR (posterior std ~0.001-0.01x the prior std), so the reduced-model
    delta-F is driven deeply negative for *every* connection -- present or
    absent alike (cluster job 55772525: a truly-absent edge scored
    delta_F = -115.9, indistinguishable from present edges by sign). Only the
    *relative ordering* of prune costs and the *separation gap* between
    essential and non-essential edges are meaningful. Inputs are cast to
    float64 internally by :func:`bayesian_model_reduction`.

    References
    ----------
    Friston, K. J. & Penny, W. D. (2011). Post hoc Bayesian model
    selection. NeuroImage, 56(4), 2089-2099.
    """
    k = len(prunable_indices)
    if k == 0:
        raise ValueError("rank_connections requires >=1 prunable index; got 0")
    d = posterior_mean.shape[0]
    for idx in prunable_indices:
        if idx < 0 or idx >= d:
            raise ValueError(
                f"prunable index {idx} out of range; "
                f"expected 0 <= index < {d}"
            )

    costs = _single_prune_costs(
        posterior_mean,
        posterior_cov,
        prior_mean,
        prior_cov,
        prunable_indices,
        shrinkage_variance,
    )

    # Sort ascending: most-essential (most-negative prune cost) first.
    ranked = sorted(costs, key=lambda c: c["prune_delta_f"])

    separation_gap = 0.0
    separation_after_rank = len(ranked)
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1
        if i + 1 < len(ranked):
            gap = ranked[i + 1]["prune_delta_f"] - entry["prune_delta_f"]
            entry["gap_to_next"] = gap
            if gap > separation_gap:
                separation_gap = gap
                separation_after_rank = i + 1
        else:
            entry["gap_to_next"] = None

    return {
        "ranked": ranked,
        "separation_gap": float(separation_gap),
        "separation_after_rank": int(separation_after_rank),
        "prunable_indices": list(prunable_indices),
    }


def temper_vl_posterior(
    sigma_post: torch.Tensor,
    tempering_factor: float = 1.0,
) -> torch.Tensor:
    """Scale a VL posterior covariance by a temperature, guarding PD.

    Multiplies the posterior covariance by ``tempering_factor`` and asserts
    the result is positive-definite via a Cholesky factorization, raising
    loudly if it is not. This is an exploratory primitive only -- the factor
    is NOT calibrated here.

    Parameters
    ----------
    sigma_post : torch.Tensor, shape (D, D)
        VL posterior covariance to temper.
    tempering_factor : float, optional
        Positive multiplicative temperature. Default is 1.0 (identity, i.e.
        backwards-compatible no-op).

    Returns
    -------
    torch.Tensor, shape (D, D)
        The symmetrized tempered covariance,
        ``0.5 * (T + T.T)`` with ``T = tempering_factor * sigma_post``,
        in float64.

    Raises
    ------
    ValueError
        If ``tempering_factor <= 0``, or if the tempered covariance is not
        positive-definite (Cholesky fails). The PD-failure message includes
        the matrix shape and the ``tempering_factor``.

    Notes
    -----
    The motivation is that VL's ReML M-step underestimates the posterior
    covariance at high SNR (see pitfall C1 / cluster job 55772525). A
    ``tempering_factor > 1`` inflates the covariance to partially restore an
    absolute-delta-F BMR regime. The calibrated factor is determined
    empirically in Phase 31 against the Phase 30 coverage curves; the default
    ``1.0`` is the identity transform and is backwards-compatible.
    """
    if tempering_factor <= 0:
        raise ValueError(
            f"tempering_factor must be > 0; got {tempering_factor}"
        )

    sigma = sigma_post.to(torch.float64)
    sigma_tempered = tempering_factor * sigma
    sigma_tempered = 0.5 * (sigma_tempered + sigma_tempered.T)

    try:
        torch.linalg.cholesky(sigma_tempered)
    except torch.linalg.LinAlgError as err:
        raise ValueError(
            f"Tempered posterior covariance (shape "
            f"{tuple(sigma_tempered.shape)}, "
            f"tempering_factor={tempering_factor}) is not positive-definite "
            f"(Cholesky failed)."
        ) from err

    return sigma_tempered
