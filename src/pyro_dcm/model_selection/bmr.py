"""Bayesian Model Reduction for DCM.

Implements post hoc Bayesian model selection from a single full-model
inversion, following [REF-070] Friston & Penny (2011).
"""

from __future__ import annotations

import logging
import warnings

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "bayesian_model_reduction",
    "make_reduced_prior_zero_connection",
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
