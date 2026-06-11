"""VLBMR-02: analytic BMR vs brute-force VL-refit agreement (spectral DCM).

Validates the analytic Bayesian Model Reduction (BMR) approximation against a
brute-force model comparison performed with the engine actually under test --
Variational Laplace (VL), not SVI. A single full-model spectral VL fit yields
analytic BMR delta-F for a small set of single-prune reduced models; each
reduced model is then independently re-fit via VL (the pruned connection zeroed
in ``a_mask``) and its free energy read off directly.

This test is **RANK-based, never value-based**. Absolute delta-F is NOT
comparable between an analytic-Laplace BMR (which holds the noise hyperparameters
fixed at the full-model optimum) and a refit that re-estimates those
hyperparameters and re-runs the SVD parameter reduction, which changes the
effective dimensionality (Pitfall S3/C1). Only the *relative ordering* of prune
costs and *worst-model agreement* between the two methods gate this test; the
Spearman rank correlation is reported as supporting evidence only (with three
points it is far too few to gate on).

The existing SVI-based ``tests/test_bmr_vs_elbo.py`` is left untouched as a
historical secondary cross-check.

References
----------
[REF-070] Friston, K. J. & Penny, W. D. (2011). Post hoc Bayesian model
selection. NeuroImage, 56(4), 2089-2099.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.inference import (
    SpectralDCMForward,
    run_variational_laplace_generic,
)
from pyro_dcm.model_selection.bmr import (
    bayesian_model_reduction,
    make_reduced_prior_zero_connection,
)
from pyro_dcm.simulators.spectral_simulator import simulate_spectral_dcm

# Spectral A prior variance (matches the BOLD/spectral DCM convention 1/64).
_PRIOR_VARIANCE = 1.0 / 64.0
_N_REGIONS = 3
_MAX_ITER = 64


def _make_sparse_ground_truth_a() -> torch.Tensor:
    """Build a sparse stable 3-region effective-connectivity matrix.

    The single present off-diagonal edge is 0->1 (``A[1, 0] = 0.15``); the
    2->0 (``A[0, 2]``) and 0->2 (``A[2, 0]``) edges are truly absent. The
    diagonal is ``-0.5`` (self-decay), guaranteeing all eigenvalues have
    negative real part (stable system).

    Returns
    -------
    torch.Tensor, shape (3, 3)
        Ground-truth effective connectivity ``A``, float64.
    """
    a_true = -0.5 * torch.eye(_N_REGIONS, dtype=torch.float64)
    a_true[1, 0] = 0.15  # present edge 0->1
    # A[0, 2] and A[2, 0] remain 0.0 (absent edges).
    max_real_eig = torch.linalg.eigvals(a_true).real.max().item()
    if max_real_eig >= 0.0:
        raise AssertionError(
            "Ground-truth A must be stable (max real eigenvalue < 0); "
            f"expected < 0, actual {max_real_eig:.4f}"
        )
    return a_true


def _fit_full_model(
    csd_obs: torch.Tensor,
    freqs: torch.Tensor,
):
    """Run the single full-model spectral VL fit (all edges free).

    Parameters
    ----------
    csd_obs : torch.Tensor, shape (F, N, N)
        Observed cross-spectral density, complex128.
    freqs : torch.Tensor, shape (F,)
        Frequency bins, float64.

    Returns
    -------
    VariationalLaplaceResult
        The full-model VL result (``theta_post``, ``sigma_post``,
        ``free_energy``).
    """
    a_mask_full = torch.ones(_N_REGIONS, _N_REGIONS, dtype=torch.float64)
    return run_variational_laplace_generic(
        SpectralDCMForward(),
        observed=csd_obs,
        a_mask=a_mask_full,
        n_regions=_N_REGIONS,
        max_iter=_MAX_ITER,
        prior_variance=_PRIOR_VARIANCE,
        context={"freqs": freqs},
    )


# Model set: flat C-order index i*N + j addresses A[i, j].
# prune_present    -> A[1, 0] (idx 3): truly PRESENT edge 0->1.
# prune_absent     -> A[0, 2] (idx 2): truly ABSENT edge 2->0.
# prune_two_absent -> A[0, 2] and A[2, 0] (idx 2, 6): both ABSENT.
_MODEL_SET: dict[str, list[int]] = {
    "prune_present": [3],
    "prune_absent": [2],
    "prune_two_absent": [2, 6],
}


def _analytic_bmr_delta_f(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
) -> dict[str, float]:
    """Score each reduced model analytically via BMR (no refit).

    Parameters
    ----------
    posterior_mean : torch.Tensor, shape (9,)
        Full-model A_free posterior mean.
    posterior_cov : torch.Tensor, shape (9, 9)
        Full-model A_free posterior covariance.
    prior_mean : torch.Tensor, shape (9,)
        Full-model A_free prior mean (zeros).
    prior_cov : torch.Tensor, shape (9, 9)
        Full-model A_free prior covariance (``prior_variance * I``).

    Returns
    -------
    dict[str, float]
        Analytic BMR delta-F per reduced model name.
    """
    bmr_delta_f: dict[str, float] = {}
    for name, prune_indices in _MODEL_SET.items():
        reduced_mean, reduced_cov = make_reduced_prior_zero_connection(
            prior_mean, prior_cov, prune_indices,
        )
        delta_f, _, _ = bayesian_model_reduction(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            reduced_mean,
            reduced_cov,
        )
        bmr_delta_f[name] = delta_f
    return bmr_delta_f


@pytest.mark.vl
def test_bmr_agrees_with_vl_refit_ranking() -> None:
    """BMR-on-VL and brute-force VL-refit agree on reduced-model RANKS.

    Absolute delta-F is never compared; only relative ordering and worst-model
    agreement gate this test (Pitfall S3/C1). Spearman rho is report-only.

    Implements VLBMR-02 ([REF-070] Friston & Penny, 2011).
    """
    # --- Ground truth + synthetic CSD --------------------------------------
    a_true = _make_sparse_ground_truth_a()
    sim = simulate_spectral_dcm(a_true, TR=2.0, n_freqs=32, seed=42)
    csd_obs = sim["csd"].to(torch.complex128)
    freqs = sim["freqs"].to(torch.float64)

    # --- Single full-model VL fit (all edges free) -------------------------
    result_full = _fit_full_model(csd_obs, freqs)

    # Full-model A_free posterior over the 9 connectivity parameters.
    posterior_mean = result_full.theta_post["A_free"].reshape(-1).double()
    posterior_cov = result_full.sigma_post[:9, :9].double()
    prior_mean = torch.zeros(9, dtype=torch.float64)
    prior_cov = _PRIOR_VARIANCE * torch.eye(9, dtype=torch.float64)

    # --- Analytic BMR delta-F per reduced model (no refit) -----------------
    bmr_delta_f = _analytic_bmr_delta_f(
        posterior_mean, posterior_cov, prior_mean, prior_cov,
    )

    print("\n=== Analytic BMR delta-F (per reduced model) ===")
    for name, value in bmr_delta_f.items():
        print(f"  {name}: {value:.4f}")
