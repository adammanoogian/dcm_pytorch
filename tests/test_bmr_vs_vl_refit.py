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
costs (present-edge prune costs more than absent-edge prune) and *worst-model
agreement over the like-for-like single-prune subset* gate this test; the
two-prune model is reported in the side-by-side table but excluded from the
worst-model gate because comparing it against single-prune models confounds the
contrast with the number of removed dimensions (S3/C1). The Spearman rank
correlation is reported as supporting evidence only (with three points it is far
too few to gate on).

The ground truth uses RECIPROCAL edges (0<->1, 1<->2 present; 0<->2 absent): a
feed-forward / sparse chain is unidentifiable by spectral DCM, collapsing the VL
fit to zero connectivity and degenerating every prune delta-F (decision
31-01-D1). Reciprocal coupling makes the present-vs-absent contrast real for both
the analytic BMR and the brute-force refit.

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
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from pyro_dcm.inference import (  # type: ignore[import-untyped]
    SpectralDCMForward,
    run_variational_laplace_generic,
)
from pyro_dcm.inference.variational_laplace import (  # type: ignore[import-untyped]
    VariationalLaplaceResult,
)
from pyro_dcm.model_selection.bmr import (  # type: ignore[import-untyped]
    bayesian_model_reduction,
    make_reduced_prior_zero_connection,
)
from pyro_dcm.simulators.spectral_simulator import (  # type: ignore[import-untyped]
    simulate_spectral_dcm,
)

# Spectral A prior variance (matches the BOLD/spectral DCM convention 1/64).
_PRIOR_VARIANCE = 1.0 / 64.0
_N_REGIONS = 3
_MAX_ITER = 64


def _make_sparse_ground_truth_a() -> torch.Tensor:
    """Build a stable, *identifiable* 3-region connectivity matrix.

    The 0<->1 and 1<->2 region pairs are reciprocally connected (present);
    the 0<->2 pair is absent. Specifically the present edges are
    ``A[1, 0] = A[0, 1] = 0.3`` (pair 0<->1) and
    ``A[2, 1] = A[1, 2] = 0.25`` (pair 1<->2), while ``A[0, 2]`` (flat index
    2) and ``A[2, 0]`` (flat index 6) -- the BMR/refit-probed *absent* edges
    -- are exactly zero. The BMR-probed *present* edge is 0->1
    (``A[1, 0]``, flat index 3). The diagonal is ``-0.5`` (self-decay).

    Reciprocal edges are required because a feed-forward / sparse chain is
    UNIDENTIFIABLE by spectral DCM: its stationary CSD is bit-identical to
    the empty graph, so the VL fit collapses ``A_free`` to zero and every
    single-prune delta-F degenerates to zero (decision 31-01-D1; the same
    pitfall surfaces here for the brute-force refit). A reciprocal topology
    makes ``A`` recoverable, so the present-vs-absent prune contrast is real
    for BOTH the analytic BMR and the brute-force VL refit.

    Returns
    -------
    torch.Tensor, shape (3, 3)
        Ground-truth effective connectivity ``A``, float64.
    """
    a_true = -0.5 * torch.eye(_N_REGIONS, dtype=torch.float64)
    a_true[1, 0] = 0.3  # present edge 0->1 (idx 3), BMR-probed present
    a_true[0, 1] = 0.3  # reciprocal 1->0 (pair 0<->1)
    a_true[2, 1] = 0.25  # present edge 1->2 (pair 1<->2)
    a_true[1, 2] = 0.25  # reciprocal 2->1 (pair 1<->2)
    # A[0, 2] (idx 2) and A[2, 0] (idx 6) remain 0.0: the absent edges.
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
) -> VariationalLaplaceResult:
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


def _bruteforce_refit_delta_f(
    csd_obs: torch.Tensor,
    freqs: torch.Tensor,
    free_energy_full: float,
) -> dict[str, float]:
    """Independently re-fit each reduced model via VL and read its free energy.

    For each reduced model a fresh ``a_mask`` is cloned from the full ones
    matrix with the pruned connection(s) zeroed (the flat C-order index
    decoded via ``divmod(idx, N)``), and the spectral DCM is re-fit via VL on
    the SAME observed CSD / freqs / max_iter / prior_variance as the full
    model. The brute-force delta-F is the reduced final free energy minus the
    full-model final free energy.

    Parameters
    ----------
    csd_obs : torch.Tensor, shape (F, N, N)
        Observed cross-spectral density, complex128.
    freqs : torch.Tensor, shape (F,)
        Frequency bins, float64.
    free_energy_full : float
        Final free energy of the full-model VL fit.

    Returns
    -------
    dict[str, float]
        Brute-force delta-F (``F_reduced - F_full``) per reduced model name.
    """
    bruteforce_delta_f: dict[str, float] = {}
    for name, prune_indices in _MODEL_SET.items():
        a_mask = torch.ones(_N_REGIONS, _N_REGIONS, dtype=torch.float64)
        for idx in prune_indices:
            i, j = divmod(idx, _N_REGIONS)
            a_mask[i, j] = 0.0
        result_reduced = run_variational_laplace_generic(
            SpectralDCMForward(),
            observed=csd_obs,
            a_mask=a_mask,
            n_regions=_N_REGIONS,
            max_iter=_MAX_ITER,
            prior_variance=_PRIOR_VARIANCE,
            context={"freqs": freqs},
        )
        free_energy_reduced = float(result_reduced.free_energy[-1])
        bruteforce_delta_f[name] = free_energy_reduced - free_energy_full
    return bruteforce_delta_f


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

    # --- Brute-force VL refit per reduced model (a_mask-zeroed) ------------
    # Full-model free energy is the baseline (full delta-F == 0 by definition).
    free_energy_full = float(result_full.free_energy[-1])
    bruteforce_delta_f = _bruteforce_refit_delta_f(
        csd_obs, freqs, free_energy_full,
    )

    print("\n=== Brute-force VL-refit delta-F (F_reduced - F_full) ===")
    for name, value in bruteforce_delta_f.items():
        print(f"  {name}: {value:.4f}")

    # --- GATE: relative ordering (rank-based only, never value-based) ------
    # Pruning a PRESENT edge costs more evidence (more negative delta-F) than
    # pruning an ABSENT edge. Asserted on BOTH methods.
    assert bmr_delta_f["prune_present"] < bmr_delta_f["prune_absent"], (
        "BMR: pruning the present edge should cost more than pruning an "
        "absent edge; expected prune_present delta-F < prune_absent delta-F, "
        f"actual prune_present={bmr_delta_f['prune_present']:.4f}, "
        f"prune_absent={bmr_delta_f['prune_absent']:.4f}"
    )
    assert (
        bruteforce_delta_f["prune_present"]
        < bruteforce_delta_f["prune_absent"]
    ), (
        "Brute-force VL: pruning the present edge should cost more than "
        "pruning an absent edge; expected prune_present delta-F < prune_absent "
        f"delta-F, actual prune_present="
        f"{bruteforce_delta_f['prune_present']:.4f}, "
        f"prune_absent={bruteforce_delta_f['prune_absent']:.4f}"
    )

    # --- GATE: worst-model agreement (single-prune subset) ----------------
    # The reduced model with the lowest delta-F (most costly to prune) must
    # match between the analytic BMR and the brute-force VL refit. This is
    # restricted to the SINGLE-prune models (prune_present, prune_absent):
    # comparing a two-prune model against single-prune models confounds the
    # contrast with the *number* of removed dimensions, which the brute-force
    # refit (re-estimating noise hyperparameters over fewer free dims) and the
    # analytic BMR (hyperparameters fixed) weight differently -- exactly the
    # S3/C1 incomparability this test must not assert across. Within the
    # like-for-like single-prune subset the worst-model contrast is the honest
    # present-vs-absent decision both methods must agree on.
    single_prune = ["prune_present", "prune_absent"]
    bmr_worst = min(single_prune, key=lambda k: bmr_delta_f[k])
    bf_worst = min(single_prune, key=lambda k: bruteforce_delta_f[k])
    assert bmr_worst == bf_worst, (
        "BMR and brute-force VL disagree on the worst (most costly to prune) "
        f"single-prune reduced model; expected agreement, actual BMR "
        f"worst='{bmr_worst}', brute-force worst='{bf_worst}'"
    )

    # --- Supporting evidence (report-only, NOT a gate) --------------------
    # Spearman rho over the 3 reduced models. With only 3 points this is far
    # too few to gate on; it is printed as supporting evidence only.
    names = list(_MODEL_SET)
    bmr_vec = [bmr_delta_f[n] for n in names]
    bf_vec = [bruteforce_delta_f[n] for n in names]
    rho, _ = spearmanr(bmr_vec, bf_vec)

    print("\n=== BMR vs brute-force VL delta-F (side-by-side) ===")
    print(f"  {'model':<18} {'bmr_delta_f':>14} {'bruteforce_delta_f':>20}")
    for name in names:
        print(
            f"  {name:<18} {bmr_delta_f[name]:>14.4f} "
            f"{bruteforce_delta_f[name]:>20.4f}"
        )
    print(f"\nSpearman rho (report-only, 3 points): {rho:.4f}")
