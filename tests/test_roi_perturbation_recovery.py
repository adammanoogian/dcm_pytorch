"""ROI-level perturbation detection via spectral DCM (Approach B).

Validates the direct pipeline: simulate ROI timeseries from known A
(OU process) -> compute empirical CSD -> fit spectral DCM (Variational
Laplace) -> recover posterior A. Then perturb A, re-simulate, re-fit,
and check that the perturbation is detectable in the posterior.

No autoencoder or PCA in the loop — this tests DCM on ROI-level data
directly, which is the foundation model / real neuroimaging path.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting
    state fMRI. NeuroImage, 94, 396-407.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.csd_computation import compute_empirical_csd
from pyro_dcm.forward_models.spectral_transfer import (
    default_frequency_grid_meg,
)
from pyro_dcm.inference.variational_laplace import (
    run_variational_laplace,
)
from pyro_dcm.simulators.meg_simulator import simulate_meg_timeseries
from pyro_dcm.simulators.spectral_simulator import make_stable_A_spectral


def _fit_spectral_dcm_from_timeseries(
    A_true: torch.Tensor,
    *,
    n_samples: int = 80,
    sfreq: float = 250.0,
    duration: float = 10.0,
    n_freqs: int = 32,
    max_iter: int = 128,
    seed: int = 42,
) -> dict:
    """Simulate ROI timeseries and fit spectral DCM via VL.

    Parameters
    ----------
    A_true : torch.Tensor
        Ground-truth connectivity, shape ``(N, N)``.
    n_samples : int
        Number of OU realizations to average CSD over.
    sfreq : float
        Sampling frequency in Hz.
    duration : float
        Duration per trial in seconds.
    n_freqs : int
        Number of frequency bins for CSD.
    max_iter : int
        Max Gauss-Newton iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``A_post`` (posterior mean A), ``A_free_post``,
        ``free_energy`` (final), ``converged``.
    """
    N = A_true.shape[0]

    result = simulate_meg_timeseries(
        A_true, sfreq=sfreq, duration=duration,
        n_samples=n_samples, seed=seed,
    )
    timeseries = result["timeseries"]

    freqs_meg = default_frequency_grid_meg(sfreq=sfreq, n_freqs=n_freqs)
    freqs_np = freqs_meg.numpy()

    csd_list = []
    for i in range(n_samples):
        trial = timeseries[i].numpy()
        csd_i = compute_empirical_csd(trial, fs=sfreq, freqs=freqs_np)
        csd_list.append(csd_i)
    csd_avg = np.mean(csd_list, axis=0)
    csd_tensor = torch.tensor(csd_avg, dtype=torch.complex128)

    a_mask = torch.ones(N, N, dtype=torch.float64)

    vl_result = run_variational_laplace(
        csd_tensor,
        freqs_meg,
        a_mask,
        max_iter=max_iter,
        tolerance=1e-2,
        prior_variance=1.0 / 16.0,
        eig_clamp=-1.0,
    )

    A_post = vl_result.theta_post["A"]
    A_free_post = vl_result.theta_post["A_free"]
    fe = vl_result.free_energy[-1] if vl_result.free_energy else float("nan")

    return {
        "A_post": A_post.detach().cpu(),
        "A_free_post": A_free_post.detach().cpu(),
        "free_energy": fe,
        "converged": vl_result.converged,
    }


@pytest.mark.slow
def test_roi_baseline_recovery() -> None:
    """Recover A from ROI timeseries without perturbation.

    Simulates a 4-region OU process, computes empirical CSD, fits
    spectral DCM via VL, and checks that the recovered A correlates
    with ground truth.
    """
    N = 4
    A_true = make_stable_A_spectral(N, connection_strength=0.08, seed=42)

    result = _fit_spectral_dcm_from_timeseries(
        A_true, n_samples=80, duration=10.0, seed=42,
    )
    A_post = result["A_post"]

    a_true_flat = A_true.flatten()
    a_post_flat = A_post.flatten().to(torch.float64)
    corr = torch.corrcoef(torch.stack([a_true_flat, a_post_flat]))[0, 1]

    print(f"\nBaseline recovery:")
    print(f"  A_true diag:  {A_true.diag().tolist()}")
    print(f"  A_post diag:  {A_post.diag().tolist()}")
    print(f"  Correlation:  {corr:.3f}")
    print(f"  Free energy:  {result['free_energy']:.2f}")
    print(f"  Converged:    {result['converged']}")

    assert corr > 0.5, (
        f"Baseline A recovery correlation {corr:.3f} < 0.5 — "
        f"spectral DCM is not recovering the connectivity structure"
    )


@pytest.mark.slow
def test_roi_perturbation_detected() -> None:
    """Perturbation to A[0,1] is detectable in the DCM posterior.

    Fits spectral DCM to baseline and perturbed (A[0,1] doubled)
    ROI timeseries. The perturbed connection should show a larger
    posterior change than the median of other connections.
    """
    N = 4
    A_base = make_stable_A_spectral(N, connection_strength=0.08, seed=42)
    A_base[0, 1] = 0.15

    eig = torch.linalg.eigvals(A_base.to(torch.complex128))
    assert eig.real.max().item() < 0, "Baseline A is unstable"

    A_perturbed = A_base.clone()
    A_perturbed[0, 1] = 0.30

    eig_p = torch.linalg.eigvals(A_perturbed.to(torch.complex128))
    assert eig_p.real.max().item() < 0, "Perturbed A is unstable"

    result_base = _fit_spectral_dcm_from_timeseries(
        A_base, n_samples=80, duration=10.0, seed=100,
    )
    result_pert = _fit_spectral_dcm_from_timeseries(
        A_perturbed, n_samples=80, duration=10.0, seed=200,
    )

    A_post_base = result_base["A_post"]
    A_post_pert = result_pert["A_post"]

    delta_A = (A_post_pert - A_post_base).abs()
    delta_01 = delta_A[0, 1].item()

    mask = torch.ones(N, N, dtype=torch.bool)
    mask[0, 1] = False
    median_other = delta_A[mask].median().item()

    print(f"\nPerturbation detection:")
    print(f"  A_base[0,1] = {A_base[0,1]:.3f}")
    print(f"  A_pert[0,1] = {A_perturbed[0,1]:.3f}")
    print(f"  Post_base[0,1] = {A_post_base[0,1]:.4f}")
    print(f"  Post_pert[0,1] = {A_post_pert[0,1]:.4f}")
    print(f"  delta_A[0,1]   = {delta_01:.4f}")
    print(f"  median(other)  = {median_other:.4f}")
    print(f"  ratio          = {delta_01 / max(median_other, 1e-8):.1f}x")

    assert delta_01 > median_other, (
        f"Perturbed connection delta ({delta_01:.4f}) should exceed "
        f"median of other connections ({median_other:.4f})"
    )

    assert delta_01 > 1e-3, (
        f"Perturbed connection delta ({delta_01:.6f}) is too small — "
        f"perturbation is not propagating through the pipeline"
    )


@pytest.mark.slow
def test_roi_no_perturbation_stable() -> None:
    """Two baseline fits with different seeds produce small delta_A.

    Verifies that the pipeline doesn't produce false alarms — running
    baseline twice with different OU realizations should give similar
    posteriors.
    """
    N = 4
    A_base = make_stable_A_spectral(N, connection_strength=0.08, seed=42)

    result_1 = _fit_spectral_dcm_from_timeseries(
        A_base, n_samples=80, duration=10.0, seed=300,
    )
    result_2 = _fit_spectral_dcm_from_timeseries(
        A_base, n_samples=80, duration=10.0, seed=400,
    )

    delta_A = (result_2["A_post"] - result_1["A_post"]).abs()
    max_delta = delta_A.max().item()

    print(f"\nStability check (no perturbation):")
    print(f"  max |delta_A| = {max_delta:.4f}")

    assert max_delta < 0.3, (
        f"max |delta_A| = {max_delta:.4f} between two baseline runs — "
        f"posterior is not stable enough for perturbation detection"
    )
