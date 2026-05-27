"""Parameter recovery tests for Variational Laplace spectral DCM.

Validates that the VL inference backend recovers known ground-truth
connectivity parameters (A matrix) from synthetic CSD data, matching
the protocol in test_spectral_dcm_recovery.py but using the
deterministic Gauss-Newton optimizer instead of Pyro SVI.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.inference.variational_laplace import (
    run_variational_laplace,
    extract_vl_posterior,
)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_mean = x.mean()
    y_mean = y.mean()
    xd = x - x_mean
    yd = y - y_mean
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()


def compute_rmse_A(
    A_true: torch.Tensor, A_inferred: torch.Tensor,
) -> float:
    return torch.sqrt(torch.mean((A_true - A_inferred) ** 2)).item()


def run_single_vl_recovery(
    seed: int,
    n_regions: int = 3,
    snr: float = 10.0,
    max_iter: int = 128,
) -> dict | None:
    """Run a single VL parameter recovery trial."""
    try:
        N = n_regions
        A_true = make_stable_A_spectral(N, seed=seed)

        sim = simulate_spectral_dcm(
            A_true, TR=2.0, n_freqs=32, seed=seed,
        )

        obs_real = decompose_csd_for_likelihood(sim["csd"])
        signal_power = obs_real.pow(2).mean().sqrt()
        noise_std = signal_power / snr
        torch.manual_seed(seed + 1000)
        noisy_obs = obs_real + noise_std * torch.randn_like(obs_real)

        F, n, _ = sim["csd"].shape
        half = F * n * n
        noisy_real = noisy_obs[:half].reshape(F, n, n)
        noisy_imag = noisy_obs[half:].reshape(F, n, n)
        noisy_csd = torch.complex(noisy_real, noisy_imag)

        a_mask = torch.ones(N, N, dtype=torch.float64)

        result = run_variational_laplace(
            noisy_csd, sim["freqs"], a_mask,
            N=N, max_iter=max_iter,
        )

        A_inferred = result.theta_post["A"]
        posterior = extract_vl_posterior(result, N, num_samples=1000)

        A_free_samples = posterior["A_free"]["samples"]
        A_free_lo = A_free_samples.quantile(0.025, dim=0)
        A_free_hi = A_free_samples.quantile(0.975, dim=0)

        diag_mask = torch.eye(N, dtype=torch.bool)
        A_lo = A_free_lo.clone()
        A_hi = A_free_hi.clone()
        A_lo[diag_mask] = -torch.exp(A_free_hi[diag_mask]) / 2.0
        A_hi[diag_mask] = -torch.exp(A_free_lo[diag_mask]) / 2.0

        return {
            "A_true": A_true,
            "A_inferred": A_inferred,
            "A_lo": A_lo,
            "A_hi": A_hi,
            "free_energy": result.free_energy,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
        }

    except (RuntimeError, ValueError) as e:
        print(f"VL recovery failed (seed={seed}): {e}")
        return None


@pytest.fixture(scope="module")
def vl_recovery_results():
    """Run 5 VL recovery trials and cache results."""
    results = []
    for seed in range(100, 105):
        r = run_single_vl_recovery(seed)
        if r is not None:
            results.append(r)
    assert len(results) >= 3, (
        f"Too many VL recovery failures: {5 - len(results)}/5 failed"
    )
    return results


class TestVLRecovery:
    """Variational Laplace parameter recovery validation."""

    def test_rmse_below_threshold(self, vl_recovery_results):
        rmses = [
            compute_rmse_A(r["A_true"], r["A_inferred"])
            for r in vl_recovery_results
        ]
        mean_rmse = sum(rmses) / len(rmses)
        assert mean_rmse < 0.10, (
            f"Mean RMSE {mean_rmse:.4f} exceeds 0.10 threshold"
        )

    def test_correlation_above_threshold(self, vl_recovery_results):
        corrs = [
            _pearson_corr(
                r["A_true"].reshape(-1), r["A_inferred"].reshape(-1)
            )
            for r in vl_recovery_results
        ]
        mean_corr = sum(corrs) / len(corrs)
        assert mean_corr > 0.70, (
            f"Mean correlation {mean_corr:.4f} below 0.70 threshold"
        )

    def test_free_energy_decreases(self, vl_recovery_results):
        for r in vl_recovery_results:
            fe = r["free_energy"]
            if len(fe) >= 5:
                assert fe[-1] >= fe[1], (
                    f"Free energy did not improve: first={fe[1]:.2f}, "
                    f"last={fe[-1]:.2f}"
                )

    def test_convergence(self, vl_recovery_results):
        n_converged = sum(1 for r in vl_recovery_results if r["converged"])
        assert n_converged >= 1, "No VL runs converged"


class TestVLPosteriorFormat:
    """Verify VL posterior matches SVI extract_posterior_params format."""

    def test_posterior_structure(self):
        N = 2
        A_true = make_stable_A_spectral(N, seed=42)
        sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=16, seed=42)
        a_mask = torch.ones(N, N, dtype=torch.float64)

        result = run_variational_laplace(
            sim["csd"], sim["freqs"], a_mask,
            N=N, max_iter=32,
        )
        posterior = extract_vl_posterior(result, N, num_samples=100)

        assert "A_free" in posterior
        assert "noise_a" in posterior
        assert "noise_b" in posterior
        assert "noise_c" in posterior
        assert "median" in posterior

        for key in ["A_free", "noise_a", "noise_b", "noise_c"]:
            assert "mean" in posterior[key]
            assert "std" in posterior[key]
            assert "samples" in posterior[key]

        assert posterior["A_free"]["mean"].shape == (N, N)
        assert posterior["A_free"]["std"].shape == (N, N)
        assert posterior["A_free"]["samples"].shape == (100, N, N)
        assert posterior["noise_a"]["mean"].shape == (2, N)
        assert posterior["noise_b"]["mean"].shape == (2, 1)
        assert posterior["noise_c"]["mean"].shape == (2, N)
