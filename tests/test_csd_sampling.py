"""Finite-sample CSD realisations for genuine multi-seed replication.

``simulate_spectral_dcm`` is deterministic: it returns the ANALYTIC cross-spectral
density and its ``seed`` argument changes nothing (verified 2026-09-08, DCCN job
55234529 fitted one problem five times and reported it as a replication). Real
CSD estimates are finite-sample averages, and ``spm_dcm_csd_Q`` explicitly assumes
a *sample* spectral density -- Camba-Mendez & Kapetanios (2005), where the
covariance of the vectorised error carries ``h = 2m + 1`` degrees of freedom for
``m`` averages.

``sample_csd_realisation`` draws such a realisation so that each seed is a
different draw around the SAME ground truth.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.simulators.spectral_simulator import (
    sample_csd_realisation,
    simulate_spectral_dcm,
)
from validation.run_vl_validation import _build_reciprocal_asymmetric_A


@pytest.fixture(scope="module")
def analytic_csd() -> torch.Tensor:
    """Analytic CSD for the N=2 reciprocal-asymmetric parity problem."""
    return simulate_spectral_dcm(A=_build_reciprocal_asymmetric_A(2), TR=2.0)["csd"]


def test_realisation_is_seed_deterministic(analytic_csd: torch.Tensor) -> None:
    """The same seed reproduces the same draw; different seeds differ."""
    a = sample_csd_realisation(analytic_csd, n_averages=32, seed=7)
    b = sample_csd_realisation(analytic_csd, n_averages=32, seed=7)
    c = sample_csd_realisation(analytic_csd, n_averages=32, seed=8)

    assert torch.equal(a, b), "same seed must reproduce the same realisation"
    diff = (a - c).abs().max().item()
    assert diff > 0.0, (
        "different seeds must give different realisations -- this is exactly the "
        f"degeneracy this function exists to remove (max|a - c| = {diff})"
    )


def test_realisation_is_hermitian(analytic_csd: torch.Tensor) -> None:
    """Every frequency slice of a sampled CSD is Hermitian."""
    sampled = sample_csd_realisation(analytic_csd, n_averages=16, seed=1)
    err = (sampled - sampled.conj().transpose(-2, -1)).abs().max().item()
    assert err < 1e-10, f"sampled CSD is not Hermitian: max|G - G^H| = {err:.3e}"


def test_realisation_is_positive_semidefinite(analytic_csd: torch.Tensor) -> None:
    """Every frequency slice is PSD, as a covariance-like quantity must be."""
    sampled = sample_csd_realisation(analytic_csd, n_averages=16, seed=2)
    eigvals = torch.linalg.eigvalsh(sampled)
    worst = eigvals.min().item()
    scale = sampled.abs().max().item()
    assert worst > -1e-10 * max(scale, 1.0), (
        f"sampled CSD has a negative eigenvalue {worst:.3e} "
        f"(scale {scale:.3e}); it is not PSD"
    )


def test_realisation_is_unbiased_as_averages_grow(
    analytic_csd: torch.Tensor,
) -> None:
    """The mean over many draws converges to the analytic CSD.

    ``m * Ghat ~ CW(m, G)`` has ``E[Ghat] = G``, so averaging independent draws
    must approach the analytic CSD. Uses a modest draw count and a tolerance
    scaled to the Monte-Carlo standard error rather than a fixed epsilon.
    """
    n_draws, n_averages = 200, 8
    acc = torch.zeros_like(analytic_csd)
    for k in range(n_draws):
        acc = acc + sample_csd_realisation(
            analytic_csd, n_averages=n_averages, seed=1000 + k,
        )
    mean = acc / n_draws

    scale = analytic_csd.abs().max().item()
    # SE of the mean ~ scale / sqrt(n_averages * n_draws); allow 5x headroom.
    tol = 5.0 * scale / (n_averages * n_draws) ** 0.5
    err = (mean - analytic_csd).abs().max().item()
    assert err < tol, (
        f"sampled CSD is biased: max|mean - analytic| = {err:.3e}, "
        f"expected < {tol:.3e} over {n_draws} draws at m={n_averages}"
    )


def test_rejects_too_few_averages(analytic_csd: torch.Tensor) -> None:
    """``n_averages`` below the region count cannot give a full-rank estimate."""
    with pytest.raises(ValueError, match="n_averages"):
        sample_csd_realisation(analytic_csd, n_averages=1, seed=0)
