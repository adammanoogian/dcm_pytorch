"""VLBMR-03 EXPLORATORY tempering-mechanics tests (Plan 31-03).

Exercises the tempering MACHINERY in ``benchmarks.bmr_recovery`` -- the
PD-guarded :func:`pyro_dcm.model_selection.bmr.temper_vl_posterior` route,
coverage-matching temperature selection, and side-by-side untempered/tempered
:func:`pyro_dcm.model_selection.bmr.rank_connections` output. This is NOT an
absolute-delta-F claim (Pitfall C1/C2): tempering is an exploratory annotation
on the PRIMARY untempered ranking (31-01), and absolute delta-F is never used
as a pass/fail criterion. No tempered delta-F magnitude or absolute threshold
is asserted -- only that the machinery runs, the PD guard fires loudly, and the
selector picks the smallest in-band temperature.

References
----------
.planning/phases/31-bmr-validation-tempering/31-03-PLAN.md
    VLBMR-03 objective (EXPLORATORY tempering, coverage-matched, PD-safe).
benchmarks.bmr_recovery.select_tempering_factor / tempered_vs_untempered_ranking
    The helpers under test.
pyro_dcm.model_selection.bmr.temper_vl_posterior
    The ONLY PD guard; all tempering routes through it (no hand-rolled
    Cholesky).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch

from benchmarks.bmr_recovery import (
    offdiag_indices,
    select_tempering_factor,
    tempered_vs_untempered_ranking,
)
from pyro_dcm.model_selection.bmr import (  # type: ignore[import-untyped]
    temper_vl_posterior,
)

pytestmark = pytest.mark.vl


def _overconfident_posterior(
    n_regions: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a small overconfident Gaussian A_free posterior for the mechanics.

    Returns a flat posterior mean ``(N*N,)``, a sharply-overconfident PD
    covariance ``(N*N, N*N)``, and the ground-truth A used for coverage. The
    covariance std is ~0.01x the parameter scale, mimicking the VL Laplace
    overconfidence regime (pitfall C1) that tempering targets, so that low
    temperatures under-cover and larger ones restore calibration.

    Parameters
    ----------
    n_regions : int
        Number of regions ``N``.
    seed : int, optional
        RNG seed for reproducibility. Default 0.

    Returns
    -------
    posterior_mean : torch.Tensor, shape (N*N,)
    posterior_cov : torch.Tensor, shape (N*N, N*N)
    a_true : torch.Tensor, shape (N, N)
    """
    torch.manual_seed(seed)
    n = n_regions
    d = n * n

    a_true = torch.zeros(n, n, dtype=torch.float64)
    a_true.fill_diagonal_(-0.5)
    # Reciprocal edges 0<->1 (identifiable structure; 31-01-D1).
    a_true[0, 1] = 0.3
    a_true[1, 0] = 0.3

    # Posterior mean slightly biased from truth so coverage is < 1 when sharp.
    posterior_mean = a_true.reshape(-1) + 0.02 * torch.randn(d, dtype=torch.float64)

    # Sharp (overconfident) PD covariance: small isotropic + tiny random PD jitter.
    jitter = 0.0005 * torch.randn(d, d, dtype=torch.float64)
    posterior_cov = jitter @ jitter.T + 1e-4 * torch.eye(d, dtype=torch.float64)
    return posterior_mean, posterior_cov, a_true


def _samples_fn_factory(
    posterior_mean: torch.Tensor, posterior_cov: torch.Tensor, n_regions: int,
    n_samples: int = 2000, seed: int = 7,
) -> Callable[[float], torch.Tensor]:
    """Return a ``samples_fn(T)`` drawing tempered A samples reshaped (S, N, N).

    Every temperature routes the covariance through ``temper_vl_posterior``
    (the PD guard) before the multivariate-normal draw -- no hand-rolled
    Cholesky anywhere.
    """
    n = n_regions

    def samples_fn(temperature: float) -> torch.Tensor:
        # Fixed seed per call so coverage differences across temperatures come
        # from the tempered covariance, not from sampling noise.
        torch.manual_seed(seed)
        cov_t = temper_vl_posterior(posterior_cov, temperature)
        dist = torch.distributions.MultivariateNormal(posterior_mean, cov_t)
        flat = dist.sample(torch.Size([n_samples]))  # shape (S, N*N)
        return flat.reshape(n_samples, n, n)

    return samples_fn


def test_temper_identity_pd_pass() -> None:
    """temper_vl_posterior(cov, 1.0) is a PD-passing no-op-magnitude transform."""
    _, posterior_cov, _ = _overconfident_posterior(3)
    tempered = temper_vl_posterior(posterior_cov, 1.0)
    # Identity factor: same magnitude, symmetric, and Cholesky succeeded.
    assert torch.allclose(tempered, posterior_cov, atol=1e-10)
    torch.linalg.cholesky(tempered)  # PD confirmed (would raise otherwise).


def test_temper_non_pd_raises_with_shape_and_factor() -> None:
    """An indefinite covariance fails the PD guard, naming shape and factor."""
    # Deliberately indefinite: a symmetric matrix with a negative eigenvalue.
    indefinite = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    over_large_t = 100.0
    with pytest.raises(ValueError) as excinfo:
        temper_vl_posterior(indefinite, over_large_t)
    message = str(excinfo.value)
    assert "(3, 3)" in message  # shape named
    assert f"tempering_factor={over_large_t}" in message  # factor named
    assert "positive-definite" in message


def test_select_tempering_factor_picks_smallest_in_band() -> None:
    """select_tempering_factor: candidate T, monotone trace, smallest in-band pick."""
    n = 3
    posterior_mean, posterior_cov, a_true = _overconfident_posterior(n)
    samples_fn = _samples_fn_factory(posterior_mean, posterior_cov, n)
    candidates = (1, 2, 5, 10, 20, 50, 100)

    result = select_tempering_factor(
        a_true, samples_fn, target=0.95, band=(0.90, 0.98), candidates=candidates,
    )

    assert result["tempering_factor"] in [float(c) for c in candidates]
    trace: dict[float, float] = result["trace"]  # type: ignore[assignment]
    assert set(trace) == {float(c) for c in candidates}

    # Coverage is non-decreasing in T (inflating the CI can only add coverage).
    cov_by_t = [trace[float(c)] for c in candidates]
    for lo, hi in zip(cov_by_t, cov_by_t[1:], strict=False):
        assert hi >= lo - 1e-9, f"coverage not monotone non-decreasing: {cov_by_t}"

    # When an in-band T exists, the SMALLEST such T must be chosen.
    if result["in_band"]:
        in_band = [
            float(c) for c in candidates if 0.90 <= trace[float(c)] <= 0.98
        ]
        assert result["tempering_factor"] == min(in_band)
        chosen_coverage: float = result["coverage"]  # type: ignore[assignment]
        assert 0.90 <= chosen_coverage <= 0.98


def test_select_tempering_factor_no_band_surfaces_closest() -> None:
    """No reachable band: the closest-to-target T is surfaced, never raised."""
    n = 3
    posterior_mean, posterior_cov, a_true = _overconfident_posterior(n)
    samples_fn = _samples_fn_factory(posterior_mean, posterior_cov, n)
    # Impossible band (low > high is rejected; an unreachable band returns
    # in_band=False). Use a band no coverage can reach.
    result = select_tempering_factor(
        a_true, samples_fn, target=0.95, band=(1.01, 1.01),
        candidates=(1, 2, 5),
    )
    assert result["in_band"] is False
    assert result["tempering_factor"] in (1.0, 2.0, 5.0)


def test_tempered_vs_untempered_ranking_finite_and_aligned() -> None:
    """Side-by-side ranking: finite separation_gap for both; equal-length lists."""
    n = 3
    posterior_mean, posterior_cov, _ = _overconfident_posterior(n)
    prior_mean = torch.zeros(n * n, dtype=torch.float64)
    prior_cov = (1.0 / 64.0) * torch.eye(n * n, dtype=torch.float64)
    prunable = offdiag_indices(n)

    out = tempered_vs_untempered_ranking(
        posterior_mean, posterior_cov, prior_mean, prior_cov, prunable,
        tempering_factor=10.0,
    )

    untempered: dict[str, object] = out["untempered"]  # type: ignore[assignment]
    tempered: dict[str, object] = out["tempered"]  # type: ignore[assignment]
    assert out["tempering_factor"] == 10.0

    # Both separation gaps finite (NOT asserting absolute delta-F magnitude).
    for block in (untempered, tempered):
        gap = block["separation_gap"]
        assert isinstance(gap, float)
        assert gap == gap  # not NaN
        assert abs(gap) != float("inf")
        ranked = cast("list[dict[str, object]]", block["ranked"])
        assert len(ranked) == len(prunable)
        # Ranked list covers the same prunable indices.
        assert {e["index"] for e in ranked} == set(prunable)
