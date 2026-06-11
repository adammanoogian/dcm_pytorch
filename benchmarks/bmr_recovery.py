"""Reusable helpers for VLBMR-01 BMR structure-recovery (Phase 31).

Provides the plumbing shared by ``tests/test_bmr_vlbmr01_recovery.py`` (and
later Plan 31-03): a SPARSE-ground-truth ``A`` builder with unambiguous true
present edges, the off-diagonal C-order flat-index set, and an extractor that
turns a Variational Laplace fit result into the four tensors
:func:`pyro_dcm.model_selection.bmr.rank_connections` consumes.

The structure-recovery helpers (``make_sparse_ground_truth_A``,
``offdiag_indices``, ``bmr_tensors_from_vl_result``) are PURE plumbing and do
NOT call ``rank_connections`` themselves. Plan 31-03 adds two tempering
helpers (``select_tempering_factor``, ``tempered_vs_untempered_ranking``) that
DO call ``rank_connections`` / ``temper_vl_posterior`` -- these are the
EXPLORATORY posterior-tempering calibration layer (absolute delta-F is never a
pass/fail criterion; all tempering routes through the PD-guarded
``temper_vl_posterior``).

References
----------
.planning/phases/31-bmr-validation-tempering/31-01-PLAN.md
    VLBMR-01 objective: relative-ranking structure recovery on a real VL
    posterior (NEVER absolute delta-F; pitfall C1 / cluster job 55772525).
cluster/scripts/lc_vl_bmr_selection.py
    The proven ``sigma_post[:N*N, :N*N]`` A_free covariance slice + OFFDIAG
    C-order flat-index convention mirrored here.
pyro_dcm.model_selection.bmr.rank_connections
    The consumer of the (mean, cov, prior_mean, prior_cov) tensors produced by
    :func:`bmr_tensors_from_vl_result`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from benchmarks.metrics import compute_coverage_from_samples
from pyro_dcm.model_selection.bmr import (  # type: ignore[import-untyped]
    rank_connections,
    temper_vl_posterior,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def make_sparse_ground_truth_A(
    n_regions: int,
    present_edges: Sequence[tuple[int, int]],
    *,
    self_connection: float = -0.5,
    strength: float = 0.15,
) -> torch.Tensor:
    """Build a sparse ground-truth effective-connectivity matrix.

    Constructs an ``(N, N)`` float64 ``A`` whose diagonal is
    ``self_connection`` and whose only non-zero off-diagonal entries are the
    explicitly listed ``present_edges`` (each set to ``strength``). All other
    off-diagonals are exactly zero, so the "true present edges" are unambiguous
    by construction (guards pitfall R2: never read true structure off a dense
    ``A``).

    Parameters
    ----------
    n_regions : int
        Number of regions ``N``.
    present_edges : Sequence[tuple[int, int]]
        Off-diagonal ``(row, col)`` index pairs that are present (non-zero).
        Each pair must have ``row != col`` (the diagonal is structural
        self-connection, never an edge).
    self_connection : float, optional
        Diagonal value (self-inhibition). Default ``-0.5``.
    strength : float, optional
        Value assigned to each present off-diagonal edge. Default ``0.15``.

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.

    Raises
    ------
    ValueError
        If any edge in ``present_edges`` lies on the diagonal
        (``row == col``), or if the resulting ``A`` is not stable
        (max real eigenvalue not ``< 0``). Messages name the offending pair
        or the offending max real eigenvalue (expected ``< 0``).
    """
    A = torch.zeros(n_regions, n_regions, dtype=torch.float64)
    A.fill_diagonal_(self_connection)
    for i, j in present_edges:
        if i == j:
            msg = (
                f"present_edges contains diagonal pair ({i}, {j}); the "
                f"diagonal is structural self-connection and is never an "
                f"edge (expected row != col, got row == col == {i})"
            )
            raise ValueError(msg)
        A[i, j] = strength

    max_real_eig = float(torch.linalg.eigvals(A).real.max())
    if not max_real_eig < 0.0:
        msg = (
            f"sparse ground-truth A is not stable: max real eigenvalue "
            f"{max_real_eig:.6f} (expected < 0); reduce strength or make "
            f"self_connection more negative"
        )
        raise ValueError(msg)

    return A


def offdiag_indices(n_regions: int) -> list[int]:
    """Return C-order flat indices of all off-diagonal entries.

    The flat index of ``A[i, j]`` is ``i * N + j`` (C-order, row-major), so
    these indices select the prunable off-diagonal connections for BMR while
    skipping the structural diagonal (guards pitfall S4: a column-major or
    transposed index silently mislabels edges). Mirrors the ``OFFDIAG``
    convention in ``cluster/scripts/lc_vl_bmr_selection.py``.

    Parameters
    ----------
    n_regions : int
        Number of regions ``N``.

    Returns
    -------
    list[int]
        The ``N * (N - 1)`` off-diagonal flat indices, in row-major order.
    """
    return [
        i * n_regions + j
        for i in range(n_regions)
        for j in range(n_regions)
        if i != j
    ]


def bmr_tensors_from_vl_result(
    result: object,
    n_regions: int,
    *,
    prior_variance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``rank_connections`` tensors from a VL fit result.

    Slices the FULL ``A_free`` posterior covariance out of the VL result and
    assembles the matching zero-mean prior. ``A_free`` is ALWAYS the first
    packed block, so its covariance is the leading ``(N*N, N*N)`` sub-block of
    ``result.sigma_post`` (mirrors ``lc_vl_bmr_selection.py:94-95``). This is
    the full off-diagonal covariance, NOT the diagonal-only marginal std that
    ``extract_vl_posterior_generic`` returns.

    Parameters
    ----------
    result : object
        A Variational Laplace fit result exposing ``theta_post["A_free"]``
        (the posterior mean, an ``(N, N)`` tensor) and ``sigma_post`` (the full
        posterior covariance, with ``A_free`` packed first).
    n_regions : int
        Number of regions ``N``.
    prior_variance : float
        Isotropic prior variance for the ``A_free`` parameters (e.g.
        ``1.0 / 64.0`` for spectral DCM).

    Returns
    -------
    posterior_mean : torch.Tensor, shape (N*N,)
        Flattened ``A_free`` posterior mean, float64.
    posterior_cov : torch.Tensor, shape (N*N, N*N)
        Full ``A_free`` posterior covariance (leading block of
        ``sigma_post``), float64.
    prior_mean : torch.Tensor, shape (N*N,)
        Zero prior mean, float64.
    prior_cov : torch.Tensor, shape (N*N, N*N)
        ``prior_variance * I`` prior covariance, float64.
    """
    n = n_regions
    d = n * n

    # Round-trip guard (pitfall S4): i*N+j must map back to (i, j) via divmod.
    sample_i, sample_j = (n - 1, 0) if n > 1 else (0, 0)
    flat = sample_i * n + sample_j
    if divmod(flat, n) != (sample_i, sample_j):
        msg = (
            f"C-order flat-index round-trip failed: divmod({flat}, {n}) = "
            f"{divmod(flat, n)} (expected ({sample_i}, {sample_j}))"
        )
        raise ValueError(msg)

    posterior_mean = result.theta_post["A_free"].reshape(-1).double()  # type: ignore[attr-defined]
    posterior_cov = result.sigma_post[:d, :d].double()  # type: ignore[attr-defined]
    prior_mean = torch.zeros(d, dtype=torch.float64)
    prior_cov = prior_variance * torch.eye(d, dtype=torch.float64)

    return posterior_mean, posterior_cov, prior_mean, prior_cov


def select_tempering_factor(
    true_vals: torch.Tensor,
    samples_fn: Callable[[float], torch.Tensor],
    *,
    target: float = 0.95,
    band: tuple[float, float] = (0.90, 0.98),
    candidates: Sequence[float] = (1, 2, 5, 10, 20, 50, 100),
) -> dict[str, object]:
    """Select a posterior tempering factor by coverage-matching (EXPLORATORY).

    Sweeps a ladder of candidate temperatures ``T`` (ascending), drawing
    tempered posterior samples via ``samples_fn(T)`` and recomputing the
    empirical 95% credible-interval coverage of ``true_vals`` against those
    samples. Returns the SMALLEST ``T`` whose coverage enters ``band`` -- the
    minimal inflation that restores nominal calibration. If no candidate lands
    in the band, returns the candidate whose coverage is closest to ``target``
    and flags ``in_band=False`` (the miss is SURFACED, never raised; research
    Section C2).

    Coverage-matching against the Phase 30 ``recovery_matrix.json`` coverage is
    the calibration the phase brief asks for. The band ``[0.90, 0.98]`` is a
    deliberate CHOICE reported as exploratory, NOT a validated tempering
    schedule (research Open Question 3). Tempering is an annotation on the
    PRIMARY untempered ranking, never a headline claim, and absolute delta-F is
    never used as a pass/fail criterion.

    Parameters
    ----------
    true_vals : torch.Tensor
        Ground-truth values, shape ``(D,)`` or ``(N, N)`` -- the same A space
        the Phase 30 coverage was computed on.
    samples_fn : Callable[[float], torch.Tensor]
        Maps a temperature ``T`` to tempered posterior samples (shape
        ``(S, D)`` or ``(S, N, N)``). The caller wires this to
        :func:`pyro_dcm.model_selection.bmr.temper_vl_posterior` plus a
        multivariate-normal draw -- ALL tempering routes through that PD guard.
    target : float, optional
        Nominal coverage to match. Default ``0.95``.
    band : tuple[float, float], optional
        Inclusive ``(low, high)`` acceptance band for the empirical 95%
        coverage. Default ``(0.90, 0.98)``.
    candidates : Sequence[float], optional
        Ascending temperature ladder. Default ``(1, 2, 5, 10, 20, 50, 100)``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``tempering_factor`` : float -- the chosen ``T``.
        - ``coverage`` : float -- empirical 95% coverage at the chosen ``T``.
        - ``in_band`` : bool -- whether the chosen coverage is inside ``band``.
        - ``trace`` : dict[float, float] -- the full ``{T: coverage}`` sweep.
        - ``band`` : tuple[float, float] -- echo of the band for provenance.
        - ``target`` : float -- echo of the target.
        - ``candidates`` : list[float] -- echo of the swept ladder.

    Raises
    ------
    ValueError
        If ``candidates`` is empty, or ``band`` is malformed
        (``low > high``).
    """
    cands = [float(c) for c in candidates]
    if not cands:
        raise ValueError("select_tempering_factor requires >=1 candidate; got 0")
    low, high = float(band[0]), float(band[1])
    if low > high:
        raise ValueError(
            f"band must satisfy low <= high; got low={low}, high={high}"
        )

    trace: dict[float, float] = {}
    for temperature in sorted(cands):
        samples = samples_fn(temperature)
        coverage = compute_coverage_from_samples(
            true_vals, samples, ci_level=0.95,
        )
        trace[temperature] = float(coverage)

    in_band_factors = [t for t, c in trace.items() if low <= c <= high]
    if in_band_factors:
        chosen = min(in_band_factors)
        return {
            "tempering_factor": chosen,
            "coverage": trace[chosen],
            "in_band": True,
            "trace": trace,
            "band": (low, high),
            "target": float(target),
            "candidates": cands,
        }

    # No candidate entered the band -- surface the closest-to-target T, do NOT
    # raise (research Section C2: the calibration miss is a reportable result).
    chosen = min(trace, key=lambda t: abs(trace[t] - float(target)))
    return {
        "tempering_factor": chosen,
        "coverage": trace[chosen],
        "in_band": False,
        "trace": trace,
        "band": (low, high),
        "target": float(target),
        "candidates": cands,
    }


def tempered_vs_untempered_ranking(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    prunable_indices: list[int],
    tempering_factor: float,
) -> dict[str, object]:
    """Rank connections untempered vs tempered, side by side (EXPLORATORY).

    Computes :func:`pyro_dcm.model_selection.bmr.rank_connections` on the raw VL
    posterior covariance and on the tempered covariance
    ``temper_vl_posterior(posterior_cov, tempering_factor)`` (THE PD-guarded
    primitive -- the tempered Cholesky is never hand-rolled here). The tempered
    ranking is reported strictly alongside the untempered one; absolute delta-F
    is never a pass/fail criterion (Pitfall C1/C2).

    Parameters
    ----------
    posterior_mean : torch.Tensor, shape (D,)
        Mean of the full-model VL posterior.
    posterior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model VL posterior.
    prior_mean : torch.Tensor, shape (D,)
        Mean of the full-model prior.
    prior_cov : torch.Tensor, shape (D, D)
        Covariance of the full-model prior.
    prunable_indices : list[int]
        Indices of parameters eligible for pruning (length ``K >= 1``).
    tempering_factor : float
        Positive temperature applied to the posterior covariance via
        :func:`temper_vl_posterior` (raises loudly if it breaks PD).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``tempering_factor`` : float -- echo of the applied temperature.
        - ``untempered`` : dict -- ``rank_connections`` output on the raw
          posterior covariance.
        - ``tempered`` : dict -- ``rank_connections`` output on the tempered
          covariance.

    Raises
    ------
    ValueError
        If ``temper_vl_posterior`` rejects the factor (non-positive or
        non-PD tempered covariance; the message names shape and factor).
    """
    untempered = rank_connections(
        posterior_mean, posterior_cov, prior_mean, prior_cov, prunable_indices,
    )
    cov_tempered = temper_vl_posterior(posterior_cov, tempering_factor)
    tempered = rank_connections(
        posterior_mean, cov_tempered, prior_mean, prior_cov, prunable_indices,
    )
    return {
        "tempering_factor": float(tempering_factor),
        "untempered": untempered,
        "tempered": tempered,
    }
