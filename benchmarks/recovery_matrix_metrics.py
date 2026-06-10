"""Hardened per-cell recovery-metric assembler for the Phase 30 sweep.

Turns ONE forward-model Variational Laplace runner result dict (the
``(BenchmarkConfig) -> dict`` output of ``benchmarks.runners.spectral_vl
.run_spectral_vl`` / ``task_vl.run_task_vl`` / ``latent_circuit_vl
.run_latent_circuit_vl``) into a flat, JSON-serializable per-cell metric block,
and provides the ground-truth-design guards the sweep depends on.

The module centralizes the metric/design hardening so the sweep driver (30-02)
and the harvest/report step (30-03) consume identical, tested logic:

- ``compute_shrinkage_ratio``: identifiability shrinkage ``std_post / std_prior``
  accepting a scalar ``std_prior`` (so it works across models with different
  ``A`` prior variances and for ``B``).
- ``assemble_cell_metrics``: per-region R-squared (NOT variance-pooled), MASKED
  sign recovery (``|true| > threshold``), 95% CI coverage, RMSE, and shrinkage,
  aggregated across the per-seed lists the runner already emits.
- ``exclude_near_boundary_A`` / ``resample_A_until_accepted``: reject ground-truth
  ``A`` whose max real eigenvalue sits in the near-stability-boundary band
  ``[-0.05, 0]`` so ``eig_clamp`` non-injectivity cannot confound recovery.
- ``snr_for_model``: map an integer SNR level onto each forward model's own noise
  mechanism (task/latent SNR kwarg vs spectral observation-noise log-amplitude).

Requirements
------------
VLREC-02 (metric hardening)
    Per-region R-squared must NOT be variance-pooled, and sign recovery must be
    masked to ``|true| > threshold``. These guard the two known artifacts:

    - **Pooled-R2 (R1).** A variance-weighted R-squared lets one high-variance
      region mask per-region failures; the sweep must report the per-region
      (uniform-average) reduction. See ``.planning/research/v0.7.0/PITFALLS.md``
      (R1) and ``.planning/research/v0.7.0/SUMMARY.md``.
    - **sign(0) (R2).** Unmasked sign comparison turns every structural zero into
      a guaranteed mismatch (``torch.sign(0) == 0``), deflating the score (the
      spurious Phase 25 HVAE-02 0.44). Masking to ``|true| > threshold`` removes
      the artifact; reuse ``pyro_dcm.models.hybrid_vae_dcm.masked_sign_recovery``.

VLREC-03 / VLROBUST-03 (design + characterization hardening)
    Ground-truth ``A`` near the stability boundary triggers ``eig_clamp``
    non-injectivity (pitfall N2): two distinct ``A_free`` map to the same clamped
    ``A``, so recovery is ill-posed by construction. Excluding the band
    ``[-0.05, 0]`` keeps ground truth comfortably inside the stable, injective
    regime. Shrinkage ``std_post / std_prior`` is the identifiability
    characterization input consumed here.

References
----------
.planning/research/v0.7.0/PITFALLS.md
    R1 (pooled R-squared), R2 (sign(0)), N2 (eig_clamp non-injectivity).
.planning/research/v0.7.0/SUMMARY.md
    Phase 30 recovery-sweep metric requirements.
benchmarks.latent_circuit_metrics.compute_trajectory_r_squared
    Per-region R-squared via ``pooled=False`` (consumed by the 30-02 driver).
pyro_dcm.models.hybrid_vae_dcm.masked_sign_recovery
    Masked sign recovery (``|true| > threshold``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import median
from typing import Any

import torch

from benchmarks.metrics import compute_summary_stats
from pyro_dcm.models.hybrid_vae_dcm import (  # type: ignore[import-untyped]
    masked_sign_recovery,
)

__all__ = [
    "compute_shrinkage_ratio",
    "assemble_cell_metrics",
    "exclude_near_boundary_A",
    "resample_A_until_accepted",
    "snr_for_model",
    "NEAR_BOUNDARY_LO",
    "NEAR_BOUNDARY_HI",
]


# ---------------------------------------------------------------------------
# Near-stability-boundary exclusion band (VLREC-03, pitfall N2)
# ---------------------------------------------------------------------------

NEAR_BOUNDARY_LO: float = -0.05
"""Lower edge of the rejected near-stability-boundary band on max Re eig.

Ground-truth ``A`` whose largest real eigenvalue falls in
``[NEAR_BOUNDARY_LO, NEAR_BOUNDARY_HI]`` = ``[-0.05, 0]`` is rejected: that band
is where ``eig_clamp`` becomes non-injective (pitfall N2) and recovery is
ill-posed by construction.
"""

NEAR_BOUNDARY_HI: float = 0.0
"""Upper edge of the rejected near-stability-boundary band on max Re eig.

A stable ``A`` already has max Re eig ``< 0``; this band-check additionally
rejects the marginally-stable sliver up to ``0`` so ground truth stays inside
the injective regime.
"""


def compute_shrinkage_ratio(
    std_post: torch.Tensor,
    std_prior: float,
) -> torch.Tensor:
    """Element-wise identifiability shrinkage ``std_post / std_prior``.

    Lower values indicate stronger posterior concentration (better
    identifiability). Mirrors ``benchmarks.bilinear_metrics.compute_shrinkage``
    but accepts a scalar ``std_prior`` (= ``sqrt(prior_variance)``) so it works
    for ``A`` (whose prior variance varies per model: ``1/64`` for BOLD,
    ``LC_A_PRIOR_VARIANCE`` for latent-circuit) and for ``B``.

    Parameters
    ----------
    std_post : torch.Tensor
        Posterior standard deviation per parameter element, any shape.
    std_prior : float
        Prior standard deviation (``sqrt(prior_variance)``). Must be ``> 0``.

    Returns
    -------
    torch.Tensor
        Element-wise shrinkage ratio, same shape as ``std_post``.

    Raises
    ------
    ValueError
        If ``std_prior`` is not strictly positive.

    References
    ----------
    .planning/REQUIREMENTS.md RECOV-07 (shrinkage soft target).
    benchmarks.bilinear_metrics.compute_shrinkage (constant-prior variant).
    """
    if not std_prior > 0.0:
        raise ValueError(
            f"std_prior must be strictly positive (expected > 0); got {std_prior!r}."
        )
    return std_post / std_prior


def _reshape_flat_to_matrices(
    flat_list: list[list[float]],
    n_regions: int,
) -> list[torch.Tensor]:
    """Reshape per-seed flat A lists back to ``(N, N)`` float64 tensors.

    Parameters
    ----------
    flat_list : list of list of float
        Per-seed flattened ``A`` matrices (row-major ``N * N`` entries each), as
        emitted by the VL runners (``a_true_list`` / ``a_inferred_list``).
    n_regions : int
        Region count ``N``; each flat list must have ``N * N`` entries.

    Returns
    -------
    list of torch.Tensor
        Per-seed ``(N, N)`` float64 matrices.

    Raises
    ------
    ValueError
        If any flat entry length is not ``n_regions ** 2``.
    """
    expected = n_regions * n_regions
    matrices: list[torch.Tensor] = []
    for i, flat in enumerate(flat_list):
        if len(flat) != expected:
            raise ValueError(
                f"flat A entry {i} has wrong length (expected {expected} for "
                f"N={n_regions}); got {len(flat)}."
            )
        matrices.append(
            torch.tensor(flat, dtype=torch.float64).reshape(n_regions, n_regions)
        )
    return matrices


def assemble_cell_metrics(
    cell_result: dict[str, Any],
    *,
    sign_threshold: float = 0.1,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Assemble one VL runner result into a hardened per-cell metric block.

    Consumes ONE forward-model VL runner result dict and aggregates its per-seed
    lists into a flat, JSON-serializable metric block. The two known artifacts
    are guarded by construction (VLREC-02):

    - ``sign_recovery_masked`` masks to ``|A_true| > sign_threshold`` (never
      ``sign(0)``), via ``masked_sign_recovery``.
    - ``r2_per_region`` is the per-region (NOT variance-pooled) R-squared; it is
      consumed from a per-seed ``r2_per_region_list`` the 30-02 driver populates
      by calling ``compute_trajectory_r_squared(..., pooled=False)``. This
      assembler never re-pools.

    Parameters
    ----------
    cell_result : dict
        A VL runner result dict. Must contain either ``rmse_list``
        (spectral/task) or ``a_rmse_list`` (latent_circuit), plus
        ``a_true_list``/``a_inferred_list`` when masked sign recovery is wanted.
        Optional driver-supplied per-seed keys: ``coverage_list``,
        ``r2_per_region_list``, ``shrinkage_list``.
    sign_threshold : float, optional
        Magnitude mask for sign recovery (``|A_true| > sign_threshold``).
        Default 0.1.
    ci_level : float, optional
        Credible-interval level associated with ``coverage_list`` (recorded in
        the output for provenance; coverage itself is computed upstream).
        Default 0.95.

    Returns
    -------
    dict
        Flat dict (Python floats / ``None`` / strings, NO tensors) with keys:
        ``variant``, ``method``, ``n_regions``, ``rmse_a`` (median/IQR via
        ``compute_summary_stats``), ``coverage_95``, ``sign_recovery_masked``,
        ``r2_per_region``, ``shrinkage``, ``n_success``, ``n_failed``,
        ``convergence_rate``, and ``*_note`` strings where a metric is absent.

    Raises
    ------
    ValueError
        If ``cell_result`` has neither ``rmse_list`` nor ``a_rmse_list``.

    References
    ----------
    .planning/research/v0.7.0/PITFALLS.md R1 (pooled R2), R2 (sign(0)).
    """
    if "rmse_list" in cell_result:
        rmse_values = cell_result["rmse_list"]
    elif "a_rmse_list" in cell_result:
        rmse_values = cell_result["a_rmse_list"]
    else:
        raise ValueError(
            "cell_result must contain 'rmse_list' (spectral/task) or "
            "'a_rmse_list' (latent_circuit); got keys "
            f"{sorted(cell_result.keys())}."
        )

    n_regions = int(cell_result.get("n_regions", 0))

    out: dict[str, Any] = {
        "variant": cell_result.get("variant"),
        "method": cell_result.get("method"),
        "n_regions": n_regions,
        "sign_threshold": float(sign_threshold),
        "ci_level": float(ci_level),
        "n_success": int(cell_result.get("n_success", len(rmse_values))),
        "n_failed": int(cell_result.get("n_failed", 0)),
        "convergence_rate": _convergence_rate(cell_result),
    }

    # RMSE: median + IQR across seeds.
    if rmse_values:
        out["rmse_a"] = compute_summary_stats([float(v) for v in rmse_values])
    else:
        out["rmse_a"] = None
        out["rmse_a_note"] = "no successful seeds; rmse list empty"

    # 95% CI coverage: median of the per-seed coverage list when present.
    cov_list = cell_result.get("coverage_list")
    if cov_list:
        out["coverage_95"] = float(median([float(v) for v in cov_list]))
    else:
        out["coverage_95"] = None
        out["coverage_note"] = (
            "no per-seed coverage_list in runner result (e.g. latent_circuit "
            "emits no A coverage); driver must supply it to populate this."
        )

    # Masked sign recovery: per-seed on reconstructed (N, N) A matrices.
    a_true_list = cell_result.get("a_true_list")
    a_inferred_list = cell_result.get("a_inferred_list")
    if a_true_list and a_inferred_list and n_regions > 0:
        if len(a_true_list) != len(a_inferred_list):
            raise ValueError(
                "a_true_list and a_inferred_list length mismatch (expected "
                f"equal); got {len(a_true_list)} vs {len(a_inferred_list)}."
            )
        true_mats = _reshape_flat_to_matrices(a_true_list, n_regions)
        pred_mats = _reshape_flat_to_matrices(a_inferred_list, n_regions)
        per_seed_sign = [
            masked_sign_recovery(pred, true, magnitude_threshold=sign_threshold)
            for pred, true in zip(pred_mats, true_mats, strict=True)
        ]
        finite = [s for s in per_seed_sign if s == s]  # drop nan
        out["sign_recovery_masked"] = float(median(finite)) if finite else float("nan")
    else:
        out["sign_recovery_masked"] = None
        out["sign_recovery_note"] = (
            "a_true_list / a_inferred_list absent or n_regions unknown; "
            "cannot reconstruct A matrices for masked sign recovery."
        )

    # Per-region R-squared (NOT pooled): median of driver-supplied per-seed list.
    r2_list = cell_result.get("r2_per_region_list")
    if r2_list:
        out["r2_per_region"] = float(median([float(v) for v in r2_list]))
    else:
        out["r2_per_region"] = None
        out["r2_note"] = (
            "no r2_per_region_list (spectral/task have no trajectory); the "
            "30-02 driver populates it for latent_circuit via "
            "compute_trajectory_r_squared(pooled=False)."
        )

    # Identifiability shrinkage: median of driver-supplied per-seed mean ratios.
    shrink_list = cell_result.get("shrinkage_list")
    if shrink_list:
        out["shrinkage"] = float(median([float(v) for v in shrink_list]))
    else:
        out["shrinkage"] = None
        out["shrinkage_note"] = (
            "no shrinkage_list; the driver supplies per-seed mean "
            "std_post/std_prior via compute_shrinkage_ratio."
        )

    return out


def _convergence_rate(cell_result: dict[str, Any]) -> float | None:
    """Extract convergence rate from a runner result dict, or ``None``.

    Prefers the runner's nested ``summary['convergence_rate']``; falls back to
    the mean of ``converged_list`` when present.

    Parameters
    ----------
    cell_result : dict
        VL runner result dict.

    Returns
    -------
    float or None
        Fraction of converged seeds, or ``None`` when unavailable.
    """
    summary = cell_result.get("summary")
    if isinstance(summary, dict) and "convergence_rate" in summary:
        return float(summary["convergence_rate"])
    converged = cell_result.get("converged_list")
    if converged:
        return float(sum(bool(c) for c in converged) / len(converged))
    return None


# ---------------------------------------------------------------------------
# Ground-truth-design guards (Task 2): near-boundary exclusion + SNR injection
# ---------------------------------------------------------------------------


def exclude_near_boundary_A(
    A: torch.Tensor,
    *,
    lo: float = NEAR_BOUNDARY_LO,
    hi: float = NEAR_BOUNDARY_HI,
) -> bool:
    """Accept ``A`` only if its max real eigenvalue is outside ``[lo, hi]``.

    Returns ``True`` (ACCEPTABLE) when the largest real eigenvalue of ``A`` is
    NOT in the near-stability-boundary band ``[lo, hi]`` (default ``[-0.05, 0]``),
    ``False`` otherwise. A genuinely stable ``A`` already has max Re eig ``< 0``;
    this band-check additionally rejects ``A`` whose max Re eig sits in
    ``[-0.05, 0]``, keeping ground truth comfortably inside the stable, injective
    regime where ``eig_clamp`` is well-defined (VLREC-03, pitfall N2).

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``.
    lo : float, optional
        Lower band edge on max Re eig. Default ``NEAR_BOUNDARY_LO`` (-0.05).
    hi : float, optional
        Upper band edge on max Re eig. Default ``NEAR_BOUNDARY_HI`` (0.0).

    Returns
    -------
    bool
        ``True`` if ``A`` is acceptable (max Re eig ``< lo`` or ``> hi``),
        ``False`` if it falls in the rejected band ``[lo, hi]``.

    References
    ----------
    .planning/research/v0.7.0/PITFALLS.md N2 (eig_clamp non-injectivity).
    """
    eigvals = torch.linalg.eigvals(A.to(torch.complex128))
    max_real = float(eigvals.real.max().item())
    in_band = lo <= max_real <= hi
    return not in_band


def resample_A_until_accepted(
    make_A_fn: Callable[[], torch.Tensor],
    *,
    max_tries: int = 50,
) -> torch.Tensor:
    """Resample ``A`` until one passes ``exclude_near_boundary_A``.

    Calls the zero-arg ``make_A_fn`` (typically a closure over
    ``make_stable_A_spectral`` / ``make_random_stable_A`` that advances a per-try
    seed) and returns the first draw whose max real eigenvalue is outside the
    near-boundary band. The 30-02 driver is responsible for making each call use
    a fresh seed so rejected draws are not regenerated identically.

    Parameters
    ----------
    make_A_fn : callable
        Zero-argument callable returning a candidate ``A`` tensor on each call.
    max_tries : int, optional
        Maximum number of draws before giving up. Default 50.

    Returns
    -------
    torch.Tensor
        The first ``A`` passing ``exclude_near_boundary_A``.

    Raises
    ------
    RuntimeError
        If no draw is accepted within ``max_tries`` attempts.

    References
    ----------
    .planning/research/v0.7.0/PITFALLS.md N2 (eig_clamp non-injectivity).
    """
    for _ in range(max_tries):
        A = make_A_fn()
        if exclude_near_boundary_A(A):
            return A
    raise RuntimeError(
        "resample_A_until_accepted exhausted attempts: expected at least one "
        f"A outside the near-boundary band [{NEAR_BOUNDARY_LO}, "
        f"{NEAR_BOUNDARY_HI}] within max_tries={max_tries}, got 0 accepted."
    )


def snr_for_model(variant: str, snr_level: float) -> dict[str, float]:
    """Map an SNR level onto a forward model's own noise knob.

    Each forward model exposes SNR through a different mechanism, so this is the
    ONE place SNR semantics diverge across the three models (keeping the matrix
    SNR axis comparable):

    - ``"task"`` / ``"latent_circuit"``: ``simulate_task_dcm`` /
      ``simulate_latent_circuit`` take an ``SNR`` kwarg directly, so return
      ``{"SNR": float(snr_level)}``.
    - ``"spectral"``: ``simulate_spectral_dcm`` has NO ``SNR`` kwarg; SNR is set
      via the observation-noise log-amplitude. Return
      ``{"noise_log_amplitude": -log(snr_level)}`` where higher SNR gives a
      more-negative log-amplitude (less observation noise). The 30-02 driver
      constructs the ``noise_params`` dict (the ``b``/``c`` observation-noise
      tensors) from this scalar.

    Parameters
    ----------
    variant : str
        Forward-model variant: ``"task"``, ``"latent_circuit"``, or
        ``"spectral"``.
    snr_level : float
        Requested signal-to-noise level (``> 0`` for the spectral mapping).

    Returns
    -------
    dict[str, float]
        Single-entry kwargs dict the driver splats into the simulate call.

    Raises
    ------
    ValueError
        On an unknown ``variant``, or non-positive ``snr_level`` for the
        spectral log-amplitude mapping.

    References
    ----------
    src.pyro_dcm.simulators.task_simulator.simulate_task_dcm (SNR kwarg).
    src.pyro_dcm.simulators.spectral_simulator.simulate_spectral_dcm
        (noise log-amplitude mechanism).
    """
    if variant in ("task", "latent_circuit"):
        return {"SNR": float(snr_level)}
    if variant == "spectral":
        if not snr_level > 0.0:
            raise ValueError(
                "spectral SNR mapping needs snr_level > 0 (log of a "
                f"non-positive value is undefined); got {snr_level!r}."
            )
        return {"noise_log_amplitude": -math.log(float(snr_level))}
    raise ValueError(
        "Unknown variant for snr_for_model (expected one of 'task', "
        f"'latent_circuit', 'spectral'); got {variant!r}."
    )
