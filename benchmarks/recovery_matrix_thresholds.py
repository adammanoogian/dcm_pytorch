"""Documented per-cell thresholds + classifier for the Phase 30 recovery sweep.

Implements VLREC-04's "pass OR identifiability-limit with evidence -- no silent
failures" rule. Each cell of the small validation grid is judged against a small
set of DOCUMENTED, deliberately-explicit thresholds and labelled either
``"pass"`` or ``"identifiability_limit"``. A failing cell is NEVER an error: it is
a documented limit carrying the evidence (shrinkage median, coverage, RMSE IQR,
convergence rate) that explains *why* it fell short. The classifier only raises
when its input is structurally malformed (a missing key the upstream
``assemble_cell_metrics`` contract guarantees).

Thresholds are provisional. The v0.7.0 threshold-research note records that NO
principled Fisher-information per-cell RMSE threshold exists yet, so this small
prove-the-harness grid uses documented defaults and audits outliers, rather than
a derived bound (see ``.planning/research/v0.7.0/SUMMARY.md``). The constants are
therefore explicit module-level values with cited provenance so a later phase can
revise them in one place.

A check whose underlying metric is ``None`` (e.g. spectral/task have no
per-region R-squared, latent_circuit emits no scalar A coverage) is SKIPPED with
``"pass": None`` and a note -- a missing metric is never auto-failed (that would
manufacture a spurious identifiability limit out of a model that simply does not
expose that quantity).

References
----------
.planning/research/v0.7.0/SUMMARY.md
    Threshold-research note: no principled per-cell RMSE threshold yet; use
    documented defaults + audit outliers. Laplace overconfidence (job 55772525)
    means shrinkage is often far below its soft target -- that is the
    identifiability signal, not a defect.
.planning/REQUIREMENTS.md
    RECOV-05 (sign recovery), RECOV-06 (coverage), RECOV-07 (shrinkage soft
    target), VLREC-04 (pass-or-documented-limit, no silent failures).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RMSE_A_THRESHOLD",
    "SIGN_RECOVERY_THRESHOLD",
    "COVERAGE_95_FLOOR",
    "SHRINKAGE_SOFT_TARGET",
    "classify_cell",
]


# ---------------------------------------------------------------------------
# Documented per-cell thresholds (provisional; revise here in one place).
# ---------------------------------------------------------------------------

RMSE_A_THRESHOLD: float = 0.05
"""Maximum acceptable median A-RMSE per cell (PROVISIONAL).

Provenance: the v0.7.0 threshold-research SUMMARY's gap-4 recommendation
("RMSE < 0.05 universally; audit outliers"). No principled Fisher-information
per-cell bound exists yet, so this is a documented default for the
prove-the-harness grid, not a derived tolerance -- it is expected to be revised
once a real identifiability bound is available. A cell with median A-RMSE
``<= RMSE_A_THRESHOLD`` passes the RMSE check.
"""

SIGN_RECOVERY_THRESHOLD: float = 0.80
"""Minimum acceptable MASKED sign-recovery fraction per cell.

Provenance: mirrors the RECOV-05 / latent-circuit convention (sign recovery
``>= 0.80``). Masked (``|A_true| > threshold``) so the ``sign(0)`` structural-zero
artifact (pitfall R2) cannot deflate it. A cell with masked sign recovery
``>= SIGN_RECOVERY_THRESHOLD`` passes the sign check.
"""

COVERAGE_95_FLOOR: float = 0.85
"""Minimum acceptable 95% credible-interval coverage per cell.

Provenance: mirrors the RECOV-06 / latent-circuit convention (coverage floor
``0.85`` for a nominal-95% interval). A cell with median 95% coverage
``>= COVERAGE_95_FLOOR`` passes the coverage check. Coverage near or above the
floor (and not wildly over 1.0) indicates calibrated uncertainty.
"""

SHRINKAGE_SOFT_TARGET: float = 0.7
"""Informational shrinkage (``std_post / std_prior``) soft target -- NOT a gate.

Provenance: RECOV-07 soft target. This is reported as EVIDENCE only and never
passes/fails a cell. Variational-Laplace overconfidence (job 55772525) means
shrinkage at high SNR is routinely FAR below this target; that very-low shrinkage
is itself the expected identifiability/overconfidence signal, not a failure. The
classifier surfaces the observed shrinkage so the overconfident regime is
documented with evidence (VLROBUST-03), but it does not reject a cell for low
shrinkage.
"""


def _check(
    observed: float | None,
    threshold: float,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Build one threshold-check record (skipped when the metric is ``None``).

    Parameters
    ----------
    observed : float or None
        The observed metric value, or ``None`` when the model does not expose it.
    threshold : float
        The documented threshold the metric is compared against.
    higher_is_better : bool
        If ``True`` the check passes when ``observed >= threshold`` (sign,
        coverage); if ``False`` it passes when ``observed <= threshold`` (RMSE).

    Returns
    -------
    dict
        ``{"observed", "threshold", "pass"}`` where ``pass`` is ``None`` (and a
        ``"note"`` is attached) when ``observed is None`` -- a missing metric is
        skipped, NOT auto-failed.
    """
    if observed is None:
        return {
            "observed": None,
            "threshold": threshold,
            "pass": None,
            "note": (
                "metric is None (model does not expose it); "
                "check skipped, not failed"
            ),
        }
    passed = observed >= threshold if higher_is_better else observed <= threshold
    return {"observed": float(observed), "threshold": threshold, "pass": bool(passed)}


def _require(cell_metrics: dict[str, Any], key: str) -> None:
    """Raise an expected-vs-actual ``KeyError`` if a contracted key is absent.

    Parameters
    ----------
    cell_metrics : dict
        The per-cell metric block produced by ``assemble_cell_metrics``.
    key : str
        A key the assembler contract guarantees to emit.

    Raises
    ------
    KeyError
        If ``key`` is missing (structurally malformed input).
    """
    if key not in cell_metrics:
        raise KeyError(
            f"cell_metrics is structurally malformed: expected key {key!r} "
            f"(contracted by assemble_cell_metrics); got keys "
            f"{sorted(cell_metrics.keys())}."
        )


def classify_cell(
    cell_metrics: dict[str, Any],
    *,
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Classify one cell as ``pass`` or ``identifiability_limit`` with evidence.

    Implements VLREC-04: every cell gets an explicit verdict and, when it falls
    short, the evidence documenting the identifiability limit. The cell passes iff
    every check whose metric is PRESENT passes (RMSE ``<=`` threshold, sign
    recovery ``>=`` threshold, coverage ``>=`` floor). A check whose metric is
    ``None`` is skipped (``"pass": None``) and does not affect the verdict -- a
    missing metric is never auto-failed. Shrinkage is reported as evidence only
    and never gates the verdict.

    Parameters
    ----------
    cell_metrics : dict
        A per-cell metric block as produced by
        ``benchmarks.recovery_matrix_metrics.assemble_cell_metrics``. Must contain
        the contracted keys ``rmse_a``, ``sign_recovery_masked``, ``coverage_95``,
        ``shrinkage``, ``convergence_rate`` (values may be ``None``).
    cell : dict
        The grid cell descriptor (``variant``, ``n_regions``, ``snr``), recorded
        in the returned ``evidence`` block for provenance.

    Returns
    -------
    dict
        ``{"status": "pass" | "identifiability_limit", "checks": {...},
        "evidence": {...}, "reason": str}``. ``checks`` maps each metric name to a
        ``{"observed", "threshold", "pass"}`` record; ``evidence`` carries the
        shrinkage median, coverage, RMSE median/IQR and convergence rate;
        ``reason`` names which checks failed (or confirms a pass).

    Raises
    ------
    KeyError
        Only if ``cell_metrics`` is structurally malformed (a contracted key is
        absent). A FAILING cell is a documented limit, never an exception.

    References
    ----------
    .planning/REQUIREMENTS.md VLREC-04 (pass-or-documented-limit; no silent
    failures).
    """
    for key in ("rmse_a", "sign_recovery_masked", "coverage_95", "shrinkage"):
        _require(cell_metrics, key)

    rmse_a = cell_metrics["rmse_a"]
    rmse_median: float | None
    rmse_iqr: tuple[float | None, float | None]
    if isinstance(rmse_a, dict):
        rmse_median = rmse_a.get("median")
        rmse_iqr = (rmse_a.get("q25"), rmse_a.get("q75"))
    else:
        rmse_median = None
        rmse_iqr = (None, None)

    checks = {
        "rmse_a": _check(rmse_median, RMSE_A_THRESHOLD, higher_is_better=False),
        "sign_recovery_masked": _check(
            cell_metrics["sign_recovery_masked"],
            SIGN_RECOVERY_THRESHOLD,
            higher_is_better=True,
        ),
        "coverage_95": _check(
            cell_metrics["coverage_95"], COVERAGE_95_FLOOR, higher_is_better=True
        ),
    }

    failed = [name for name, rec in checks.items() if rec["pass"] is False]
    skipped = [name for name, rec in checks.items() if rec["pass"] is None]

    shrinkage = cell_metrics["shrinkage"]
    evidence = {
        "variant": cell.get("variant"),
        "n_regions": cell.get("n_regions"),
        "snr": cell.get("snr"),
        "shrinkage_median": shrinkage,
        "shrinkage_soft_target": SHRINKAGE_SOFT_TARGET,
        "coverage_95": cell_metrics["coverage_95"],
        "rmse_a_median": rmse_median,
        "rmse_a_iqr": rmse_iqr,
        "convergence_rate": cell_metrics.get("convergence_rate"),
        "r2_per_region": cell_metrics.get("r2_per_region"),
        "overconfident_low_shrinkage": (
            shrinkage is not None and shrinkage < SHRINKAGE_SOFT_TARGET
        ),
    }

    if failed:
        reason = "identifiability limit: failed checks " + ", ".join(
            f"{name}(observed={checks[name]['observed']}, "
            f"threshold={checks[name]['threshold']})"
            for name in failed
        )
        if skipped:
            reason += f"; skipped (metric None): {', '.join(skipped)}"
        return {
            "status": "identifiability_limit",
            "checks": checks,
            "evidence": evidence,
            "reason": reason,
        }

    reason = "pass: all present checks met documented thresholds"
    if skipped:
        reason += f" (skipped, metric None: {', '.join(skipped)})"
    return {
        "status": "pass",
        "checks": checks,
        "evidence": evidence,
        "reason": reason,
    }
