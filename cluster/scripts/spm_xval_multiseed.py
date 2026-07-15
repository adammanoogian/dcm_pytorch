"""Multi-seed VL-vs-SPM12 cross-validation divergence investigation (Phase 32-03).

Runs :func:`validation.run_vl_validation.run_vl_spectral_dcm_validation` across
several seeds on M3 (where MATLAB R2022a + SPM12 are licensed) and aggregates the
two open questions from the single-seed run (job 56407192):

1. **Is the matched-F gap a constant normalization offset?** If
   ``vl_F - spm_F`` is (nearly) constant across seeds, the two engines compute
   the same free energy up to a fixed additive constant -- the strict-5% absolute
   gate is then infeasible *by convention* (research pitfall S3), but a
   *relative* / offset-corrected F agreement is the meaningful quantity.

2. **Is the posterior-mean divergence systematic?** Per-seed off-diagonal
   (free-parameter) agreement vs ground truth for BOTH engines tells us whether
   SPM systematically lands off-truth on the injected analytic CSD, or whether
   seed 42 was an outlier.

NEVER compares element-wise ``Cp`` nor absolute F across *different models* (S3);
the only cross-model statistic is the relative ranking agreement, carried through
unchanged from the per-seed orchestrator.

Run via ``cluster/sbatch/spm_xval_multiseed.sbatch`` (sets ``MATLAB_PATH`` +
``SPM12_PATH``). Seeds come from the ``XVAL_SEEDS`` env var (CSV; default
``42,43,44,45,46``). Output:
``cluster/results/spm_xval_multiseed_<jobid>.json``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from config import CLUSTER_RESULTS_DIR  # noqa: E402

from validation.run_vl_validation import (  # noqa: E402
    run_vl_spectral_dcm_validation,
)


def _offdiag_free_param_errors(
    vl_a_free: np.ndarray,
    spm_ep_a: np.ndarray,
    a_true: np.ndarray,
) -> dict:
    """Off-diagonal (coupling) free-parameter agreement, S1-safe.

    The self-connection diagonal is excluded: VL reports it as a deviation from
    the fixed ``-0.5`` baseline (``~0``) while SPM reports the raw free parameter
    (``Ep.A`` diagonal under the ``-exp(x)/2`` convention), so the two are not
    element-wise comparable. The off-diagonal coupling entries ARE the shared
    free-parameter space and the only honest mean-agreement signal.

    Parameters
    ----------
    vl_a_free : np.ndarray
        VL posterior mean A in free-parameter space, shape ``(N, N)``.
    spm_ep_a : np.ndarray
        SPM12 ``Ep.A`` posterior mean, shape ``(N, N)``.
    a_true : np.ndarray
        Ground-truth A, shape ``(N, N)``.

    Returns
    -------
    dict
        Off-diagonal VL/SPM values, their pairwise relative error, and each
        engine's relative error against the ground-truth off-diagonals.
    """
    n = vl_a_free.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    vl_off = vl_a_free[offdiag]
    spm_off = spm_ep_a[offdiag]
    true_off = a_true[offdiag]

    def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)

    return {
        "vl_offdiag": vl_off.tolist(),
        "spm_offdiag": spm_off.tolist(),
        "true_offdiag": true_off.tolist(),
        "vl_vs_spm_rel_error": _rel(vl_off, spm_off).tolist(),
        "vl_vs_true_rel_error": _rel(vl_off, true_off).tolist(),
        "spm_vs_true_rel_error": _rel(spm_off, true_off).tolist(),
        "vl_vs_spm_max_rel_error": float(_rel(vl_off, spm_off).max()),
    }


def main() -> int:
    """Run the multi-seed cross-validation sweep and aggregate divergence stats.

    Returns
    -------
    int
        ``0`` on success (a per-seed fit failure is RECORDED, not fatal);
        non-zero only on an unexpected top-level exception.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    seeds_csv = os.environ.get("XVAL_SEEDS", "42,43,44,45,46")
    seeds = [int(s) for s in seeds_csv.split(",") if s.strip()]

    out_dir = CLUSTER_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spm_xval_multiseed_{job_id}.json"

    print(f"Multi-seed VL-vs-SPM12 cross-validation; seeds={seeds}")

    per_seed: list[dict] = []
    f_offsets: list[float] = []
    ranking_rates: list[float] = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        try:
            res = run_vl_spectral_dcm_validation(
                seed=seed, n_regions=2, max_iter=64,
            )
            vl_a_free = np.asarray(res["vl_A_free"])
            spm_ep_a = np.asarray(res["spm_Ep_A"])
            a_true = np.asarray(res["A_true"])
            off = _offdiag_free_param_errors(vl_a_free, spm_ep_a, a_true)
            f_off = float(res["vl_F"]) - float(res["spm_F"])
            rate = float(res["ranking"]["agreement_rate"])
            f_offsets.append(f_off)
            ranking_rates.append(rate)
            per_seed.append({
                "seed": seed,
                "status": "ok",
                "vl_F": float(res["vl_F"]),
                "spm_F": float(res["spm_F"]),
                "f_offset_vl_minus_spm": f_off,
                "ranking_agreement_rate": rate,
                "matched_f_relative_error": float(
                    res["matched_f_comparison"]["relative_error"]
                ),
                "offdiag_free_params": off,
            })
            print(
                f"  vl_F={res['vl_F']:.3f} spm_F={res['spm_F']:.3f} "
                f"offset={f_off:.3f} ranking={rate:.2f} "
                f"offdiag_vl_vs_spm_max={off['vl_vs_spm_max_rel_error']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - record, don't crash a seed
            print(f"  seed {seed} FAILED: {exc}")
            per_seed.append({
                "seed": seed,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    # --- Aggregate the two investigation questions -------------------------
    summary: dict = {"n_seeds_ok": len(f_offsets)}
    if f_offsets:
        arr = np.asarray(f_offsets)
        summary["f_offset_mean"] = float(arr.mean())
        summary["f_offset_std"] = float(arr.std())
        summary["f_offset_cv"] = float(
            arr.std() / max(abs(arr.mean()), 1e-12)
        )
        # If the offset is ~constant, the engines agree up to a constant: an
        # offset-corrected (relative-to-mean) F gate is the honest test.
        summary["f_offset_is_constant"] = bool(
            arr.std() / max(abs(arr.mean()), 1e-12) < 0.05
        )
        summary["ranking_agreement_all_seeds"] = ranking_rates
        summary["ranking_agreement_min"] = float(min(ranking_rates))

    payload = {
        "status": "ok",
        "job_id": job_id,
        "seeds": seeds,
        "note": (
            "Phase 32-03 divergence investigation. Q1: is vl_F - spm_F a "
            "constant offset (f_offset_is_constant)? Q2: is the posterior-mean "
            "divergence systematic (per-seed offdiag_free_params vs truth)? "
            "Absolute F across DIFFERENT models is NEVER compared (S3); ranking "
            "is the only cross-model statistic."
        ),
        "summary": summary,
        "per_seed": per_seed,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(f"SUMMARY: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
