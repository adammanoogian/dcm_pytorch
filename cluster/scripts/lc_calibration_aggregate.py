"""Aggregate latent circuit calibration array job results.

Reads per-task JSON files from cluster/results/lc_calib_*.json,
merges into a single summary, and prints the top-10 table.

Usage:
    python cluster/scripts/lc_calibration_aggregate.py [job_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLUSTER_RESULTS_DIR  # noqa: E402


def aggregate(job_id: str | None = None) -> list[dict]:
    """Collect per-task results into a sorted list.

    Parameters
    ----------
    job_id : str or None
        SLURM job ID to filter files. None collects all lc_calib_*.json.

    Returns
    -------
    list[dict]
        Combined results sorted by task_id.
    """
    results_dir = CLUSTER_RESULTS_DIR
    if job_id:
        pattern = f"lc_calib_{job_id}_*.json"
    else:
        pattern = "lc_calib_*.json"

    files = sorted(results_dir.glob(pattern))
    if not files:
        print(f"No files matching {pattern} in {results_dir}")
        return []

    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))

    results.sort(key=lambda r: r.get("task_id", 0))
    return results


def print_summary(results: list[dict]) -> None:
    """Print top-10 table sorted by B-RMSE.

    Parameters
    ----------
    results : list[dict]
        Combined sweep results.
    """
    ok_results = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    print(f"\nTotal: {len(results)} combos, {len(ok_results)} OK, "
          f"{len(failed)} failed")

    if failed:
        print("\nFailed combos:")
        for r in failed:
            print(f"  task {r.get('task_id', '?')}: "
                  f"a_var={r['a_prior_var']:.4f}, "
                  f"b_var={r['b_prior_var']:.2f}, "
                  f"init={r['init_scale']:.3f} -- {r.get('error', '?')}")

    if not ok_results:
        print("\nNo successful combos!")
        return

    ok_results.sort(key=lambda r: r["b_rmse"])

    print("\n" + "=" * 90)
    print("  CALIBRATION SWEEP RESULTS (top 10 by B-RMSE)")
    print("=" * 90)
    print(
        f"{'Rank':<5} {'A_var':<8} {'B_var':<7} {'Init':<6} "
        f"{'B-RMSE':<9} {'A-RMSE':<9} {'R2':<8} {'Shrink_B':<10} "
        f"{'Sign':<6} {'Time':<6}"
    )
    print("-" * 90)

    for i, r in enumerate(ok_results[:10]):
        shrink_b = r.get("shrinkage_B")
        shrink_str = f"{shrink_b:.3f}" if shrink_b is not None else "N/A"
        sign_str = (
            f"{r['sign_recovery']:.2f}" if r.get("sign_recovery") else "N/A"
        )
        elapsed = r.get("elapsed_s", 0)
        print(
            f"{i + 1:<5} {r['a_prior_var']:<8.4f} {r['b_prior_var']:<7.2f} "
            f"{r['init_scale']:<6.3f} {r['b_rmse']:<9.4f} "
            f"{r['a_rmse']:<9.4f} {r['trajectory_r_squared']:<8.3f} "
            f"{shrink_str:<10} {sign_str:<6} {elapsed:<6.0f}"
        )

    print("-" * 90)

    valid = [
        r for r in ok_results
        if r.get("shrinkage_B") is None or r["shrinkage_B"] < 0.95
    ]
    if valid:
        winner = valid[0]
        print(
            f"\n  WINNER: a_var={winner['a_prior_var']:.6f}, "
            f"b_var={winner['b_prior_var']:.2f}, "
            f"init_scale={winner['init_scale']:.3f}"
        )
        print(
            f"  B-RMSE={winner['b_rmse']:.4f}, "
            f"A-RMSE={winner['a_rmse']:.4f}, "
            f"R2={winner['trajectory_r_squared']:.3f}, "
            f"sign={winner['sign_recovery']:.3f}"
        )
    else:
        print("\n  WARNING: No valid winner (all shrinkage_B > 0.95)")

    print("=" * 90)

    merged_path = CLUSTER_RESULTS_DIR / "lc_calibration_sweep_merged.json"
    with open(merged_path, "w") as f:
        json.dump(
            {
                "results": results,
                "n_total": len(results),
                "n_ok": len(ok_results),
                "winner": valid[0] if valid else None,
            },
            f,
            indent=2,
        )
    print(f"\nMerged results saved to: {merged_path}")


if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    results = aggregate(job_id)
    if results:
        print_summary(results)
