r"""Phase 16.1 diagnostic: single-seed init_scale sweep for RECOV-04 B-RMSE.

Sweeps ``init_scale`` values on a single seed to identify the root cause of
the RECOV-04 B-RMSE=0.34 acceptance failure. The hypothesis (HIGH confidence)
is that ``init_scale=0.005`` is 0.5% of B's prior std of 1.0, causing the
AutoNormal guide to stick near its initialization basin.

This is a one-shot diagnostic script (NOT imported by the package, NOT
imported by tests). It reuses the existing benchmark helpers verbatim to
ensure bit-identical fixture construction and fitting.

Usage
-----
Local pre-check (single init_scale, <5 min)::

    python scripts/diagnose_phase16_init_scale.py \
        --seed 42 --init-scales 0.005 \
        --num-steps 500 \
        --output-json /tmp/phase16_1_local_precheck.json \
        --output-md /tmp/phase16_1_local_precheck.md

Full 4-init sweep (cluster dispatch via SLURM)::

    python scripts/diagnose_phase16_init_scale.py \
        --seed 42 --init-scales 0.005,0.05,0.1,0.5 \
        --num-steps 500 \
        --output-json cluster/results/phase16_1_init_scale_sweep_$SLURM_JOB_ID.json \
        --output-md cluster/results/phase16_1_init_scale_sweep_$SLURM_JOB_ID.md

References
----------
.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-RESEARCH.md
.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-01-PLAN.md Task 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so benchmarks/ and src/ are importable
# when the script is invoked directly (e.g., `python scripts/...`).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pyro  # noqa: E402
import torch  # noqa: E402

from benchmarks.bilinear_metrics import compute_shrinkage  # noqa: E402
from benchmarks.metrics import compute_rmse  # noqa: E402
from benchmarks.runners.task_bilinear import (  # noqa: E402
    _DT_MODEL,
    _DURATION,
    _TR,
    _fit_and_extract,
    _make_bilinear_ground_truth,
)
from pyro_dcm.forward_models.neural_state import parameterize_A  # noqa: E402


def _run_single_init_scale(
    seed: int,
    init_scale: float,
    num_steps: int,
    data: dict[str, Any],
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Fit one init_scale and extract diagnostics.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    init_scale : float
        AutoNormal guide init_scale to test.
    num_steps : int
        Number of SVI steps.
    data : dict
        Ground-truth fixture from ``_make_bilinear_ground_truth``.
    model_args : tuple
        Positional args for ``task_dcm_model``.
    model_kwargs : dict
        Keyword args (b_masks, stim_mod) for ``task_dcm_model``.

    Returns
    -------
    dict
        Per-init_scale diagnostic record.
    """
    # Seed isolation (same contract as _fit_and_extract).
    pyro.clear_param_store()
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    row: dict[str, Any] = {"init_scale": init_scale}
    t0 = time.time()

    try:
        posterior_bi, elapsed = _fit_and_extract(
            model_args,
            model_kwargs,
            guide_type="auto_normal",
            init_scale=init_scale,
            num_steps=num_steps,
            elbo_type="trace_elbo",
        )
        row["nan_at_step"] = None
        row["wall_seconds"] = elapsed

        # B posterior diagnostics.
        b_post_mean = posterior_bi["B_free_0"]["mean"]  # (N, N) tensor
        b_post_std = posterior_bi["B_free_0"]["std"]  # (N, N) tensor
        b_mask_0 = data["b_mask_0"]
        b_true_0 = data["B_true"][0]  # (N, N)

        # B-RMSE on non-null mask.
        nonnull_mask = b_mask_0 > 0
        if nonnull_mask.any():
            diff = (b_post_mean * b_mask_0 - b_true_0)[nonnull_mask]
            b_rmse_nonnull = float(torch.sqrt((diff**2).mean()).item())
        else:
            b_rmse_nonnull = 0.0
        row["b_rmse_nonnull"] = b_rmse_nonnull

        # A-RMSE.
        a_inferred = parameterize_A(posterior_bi["A_free"]["mean"])
        a_rmse_bi = float(compute_rmse(data["A_true"], a_inferred))
        row["a_rmse_bi"] = a_rmse_bi

        # Per-element B posterior mean and std.
        row["B_post_mean_per_element"] = b_post_mean.detach().tolist()
        row["B_post_std_per_element"] = b_post_std.detach().tolist()

        # Shrinkage on non-null elements (sigma_prior=1.0).
        shrinkage = compute_shrinkage(b_post_std.unsqueeze(0))  # (1, N, N)
        shrinkage_nonnull = shrinkage[0][nonnull_mask]
        row["shrinkage_nonnull_mean"] = float(shrinkage_nonnull.mean().item())
        row["shrinkage_nonnull_values"] = shrinkage_nonnull.tolist()

        # Final loss tail.
        row["final_loss_tail"] = posterior_bi.get("final_losses", [])

    except RuntimeError as err:
        err_str = str(err)
        row["wall_seconds"] = time.time() - t0
        if "NaN ELBO at step" in err_str:
            # Extract step number.
            import re

            match = re.search(r"NaN ELBO at step (\d+)", err_str)
            nan_step = int(match.group(1)) if match else -1
            row["nan_at_step"] = nan_step
        else:
            row["nan_at_step"] = -1
            row["error"] = err_str
        row["b_rmse_nonnull"] = None
        row["a_rmse_bi"] = None
        row["B_post_mean_per_element"] = None
        row["B_post_std_per_element"] = None
        row["shrinkage_nonnull_mean"] = None
        row["shrinkage_nonnull_values"] = None
        row["final_loss_tail"] = None

    return row


def _pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the winning init_scale from sweep results.

    Criteria (all three must hold):
    - ``b_rmse_nonnull < 0.20``
    - ``shrinkage_nonnull_mean in [0.05, 0.6]``
    - ``a_rmse_bi <= 0.15``

    Tie-break: smallest init_scale satisfying all three.

    Parameters
    ----------
    rows : list of dict
        Per-init_scale diagnostic records.

    Returns
    -------
    dict
        Winner info with ``chosen_init_scale`` (float or None) and
        ``escalation`` block if no winner.
    """
    candidates = []
    per_candidate: list[dict[str, Any]] = []

    for r in rows:
        b_rmse = r.get("b_rmse_nonnull")
        shrinkage = r.get("shrinkage_nonnull_mean")
        a_rmse = r.get("a_rmse_bi")

        gates: dict[str, Any] = {
            "init_scale": r["init_scale"],
            "b_rmse_pass": b_rmse is not None and b_rmse < 0.20,
            "shrinkage_pass": (
                shrinkage is not None and 0.05 <= shrinkage <= 0.6
            ),
            "a_rmse_pass": a_rmse is not None and a_rmse <= 0.15,
            "b_rmse_nonnull": b_rmse,
            "shrinkage_nonnull_mean": shrinkage,
            "a_rmse_bi": a_rmse,
        }
        per_candidate.append(gates)

        if gates["b_rmse_pass"] and gates["shrinkage_pass"] and gates["a_rmse_pass"]:
            candidates.append(r["init_scale"])

    if candidates:
        winner = min(candidates)
        return {
            "chosen_init_scale": winner,
            "escalation": None,
            "per_candidate_gates": per_candidate,
        }

    return {
        "chosen_init_scale": None,
        "escalation": {
            "message": (
                "No init_scale satisfied all three gates "
                "(b_rmse < 0.20, shrinkage in [0.05, 0.6], a_rmse <= 0.15)."
            ),
            "per_candidate_gates": per_candidate,
        },
        "per_candidate_gates": per_candidate,
    }


def _write_json(path: Path, output: dict[str, Any]) -> None:
    """Write machine-readable JSON output.

    Parameters
    ----------
    path : Path
        Output file path.
    output : dict
        Full diagnostic output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote JSON: {path}")


def _write_md(path: Path, output: dict[str, Any]) -> None:
    """Write human-readable Markdown table.

    Parameters
    ----------
    path : Path
        Output file path.
    output : dict
        Full diagnostic output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 16.1 init_scale Sweep Diagnostic",
        "",
        f"**Seed:** {output['seed']}",
        f"**Steps:** {output['num_steps']}",
        "",
        "## Results",
        "",
        "| init_scale | b_rmse_nonnull | a_rmse_bi | shrinkage_nonnull_mean "
        "| final_loss_last | nan_at_step | wall_s |",
        "|------------|----------------|-----------|------------------------"
        "|-----------------|-------------|--------|",
    ]

    for r in output["rows"]:
        b_rmse = r.get("b_rmse_nonnull")
        a_rmse = r.get("a_rmse_bi")
        shrinkage = r.get("shrinkage_nonnull_mean")
        tail = r.get("final_loss_tail")
        last_loss = f"{tail[-1]:.2f}" if tail and len(tail) > 0 else "N/A"
        nan_step = r.get("nan_at_step")
        nan_str = str(nan_step) if nan_step is not None else "none"
        wall = r.get("wall_seconds", 0)

        lines.append(
            f"| {r['init_scale']:.4f} "
            f"| {b_rmse:.4f} " if b_rmse is not None else f"| {'N/A':>14s} "
        )
        # Build row properly.
        b_str = f"{b_rmse:.4f}" if b_rmse is not None else "N/A"
        a_str = f"{a_rmse:.4f}" if a_rmse is not None else "N/A"
        s_str = f"{shrinkage:.4f}" if shrinkage is not None else "N/A"
        lines[-1] = (
            f"| {r['init_scale']:.4f} | {b_str:>14s} | {a_str:>9s} "
            f"| {s_str:>22s} | {last_loss:>15s} | {nan_str:>11s} "
            f"| {wall:>6.1f} |"
        )

    lines.append("")

    # Chosen init_scale.
    chosen = output["winner"]["chosen_init_scale"]
    if chosen is not None:
        lines.append(f"**Chosen init_scale: {chosen}**")
        lines.append("")
        # Arithmetic check vs research prediction.
        baseline_row = next(
            (r for r in output["rows"] if r["init_scale"] == 0.005),
            None,
        )
        if baseline_row and baseline_row.get("b_rmse_nonnull") is not None:
            baseline_b = baseline_row["b_rmse_nonnull"]
            collapse_to_zero = math.sqrt((0.4**2 + 0.3**2) / 2)
            ratio = baseline_b / collapse_to_zero * 100
            lines.append(
                f"Arithmetic check: baseline B-RMSE={baseline_b:.4f} is "
                f"{ratio:.1f}% of collapse-to-zero ({collapse_to_zero:.4f})."
            )
    else:
        lines.append("**Chosen init_scale: NONE -- escalation required.**")
        lines.append("")
        if output["winner"].get("escalation"):
            lines.append(
                f"Escalation: {output['winner']['escalation']['message']}"
            )
            lines.append("")
            lines.append("Per-candidate gate details:")
            for g in output["winner"]["escalation"]["per_candidate_gates"]:
                lines.append(
                    f"  - init_scale={g['init_scale']}: "
                    f"b_rmse_pass={g['b_rmse_pass']}, "
                    f"shrinkage_pass={g['shrinkage_pass']}, "
                    f"a_rmse_pass={g['a_rmse_pass']} "
                    f"(b_rmse={g['b_rmse_nonnull']}, "
                    f"shrinkage={g['shrinkage_nonnull_mean']}, "
                    f"a_rmse={g['a_rmse_bi']})"
                )

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote MD: {path}")


def main() -> None:
    """CLI entry point for the init_scale sweep diagnostic."""
    parser = argparse.ArgumentParser(
        description="Phase 16.1: single-seed init_scale sweep diagnostic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fixture generation and SVI (default: 42)",
    )
    parser.add_argument(
        "--init-scales",
        type=str,
        default="0.005,0.05,0.1,0.5",
        help="Comma-separated init_scale values to sweep (default: 0.005,0.05,0.1,0.5)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of SVI steps per fit (default: 500)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path for machine-readable JSON output",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        required=True,
        help="Path for human-readable Markdown output",
    )
    args = parser.parse_args()

    init_scales = [float(x.strip()) for x in args.init_scales.split(",")]
    seed = args.seed
    num_steps = args.num_steps

    print("Phase 16.1 init_scale sweep diagnostic")
    print(f"  seed={seed}, init_scales={init_scales}, num_steps={num_steps}")
    print()

    # Generate ground-truth fixture (bit-identical to benchmark runner).
    print("Generating ground-truth fixture...")
    data = _make_bilinear_ground_truth(3, seed)

    # Pre-flight: verify fixture is clean.
    if torch.isnan(data["bold"]).any() or torch.isinf(data["bold"]).any():
        print(f"ERROR: seed {seed} produces NaN/Inf BOLD (corrupt fixture).")
        print("Choose a different seed or investigate fixture generation.")
        raise SystemExit(1)

    print(f"  A_true shape: {data['A_true'].shape}")
    print(f"  B_true[0] non-null: B[1,0]={data['B_true'][0, 1, 0]:.3f}, "
          f"B[2,1]={data['B_true'][0, 2, 1]:.3f}")
    print()

    # Construct model_args exactly as in run_task_bilinear_svi.
    n_regions = 3
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.zeros(n_regions, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    t_eval = torch.arange(0, _DURATION, _DT_MODEL, dtype=torch.float64)
    model_args = (
        data["bold"],
        data["stimulus"],
        a_mask,
        c_mask,
        t_eval,
        _TR,
        _DT_MODEL,
    )
    model_kwargs = {
        "b_masks": [data["b_mask_0"]],
        "stim_mod": data["stim_mod"],
    }

    # Sweep init_scales.
    rows: list[dict[str, Any]] = []
    for i, init_scale in enumerate(init_scales):
        print(
            f"[{i + 1}/{len(init_scales)}] Fitting with init_scale={init_scale}..."
        )
        row = _run_single_init_scale(
            seed, init_scale, num_steps, data, model_args, model_kwargs,
        )
        rows.append(row)

        b_rmse = row.get("b_rmse_nonnull")
        nan_step = row.get("nan_at_step")
        wall = row.get("wall_seconds", 0)
        shrinkage = row.get("shrinkage_nonnull_mean")
        a_rmse = row.get("a_rmse_bi")

        b_str = f"{b_rmse:.4f}" if b_rmse is not None else "NaN"
        a_str = f"{a_rmse:.4f}" if a_rmse is not None else "NaN"
        s_str = f"{shrinkage:.4f}" if shrinkage is not None else "NaN"
        nan_str = f"step {nan_step}" if nan_step is not None else "none"
        print(
            f"  b_rmse_nonnull={b_str}, a_rmse_bi={a_str}, "
            f"shrinkage_nonnull={s_str}, nan={nan_str}, wall={wall:.1f}s"
        )
        print()

    # Pick winner.
    winner = _pick_winner(rows)
    chosen = winner["chosen_init_scale"]
    if chosen is not None:
        print(f"WINNER: init_scale={chosen}")
    else:
        print("NO WINNER: escalation required.")
        if winner.get("escalation"):
            print(f"  {winner['escalation']['message']}")

    # Assemble output.
    output: dict[str, Any] = {
        "seed": seed,
        "num_steps": num_steps,
        "init_scales": init_scales,
        "rows": rows,
        "winner": winner,
    }

    # Write outputs.
    _write_json(Path(args.output_json), output)
    _write_md(Path(args.output_md), output)

    print()
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
