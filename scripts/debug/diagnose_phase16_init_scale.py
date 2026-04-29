"""Phase 16.1 single-seed init_scale sweep diagnostic.

Run a deterministic single-seed sweep of the bilinear task-DCM SVI fit at
multiple guide ``init_scale`` values, capturing posterior B mean / std,
B-RMSE on the non-null mask, A-RMSE, ELBO tail, and NaN status. Used to
root-cause the Phase 16 RECOV-04 cluster failure (B-RMSE 0.3424 across all
10 seeds -- suspected systematic posterior collapse to AutoNormal init at
``init_scale = 0.005``) and pick the smallest init_scale that pushes
B-RMSE below 0.20 AND moves posterior std on non-null elements off the
init basin (target shrinkage_nonnull mean in [0.05, 0.6]).

The chosen winner is consumed verbatim by Plan 16.1-02 to update the
runner-level ``_BILINEAR_INIT_SCALE`` constant.

References
----------
.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-01-PLAN.md
.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-RESEARCH.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyro
import torch

# Reuse the existing harness verbatim -- DO NOT FORK ground-truth construction.
from benchmarks.bilinear_metrics import compute_b_rmse_magnitude, compute_shrinkage
from benchmarks.metrics import compute_rmse
from benchmarks.runners.task_bilinear import (
    _DT_MODEL,
    _DURATION,
    _TR,
    _fit_and_extract,
    _make_bilinear_ground_truth,
)
from pyro_dcm.forward_models.neural_state import parameterize_A

# Winner-selection gates (Plan 16.1-01 must_haves + RESEARCH.md Sec 7).
_B_RMSE_PASS_THRESHOLD: float = 0.20
_SHRINKAGE_NONNULL_LO: float = 0.05
_SHRINKAGE_NONNULL_HI: float = 0.60
_A_RMSE_MAX: float = 0.15

# RECOV-04 magnitude mask (B_true[1,0]=0.4 and B_true[2,1]=0.3 are above 0.1).
_RECOV_04_MAGNITUDE_MASK: float = 0.1

# Research-prediction baseline at init_scale=0.005 (cluster job 54933838 mean).
_BASELINE_B_RMSE_REF: float = 0.3424
_BASELINE_SHRINKAGE_REF: float = 0.008


def _parse_init_scales(spec: str) -> list[float]:
    """Parse a comma-separated init_scale list (CLI helper)."""
    try:
        scales = [float(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError as err:  # pragma: no cover (CLI parse path)
        raise argparse.ArgumentTypeError(
            f"--init-scales must be comma-separated floats; got '{spec}'"
        ) from err
    if not scales:
        raise argparse.ArgumentTypeError(
            f"--init-scales produced empty list; got '{spec}'"
        )
    return scales


def _build_model_args(
    data: dict[str, Any], n_regions: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Construct task_dcm_model positional + bilinear kwargs.

    Mirrors ``run_task_bilinear_svi`` (task_bilinear.py:706-726) verbatim so
    the diagnostic harness produces a bit-identical fit to the cluster runner.
    """
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
    bilinear_kwargs = {
        "b_masks": [data["b_mask_0"]],
        "stim_mod": data["stim_mod"],
    }
    return model_args, bilinear_kwargs


def _compute_metrics(
    posterior: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Compute b_rmse_nonnull, a_rmse_bi, posterior B stats from a fit."""
    # Posterior B (J=1 modulator). post["B_free_0"]["mean"] shape (N, N).
    b_mean_nn = torch.tensor(
        posterior["B_free_0"]["mean"], dtype=torch.float64
    )
    b_std_nn = torch.tensor(
        posterior["B_free_0"]["std"], dtype=torch.float64
    )
    # Wrap to (J, N, N) = (1, N, N) for compute_b_rmse_magnitude.
    b_mean_jnn = b_mean_nn.unsqueeze(0)
    b_std_jnn = b_std_nn.unsqueeze(0)

    b_rmse_nonnull = compute_b_rmse_magnitude(
        data["B_true"],
        b_mean_jnn,
        magnitude_threshold=_RECOV_04_MAGNITUDE_MASK,
    )

    # A-RMSE on the bilinear posterior (parameterize_A enforces stability).
    a_post = parameterize_A(
        torch.tensor(posterior["A_free"]["mean"], dtype=torch.float64)
    )
    a_rmse_bi = compute_rmse(data["A_true"], a_post)

    # Shrinkage (std_post / sigma_prior=1.0) on non-null mask only.
    shrinkage_jnn = compute_shrinkage(b_std_jnn)
    nonnull_mask = torch.abs(data["B_true"]) > _RECOV_04_MAGNITUDE_MASK
    if nonnull_mask.any():
        shrinkage_nonnull_mean = float(shrinkage_jnn[nonnull_mask].mean().item())
    else:
        shrinkage_nonnull_mean = 0.0

    return {
        "b_rmse_nonnull": b_rmse_nonnull,
        "a_rmse_bi": a_rmse_bi,
        "B_post_mean_per_element": b_mean_nn.tolist(),
        "B_post_std_per_element": b_std_nn.tolist(),
        "shrinkage_nonnull_mean": shrinkage_nonnull_mean,
    }


def run_one(
    init_scale: float,
    seed: int,
    num_steps: int,
) -> dict[str, Any]:
    """Run a single SVI fit at one init_scale; never raises.

    Catches ``RuntimeError`` from ``run_svi`` (NaN ELBO at step k) and records
    the failing step so the diagnostic table surfaces NaN seeds.
    """
    # Match _fit_and_extract's contract: clear param store + reseed BEFORE
    # the fit. _fit_and_extract itself calls clear_param_store, but we add
    # the seed reset here so the diagnostic is reproducible across init_scales
    # (each init_scale gets a fresh, identically-seeded fixture + guide init).
    pyro.clear_param_store()
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)

    # Build the fixture for THIS seed (same builder as the runner).
    data = _make_bilinear_ground_truth(3, seed)

    # Pre-flight corruption check: a divergent fixture corrupts the diagnostic.
    if (
        torch.isnan(data["bold"]).any().item()
        or torch.isinf(data["bold"]).any().item()
    ):
        return {
            "init_scale": init_scale,
            "b_rmse_nonnull": None,
            "a_rmse_bi": None,
            "B_post_mean_per_element": None,
            "B_post_std_per_element": None,
            "shrinkage_nonnull_mean": None,
            "final_loss_tail": [],
            "nan_at_step": -1,
            "wall_seconds": 0.0,
            "error": (
                "fixture_corrupt: ground-truth BOLD contains NaN/Inf at "
                f"seed {seed} (A+B unstable under sustained u_mod=1)"
            ),
        }

    model_args, bilinear_kwargs = _build_model_args(data, n_regions=3)

    t0 = time.time()
    try:
        posterior, elapsed = _fit_and_extract(
            model_args,
            bilinear_kwargs,
            guide_type="auto_normal",
            init_scale=init_scale,
            num_steps=num_steps,
            elbo_type="trace_mean_field",
        )
    except RuntimeError as err:
        msg = str(err)
        # _run_svi raises "NaN ELBO at step {step}" -- extract the step.
        nan_step: int | None = None
        if "NaN ELBO at step" in msg:
            try:
                nan_step = int(msg.split("step")[-1].split()[0].strip())
            except (ValueError, IndexError):
                nan_step = -1
        return {
            "init_scale": init_scale,
            "b_rmse_nonnull": None,
            "a_rmse_bi": None,
            "B_post_mean_per_element": None,
            "B_post_std_per_element": None,
            "shrinkage_nonnull_mean": None,
            "final_loss_tail": [],
            "nan_at_step": nan_step if nan_step is not None else -1,
            "wall_seconds": time.time() - t0,
            "error": msg,
        }

    metrics = _compute_metrics(posterior, data)
    final_losses = list(posterior.get("final_losses", []))

    return {
        "init_scale": init_scale,
        "b_rmse_nonnull": metrics["b_rmse_nonnull"],
        "a_rmse_bi": metrics["a_rmse_bi"],
        "B_post_mean_per_element": metrics["B_post_mean_per_element"],
        "B_post_std_per_element": metrics["B_post_std_per_element"],
        "shrinkage_nonnull_mean": metrics["shrinkage_nonnull_mean"],
        "final_loss_tail": final_losses,
        "nan_at_step": None,
        "wall_seconds": elapsed,
        "error": None,
    }


def pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the smallest init_scale satisfying all three winner gates.

    Gates (Plan 16.1-01 must_have #2 + RESEARCH Sec 7):
      - ``b_rmse_nonnull`` < 0.20
      - ``shrinkage_nonnull_mean`` in [0.05, 0.60]
      - ``a_rmse_bi`` <= 0.15

    Returns
    -------
    dict
        ``chosen_init_scale`` (float | None) plus a structured ``escalation``
        block listing per-candidate gate failures.
    """
    candidates: list[tuple[float, dict[str, Any]]] = []
    per_candidate_gate_status: list[dict[str, Any]] = []
    for row in rows:
        if row.get("error") is not None:
            per_candidate_gate_status.append(
                {
                    "init_scale": row["init_scale"],
                    "gate_b_rmse": False,
                    "gate_shrinkage": False,
                    "gate_a_rmse": False,
                    "reason": f"errored: {row['error']}",
                }
            )
            continue
        b_rmse = row["b_rmse_nonnull"]
        shrink = row["shrinkage_nonnull_mean"]
        a_rmse = row["a_rmse_bi"]
        gate_b = b_rmse is not None and b_rmse < _B_RMSE_PASS_THRESHOLD
        gate_s = (
            shrink is not None
            and _SHRINKAGE_NONNULL_LO <= shrink <= _SHRINKAGE_NONNULL_HI
        )
        gate_a = a_rmse is not None and a_rmse <= _A_RMSE_MAX
        per_candidate_gate_status.append(
            {
                "init_scale": row["init_scale"],
                "gate_b_rmse": bool(gate_b),
                "gate_shrinkage": bool(gate_s),
                "gate_a_rmse": bool(gate_a),
                "b_rmse": b_rmse,
                "shrinkage_nonnull": shrink,
                "a_rmse": a_rmse,
            }
        )
        if gate_b and gate_s and gate_a:
            candidates.append((row["init_scale"], row))
    if not candidates:
        return {
            "chosen_init_scale": None,
            "escalation": {
                "reason": (
                    "no init_scale satisfies all three gates "
                    f"(b_rmse < {_B_RMSE_PASS_THRESHOLD}, "
                    f"shrinkage_nonnull in [{_SHRINKAGE_NONNULL_LO}, "
                    f"{_SHRINKAGE_NONNULL_HI}], a_rmse_bi <= {_A_RMSE_MAX})"
                ),
                "per_candidate": per_candidate_gate_status,
            },
        }
    # Tie-break: smallest init_scale.
    candidates.sort(key=lambda kv: kv[0])
    return {
        "chosen_init_scale": candidates[0][0],
        "escalation": None,
        "per_candidate": per_candidate_gate_status,
    }


def write_md_table(
    rows: list[dict[str, Any]],
    chosen: dict[str, Any],
    seed: int,
    num_steps: int,
    output_md: Path,
) -> None:
    """Emit the human-readable diagnostic table + chosen-knob line."""
    lines: list[str] = []
    lines.append(f"# Phase 16.1 init_scale Sweep Diagnostic (seed {seed})")
    lines.append("")
    lines.append(f"- num_steps: {num_steps}")
    lines.append(f"- guide: AutoNormal")
    lines.append(f"- elbo: trace_mean_field")
    lines.append(f"- pass gates: b_rmse_nonnull < {_B_RMSE_PASS_THRESHOLD}, "
                 f"shrinkage_nonnull in [{_SHRINKAGE_NONNULL_LO}, "
                 f"{_SHRINKAGE_NONNULL_HI}], a_rmse_bi <= {_A_RMSE_MAX}")
    lines.append("")
    header = (
        "| init_scale | b_rmse_nonnull | a_rmse_bi | shrinkage_nonnull_mean "
        "| final_loss_last | nan_at_step | wall_s |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        last_loss = (
            f"{row['final_loss_tail'][-1]:.4f}"
            if row.get("final_loss_tail")
            else "NA"
        )
        b_rmse_s = (
            f"{row['b_rmse_nonnull']:.4f}"
            if row.get("b_rmse_nonnull") is not None
            else "NA"
        )
        a_rmse_s = (
            f"{row['a_rmse_bi']:.4f}"
            if row.get("a_rmse_bi") is not None
            else "NA"
        )
        shrink_s = (
            f"{row['shrinkage_nonnull_mean']:.4f}"
            if row.get("shrinkage_nonnull_mean") is not None
            else "NA"
        )
        nan_s = (
            "-"
            if row.get("nan_at_step") is None
            else str(row["nan_at_step"])
        )
        lines.append(
            f"| {row['init_scale']:.4f} | {b_rmse_s} | {a_rmse_s} | "
            f"{shrink_s} | {last_loss} | {nan_s} | "
            f"{row['wall_seconds']:.1f} |"
        )
    lines.append("")
    chosen_value = chosen.get("chosen_init_scale")
    if chosen_value is None:
        lines.append("**Chosen init_scale: NONE -- escalation required**")
        lines.append("")
        esc = chosen.get("escalation", {})
        lines.append(f"Reason: {esc.get('reason', 'unspecified')}")
    else:
        lines.append(f"**Chosen init_scale: {chosen_value}**")
    lines.append("")
    # Arithmetic check vs research prediction at init_scale=0.005.
    baseline = next(
        (r for r in rows if abs(r["init_scale"] - 0.005) < 1e-9),
        None,
    )
    if baseline is not None and baseline.get("b_rmse_nonnull") is not None:
        ratio = baseline["b_rmse_nonnull"] / _BASELINE_B_RMSE_REF
        lines.append(
            f"Harness check: init_scale=0.005 b_rmse_nonnull = "
            f"{baseline['b_rmse_nonnull']:.4f} "
            f"({ratio * 100:.1f}% of cluster reference {_BASELINE_B_RMSE_REF})."
        )
        lines.append(
            f"Research prediction: collapse-to-zero RMSE on non-null mask = "
            f"sqrt(mean([0.4^2, 0.3^2])) = 0.354. Cluster mean 0.3424 is "
            "96.8% of that collapse value, consistent with H1."
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on completion (winner OR no-winner)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 16.1 single-seed init_scale sweep diagnostic for the "
            "bilinear task-DCM SVI fit."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-scales",
        type=_parse_init_scales,
        default=_parse_init_scales("0.005,0.05,0.1,0.5"),
        help="Comma-separated init_scale values (default: 0.005,0.05,0.1,0.5)",
    )
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Output JSON path."
    )
    parser.add_argument(
        "--output-md", type=Path, required=True, help="Output MD path."
    )
    args = parser.parse_args(argv)

    # Silence the bilinear early-SVI stability WARNING spam (D4 log-only).
    logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)

    rows: list[dict[str, Any]] = []
    for init_scale in args.init_scales:
        print(
            f"[diagnose] Fitting seed={args.seed} init_scale={init_scale} "
            f"num_steps={args.num_steps}...",
            flush=True,
        )
        row = run_one(
            init_scale=init_scale,
            seed=args.seed,
            num_steps=args.num_steps,
        )
        if row.get("error") is not None:
            print(
                f"  ERROR: {row['error']} "
                f"(nan_at_step={row['nan_at_step']}, "
                f"wall={row['wall_seconds']:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  b_rmse_nonnull={row['b_rmse_nonnull']:.4f}, "
                f"a_rmse_bi={row['a_rmse_bi']:.4f}, "
                f"shrinkage_nonnull={row['shrinkage_nonnull_mean']:.4f}, "
                f"wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
        rows.append(row)

    chosen = pick_winner(rows)

    record = {
        "seed": args.seed,
        "num_steps": args.num_steps,
        "init_scales": args.init_scales,
        "rows": rows,
        "chosen_init_scale": chosen["chosen_init_scale"],
        "escalation": chosen.get("escalation"),
        "per_candidate_gate_status": chosen.get("per_candidate"),
        "winner_gates": {
            "b_rmse_threshold": _B_RMSE_PASS_THRESHOLD,
            "shrinkage_lo": _SHRINKAGE_NONNULL_LO,
            "shrinkage_hi": _SHRINKAGE_NONNULL_HI,
            "a_rmse_max": _A_RMSE_MAX,
        },
        "research_baseline": {
            "init_scale_baseline": 0.005,
            "b_rmse_cluster_mean": _BASELINE_B_RMSE_REF,
            "shrinkage_cluster_mean": _BASELINE_SHRINKAGE_REF,
            "tolerance_b_rmse_pct": 5.0,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    write_md_table(rows, chosen, args.seed, args.num_steps, args.output_md)

    print(
        f"[diagnose] Wrote JSON: {args.output_json}",
        flush=True,
    )
    print(
        f"[diagnose] Wrote MD:   {args.output_md}",
        flush=True,
    )
    print(
        f"[diagnose] chosen_init_scale = {chosen['chosen_init_scale']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
