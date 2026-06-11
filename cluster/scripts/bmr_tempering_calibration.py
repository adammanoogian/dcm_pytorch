"""VLBMR-03 EXPLORATORY tempering calibration on the task-N4 stress cell (M3).

Single-seed M3 job (NOT an array). Re-fits ONE representative seed of the Phase
30 task-N4 stress cell -- the strongest VL Laplace-overconfidence regime
(``recovery_matrix.json``: variant=="task", n_regions==4, coverage_95 == 0.0) --
to obtain a real ``A_free`` posterior mean + full covariance, then:

1. Reads ``benchmarks/results/recovery_matrix.json`` to locate the task-N4
   stress cell and one HELD-OUT cell (task N=2, the cross-condition).
2. Re-fits one task-N4 seed via the SAME simulate/forward symbols the Phase 30
   driver (``benchmarks.recovery_matrix_grid._run_task_cell``) uses -- fixed-step
   ``rk4`` ground-truth sim (the dopri5-underflow fix, commit c0a7616) and
   ``dt >= 0.1`` (Pitfall N1). Phase 30 did not persist Ep/Cp, so a fresh single
   re-fit is required (research Open Question 4).
3. Builds the BMR tensors via the 31-01 helpers
   (``bmr_tensors_from_vl_result``, ``offdiag_indices``).
4. Calibrates the temperature ``T`` by coverage-matching against the cell's
   nominal 0.95 target (band [0.90, 0.98]); ``samples_fn(T)`` draws ``A_free``
   samples from ``MultivariateNormal(mean, temper_vl_posterior(Cp, T))``,
   parameterizes them to the SAME A space Phase 30 measured coverage on, and
   recomputes empirical 95% coverage.
5. Applies the chosen ``T`` via ``tempered_vs_untempered_ranking`` and collects
   the side-by-side ranked lists + separation gaps.
6. Cross-condition: re-fits one held-out (task N=2) seed and applies the SAME
   chosen ``T``, recording whether the tempered ranking preserves its top-K
   (research Section C2a -- the chosen T must not catastrophically over-inflate;
   this is NOT a universal-schedule claim).
7. Writes ``cluster/results/bmr_tempering_calibration_<jobid>.json`` with the
   chosen ``T``, the ``{T: coverage}`` trace, the untempered + tempered rankings
   for the stress cell, the held-out cross-condition result, and a top-level
   EXPLORATORY note. A non-PD reduced posterior (``delta_f == -inf``) inside BMR
   is the expected C2c failure mode and is surfaced, never masked.

ALL tempering routes through ``temper_vl_posterior`` (the loud PD guard); no
hand-rolled Cholesky. Absolute delta-F is NEVER a pass/fail criterion (Pitfall
C1/C2). The task fit is minutes -> M3 per the >3-min routing rule.

Environment variables
---------------------
SLURM_JOB_ID : str, default "local"
    Job id used in the output filename.
BMR_TEMPER_SEED : int, default 42
    Representative seed re-fit for the task-N4 stress cell (and N=2 held-out).
BMR_TEMPER_MAX_ITER : int, default 64
    VL Gauss-Newton iteration cap.

References
----------
cluster/scripts/recovery_matrix_cell.py
    The mirrored SLURM-glue structure (sys.path, env knobs, status JSON).
benchmarks.recovery_matrix_grid
    The Phase 30 per-cell fit logic reused for the single-seed re-fits.
benchmarks.bmr_recovery
    select_tempering_factor / tempered_vs_untempered_ranking / the 31-01 helpers.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.bmr_recovery import (  # noqa: E402
    bmr_tensors_from_vl_result,
    offdiag_indices,
    select_tempering_factor,
    tempered_vs_untempered_ranking,
)
from benchmarks.metrics import compute_coverage_from_samples  # noqa: E402
from benchmarks.recovery_matrix_metrics import (  # noqa: E402
    resample_A_until_accepted,
    snr_for_model,
)
from pyro_dcm.forward_models.neural_state import (  # type: ignore[import-untyped]  # noqa: E402
    parameterize_A,
)
from pyro_dcm.inference import (  # type: ignore[import-untyped]  # noqa: E402
    TaskDCMForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.model_selection.bmr import (  # type: ignore[import-untyped]  # noqa: E402
    rank_connections,
    temper_vl_posterior,
)
from pyro_dcm.simulators.task_simulator import (  # type: ignore[import-untyped]  # noqa: E402
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)

_A_PRIOR_VARIANCE_BOLD = 1.0 / 64.0
_DT_TASK = 0.1
_TR_TASK = 2.0
_N_SAMPLES = 4000
_TEMPER_CANDIDATES: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100)
_TEMPER_BAND: tuple[float, float] = (0.90, 0.98)
_TEMPER_TARGET = 0.95


def _seeded_A_factory(n_regions: int, seed_base: int):  # type: ignore[no-untyped-def]
    """Return a zero-arg ``A`` maker advancing a fresh seed per call.

    Mirrors ``benchmarks.recovery_matrix_grid._seeded_A_factory`` so
    ``resample_A_until_accepted`` draws a DIFFERENT near-boundary-rejected ``A``
    on each retry.
    """
    counter = {"n": 0}

    def _draw() -> torch.Tensor:
        counter["n"] += 1
        A: torch.Tensor = make_random_stable_A(
            n_regions, density=0.5, seed=seed_base + counter["n"],
        )
        return A

    return _draw


def _refit_task_seed(
    n_regions: int, snr: float, seed: int, max_iter: int,
) -> dict[str, Any]:
    """Re-fit ONE task-DCM seed and return the VL result + ground truth.

    Replicates the single-seed body of
    ``benchmarks.recovery_matrix_grid._run_task_cell`` (same simulate/forward
    symbols, fixed-step ``rk4`` ground-truth sim, ``dt >= 0.1``) but returns the
    raw VL ``result`` (with ``theta_post`` / ``sigma_post``) plus ``A_true`` and
    the parameterized posterior coverage so the tempering layer can consume them.

    Parameters
    ----------
    n_regions : int
        Number of regions ``N``.
    snr : float
        Cell SNR (mapped via ``snr_for_model("task", snr)``).
    seed : int
        Representative seed re-fit.
    max_iter : int
        VL Gauss-Newton iteration cap.

    Returns
    -------
    dict
        Keys: ``result`` (VL fit), ``A_true`` (N, N), ``A_free_mean`` (N, N),
        ``coverage_untempered`` (float, the sharp posterior's empirical 95%
        coverage), ``a_mask`` (N, N).

    Raises
    ------
    AssertionError
        If ``_DT_TASK < 0.1`` (precision floor, Pitfall N1).
    """
    if not _DT_TASK >= 0.1:
        raise AssertionError(
            f"task VL dt must be >= 0.1 (precision floor); got {_DT_TASK}."
        )
    N = n_regions
    M = 1
    task_snr = snr_for_model("task", snr)["SNR"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    A_true = resample_A_until_accepted(
        _seeded_A_factory(N, seed_base=seed * 1000),
    )
    C_true = torch.zeros(N, M, dtype=torch.float64)
    C_true[0, 0] = 1.0
    stim = make_block_stimulus(
        n_blocks=3, block_duration=15.0, rest_duration=15.0, n_inputs=M,
    )
    sim = simulate_task_dcm(
        A_true, C_true, stim,
        duration=120.0, dt=0.01, TR=_TR_TASK, SNR=task_snr,
        seed=seed, solver="rk4",
    )
    bold = sim["bold"].to(torch.float64)
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, M, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    t_eval = torch.arange(
        0.0, bold.shape[0] * _TR_TASK, _TR_TASK, dtype=torch.float64,
    )[: bold.shape[0]]

    forward = TaskDCMForward(
        stimulus_fn=sim["stimulus"], c_mask=c_mask, t_eval=t_eval, dt=_DT_TASK,
    )
    result = run_variational_laplace_generic(
        forward,
        observed=bold,
        a_mask=a_mask,
        n_regions=N,
        max_iter=max_iter,
        prior_variance=_A_PRIOR_VARIANCE_BOLD,
        context={"a_mask": a_mask},
    )
    posterior = extract_vl_posterior_generic(result, forward, N)
    A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
    A_free_samples = posterior["A_free"]["samples"].to(torch.float64)
    A_param_samples = torch.stack(
        [parameterize_A(s * a_mask) for s in A_free_samples],
    )
    coverage_untempered = compute_coverage_from_samples(
        A_true.to(torch.float64), A_param_samples, ci_level=0.95,
    )
    return {
        "result": result,
        "A_true": A_true.to(torch.float64),
        "A_free_mean": A_free_mean,
        "coverage_untempered": float(coverage_untempered),
        "a_mask": a_mask,
    }


def _make_samples_fn(
    posterior_mean: torch.Tensor,
    posterior_cov: torch.Tensor,
    a_mask: torch.Tensor,
    n_regions: int,
    seed: int = 7,
) -> Callable[[float], torch.Tensor]:
    """Build ``samples_fn(T)`` drawing parameterized tempered A samples.

    Draws ``A_free`` from ``MultivariateNormal(mean, temper_vl_posterior(Cp,
    T))`` (THE PD guard) and parameterizes each draw to the SAME A space Phase
    30 computed coverage on, reshaped ``(S, N, N)``.
    """
    n = n_regions

    def samples_fn(temperature: float) -> torch.Tensor:
        torch.manual_seed(seed)
        cov_t = temper_vl_posterior(posterior_cov, temperature)
        dist = torch.distributions.MultivariateNormal(posterior_mean, cov_t)
        flat = dist.sample(torch.Size([_N_SAMPLES]))  # (S, N*N)
        free = flat.reshape(_N_SAMPLES, n, n)
        return torch.stack([parameterize_A(s * a_mask) for s in free])

    return samples_fn


def _rank_block_to_json(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``rank_connections`` dict to a JSON-serializable block.

    Surfaces any ``delta_f == -inf`` (the expected C2c non-PD reduced-posterior
    failure mode) as the string ``"-inf"`` rather than masking it.
    """
    ranked = []
    for entry in block["ranked"]:
        df = entry["prune_delta_f"]
        ranked.append(
            {
                "index": int(entry["index"]),
                "prune_delta_f": (
                    "-inf" if math.isinf(df) and df < 0 else float(df)
                ),
                "rank": int(entry["rank"]),
                "gap_to_next": (
                    None if entry["gap_to_next"] is None
                    else float(entry["gap_to_next"])
                ),
            }
        )
    return {
        "ranked": ranked,
        "separation_gap": float(block["separation_gap"]),
        "separation_after_rank": int(block["separation_after_rank"]),
        "prunable_indices": [int(i) for i in block["prunable_indices"]],
    }


def _find_cell(
    rows: list[dict[str, Any]], variant: str, n_regions: int,
) -> dict[str, Any]:
    """Return the recovery-matrix row for ``(variant, n_regions)``.

    Raises
    ------
    ValueError
        If no matching cell exists (message names the expected key).
    """
    for row in rows:
        if row.get("variant") == variant and row.get("n_regions") == n_regions:
            return row
    raise ValueError(
        f"no recovery_matrix cell for variant={variant!r} n_regions={n_regions} "
        f"(expected at least one matching row)."
    )


def _topk_indices(block: dict[str, Any], k: int) -> list[int]:
    """Return the top-``k`` essential indices (most-negative prune cost first)."""
    return [int(e["index"]) for e in block["ranked"][:k]]


def main() -> None:
    """Run the task-N4 tempering calibration + held-out cross-condition check."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    seed = int(os.environ.get("BMR_TEMPER_SEED", "42"))
    max_iter = int(os.environ.get("BMR_TEMPER_MAX_ITER", "64"))

    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bmr_tempering_calibration_{job_id}.json"

    note = (
        "EXPLORATORY -- tempering is NOT a headline claim; absolute delta-F is "
        "never a pass/fail criterion (Pitfall C1/C2)."
    )
    t0 = time.time()

    entry: dict[str, Any]
    try:
        matrix_path = Path("benchmarks/results/recovery_matrix.json")
        with open(matrix_path) as f:
            matrix = json.load(f)
        rows = matrix["rows"]

        stress = _find_cell(rows, "task", 4)
        held_out = _find_cell(rows, "task", 2)
        stress_snr = float(stress["snr"])
        held_out_snr = float(held_out["snr"])

        print(
            f"Stress cell: task N=4 SNR={stress_snr} "
            f"(Phase 30 coverage_95={stress['coverage_95']}); "
            f"held-out: task N=2 SNR={held_out_snr} "
            f"(coverage_95={held_out['coverage_95']})."
        )

        # --- Re-fit the task-N4 stress seed -------------------------------
        print(f"Re-fitting task N=4 seed {seed} (max_iter={max_iter})...")
        fit = _refit_task_seed(4, stress_snr, seed, max_iter)
        result = fit["result"]
        A_true = fit["A_true"]
        a_mask = fit["a_mask"]
        N = 4

        posterior_mean, posterior_cov, prior_mean, prior_cov = (
            bmr_tensors_from_vl_result(
                result, N, prior_variance=_A_PRIOR_VARIANCE_BOLD,
            )
        )
        prunable = offdiag_indices(N)

        # --- Calibrate T by coverage-matching -----------------------------
        samples_fn = _make_samples_fn(posterior_mean, posterior_cov, a_mask, N)
        calib = select_tempering_factor(
            A_true, samples_fn,
            target=_TEMPER_TARGET, band=_TEMPER_BAND,
            candidates=_TEMPER_CANDIDATES,
        )
        chosen_t = float(cast("float", calib["tempering_factor"]))
        tempered_cov95 = float(cast("float", calib["coverage"]))
        calib_trace = cast("dict[float, float]", calib["trace"])
        print(
            f"  untempered coverage_95={fit['coverage_untempered']:.3f}; "
            f"chosen T={chosen_t} -> tempered coverage_95="
            f"{tempered_cov95:.3f} (in_band={calib['in_band']})"
        )

        # --- Side-by-side untempered/tempered ranking ---------------------
        ranking = tempered_vs_untempered_ranking(
            posterior_mean, posterior_cov, prior_mean, prior_cov,
            prunable, chosen_t,
        )
        untempered_block = cast("dict[str, Any]", ranking["untempered"])
        tempered_block = cast("dict[str, Any]", ranking["tempered"])

        # --- Held-out cross-condition (same T) ----------------------------
        # The chosen T is tuned on the N=4 stress cell; applying it UNCHANGED to
        # the N=2 held-out posterior is the cross-condition probe (research
        # Section C2a). If T pushes the held-out covariance non-PD, that is the
        # EXPECTED C2c failure mode -- we RECORD it (do not abort), because the
        # stress-cell calibration already succeeded.
        print(f"Re-fitting held-out task N=2 seed {seed}...")
        held_fit = _refit_task_seed(2, held_out_snr, seed, max_iter)
        held_result = held_fit["result"]
        N2 = 2
        (
            held_mean, held_cov, held_prior_mean, held_prior_cov,
        ) = bmr_tensors_from_vl_result(
            held_result, N2, prior_variance=_A_PRIOR_VARIANCE_BOLD,
        )
        held_prunable = offdiag_indices(N2)
        # Untempered ranking always succeeds (identity covariance).
        held_untempered_block = rank_connections(
            held_mean, held_cov, held_prior_mean, held_prior_cov, held_prunable,
        )
        k_held = len(held_prunable)
        held_untempered_topk = _topk_indices(held_untempered_block, k_held)

        held_out_block: dict[str, Any] = {
            "variant": "task",
            "n_regions": 2,
            "snr": held_out_snr,
            "phase30_coverage_95": held_out["coverage_95"],
            "applied_tempering_factor": chosen_t,
            "untempered_topk": held_untempered_topk,
            "untempered_ranking": _rank_block_to_json(held_untempered_block),
        }
        try:
            held_cov_tempered = temper_vl_posterior(held_cov, chosen_t)
            held_tempered_block = rank_connections(
                held_mean, held_cov_tempered, held_prior_mean,
                held_prior_cov, held_prunable,
            )
            held_tempered_topk = _topk_indices(held_tempered_block, k_held)
            held_topk_preserved = held_untempered_topk == held_tempered_topk
            held_out_block["cross_condition_non_pd"] = False
            held_out_block["tempered_topk"] = held_tempered_topk
            held_out_block["topk_preserved"] = held_topk_preserved
            held_out_block["tempered_ranking"] = _rank_block_to_json(
                held_tempered_block,
            )
            print(
                f"  held-out top-K preserved under T={chosen_t}: "
                f"{held_topk_preserved}"
            )
        except ValueError as pd_err:
            # Expected C2c: the stress-cell T is not PD-safe on this condition.
            held_out_block["cross_condition_non_pd"] = True
            held_out_block["tempered_topk"] = None
            held_out_block["topk_preserved"] = False
            held_out_block["tempered_ranking"] = None
            held_out_block["non_pd_message"] = str(pd_err)
            print(
                f"  held-out C2c: T={chosen_t} broke PD on the N=2 posterior "
                f"-> {pd_err}"
            )

        entry = {
            "status": "ok",
            "job_id": job_id,
            "note": note,
            "seed": seed,
            "max_iter": max_iter,
            "tempering_factor": chosen_t,
            "in_band": bool(calib["in_band"]),
            "coverage_target": _TEMPER_TARGET,
            "coverage_band": list(_TEMPER_BAND),
            "stress_cell": {
                "variant": "task",
                "n_regions": 4,
                "snr": stress_snr,
                "phase30_coverage_95": stress["coverage_95"],
                "refit_untempered_coverage_95": fit["coverage_untempered"],
                "tempered_coverage_95": tempered_cov95,
                "coverage_trace": {
                    str(k): float(v) for k, v in calib_trace.items()
                },
                "untempered_ranking": _rank_block_to_json(untempered_block),
                "tempered_ranking": _rank_block_to_json(tempered_block),
            },
            "held_out_cell": held_out_block,
        }
        print("  OK -- tempering calibration complete.")
    except Exception as e:  # noqa: BLE001 -- record any failure for triage
        entry = {
            "status": "error",
            "job_id": job_id,
            "note": note,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
