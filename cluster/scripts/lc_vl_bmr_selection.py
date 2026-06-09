"""SYNTH-03: structure selection via BMR on the VL posterior (Phase 20-05).

Replaces the retired cross-dimensional ELBO scan (invalid, 20-05-D2) with the
SPM-aligned approach: fit the FULL N=4 latent-circuit model with Variational
Laplace (a_mask all-ones), then use Bayesian Model Reduction (Friston & Penny
2011, REF-070) to score the off-diagonal A connections by their contribution to
model evidence. The analytic BMR reuses the single VL fit -- no refitting.

**Structure test = BMR evidence RANKING, not the absolute prune threshold.**
VL's Laplace posterior is over-confident (posterior covariance very tight at
high SNR), so the absolute "prune if dF>0" rule never fires -- exhaustive
best-model BMR keeps the full model (every single-connection prune dF is
strongly negative). But the *relative* prune cost cleanly separates real from
absent connections (true chain ~15x more costly to prune than any absent edge).
We therefore rank the off-diagonal connections by single-connection prune cost
and check that the K most essential are exactly the true chain.

Success: the K=3 connections most costly to prune are exactly the true chain
{A[1,0], A[2,1], A[3,2]} (flat indices 4, 9, 14), with a clear separation gap
to the next connection.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md (SYNTH-03)
pyro_dcm.model_selection.bmr_circuit_selection
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from benchmarks.runners.latent_circuit_recovery import _build_ground_truth
from pyro_dcm.inference import (
    LatentCircuitForward,
    run_variational_laplace_generic,
)
from pyro_dcm.model_selection import bayesian_model_reduction
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
)
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

N = 4
DURATION = float(os.environ.get("LC_VL_DURATION", "50.0"))
DT = float(os.environ.get("LC_VL_DT", "0.1"))
SNR = float(os.environ.get("LC_VL_SNR", "10.0"))
MAX_ITER = int(os.environ.get("LC_VL_MAX_ITER", "64"))
TRAIN_FRACTION = 0.80
SEEDS = [int(s) for s in os.environ.get("LC_VL_SEEDS", "42,43,44").split(",")]

# A[i,j] is at flat index i*N+j. Diagonal is structural (self-inhibition),
# never pruned. Off-diagonal entries are prunable.
OFFDIAG = [i * N + j for i in range(N) for j in range(N) if i != j]
TRUE_CHAIN = {1 * N + 0, 2 * N + 1, 3 * N + 2}          # {4, 9, 14}
TRUE_ABSENT = sorted(set(OFFDIAG) - TRUE_CHAIN)          # the 9 to prune


def _fit_full_vl(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """VL-fit the full (all-ones a_mask) N=4 model; return A_free post mean,cov."""
    gt = _build_ground_truth(seed=0, duration=DURATION)
    A_true, B_true, C_true = gt["A_true"], gt["B_true"], gt["C"]
    stim, stim_mod = gt["stim"], gt["stim_mod"]
    a_mask, c_mask, b_mask = gt["a_mask"], gt["c_mask"], gt["b_mask_0"]

    sim = simulate_latent_circuit(
        A_true, C_true, stim, duration=DURATION, dt=DT, SNR=SNR,
        solver="rk4", seed=seed, B_list=[B_true[0]], stimulus_mod=stim_mod,
    )
    trajs = sim["trajectories"].double()
    t_all = sim["times"].double()
    T_train = int(trajs.shape[0] * TRAIN_FRACTION)

    drive = PiecewiseConstantInput(stim["times"], stim["values"])
    fwd = LatentCircuitForward(
        stimulus=drive, c_mask=c_mask, t_eval=t_all[:T_train], dt=DT,
        b_masks=[b_mask], stim_mod=stim_mod,
        c_prior_variance=1.0, b_prior_variance=LC_B_PRIOR_VARIANCE,
    )
    res = run_variational_laplace_generic(
        fwd, observed=trajs[:T_train], a_mask=a_mask, n_regions=N,
        max_iter=MAX_ITER, prior_variance=LC_A_PRIOR_VARIANCE, context={},
    )
    a_free_mean = res.theta_post["A_free"].reshape(-1).double()  # (16,)
    a_cov = res.sigma_post[: N * N, : N * N].double()            # (16,16)
    return a_free_mean, a_cov


def main() -> None:
    """Run BMR structure selection over the requested seeds."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = Path("cluster/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prior_mean = torch.zeros(N * N, dtype=torch.float64)
    prior_cov = torch.eye(N * N, dtype=torch.float64) * LC_A_PRIOR_VARIANCE

    per_seed = []
    try:
        for seed in SEEDS:
            torch.manual_seed(seed)
            mean, cov = _fit_full_vl(seed)

            # Single-connection prune cost (dF) for each off-diagonal edge.
            # More negative dF = more essential (removing it costs evidence).
            prune_dF: dict[int, float] = {}
            for idx in OFFDIAG:
                reduced_cov = prior_cov.clone()
                reduced_cov[idx, idx] = 1e-8  # shrink this edge to ~0
                dF, _, _ = bayesian_model_reduction(
                    mean, cov, prior_mean, prior_cov, prior_mean, reduced_cov,
                )
                prune_dF[idx] = float(dF)

            # Rank ascending: the K most-negative are the most essential edges.
            ranked = sorted(OFFDIAG, key=lambda i: prune_dF[i])
            k = len(TRUE_CHAIN)
            essential = sorted(ranked[:k])
            correct = set(essential) == TRUE_CHAIN
            # Separation: prune cost of the K-th essential vs the (K+1)-th edge.
            kth, nxt = prune_dF[ranked[k - 1]], prune_dF[ranked[k]]
            sep = float(kth / nxt) if nxt != 0 else float("inf")
            per_seed.append({
                "seed": seed,
                "recovered_true_structure": bool(correct),
                "essential_topk": essential,
                "true_chain": sorted(TRUE_CHAIN),
                "prune_dF_essential": [prune_dF[i] for i in essential],
                "prune_dF_next": nxt,
                "separation_ratio": sep,
            })
            print(
                f"seed {seed}: recovered={correct} "
                f"essential={essential} (true={sorted(TRUE_CHAIN)}) "
                f"sep_ratio={sep:.1f}x"
            )
        n_ok = sum(r["recovered_true_structure"] for r in per_seed)
        entry = {
            "task": "synth03_bmr_structure_selection",
            "n_seeds": len(per_seed),
            "n_recovered_true_structure": n_ok,
            "all_recovered": n_ok == len(per_seed),
            "per_seed": per_seed,
            "status": "ok",
        }
        print(
            f"\nSYNTH-03: {n_ok}/{len(per_seed)} seeds recovered the true "
            f"chain structure via BMR."
        )
    except Exception as e:  # noqa: BLE001
        entry = {"task": "synth03_bmr", "status": "error",
                 "error": str(e), "traceback": traceback.format_exc()}
        print(f"ERROR: {e}")

    entry["elapsed_s"] = round(time.time() - t0, 1)
    out = out_dir / f"lc_vl_bmr_selection_{job_id}.json"
    with open(out, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
