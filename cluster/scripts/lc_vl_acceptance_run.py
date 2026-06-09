"""Latent-circuit DCM acceptance run via VARIATIONAL LAPLACE on M3.

The Phase 20-05 rework (Tier B): fit the N=4 bilinear ground truth with
``LatentCircuitForward`` + Variational Laplace instead of the mean-field SVI
runner that collapsed the B posterior. VL returns a full posterior covariance
(structured posterior) and the SPM free energy.

Each SLURM array task runs ONE seed and writes a per-seed JSON.

Key configuration choices (vs the SVI runner):
- ``dt = 0.1`` (not 0.01). VL's time-domain observation precision is a dense
  ``(T*N) x (T*N)`` matrix that is inverted in the ReML M-step, so the SVI
  grid (100s @ dt=0.01 -> 10000 pts -> 40000x40000 inverse) is intractable.
  The latent dynamics are slow (tau ~ 1-2s), so dt=0.1 is accurate.
- ``duration = 50s`` with the Tier-A modulator retiming -> all three modulator
  epochs fall inside the 80% training split (see 20-05-SUMMARY root cause 1).
- No restarts: VL is Gauss-Newton from the prior mean (SPM convention).

References
----------
.planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md (Tier B)
pyro_dcm.inference.LatentCircuitForward
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

from benchmarks.metrics import compute_rmse
from benchmarks.latent_circuit_metrics import compute_trajectory_r_squared
from benchmarks.runners.latent_circuit_recovery import (
    _build_ground_truth,
    _compute_coverage_of_zero_single,
    _compute_sign_recovery_pooled,
    _predict_trajectories,
)
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.inference import (
    LatentCircuitForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
)
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# --- Configuration (env-overridable) ---
N_REGIONS = 4
DURATION = float(os.environ.get("LC_VL_DURATION", "50.0"))
DT = float(os.environ.get("LC_VL_DT", "0.1"))
SNR = float(os.environ.get("LC_VL_SNR", "10.0"))
MAX_ITER = int(os.environ.get("LC_VL_MAX_ITER", "64"))
TRAIN_FRACTION = 0.80
BASE_SEED = 42


def main() -> None:
    """Run a single-seed VL acceptance fit from SLURM_ARRAY_TASK_ID."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    seed_offset = int(task_id_str) if task_id_str is not None else 0
    seed = BASE_SEED + seed_offset

    print(f"VL acceptance: seed={seed} (offset={seed_offset})")
    print(
        f"Config: duration={DURATION}s, dt={DT}, SNR={SNR}, "
        f"max_iter={MAX_ITER}, A_prior={LC_A_PRIOR_VARIANCE:.4f}, "
        f"B_prior={LC_B_PRIOR_VARIANCE:.4f}"
    )

    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        torch.manual_seed(seed)

        # --- Ground truth (duration-aware Tier-A retiming) ---
        gt = _build_ground_truth(seed=0, duration=DURATION)
        A_true = gt["A_true"]
        B_true = gt["B_true"]            # (1, N, N)
        C_true = gt["C"]
        b_mask_0 = gt["b_mask_0"]
        stim = gt["stim"]
        stim_mod = gt["stim_mod"]
        a_mask = gt["a_mask"]
        c_mask = gt["c_mask"]
        N = A_true.shape[0]

        # --- Simulate at the (coarse) VL grid ---
        sim = simulate_latent_circuit(
            A_true, C_true, stim,
            duration=DURATION, dt=DT, SNR=SNR,
            solver="rk4", seed=seed,
            B_list=[B_true[0]], stimulus_mod=stim_mod,
        )
        trajs = sim["trajectories"].to(torch.float64)   # (T, N)
        t_all = sim["times"].to(torch.float64)
        if torch.isnan(trajs).any() or torch.isinf(trajs).any():
            raise ValueError("Simulated trajectories contain NaN/Inf.")

        T_total = trajs.shape[0]
        T_train = int(T_total * TRAIN_FRACTION)
        trajs_train = trajs[:T_train]
        trajs_test = trajs[T_train:]
        t_eval_train = t_all[:T_train]

        driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

        # --- Variational Laplace fit ---
        forward = LatentCircuitForward(
            stimulus=driving_stim,
            c_mask=c_mask,
            t_eval=t_eval_train,
            dt=DT,
            b_masks=[b_mask_0],
            stim_mod=stim_mod,
            c_prior_variance=1.0,
            b_prior_variance=LC_B_PRIOR_VARIANCE,
        )
        result = run_variational_laplace_generic(
            forward,
            observed=trajs_train,
            a_mask=a_mask,
            n_regions=N,
            max_iter=MAX_ITER,
            prior_variance=LC_A_PRIOR_VARIANCE,
            context={},
        )
        posterior = extract_vl_posterior_generic(result, forward, N)

        # --- Posterior means ---
        A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
        C_mean = posterior["C_free"]["mean"].to(torch.float64) * c_mask
        B_free_mean = posterior["B_free"]["mean"].to(torch.float64)  # (J,N,N)
        A_inferred = parameterize_A(A_free_mean * a_mask)

        # --- A-RMSE ---
        a_rmse = float(compute_rmse(A_true.to(torch.float64), A_inferred))

        # --- B-RMSE (magnitude-masked on |B_true| > 0.1) ---
        B_inferred = parameterize_B(B_free_mean, b_mask_0.unsqueeze(0))  # (1,N,N)
        b_eligible = (B_true.to(torch.float64).abs() > 0.1).float()
        b_rmse = float(
            (((B_true.to(torch.float64) - B_inferred) ** 2 * b_eligible).sum()
             / b_eligible.sum().clamp(min=1.0)) ** 0.5
        )

        # --- Sign recovery (pooled, |B_true| > 0.1) ---
        sign_recovery = float(
            _compute_sign_recovery_pooled(
                [B_true.to(torch.float64)], [B_inferred.to(torch.float64)],
            )
        )

        # --- 95% CI coverage of zero on null B elements ---
        B_samples = posterior["B_free"]["samples"].to(torch.float64)  # (S,J,N,N)
        B_samples = parameterize_B(
            B_samples.reshape(-1, N, N), b_mask_0.unsqueeze(0).expand(
                B_samples.shape[0] * B_samples.shape[1], N, N,
            ),
        ).reshape(B_samples.shape)
        ci_coverage_95 = float(
            _compute_coverage_of_zero_single(B_true.to(torch.float64), B_samples)
        )

        # --- Shrinkage (posterior std / prior std) ---
        a_std = float(posterior["A_free"]["std"].float().mean().item())
        b_std = float(posterior["B_free"]["std"].float().mean().item())
        shrinkage_A = a_std / (LC_A_PRIOR_VARIANCE ** 0.5)
        shrinkage_B = b_std / (LC_B_PRIOR_VARIANCE ** 0.5)

        # --- Held-out trajectory R-squared (integrate from t=0) ---
        predicted_test = _predict_trajectories(
            A_free_mean, C_mean, B_free_mean[0], b_mask_0,
            driving_stim, stim_mod, t_all, DT, T_train,
        )
        traj_r2 = float(compute_trajectory_r_squared(predicted_test, trajs_test))

        free_energy = [float(f) for f in result.free_energy]
        entry = {
            "seed": seed,
            "seed_offset": seed_offset,
            "inference": "variational_laplace",
            "a_rmse": a_rmse,
            "b_rmse": b_rmse,
            "sign_recovery": sign_recovery,
            "ci_coverage_95": ci_coverage_95,
            "trajectory_r_squared": traj_r2,
            "shrinkage_A": shrinkage_A,
            "shrinkage_B": shrinkage_B,
            "final_free_energy": free_energy[-1] if free_energy else None,
            "free_energy_improved": (
                bool(free_energy[-1] >= free_energy[0])
                if len(free_energy) > 1 else None
            ),
            "n_iterations": result.n_iterations,
            "converged": bool(result.converged),
            "n_free_energy_evals": len(free_energy),
            "config": {
                "duration": DURATION, "dt": DT, "snr": SNR,
                "max_iter": MAX_ITER, "T_train": T_train, "T_total": T_total,
            },
            "status": "ok",
        }
        print(
            f"  A-RMSE={a_rmse:.4f} B-RMSE={b_rmse:.4f} "
            f"sign={sign_recovery:.3f} cov95={ci_coverage_95:.3f} "
            f"R2={traj_r2:.4f} iters={result.n_iterations} "
            f"converged={result.converged}"
        )

    except Exception as e:  # noqa: BLE001 -- record any failure for triage
        entry = {
            "seed": seed,
            "seed_offset": seed_offset,
            "inference": "variational_laplace",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)
    out_path = output_dir / f"lc_vl_acceptance_{job_id}_{seed_offset}.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
