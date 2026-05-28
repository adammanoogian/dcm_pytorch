"""ELBO model selection for latent-circuit DCM (Phase 20-05, SYNTH-03).

Tests whether ELBO correctly identifies the true number of coupled
dimensions (N=4) from candidates {2, 3, 4, 5, 6}. SLURM array job:
each task handles one (N_candidate, seed) combination.

Array task mapping (15 tasks total):
    task_id = seed_idx * 5 + n_idx
    seed_idx in {0, 1, 2}, n_idx in {0, 1, 2, 3, 4}
    N_CANDIDATES = [2, 3, 4, 5, 6]
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import pyro
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.runners.latent_circuit_recovery import (
    _build_ground_truth,
    _DT,
    _SNR,
)
from pyro_dcm.models import create_guide, run_svi
from pyro_dcm.models.latent_circuit_dcm_model import latent_circuit_dcm_model
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

N_CANDIDATES = [2, 3, 4, 5, 6]
N_SEEDS = 3
N_SVI_STEPS = 300
N_RESTARTS = 3
INIT_SCALE = 0.1
BASE_SEED = 42
DURATION = 50.0
TRAIN_FRACTION = 0.80


def _prepare_data_for_n(
    trajs: torch.Tensor, n_candidate: int, n_true: int, seed: int,
) -> torch.Tensor:
    """Prepare N-dimensional data from 4D ground truth.

    Parameters
    ----------
    trajs : torch.Tensor, shape (T, 4)
        Ground-truth trajectories.
    n_candidate : int
        Target dimensionality.
    n_true : int
        True dimensionality (4).
    seed : int
        Random seed for noise columns.

    Returns
    -------
    torch.Tensor, shape (T, n_candidate)
    """
    if n_candidate <= n_true:
        return trajs[:, :n_candidate]

    rng = np.random.RandomState(seed + 1000)
    noise_std = trajs.std().item() * 0.1
    n_extra = n_candidate - n_true
    noise = torch.tensor(
        rng.randn(trajs.shape[0], n_extra) * noise_std,
        dtype=trajs.dtype,
    )
    return torch.cat([trajs, noise], dim=1)


def main() -> None:
    """Run single (N_candidate, seed) ELBO evaluation."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    if task_id_str is not None:
        task_id = int(task_id_str)
    else:
        task_id = 0

    seed_idx = task_id // len(N_CANDIDATES)
    n_idx = task_id % len(N_CANDIDATES)
    n_candidate = N_CANDIDATES[n_idx]
    seed = BASE_SEED + seed_idx

    print(f"ELBO model selection: N={n_candidate}, seed={seed}")
    print(
        f"Config: steps={N_SVI_STEPS}, restarts={N_RESTARTS}, "
        f"duration={DURATION}s"
    )

    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pyro.set_rng_seed(seed)
        pyro.enable_validation(False)

        gt = _build_ground_truth(seed=0)
        A_true = gt["A_true"]
        B_true = gt["B_true"]
        C_true = gt["C"]
        stim = gt["stim"]
        stim_mod = gt["stim_mod"]

        sim = simulate_latent_circuit(
            A_true, C_true, stim,
            duration=DURATION, dt=_DT, SNR=_SNR,
            solver="rk4", seed=seed,
            B_list=[B_true[0]],
            stimulus_mod=stim_mod,
        )
        trajs_full = sim["trajectories"]
        t_all = sim["times"]

        trajs_n = _prepare_data_for_n(trajs_full, n_candidate, 4, seed)

        T_total = trajs_n.shape[0]
        T_train = int(T_total * TRAIN_FRACTION)
        trajs_train = trajs_n[:T_train]
        t_eval_train = t_all[:T_train]

        a_mask = torch.ones(n_candidate, n_candidate, dtype=torch.float64)
        c_mask = torch.zeros(n_candidate, 1, dtype=torch.float64)
        c_mask[0, 0] = 1.0
        b_mask = torch.ones(n_candidate, n_candidate, dtype=torch.float64)

        driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

        model_args = (
            trajs_train,
            driving_stim,
            a_mask,
            c_mask,
            t_eval_train,
            _DT,
        )
        model_kwargs = {
            "b_masks": [b_mask],
            "stim_mod": stim_mod,
        }

        guide_factory = partial(
            create_guide,
            latent_circuit_dcm_model,
            guide_type="auto_normal",
            init_scale=INIT_SCALE,
        )

        svi_result = run_svi(
            latent_circuit_dcm_model,
            guide_factory(),
            model_args,
            num_steps=N_SVI_STEPS,
            lr=0.005,
            clip_norm=10.0,
            lr_decay_factor=0.01,
            elbo_type="trace_elbo",
            guide_type="auto_normal",
            model_kwargs=model_kwargs,
            n_restarts=N_RESTARTS,
            guide_factory=guide_factory,
        )

        best_elbo = -svi_result["final_loss"]

        entry = {
            "n_candidate": n_candidate,
            "seed": seed,
            "seed_idx": seed_idx,
            "best_elbo": float(best_elbo),
            "final_loss": float(svi_result["final_loss"]),
            "num_steps": int(svi_result["num_steps"]),
            "status": "ok",
        }
        print(f"  N={n_candidate}  ELBO={best_elbo:.2f}")

    except Exception as e:
        entry = {
            "n_candidate": n_candidate,
            "seed": seed,
            "seed_idx": seed_idx,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)

    output_path = output_dir / f"lc_elbo_{job_id}_{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {output_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
