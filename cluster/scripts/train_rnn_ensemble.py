"""Cluster-side wrapper for Phase 21 RNN ensemble training (array job).

Reads SLURM_ARRAY_TASK_ID for the per-seed index. Each array job trains
one ContinuousTimeRNN on CDDM, saves weights and metadata under
checkpoints/rnn/, and extracts hidden-state trajectories to
data/rnn_trajectories/.

Configurable via environment variables (for sbatch --export overrides):

    RNN_HIDDEN       int   hidden units H (default: 256)
    RNN_N_STEPS      int   max training steps (default: 3000)
    RNN_LR           float Adam learning rate (default: 1e-3)
    RNN_BATCH_SIZE   int   training batch size (default: 32)
    RNN_SEQ_LEN      int   sequence length per trial (default: 100)
    RNN_GRAD_CLIP    float gradient clip norm (default: 1.0)
    RNN_NOISE_STD    float hidden-state noise std (default: 0.05)
    RNN_TAU          float CT-RNN time constant (default: 1.0)
    RNN_DT           float CT-RNN Euler step dt (default: 0.1)
    RNN_CRITERION    float early-stopping accuracy (default: 0.85)
    RNN_N_EVAL       int   evaluation batches (default: 100)
    RNN_OUTPUT_DIR   str   output root dir (default: ".")

Usage (from project root on M3)::

    sbatch cluster/sbatch/rnn_train_array.sbatch

Or dry-run a single seed locally::

    SLURM_ARRAY_TASK_ID=0 python cluster/scripts/train_rnn_ensemble.py

References
----------
Langdon & Engel (2025) trainRNNbrain (interim cite; formal REF-ID Phase 25).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Resolve project root: this script lives two levels below root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pyro_dcm.rnn import (  # noqa: E402
    ContinuousTimeRNN,
    eval_rnn_performance,
    extract_trajectories,
    train_rnn,
)

TASK = "ContextDecisionMaking-v0"
ENV_KWARGS: dict = {"dt": 100}  # 100 ms per step


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Default value if the variable is absent.

    Returns
    -------
    int
        Parsed integer value.
    """
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Default value if the variable is absent.

    Returns
    -------
    float
        Parsed float value.
    """
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable with a default.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Default value if the variable is absent.

    Returns
    -------
    str
        Environment variable value or default.
    """
    return os.environ.get(name, default)


def _get_task_dims() -> tuple[int, int]:
    """Query CDDM task observation and action dimensions.

    Returns
    -------
    ob_size : int
        Observation space dimension.
    act_size : int
        Discrete action space size.
    """
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required. Install with: pip install 'pyro-dcm[latent]'"
        ) from e

    dataset = ngym.Dataset(
        TASK,
        env_kwargs=ENV_KWARGS,
        batch_size=1,
        seq_len=10,
    )
    ob_size: int = dataset.env.observation_space.shape[0]
    act_size: int = dataset.env.action_space.n
    return ob_size, act_size


def _make_env():
    """Create a CDDM neurogym environment for trajectory extraction.

    Returns
    -------
    env : neurogym.Env
        Initialized ContextDecisionMaking-v0 environment.
    """
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required. Install with: pip install 'pyro-dcm[latent]'"
        ) from e

    return ngym.make(TASK, **ENV_KWARGS)


def _log_memory() -> None:
    """Log current process RSS memory usage if psutil is available."""
    try:
        import psutil

        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / 1024**2
        print(f"Memory (RSS): {rss_mb:.0f} MB")
    except ImportError:
        print("Memory: psutil not available")


def main() -> None:
    """Run single-seed training for the array task ID from SLURM environment."""
    t_start = time.time()

    # --- Read SLURM array task ID ---
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print(
            "WARNING: SLURM_ARRAY_TASK_ID not set. "
            "Using seed=0 (dry-run mode)."
        )
        seed = 0
    else:
        seed = int(task_id_str)

    # --- Configuration from environment variables ---
    hidden = _env_int("RNN_HIDDEN", 256)
    n_steps = _env_int("RNN_N_STEPS", 3000)
    lr = _env_float("RNN_LR", 1e-3)
    batch_size = _env_int("RNN_BATCH_SIZE", 32)
    seq_len = _env_int("RNN_SEQ_LEN", 100)
    grad_clip = _env_float("RNN_GRAD_CLIP", 1.0)
    noise_std = _env_float("RNN_NOISE_STD", 0.05)
    tau = _env_float("RNN_TAU", 1.0)
    dt = _env_float("RNN_DT", 0.1)
    criterion_acc = _env_float("RNN_CRITERION", 0.85)
    n_eval = _env_int("RNN_N_EVAL", 100)
    output_dir = Path(_env_str("RNN_OUTPUT_DIR", "."))

    checkpoints_dir = output_dir / "checkpoints" / "rnn"
    trajectories_dir = output_dir / "data" / "rnn_trajectories"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    print("=" * 64)
    print("  RNN Ensemble Training (array job)")
    print("=" * 64)
    print(f"  SLURM_JOB_ID:          {job_id}")
    print(f"  SLURM_ARRAY_TASK_ID:   {array_id}")
    print(f"  Seed:                  {seed}")
    print(f"  H:                     {hidden}")
    print(f"  n_steps:               {n_steps}")
    print(f"  lr:                    {lr}")
    print(f"  batch_size:            {batch_size}")
    print(f"  seq_len:               {seq_len}")
    print(f"  grad_clip:             {grad_clip}")
    print(f"  noise_std:             {noise_std}")
    print(f"  tau:                   {tau}  dt:  {dt}  alpha: {dt/tau:.4f}")
    print(f"  criterion_acc:         {criterion_acc}")
    print(f"  output_dir:            {output_dir.resolve()}")
    print(f"  torch:                 {torch.__version__}")
    print("=" * 64)

    # --- Seeding ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Build RNN ---
    ob_size, act_size = _get_task_dims()
    rnn = ContinuousTimeRNN(
        n_input=ob_size,
        n_hidden=hidden,
        n_output=act_size,
        tau=tau,
        dt=dt,
        noise_std=noise_std,
    )
    n_params = sum(p.numel() for p in rnn.parameters())
    print(
        f"\nContinuousTimeRNN built: n_input={ob_size}, n_hidden={hidden}, "
        f"n_output={act_size}, params={n_params:,}"
    )

    # --- Train ---
    print(f"\nStarting training at {time.strftime('%H:%M:%S')}...")
    t_train = time.time()
    train_result = train_rnn(
        rnn,
        task=TASK,
        env_kwargs=ENV_KWARGS,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        grad_clip=grad_clip,
        criterion_acc=criterion_acc,
    )
    train_elapsed = time.time() - t_train
    n_steps_done = train_result["n_steps_completed"]
    print(
        f"\nTraining done: {n_steps_done} steps in {train_elapsed:.1f}s "
        f"({train_elapsed/n_steps_done*1000:.1f} ms/step)"
    )
    _log_memory()

    # --- Evaluate ---
    print("\nEvaluating...")
    eval_result = eval_rnn_performance(
        rnn,
        task=TASK,
        env_kwargs=ENV_KWARGS,
        n_eval_steps=n_eval,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    accuracy = eval_result["accuracy"]
    criterion_met = accuracy >= criterion_acc
    status = "PASS" if criterion_met else "FAIL"
    print(
        f"Eval accuracy: {accuracy:.4f} "
        f"(criterion={criterion_acc}) [{status}]"
    )

    # --- Save weights ---
    weights_stem = f"seed_{seed:04d}_H{hidden:03d}"
    weights_path = checkpoints_dir / f"{weights_stem}.pt"
    torch.save(rnn.state_dict(), weights_path)
    print(f"\nWeights saved: {weights_path}")

    # Construct env for trajectory extraction and metadata
    env = _make_env()
    dt_seconds: float = float(getattr(env, "dt", 100.0)) / 1000.0

    metadata: dict = {
        "seed": seed,
        "H": hidden,
        "accuracy": float(accuracy),
        "criterion_acc": criterion_acc,
        "criterion_met": bool(criterion_met),
        "n_steps": int(n_steps_done),
        "final_loss": float(train_result["final_loss"]),
        "tau": tau,
        "dt": dt,
        "alpha": float(rnn.alpha),
        "dt_seconds": dt_seconds,
        "noise_std": noise_std,
        "task": TASK,
        "ob_size": ob_size,
        "act_size": act_size,
        "n_params": n_params,
        "slurm_job_id": job_id,
        "slurm_array_task_id": array_id,
        "train_elapsed_s": round(train_elapsed, 1),
    }
    meta_path = checkpoints_dir / f"{weights_stem}.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Metadata saved: {meta_path}")

    # --- Trajectory extraction ---
    print("\nExtracting trajectories (n_trials_per_condition=50)...")
    t_traj = time.time()
    trajs = extract_trajectories(
        rnn,
        env=env,
        n_trials_per_condition=50,
    )
    traj_elapsed = time.time() - t_traj
    meta_dict = trajs["__meta__"]
    traj_keys = [k for k in trajs if k != "__meta__"]
    for key in traj_keys:
        arr = trajs[key]
        print(f"  condition='{key}': shape {arr.shape}")
    print(f"Trajectories extracted in {traj_elapsed:.1f}s")

    npz_path = trajectories_dir / f"{weights_stem}_trajectories.npz"
    save_dict: dict = {k: trajs[k] for k in traj_keys}
    # Metadata keys required by pitfall LC10 (time-grid alignment)
    save_dict["dt_seconds"] = np.array(meta_dict["dt_seconds"], dtype=np.float64)
    save_dict["tau"] = np.array(meta_dict["tau"], dtype=np.float64)
    save_dict["alpha"] = np.array(meta_dict["alpha"], dtype=np.float64)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Trajectories saved: {npz_path}")
    _log_memory()

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 64)
    print("  COMPLETE")
    print(f"  seed={seed}  H={hidden}  acc={accuracy:.4f}  [{status}]")
    print(
        f"  Steps: {n_steps_done}  Train: {train_elapsed:.1f}s  "
        f"Total: {total_elapsed:.1f}s"
    )
    print(f"  Weights: {weights_path}")
    print(f"  Trajectories: {npz_path}")
    print("=" * 64)

    if not criterion_met:
        print(
            f"\nWARNING: accuracy {accuracy:.4f} < criterion {criterion_acc}. "
            "Consider more training steps or check convergence."
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
