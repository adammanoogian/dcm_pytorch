"""Single-seed CT-RNN training CLI for Phase 21 ensemble infrastructure.

Trains one ContinuousTimeRNN on the CDDM task (ContextDecisionMaking-v0),
saves weights and metadata, extracts hidden-state trajectories, and
optionally runs PCA with the output-R-squared quality gate.

Typical usage::

    python scripts/train_rnn_seed.py --seed 0 --hidden 256 --n-steps 3000

Cluster usage: see cluster/sbatch/rnn_train_array.sbatch which invokes
cluster/scripts/train_rnn_ensemble.py (reads SLURM_ARRAY_TASK_ID for seed).

References
----------
Langdon & Engel (2025) trainRNNbrain (interim cite; formal REF-ID Phase 25).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pyro_dcm.rnn import (
    ContinuousTimeRNN,
    eval_rnn_performance,
    extract_trajectories,
    output_r_squared_gate,
    pca_reduce,
    train_rnn,
    variance_explained_diagnostic,
)

TASK = "ContextDecisionMaking-v0"
ENV_KWARGS: dict = {"dt": 100}  # 100 ms per step


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-seed RNN training.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace with all training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train a single CT-RNN seed on CDDM and extract trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for torch and numpy."
    )
    parser.add_argument(
        "--hidden", type=int, default=256, help="Number of hidden units H."
    )
    parser.add_argument(
        "--n-steps", type=int, default=3000, help="Maximum training gradient steps."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--seq-len", type=int, default=100, help="Sequence length (steps per trial)."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clip norm."
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Additive Gaussian noise std on hidden states during training.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="CT-RNN time constant (normalized units).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="CT-RNN Euler integration step (alpha = dt/tau).",
    )
    parser.add_argument(
        "--criterion-acc",
        type=float,
        default=0.85,
        help="Early-stopping accuracy threshold.",
    )
    parser.add_argument(
        "--n-eval-trials",
        type=int,
        default=100,
        help="Number of evaluation batches for final accuracy estimate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory. Weights go to {output_dir}/checkpoints/rnn/; "
        "trajectories go to {output_dir}/data/rnn_trajectories/.",
    )
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help="Skip trajectory extraction (weights + metadata only).",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=None,
        help="If set, run PCA on extracted trajectories and report variance explained "
        "and output-R-squared gate.",
    )
    return parser.parse_args()


def _get_task_dims() -> tuple[int, int]:
    """Query CDDM task observation and action dimensions.

    Returns
    -------
    ob_size : int
        Observation space dimension.
    act_size : int
        Number of discrete actions.
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


def main() -> None:
    """Run single-seed training, evaluation, and trajectory extraction."""
    args = parse_args()
    t_start = time.time()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints" / "rnn"
    trajectories_dir = output_dir / "data" / "rnn_trajectories"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed: {args.seed} | H: {args.hidden} | steps: {args.n_steps}")
    print(f"Output dir: {output_dir.resolve()}")
    print()

    # --- Build RNN ---
    ob_size, act_size = _get_task_dims()
    rnn = ContinuousTimeRNN(
        n_input=ob_size,
        n_hidden=args.hidden,
        n_output=act_size,
        tau=args.tau,
        dt=args.dt,
        noise_std=args.noise_std,
    )
    n_params = sum(p.numel() for p in rnn.parameters())
    print(
        f"ContinuousTimeRNN: n_input={ob_size}, n_hidden={args.hidden}, "
        f"n_output={act_size}, alpha={rnn.alpha:.4f}, params={n_params:,}"
    )

    # --- Train ---
    print(f"\nTraining (lr={args.lr}, batch={args.batch_size}, "
          f"seq_len={args.seq_len}, grad_clip={args.grad_clip})...")
    train_result = train_rnn(
        rnn,
        task=TASK,
        env_kwargs=ENV_KWARGS,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        grad_clip=args.grad_clip,
        criterion_acc=args.criterion_acc,
    )
    n_steps_done = train_result["n_steps_completed"]
    train_acc = train_result["final_accuracy"]
    print(
        f"\nTraining complete: {n_steps_done} steps, "
        f"final_loss={train_result['final_loss']:.4f}, "
        f"final_acc={train_acc:.3f}"
    )

    # --- Evaluate ---
    print(f"\nEvaluating on {args.n_eval_trials} batches...")
    eval_result = eval_rnn_performance(
        rnn,
        task=TASK,
        env_kwargs=ENV_KWARGS,
        n_eval_steps=args.n_eval_trials,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    accuracy = eval_result["accuracy"]
    criterion_met = accuracy >= args.criterion_acc
    status = "PASS" if criterion_met else "FAIL"
    print(
        f"Eval accuracy: {accuracy:.4f} "
        f"(criterion={args.criterion_acc}) [{status}]"
    )

    # --- Save weights ---
    weights_stem = f"seed_{args.seed:04d}_H{args.hidden:03d}"
    weights_path = checkpoints_dir / f"{weights_stem}.pt"
    torch.save(rnn.state_dict(), weights_path)
    print(f"\nWeights saved: {weights_path}")

    # env.dt is in ms; convert to seconds for LC10 metadata
    env = _make_env()
    dt_seconds: float = float(getattr(env, "dt", 100.0)) / 1000.0

    metadata: dict = {
        "seed": int(args.seed),
        "H": int(args.hidden),
        "accuracy": float(accuracy),
        "criterion_acc": float(args.criterion_acc),
        "criterion_met": bool(criterion_met),
        "n_steps": int(n_steps_done),
        "final_loss": float(train_result["final_loss"]),
        "tau": float(args.tau),
        "dt": float(args.dt),
        "alpha": float(rnn.alpha),
        "dt_seconds": float(dt_seconds),
        "noise_std": float(args.noise_std),
        "task": TASK,
        "ob_size": int(ob_size),
        "act_size": int(act_size),
        "n_params": int(n_params),
    }
    meta_path = checkpoints_dir / f"{weights_stem}.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Metadata saved: {meta_path}")

    # --- Trajectory extraction ---
    if not args.skip_trajectories:
        print("\nExtracting trajectories...")
        trajs = extract_trajectories(
            rnn,
            env=env,
            n_trials_per_condition=50,
        )
        meta_dict = trajs["__meta__"]
        traj_keys = [k for k in trajs if k != "__meta__"]
        for key in traj_keys:
            arr = trajs[key]
            print(f"  condition='{key}': shape {arr.shape}")

        npz_path = trajectories_dir / f"{weights_stem}_trajectories.npz"
        save_dict: dict = {k: trajs[k] for k in traj_keys}
        # Store LC10 metadata as scalars in the npz file
        save_dict["dt_seconds"] = np.array(meta_dict["dt_seconds"], dtype=np.float64)
        save_dict["tau"] = np.array(meta_dict["tau"], dtype=np.float64)
        save_dict["alpha"] = np.array(meta_dict["alpha"], dtype=np.float64)
        np.savez_compressed(npz_path, **save_dict)
        print(f"Trajectories saved: {npz_path}")

        # --- Optional PCA ---
        if args.n_pca_components is not None:
            print(f"\nRunning PCA (n_components={args.n_pca_components})...")
            # Stack all trajectory data for PCA fitting
            all_trajs = np.concatenate(
                [trajs[k].reshape(-1, args.hidden) for k in traj_keys], axis=0
            )
            n_components = min(args.n_pca_components, all_trajs.shape[1])
            pca, projected = pca_reduce(all_trajs, n_components=n_components)
            diag = variance_explained_diagnostic(pca)
            cum_var = diag["cumulative"][-1] if len(diag["cumulative"]) > 0 else 0.0
            print(
                f"  PCA: {n_components} components, "
                f"cumulative variance explained: {cum_var:.3f}"
            )
            print(f"  Recommended N (5% elbow): {diag['recommended_n']}")

            # Output-R2 gate using PCA-projected vs true readout
            # Use the same all_trajs (no held-out split at this scale; informational)
            w_out = rnn.W_out.detach().cpu().numpy()
            # Build true z by running a forward pass on the stacked trajectories
            # Compute W_out @ h.T directly (no forward pass needed for logits check)
            z_true = all_trajs @ w_out.T  # (n_samples, act_size)
            gate = output_r_squared_gate(projected, z_true, w_out, pca)
            gate_status = "PASS" if gate["passed"] else "FAIL"
            print(
                f"  Output-R2 gate: R2={gate['r_squared']:.4f} "
                f"(threshold={gate['threshold']}) [{gate_status}]"
            )
    else:
        print("\nTrajectory extraction skipped (--skip-trajectories set).")

    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    print(
        f"\nSummary: seed={args.seed} H={args.hidden} "
        f"acc={accuracy:.4f} steps={n_steps_done} [{status}]"
    )


if __name__ == "__main__":
    main()
