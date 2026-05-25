---
phase: 21
plan: 04
subsystem: rnn-cluster-ensemble
tags: [rnn-training, cluster, slurm, ensemble, trajectories, cddm]
one-liner: "Single-seed RNN training CLI + 20-seed SLURM array job for CDDM ensemble training and trajectory extraction (RNN-03)"

dependency-graph:
  requires:
    - "21-01 (ContinuousTimeRNN module)"
    - "21-02 (train_rnn, eval_rnn_performance)"
    - "21-03 (extract_trajectories, pca_reduce)"
  provides:
    - "scripts/train_rnn_seed.py -- CLI for single-seed training + trajectories"
    - "cluster/scripts/train_rnn_ensemble.py -- SLURM wrapper for array jobs"
    - "cluster/sbatch/rnn_train_array.sbatch -- 20-seed array job"
  affects:
    - "Phase 22 (PIPE-03: consumes checkpoints/rnn/*.pt and data/rnn_trajectories/*.npz)"

tech-stack:
  - "torch (ContinuousTimeRNN)"
  - "neurogym (ContextDecisionMaking-v0)"
  - "scikit-learn (PCA, optional)"
  - "numpy (trajectory npz storage)"
  - "argparse (CLI), SLURM (cluster array jobs)"

key-decisions:
  - "[21-04-D1] SLURM wrapper reads SLURM_ARRAY_TASK_ID as seed (0-19). All config via env vars (RNN_HIDDEN, RNN_N_STEPS, etc.) for sbatch --export overrides."
  - "[21-04-D2] Trajectory npz includes dt_seconds, tau, alpha metadata scalars (pitfall LC10 time-grid alignment)."
  - "[21-04-D3] Naming convention: seed_NNNN_HHHH.{pt,json,_trajectories.npz} for discoverable filenames."

artifacts:
  - path: "scripts/train_rnn_seed.py"
    lines: 347
    role: "CLI script: argparse, train, evaluate, save weights + metadata + trajectories, optional PCA"
  - path: "cluster/scripts/train_rnn_ensemble.py"
    lines: 377
    role: "SLURM wrapper: reads array task ID, env var config, logs timing/memory"
  - path: "cluster/sbatch/rnn_train_array.sbatch"
    lines: 78
    role: "SLURM array job: seeds 0-19, H=256, 1.5h/8GB/4CPU per task"

test-evidence:
  - "scripts/train_rnn_seed.py --seed 0 --hidden 16 --n-steps 10 --skip-trajectories completes in ~15s with eval acc=0.927"
  - "ruff check passes on all 3 files"
  - "cluster sbatch follows cluster_env.sh pattern (crlf_guard, activate_env, verify_torch, print_job_header)"

state-updates:
  - key: "phase-21-plan-04"
    value: "complete"
    reason: "Training CLI + cluster sbatch ready. Cluster submission pending."
---

## What Was Done

Created three files for the Phase 21 RNN ensemble training infrastructure:

1. **`scripts/train_rnn_seed.py`** (347 lines) -- Full CLI for single-seed RNN training:
   - Argparse with all hyperparameters (seed, hidden, n_steps, lr, batch_size, etc.)
   - Creates ContinuousTimeRNN, trains on CDDM, evaluates, saves weights + JSON metadata
   - Optional trajectory extraction via `extract_trajectories()`
   - Optional PCA with `pca_reduce()` + `output_r_squared_gate()`
   - Includes `dt_seconds`, `tau`, `alpha` in saved npz (pitfall LC10)

2. **`cluster/scripts/train_rnn_ensemble.py`** (377 lines) -- SLURM array wrapper:
   - Reads `SLURM_ARRAY_TASK_ID` as seed (0-19)
   - All config via env vars (RNN_HIDDEN, RNN_N_STEPS, etc.)
   - Full training + evaluation + trajectory extraction pipeline
   - Timing/memory logging, exit code reflects accuracy gate

3. **`cluster/sbatch/rnn_train_array.sbatch`** (78 lines) -- SLURM directives:
   - `--array=0-19` for 20 concurrent seeds
   - 1.5h/8GB/4CPU per task, comp partition
   - Sources `cluster_env.sh`, installs package, runs ensemble script

## What's Left

Submit the array job to M3: `sbatch cluster/sbatch/rnn_train_array.sbatch`
