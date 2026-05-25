#!/bin/bash
# =============================================================================
# Phase 21 CT-RNN Ensemble Training (array job, 20 seeds)
# =============================================================================
# Trains one ContinuousTimeRNN per array task (SLURM_ARRAY_TASK_ID = seed).
# Seeds 0-19 => 20 RNNs on CDDM (ContextDecisionMaking-v0), H=256 hidden.
#
# Each job:
#   1. Trains one RNN for up to 3000 steps (early-stop at 85% accuracy)
#   2. Evaluates final accuracy on 100 evaluation batches
#   3. Saves weights to checkpoints/rnn/seed_NNNN_H256.pt
#   4. Saves metadata JSON to checkpoints/rnn/seed_NNNN_H256.json
#   5. Extracts hidden-state trajectories to data/rnn_trajectories/
#
# Estimated runtime: 30-60 min per seed (H=256, 3000 steps, 100ms neurogym dt)
# Parallel: 20 seeds run concurrently => ~60 min wall time total
# Storage estimate: ~100 MB per seed x 20 = ~2 GB total (trajectories + weights)
#
# Usage:
#   sbatch cluster/sbatch/rnn_train_array.sbatch
#
# Overrides (via sbatch --export=ALL,...):
#   RNN_HIDDEN=128         reduce hidden units (faster, less accurate)
#   RNN_N_STEPS=1000       fewer training steps
#   RNN_OUTPUT_DIR=/path   custom output directory
#
# Outputs:
#   cluster/logs/rnn_train_<JOBID>_<TASKID>.out  per-seed stdout
#   cluster/logs/rnn_train_<JOBID>_<TASKID>.err  per-seed stderr
#   checkpoints/rnn/seed_NNNN_H256.pt            trained weights
#   checkpoints/rnn/seed_NNNN_H256.json          training metadata
#   data/rnn_trajectories/seed_NNNN_H256_trajectories.npz  h(t) arrays
# =============================================================================

#SBATCH --job-name=rnn_train_phase21
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/rnn_train_%A_%a.out
#SBATCH --error=cluster/logs/rnn_train_%A_%a.err
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

echo "Installing package..."
pip install -e ".[benchmark,dev,latent]" --quiet 2>&1 | tail -3

verify_torch
print_job_header "Phase 21 RNN Training (seed=${SLURM_ARRAY_TASK_ID})"

mkdir -p cluster/logs checkpoints/rnn data/rnn_trajectories

echo ""
echo "Starting RNN training for seed ${SLURM_ARRAY_TASK_ID}..."
echo ""

python cluster/scripts/train_rnn_ensemble.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  SEED ${SLURM_ARRAY_TASK_ID} COMPLETE (exit 0)"
else
    echo "  SEED ${SLURM_ARRAY_TASK_ID} FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
