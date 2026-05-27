#!/bin/bash
#SBATCH --job-name=meeg_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/meeg_extract_%j.out
#SBATCH --error=cluster/logs/meeg_extract_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# ============================================================
# Phase 24: M/EEG Foundation Model Extraction + DCM Fitting
# ============================================================
# Extracts latent dynamics from M/EEG data using LaBraM or
# BrainOmni, then fits spectral DCM to the latent dynamics.
#
# Usage:
#   sbatch --export=INPUT_FIF=/path/to/epochs-epo.fif \
#          cluster/sbatch/24_meeg_extract.slurm
#
# Optional environment variables:
#   MODEL       - "labram" (default) or "brainomni"
#   N_COMP      - PCA components for DCM (default: 4)
#   DCM_TYPE    - "spectral" (default) or "latent_circuit"
#   NUM_STEPS   - SVI steps (default: 2000)
#   N_RESTARTS  - Multi-start restarts (default: 10)
#   CHECKPOINT  - Path to local model weights (optional)
# ============================================================

set -euo pipefail

# --- Environment Setup ---
module purge
module load cuda/12.1
module load python/3.12.2

# Activate project virtualenv (adjust path as needed)
source "${HOME}/.venvs/dcm/bin/activate" 2>/dev/null || true

# Install braindecode if not present (single job, not array)
pip install --quiet braindecode>=1.3.0 huggingface_hub>=0.20

# --- Configuration ---
MODEL="${MODEL:-labram}"
N_COMP="${N_COMP:-4}"
DCM_TYPE="${DCM_TYPE:-spectral}"
NUM_STEPS="${NUM_STEPS:-2000}"
N_RESTARTS="${N_RESTARTS:-10}"
OUTPUT_DIR="results/phase24_meeg/"

echo "=== M/EEG Foundation Model Extraction ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:        ${MODEL}"
echo "Input:        ${INPUT_FIF}"
echo "Components:   ${N_COMP}"
echo "DCM type:     ${DCM_TYPE}"
echo "SVI steps:    ${NUM_STEPS}"
echo "Restarts:     ${N_RESTARTS}"
echo ""

# --- Step 1: Extract latent dynamics ---
echo "--- Step 1: Latent Extraction ---"
EXTRACT_CMD="python scripts/24_extract_meeg_latents.py \
    --model ${MODEL} \
    --input-fif ${INPUT_FIF} \
    --output-dir ${OUTPUT_DIR} \
    --n-components ${N_COMP}"

if [ -n "${CHECKPOINT:-}" ]; then
    EXTRACT_CMD="${EXTRACT_CMD} --checkpoint ${CHECKPOINT}"
fi

echo "Running: ${EXTRACT_CMD}"
eval "${EXTRACT_CMD}"
echo ""

# --- Step 2: Fit DCM ---
echo "--- Step 2: DCM Fitting ---"
DCM_CMD="python scripts/24_fit_dcm_meeg.py \
    --input-npz ${OUTPUT_DIR}/meeg_latent_dynamics.npz \
    --output-dir ${OUTPUT_DIR} \
    --dcm-type ${DCM_TYPE} \
    --num-steps ${NUM_STEPS} \
    --n-restarts ${N_RESTARTS}"

echo "Running: ${DCM_CMD}"
eval "${DCM_CMD}"
echo ""

echo "=== Pipeline Complete ==="
echo "Results in: ${OUTPUT_DIR}"
