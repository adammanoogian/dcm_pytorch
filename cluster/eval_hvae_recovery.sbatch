#!/bin/bash
# =============================================================================
# HVAE-02 eval-only: recompute MASKED A sign recovery on the trained checkpoint
# =============================================================================
# Loads results/hybrid_vae_dcm/encoder_checkpoint.pt, regenerates the identical
# held-out set, and recomputes masked vs unmasked sign recovery. No retraining.
# Closes the HVAE-02 audit gap (.planning/v0.6.0-AUDIT.md). Quick (<15 min).
#
# Usage:  sbatch cluster/eval_hvae_recovery.sbatch
# =============================================================================
#SBATCH --job-name=eval_hvae
#SBATCH --output=cluster/logs/eval_hvae_%j.out
#SBATCH --error=cluster/logs/eval_hvae_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cluster/logs

source cluster/lib/cluster_env.sh
crlf_guard
activate_env

if ! python -c "import torch, pyro_dcm" 2>/dev/null; then
    echo "ERROR: import check failed"
    python -c "import torch, pyro_dcm" || true
    exit 1
fi

echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python --version

python scripts/eval_hybrid_vae_dcm.py \
    --checkpoint results/hybrid_vae_dcm/encoder_checkpoint.pt \
    --output-dir results/hybrid_vae_dcm

echo "Exit code: $?"
