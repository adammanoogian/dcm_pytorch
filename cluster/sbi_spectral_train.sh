#!/bin/bash
# =============================================================================
# SBI Spectral DCM: NPE Training + SBC Validation (CPU)
# =============================================================================
# Trains an NPE density estimator on 50k simulated CSDs and validates
# calibration via 200 SBC trials. Expected wall time: ~30-60 min.
#
# Usage:
#   sbatch cluster/sbi_spectral_train.sh
#
# Overrides:
#   sbatch --export=ENV_NAME=my_env cluster/sbi_spectral_train.sh
#   sbatch --export=N_SIMS=100000 cluster/sbi_spectral_train.sh
#
# Outputs:
#   results/sbi_spectral_${SLURM_JOB_ID}/estimator.pt
#   results/sbi_spectral_${SLURM_JOB_ID}/sbc_ranks.pt
#   results/sbi_spectral_${SLURM_JOB_ID}/training_metadata.pt
# =============================================================================

#SBATCH --job-name=sbi_spectral
#SBATCH --output=cluster/logs/sbi_spectral_%j.out
#SBATCH --error=cluster/logs/sbi_spectral_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs results

START_ISO="$(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

echo "============================================================"
echo "SBI Spectral DCM: NPE Training + SBC Validation"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "Project root: $PROJECT_ROOT"
echo "Commit:       $COMMIT_SHA"
echo "Start time:   $START_ISO"
echo "============================================================"

# =============================================================================
# Configuration
# =============================================================================
_PROJECT="${PROJECT:-fc37}"
ENV_NAME="${ENV_NAME:-actinf-py-scripts}"
N_SIMS="${N_SIMS:-50000}"
N_SBC="${N_SBC:-200}"

# =============================================================================
# Conda Activation
# =============================================================================
conda deactivate 2>/dev/null || true

if conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Activated env: $ENV_NAME (by name)"
elif conda activate "/scratch/${_PROJECT}/${USER}/conda/envs/${ENV_NAME}" 2>/dev/null; then
    echo "Activated env: $ENV_NAME (scratch)"
else
    echo "ERROR: cannot activate env $ENV_NAME"
    exit 1
fi

# Install project + sbi dependency
echo "Syncing deps..."
pip install -q -e ".[dev]"
pip install -q "sbi>=0.22"

# Verify imports
if ! python -c "import torch, sbi, scipy, pyro_dcm" 2>/dev/null; then
    echo "ERROR: import check failed"
    python -c "import torch, sbi, scipy, pyro_dcm" || true
    exit 1
fi

echo "Python: $(python --version)"
echo "sbi:    $(python -c 'import sbi; print(sbi.__version__)')"
echo ""

# =============================================================================
# Run Training + SBC
# =============================================================================
OUTPUT_DIR="results/sbi_spectral_${SLURM_JOB_ID:-local}"

python scripts/train_sbi_spectral.py \
    --n-sims "$N_SIMS" \
    --n-sbc "$N_SBC" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

# =============================================================================
# Report
# =============================================================================
END_ISO="$(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
echo ""
echo "============================================================"
echo "SBI Training Complete"
echo "============================================================"
echo "Exit code:  $EXIT_CODE"
echo "End time:   $END_ISO"
echo "Output dir: $OUTPUT_DIR"
echo "============================================================"

exit $EXIT_CODE
