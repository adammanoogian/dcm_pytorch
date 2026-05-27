#!/bin/bash
# =============================================================================
# Hybrid VAE-DCM: Full-Scale Training on M3 Cluster
# =============================================================================
# Trains a physics-informed VAE with DCM ODE decoder on 1000 synthetic
# examples over 200 epochs. Validates amortized parameter recovery on 50
# held-out test examples.
#
# Expected runtime: ~2-3 hours on CPU (4 cores, 16GB).
#
# Usage:
#   sbatch cluster/sbatch_hybrid_vae_dcm.sh
#
# Overrides:
#   sbatch --export=N_SAMPLES=2000,N_EPOCHS=300 cluster/sbatch_hybrid_vae_dcm.sh
#   sbatch --export=ENV_NAME=my_env cluster/sbatch_hybrid_vae_dcm.sh
#
# Outputs:
#   results/hybrid_vae_dcm/encoder_checkpoint.pt
#   results/hybrid_vae_dcm/recovery_report.json
#   results/hybrid_vae_dcm/training_loss.png
# =============================================================================

#SBATCH --job-name=hvae_dcm_train
#SBATCH --output=cluster/logs/hvae_dcm_%j.out
#SBATCH --error=cluster/logs/hvae_dcm_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
source cluster/lib/cluster_env.sh
crlf_guard

# Configuration (overridable via --export)
ENV_NAME="${ENV_NAME:-actinf-py-scripts}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_EPOCHS="${N_EPOCHS:-200}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
N_REGIONS="${N_REGIONS:-4}"
N_INPUTS="${N_INPUTS:-1}"
DURATION="${DURATION:-5.0}"
DT="${DT:-0.01}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
N_TEST="${N_TEST:-50}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs results

# Activate environment
activate_env "$ENV_NAME"

# Thread control
setup_torch_threads "${SLURM_CPUS_ON_NODE:-4}"

# Install project in dev mode (no resolver overhead for array jobs)
pip install -q --no-deps -e .

# Verify imports
verify_torch

# Print job header
print_job_header "Hybrid VAE-DCM Training"

echo "Configuration:"
echo "  N_SAMPLES:     $N_SAMPLES"
echo "  N_EPOCHS:      $N_EPOCHS"
echo "  WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "  N_REGIONS:     $N_REGIONS"
echo "  N_INPUTS:      $N_INPUTS"
echo "  DURATION:      $DURATION"
echo "  DT:            $DT"
echo "  LR:            $LR"
echo "  SEED:          $SEED"
echo "  N_TEST:        $N_TEST"
echo ""

# =============================================================================
# Run Training
# =============================================================================
OUTPUT_DIR="results/hybrid_vae_dcm"

python scripts/train_hybrid_vae_dcm.py \
    --n_samples "$N_SAMPLES" \
    --n_epochs "$N_EPOCHS" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --n_regions "$N_REGIONS" \
    --n_inputs "$N_INPUTS" \
    --duration "$DURATION" \
    --dt "$DT" \
    --lr "$LR" \
    --seed "$SEED" \
    --n_test "$N_TEST" \
    --output_dir "$OUTPUT_DIR" \
    --save_encoder \
    --save_recovery_report

EXIT_CODE=$?

# =============================================================================
# Report
# =============================================================================
END_ISO="$(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
echo ""
echo "============================================================"
echo "Hybrid VAE-DCM Training Complete"
echo "============================================================"
echo "Exit code:  $EXIT_CODE"
echo "End time:   $END_ISO"
echo "Output dir: $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/recovery_report.json" ]]; then
    echo ""
    echo "Recovery report:"
    cat "$OUTPUT_DIR/recovery_report.json" | python -m json.tool 2>/dev/null \
        || cat "$OUTPUT_DIR/recovery_report.json"
fi
echo "============================================================"

exit $EXIT_CODE
