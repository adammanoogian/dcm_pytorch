#!/bin/bash
# =============================================================================
# Phase 22 Perturbation Experiment (single job)
# =============================================================================
# Runs the full perturbation sweep: baseline DCM + 9 perturbation conditions.
# Each condition: generate data -> extract latents -> fit spectral DCM.
#
# Estimated runtime: 9 conditions x ~60 min each = ~9 hours (12h budget)
# Memory: spectral DCM with 20 latent dims + LSTM-AE training = ~16 GB peak
#
# Usage:
#   sbatch cluster/sbatch/22_perturbation.sbatch
#
# Overrides (via sbatch --export=ALL,...):
#   PERTURB_SVI_STEPS=500    fewer SVI steps (faster, less accurate)
#   PERTURB_N_RESTARTS=3     fewer restarts
#   PERTURB_AE_EPOCHS=50     fewer AE training epochs
#   PERTURB_SEED=42          random seed
#   PERTURB_OUTPUT_DIR=/path  custom output directory
#
# Outputs:
#   cluster/logs/perturb_22_<JOBID>.out   stdout
#   cluster/logs/perturb_22_<JOBID>.err   stderr
#   $PERTURB_OUTPUT_DIR/perturbation_results.npz   experiment results
#   $PERTURB_OUTPUT_DIR/ae_checkpoint.pt            autoencoder weights
# =============================================================================

#SBATCH --job-name=perturb_22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/perturb_22_%j.out
#SBATCH --error=cluster/logs/perturb_22_%j.err
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# Package is pre-installed (editable).  NEVER run pip install inside
# SLURM jobs -- concurrent resolvers OOM + corrupt env.

verify_torch
print_job_header "Phase 22 Perturbation Experiment"

# Configurable parameters with defaults
SVI_STEPS="${PERTURB_SVI_STEPS:-1000}"
N_RESTARTS="${PERTURB_N_RESTARTS:-10}"
AE_EPOCHS="${PERTURB_AE_EPOCHS:-150}"
SEED="${PERTURB_SEED:-42}"
OUTPUT_DIR="${PERTURB_OUTPUT_DIR:-results/perturbation}"

mkdir -p cluster/logs "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  SVI steps:    $SVI_STEPS"
echo "  N restarts:   $N_RESTARTS"
echo "  AE epochs:    $AE_EPOCHS"
echo "  Seed:         $SEED"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

python scripts/22_perturbation_experiment.py \
    --output-dir "$OUTPUT_DIR" \
    --svi-steps "$SVI_STEPS" \
    --n-restarts "$N_RESTARTS" \
    --ae-epochs "$AE_EPOCHS" \
    --seed "$SEED"

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  PERTURBATION EXPERIMENT COMPLETE (exit 0)"
    echo "  Results: $OUTPUT_DIR/perturbation_results.npz"
else
    echo "  PERTURBATION EXPERIMENT FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
