#!/bin/bash
# =============================================================================
# ELBO Model Selection for Latent Circuit DCM (Phase 20-05, SYNTH-03)
# =============================================================================
# Tests N_candidate in {2,3,4,5,6} × 3 seeds = 15 array tasks.
# Each task fits one (N, seed) combo with 300 steps, 3 restarts, 50s duration.
#
# Usage:
#   sbatch cluster/sbatch/lc_elbo.sbatch
# =============================================================================

#SBATCH --job-name=lc_elbo
#SBATCH --array=0-14
#SBATCH --output=cluster/logs/lc_elbo_%A_%a.out
#SBATCH --error=cluster/logs/lc_elbo_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# Package is pre-installed (editable) or found via sys.path in the
# Python script.  NEVER run pip install inside array jobs -- N
# concurrent resolvers race on the shared conda env (OOM + .pth
# corruption).  See m3-hpc.skill convention 6.

verify_torch
print_job_header "LC ELBO Selection (task ${SLURM_ARRAY_TASK_ID}/14)"

mkdir -p cluster/logs cluster/results

python cluster/scripts/lc_elbo_model_selection.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  TASK ${SLURM_ARRAY_TASK_ID} COMPLETE (exit 0)"
else
    echo "  TASK ${SLURM_ARRAY_TASK_ID} FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
