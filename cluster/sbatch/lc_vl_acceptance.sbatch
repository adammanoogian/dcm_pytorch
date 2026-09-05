#!/bin/bash
# =============================================================================
# Latent Circuit VL Acceptance Run (Phase 20-05 Tier-B rework)
# =============================================================================
# Each array task runs ONE seed: fit the N=4 bilinear ground truth with
# LatentCircuitForward + Variational Laplace (no random restarts; Gauss-Newton
# from the prior mean). dt=0.1 / 50s keeps the VL precision matrix tractable.
#
# Usage:
#   sbatch cluster/sbatch/lc_vl_acceptance.sbatch
#
# Outputs:
#   cluster/results/lc_vl_acceptance_<jobid>_<taskid>.json  Per-seed result
#   cluster/logs/lc_vl_accept_<jobid>_<taskid>.{out,err}    SLURM logs
# =============================================================================

#SBATCH --job-name=lc_vl_accept
#SBATCH --array=0-9
#SBATCH --output=cluster/logs/lc_vl_accept_%A_%a.out
#SBATCH --error=cluster/logs/lc_vl_accept_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env

# NEVER pip install inside an array job (N concurrent resolvers race on the
# shared venv -> OOM + .pth corruption). Package is pre-installed editable
# / found via sys.path in the Python script. See m3-hpc.skill convention.

verify_torch
print_job_header "LC VL Acceptance (seed ${SLURM_ARRAY_TASK_ID}/9)"

mkdir -p cluster/logs cluster/results

python cluster/scripts/lc_vl_acceptance_run.py
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
