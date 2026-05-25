#!/bin/bash
# =============================================================================
# Latent Circuit 10-Seed Acceptance Run (Phase 20-05, Task 3)
# =============================================================================
# Each array task runs ONE seed with 1000 SVI steps, 10 restarts, full
# 100s duration. Expected wall time: 12-24h per seed (conservative).
#
# Usage:
#   sbatch cluster/sbatch/lc_acceptance.sbatch
#
# Outputs:
#   cluster/results/lc_acceptance_<jobid>_<taskid>.json  Per-seed result
#   cluster/logs/lc_accept_<jobid>_<taskid>.out          SLURM stdout
#   cluster/logs/lc_accept_<jobid>_<taskid>.err          SLURM stderr
# =============================================================================

#SBATCH --job-name=lc_accept
#SBATCH --array=0-9
#SBATCH --output=cluster/logs/lc_accept_%A_%a.out
#SBATCH --error=cluster/logs/lc_accept_%A_%a.err
#SBATCH --time=24:00:00
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
print_job_header "LC Acceptance (seed ${SLURM_ARRAY_TASK_ID}/9)"

mkdir -p cluster/logs cluster/results

python cluster/scripts/lc_acceptance_run.py
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
