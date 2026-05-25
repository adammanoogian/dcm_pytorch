#!/bin/bash
# =============================================================================
# Latent Circuit Prior Calibration Sweep (Phase 20-05, Task 1) -- Array Job
# =============================================================================
# Sweeps 48 combinations of (LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE,
# init_scale) as a SLURM array job. Each task runs ONE combination with
# seed=42, 500 SVI steps, 3 restarts, 10s simulation duration.
#
# Usage:
#   sbatch cluster/sbatch/lc_calibration.sbatch
#
# After completion, aggregate results:
#   python cluster/scripts/lc_calibration_aggregate.py <job_id>
#
# Outputs:
#   cluster/results/lc_calib_<jobid>_<taskid>.json   Per-task result
#   cluster/logs/lc_calib_<jobid>_<taskid>.out        SLURM stdout
#   cluster/logs/lc_calib_<jobid>_<taskid>.err        SLURM stderr
# =============================================================================

#SBATCH --job-name=lc_calib
#SBATCH --array=0-47
#SBATCH --output=cluster/logs/lc_calib_%A_%a.out
#SBATCH --error=cluster/logs/lc_calib_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

echo "Installing package..."
pip install -e ".[benchmark,dev]" --quiet 2>&1 | tail -3

verify_torch
print_job_header "LC Calibration (task ${SLURM_ARRAY_TASK_ID}/47)"

mkdir -p cluster/logs cluster/results

python cluster/scripts/lc_prior_calibration_sweep.py
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
