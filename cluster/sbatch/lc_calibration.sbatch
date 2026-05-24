#!/bin/bash
# =============================================================================
# Latent Circuit Prior Calibration Sweep (Phase 20-05, Task 1)
# =============================================================================
# Sweeps 48 combinations of (LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE, init_scale)
# on single seed=42 with 1000 SVI steps, 3 restarts.
# Expected wall time: 2-6 hours (48 combos x 2-5 min each).
# =============================================================================

#SBATCH --job-name=lc_calibration_sweep
#SBATCH --output=cluster/logs/lc_calibration_%j.out
#SBATCH --error=cluster/logs/lc_calibration_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
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
print_job_header "LC Prior Calibration Sweep"

mkdir -p cluster/logs cluster/results

echo ""
echo "Starting calibration sweep..."
echo ""

python cluster/scripts/lc_prior_calibration_sweep.py 2>&1 | tee "cluster/logs/lc_calibration_detail_${SLURM_JOB_ID}.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CALIBRATION SWEEP COMPLETE (exit 0)"
else
    echo "  CALIBRATION SWEEP FAILED (exit ${EXIT_CODE})"
fi
echo "  Results: cluster/results/lc_calibration_sweep_${SLURM_JOB_ID}.json"
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
