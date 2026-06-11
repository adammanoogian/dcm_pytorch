#!/bin/bash
# =============================================================================
# BMR Tempering Calibration (Phase 31-03, VLBMR-03 EXPLORATORY)
# =============================================================================
# Single task (NOT an array): re-fits ONE representative seed of the Phase 30
# task-N4 stress cell (the strongest VL Laplace-overconfidence regime,
# coverage_95 == 0.0), calibrates a posterior tempering factor T by
# coverage-matching against recovery_matrix.json, emits side-by-side untempered
# vs tempered rank_connections output, and validates T on a held-out task-N2
# cell. ALL tempering routes through temper_vl_posterior (the loud PD guard).
# Absolute delta-F is NEVER a pass/fail criterion (Pitfall C1/C2).
#
# dt>=0.1 is enforced INSIDE the script (precision floor, Pitfall N1); the
# task ground-truth sim uses fixed-step rk4 (the dopri5-underflow fix,
# commit c0a7616). This job only touches task DCM, so it is unaffected by the
# Mutagen `models/` anchored-ignore footgun.
#
# Usage:
#   sbatch cluster/sbatch/bmr_tempering_calibration.sbatch
#
# Outputs:
#   cluster/results/bmr_tempering_calibration_<jobid>.json   Calibration result
#   cluster/logs/bmr_temper_<jobid>.{out,err}                SLURM logs
# =============================================================================

#SBATCH --job-name=bmr_temper
#SBATCH --output=cluster/logs/bmr_temper_%j.out
#SBATCH --error=cluster/logs/bmr_temper_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# NEVER pip install inside this job. The package is found via sys.path in the
# Python script; there is no install step here. Concurrent resolvers against
# the shared conda env race -> OOM + .pth corruption (project hard rule). See
# m3-hpc.skill convention and cluster/sbatch/recovery_matrix_sweep.sbatch.

verify_torch
print_job_header "BMR Tempering Calibration (task-N4 stress)"

mkdir -p cluster/logs cluster/results

python cluster/scripts/bmr_tempering_calibration.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  BMR TEMPERING CALIBRATION COMPLETE (exit 0)"
else
    echo "  BMR TEMPERING CALIBRATION FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
