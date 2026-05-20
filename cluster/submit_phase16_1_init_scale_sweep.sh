#!/bin/bash
# =============================================================================
# Phase 16.1 init_scale Sweep -- Submit Wrapper
# =============================================================================
# Submits the init_scale sweep sbatch job. Results sync back to the local
# machine via Mutagen (no git autopush needed).
#
# Usage (from M3, in the project directory):
#   bash cluster/submit_phase16_1_init_scale_sweep.sh
#
# Usage (from local machine via SSH):
#   ssh m3 "cd ~/fc37/adam/projects/dcm_pytorch && bash cluster/submit_phase16_1_init_scale_sweep.sh"
#
# Overrides (env vars):
#   ENV_NAME=my_env bash cluster/submit_phase16_1_init_scale_sweep.sh
#   PROJECT=ft29 bash cluster/submit_phase16_1_init_scale_sweep.sh
# =============================================================================

set -u

# --- 1. Strip CRLF (safety for cross-OS Mutagen sync) --------------------
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# --- 2. Pre-flight --------------------------------------------------------
if [[ ! -f cluster/run_phase16_1_init_scale_sweep.slurm ]]; then
    echo "ERROR: missing cluster/run_phase16_1_init_scale_sweep.slurm" >&2
    exit 1
fi

# --- 3. Build --export args -----------------------------------------------
EXPORT_ARGS="ALL"
[[ -n "${ENV_NAME:-}" ]]     && EXPORT_ARGS="${EXPORT_ARGS},ENV_NAME=${ENV_NAME}"
[[ -n "${ENV_FALLBACK:-}" ]] && EXPORT_ARGS="${EXPORT_ARGS},ENV_FALLBACK=${ENV_FALLBACK}"
[[ -n "${PROJECT:-}" ]]      && EXPORT_ARGS="${EXPORT_ARGS},PROJECT=${PROJECT}"

# --- 4. Submit diagnostic job ---------------------------------------------
echo "Submitting Phase 16.1 init_scale sweep job (--export=${EXPORT_ARGS})..."
JOB1=$(sbatch --export="${EXPORT_ARGS}" --parsable cluster/run_phase16_1_init_scale_sweep.slurm)
if [[ -z "$JOB1" ]]; then
    echo "ERROR: sbatch did not return a job ID" >&2
    exit 1
fi

# --- 5. Report ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 16.1 init_scale sweep dispatched"
echo "============================================================"
echo "Job ID: $JOB1"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/phase16_1_init_scale_sweep_${JOB1}.out"
echo "  tail -f cluster/logs/phase16_1_init_scale_sweep_${JOB1}.err"
echo ""
echo "Results sync back automatically via Mutagen to:"
echo "  cluster/results/phase16_1_init_scale_sweep_${JOB1}.json"
echo "  cluster/results/phase16_1_init_scale_sweep_${JOB1}.md"
echo "============================================================"
