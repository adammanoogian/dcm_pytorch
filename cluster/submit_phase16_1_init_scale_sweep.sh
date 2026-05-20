#!/bin/bash
# =============================================================================
# Phase 16.1 init_scale Sweep -- Submit Wrapper
# =============================================================================
# Submits the init_scale sweep sbatch job + a dependent push job. Strips CRLF
# first (gotcha #1 for Windows-cloned repos -- and THIS repo is cloned from a
# Windows checkout).
#
# Usage:
#   bash cluster/submit_phase16_1_init_scale_sweep.sh
#
# Overrides (pass through sbatch --export):
#   ENV_NAME=my_env bash cluster/submit_phase16_1_init_scale_sweep.sh
#   ENV_FALLBACK=other_env bash cluster/submit_phase16_1_init_scale_sweep.sh
#   PROJECT=ft29 bash cluster/submit_phase16_1_init_scale_sweep.sh
#   PUSH_TO_MAIN=true bash cluster/submit_phase16_1_init_scale_sweep.sh  # NOT recommended
# =============================================================================

set -u  # undefined vars are errors; no -e because we want to continue past
        # non-fatal warnings (e.g. sed returning non-zero on a clean file).

# --- 1. Strip CRLF (HPC README gotcha #1) ---------------------------------
# Idempotent; safe on already-clean files.
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# --- 2. Pre-flight: ensure the scripts exist ------------------------------
for f in cluster/run_phase16_1_init_scale_sweep.slurm cluster/99_push_phase16_1_results.slurm; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required script: $f" >&2
        exit 1
    fi
done

# --- 3. Build --export args, forwarding any overrides the user set --------
EXPORT_ARGS="ALL"
[[ -n "${ENV_NAME:-}" ]]     && EXPORT_ARGS="${EXPORT_ARGS},ENV_NAME=${ENV_NAME}"
[[ -n "${ENV_FALLBACK:-}" ]] && EXPORT_ARGS="${EXPORT_ARGS},ENV_FALLBACK=${ENV_FALLBACK}"
[[ -n "${PROJECT:-}" ]]      && EXPORT_ARGS="${EXPORT_ARGS},PROJECT=${PROJECT}"

# --- 4. Submit main diagnostic job ----------------------------------------
echo "Submitting Phase 16.1 init_scale sweep job (--export=${EXPORT_ARGS})..."
JOB1=$(sbatch --export="${EXPORT_ARGS}" --parsable cluster/run_phase16_1_init_scale_sweep.slurm)
if [[ -z "$JOB1" ]]; then
    echo "ERROR: sbatch did not return a job ID for the diagnostic job" >&2
    exit 1
fi
echo "  Diagnostic job ID: $JOB1"

# --- 5. Submit push job depending on diagnostic ----------------------------
echo "Submitting Phase 16.1 push job (depends on afterany:${JOB1})..."
PUSH_JOB=$(sbatch --dependency=afterany:${JOB1} \
    --export=ALL,PARENT_JOBS="$JOB1" \
    --parsable cluster/99_push_phase16_1_results.slurm)
if [[ -z "$PUSH_JOB" ]]; then
    echo "ERROR: sbatch did not return a job ID for the push job" >&2
    exit 1
fi
echo "  Push job ID: $PUSH_JOB"

# --- 6. Report ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 16.1 init_scale sweep dispatched"
echo "============================================================"
echo "Diagnostic: $JOB1"
echo "Push:       $PUSH_JOB (runs afterany regardless of diagnostic pass/fail)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/phase16_1_init_scale_sweep_${JOB1}.out"
echo "  tail -f cluster/logs/phase16_1_init_scale_sweep_${JOB1}.err"
echo ""
echo "Retrieve results after both jobs complete:"
echo "  git fetch origin"
echo "  git log origin/results/phase16_1-init-scale-sweep-* --oneline"
echo "============================================================"
