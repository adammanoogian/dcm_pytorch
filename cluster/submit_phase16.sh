#!/bin/bash
# =============================================================================
# Phase 16 Acceptance -- Submit Wrapper
# =============================================================================
# Submits the acceptance sbatch job + a dependent push job. Strips CRLF first
# (gotcha #1 for Windows-cloned repos -- and THIS repo is cloned from a Windows
# checkout).
#
# Usage:
#   bash cluster/submit_phase16.sh
#
# Overrides (pass through sbatch --export):
#   ENV_NAME=my_env bash cluster/submit_phase16.sh       # primary conda env
#   ENV_FALLBACK=other_env bash cluster/submit_phase16.sh # secondary env
#   PROJECT=ft29 bash cluster/submit_phase16.sh           # /scratch project code
#   PUSH_TO_MAIN=true bash cluster/submit_phase16.sh      # NOT recommended
# =============================================================================

set -u  # undefined vars are errors; no -e because we want to continue past
        # non-fatal warnings (e.g. sed returning non-zero on a clean file).

# --- 1. Strip CRLF (HPC README gotcha #1) ---------------------------------
# Idempotent; safe on already-clean files.
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# --- 2. Pre-flight: ensure the scripts exist ------------------------------
for f in cluster/run_phase16_acceptance.slurm cluster/99_push_phase16_results.slurm; do
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

# --- 4. Submit main acceptance job ---------------------------------------
echo "Submitting Phase 16 acceptance job (--export=${EXPORT_ARGS})..."
JOB1=$(sbatch --export="${EXPORT_ARGS}" --parsable cluster/run_phase16_acceptance.slurm)
if [[ -z "$JOB1" ]]; then
    echo "ERROR: sbatch did not return a job ID for the acceptance job" >&2
    exit 1
fi
echo "  Acceptance job ID: $JOB1"

# --- 5. Submit push job depending on acceptance ---------------------------
echo "Submitting Phase 16 push job (depends on afterany:${JOB1})..."
PUSH_JOB=$(sbatch --dependency=afterany:${JOB1} \
    --export=ALL,PARENT_JOBS="$JOB1" \
    --parsable cluster/99_push_phase16_results.slurm)
if [[ -z "$PUSH_JOB" ]]; then
    echo "ERROR: sbatch did not return a job ID for the push job" >&2
    exit 1
fi
echo "  Push job ID: $PUSH_JOB"

# --- 6. Report ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 16 acceptance dispatched"
echo "============================================================"
echo "Acceptance: $JOB1"
echo "Push:       $PUSH_JOB (runs afterany regardless of acceptance pass/fail)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/phase16_${JOB1}.out"
echo "  tail -f cluster/logs/phase16_${JOB1}.err"
echo ""
echo "Retrieve results after both jobs complete:"
echo "  git fetch origin"
echo "  git log origin/results/phase16-acceptance-* --oneline"
echo "============================================================"
