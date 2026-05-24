#!/bin/bash
# =============================================================================
# submit_pytest.sh -- Submit pytest job to M3 via SSH
# =============================================================================
# Usage (from local machine):
#   bash cluster/submit_pytest.sh "tests/test_latent_circuit_forward.py -x -v"
#   bash cluster/submit_pytest.sh "tests/test_multi_start_svi.py -v" --label multi_start
# =============================================================================
set -u

TEST_TARGET=""
JOB_LABEL=""
TIME_OVERRIDE=""
MEM_OVERRIDE=""
INSTALL_DEPS="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)      JOB_LABEL="$2"; shift 2 ;;
        --time)       TIME_OVERRIDE="$2"; shift 2 ;;
        --mem)        MEM_OVERRIDE="$2"; shift 2 ;;
        --no-install) INSTALL_DEPS="false"; shift ;;
        -*)           echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if [[ -z "$TEST_TARGET" ]]; then
                TEST_TARGET="$1"
            else
                TEST_TARGET="$TEST_TARGET $1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$TEST_TARGET" ]]; then
    echo "Usage: submit_pytest.sh <test_target> [--label NAME] [--time HH:MM:SS]"
    exit 2
fi

if [[ -z "$JOB_LABEL" ]]; then
    JOB_LABEL=$(echo "$TEST_TARGET" | grep -oP 'test_\K[a-z_]+' | head -1 || echo "pytest")
    [[ -z "$JOB_LABEL" ]] && JOB_LABEL="pytest"
fi

SBATCH_OVERRIDES=""
[[ -n "$TIME_OVERRIDE" ]] && SBATCH_OVERRIDES="$SBATCH_OVERRIDES --time=$TIME_OVERRIDE"
[[ -n "$MEM_OVERRIDE" ]] && SBATCH_OVERRIDES="$SBATCH_OVERRIDES --mem=$MEM_OVERRIDE"

REMOTE_DIR="~/fc37/adam/projects/dcm_pytorch"

echo "Submitting pytest job to M3..."
echo "  Target:  $TEST_TARGET"
echo "  Label:   $JOB_LABEL"
echo ""

JOB_ID=$(ssh m3 "cd $REMOTE_DIR && \
    sbatch --parsable \
        --job-name=${JOB_LABEL} \
        --export=ALL,TEST_TARGET='${TEST_TARGET}',JOB_LABEL='${JOB_LABEL}',INSTALL_DEPS='${INSTALL_DEPS}' \
        ${SBATCH_OVERRIDES} \
        cluster/run_pytest.slurm" 2>&1)

if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "============================================================"
    echo "  Job submitted: $JOB_ID"
    echo "============================================================"
    echo "  Monitor: ssh m3 \"squeue -j $JOB_ID\""
    echo "  Logs:    cluster/logs/pytest_${JOB_LABEL}_${JOB_ID}.log"
    echo "============================================================"
else
    echo "ERROR: sbatch failed: $JOB_ID" >&2
    exit 1
fi
