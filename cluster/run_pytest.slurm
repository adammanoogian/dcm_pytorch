#!/bin/bash
# =============================================================================
# Generic Pytest Runner for SLURM (dcm_pytorch)
# =============================================================================
# Runs a single pytest invocation on M3. Parameterized via environment variables.
#
# Required env vars:
#   TEST_TARGET  -- pytest target (e.g. "tests/test_foo.py -x -v")
#
# Optional env vars:
#   ENV_NAME     -- conda env (default: actinf-py-scripts)
#   JOB_LABEL    -- label for logs (default: pytest)
#   INSTALL_DEPS -- "true" to pip install -e . (default: true)
#
# Usage:
#   sbatch --export=ALL,TEST_TARGET="tests/test_foo.py -v" cluster/run_pytest.slurm
# =============================================================================

#SBATCH --job-name=pytest
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"
JOB_LABEL="${JOB_LABEL:-pytest}"
INSTALL_DEPS="${INSTALL_DEPS:-true}"

if [[ -z "${TEST_TARGET:-}" ]]; then
    echo "ERROR: TEST_TARGET environment variable is required"
    exit 1
fi

setup_torch_threads 4
activate_env "$ENV_NAME"

if [[ "$INSTALL_DEPS" == "true" ]]; then
    echo "Installing package in editable mode..."
    pip install -e ".[benchmark,dev]" --quiet 2>&1 | tail -3
fi

verify_torch
print_job_header "Pytest: ${JOB_LABEL}"

mkdir -p cluster/logs cluster/results

PYTEST_LOG="cluster/logs/pytest_${JOB_LABEL}_${SLURM_JOB_ID}.log"
RESULT_JSON="cluster/results/${JOB_LABEL}_${SLURM_JOB_ID}.json"

echo ""
echo "Running: python -m pytest ${TEST_TARGET}"
echo "Log: ${PYTEST_LOG}"
echo ""

START_TIME=$(date +%s)

python -m pytest ${TEST_TARGET} 2>&1 | tee "$PYTEST_LOG"
PYTEST_EXIT=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

cat > "$RESULT_JSON" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_label": "${JOB_LABEL}",
  "test_target": "${TEST_TARGET}",
  "exit_code": ${PYTEST_EXIT},
  "elapsed_seconds": ${ELAPSED},
  "timestamp": "$(date -Iseconds)",
  "node": "${SLURMD_NODENAME:-unknown}",
  "commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
}
EOF

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $PYTEST_EXIT -eq 0 ]]; then
    echo "  PYTEST PASSED (exit 0) in ${ELAPSED}s"
else
    echo "  PYTEST FAILED (exit ${PYTEST_EXIT}) in ${ELAPSED}s"
fi
echo "  Log: ${PYTEST_LOG}"
echo "  Result: ${RESULT_JSON}"
echo "════════════════════════════════════════════════════════"

exit $PYTEST_EXIT
