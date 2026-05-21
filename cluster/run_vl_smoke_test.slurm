#!/bin/bash
# =============================================================================
# VL Smoke Test -- Variational Laplace recovery on simulated bilinear DCM
# =============================================================================
# Runs the @slow VL tests:
#   tests/test_variational_laplace.py -m slow
#
# Expected wall time: 5-15 min (deterministic GN, no sampling).
# Wall limit: 30 min.
#
# Usage:
#   sbatch cluster/run_vl_smoke_test.slurm
# =============================================================================

#SBATCH --job-name=vl_smoke_test
#SBATCH --output=cluster/logs/vl_smoke_%j.out
#SBATCH --error=cluster/logs/vl_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"
mkdir -p cluster/logs cluster/results

echo "============================================================"
echo "VL Smoke Test -- SLURM Job"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "Project root: $PROJECT_ROOT"
echo "Branch:       $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "Commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Start time:   $(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
echo "============================================================"

_PROJECT="${PROJECT:-fc37}"
ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

conda deactivate 2>/dev/null || true
if conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Activated env: $ENV_NAME (by name)"
elif conda activate "/scratch/${_PROJECT}/${USER}/conda/envs/${ENV_NAME}" 2>/dev/null; then
    echo "Activated env: $ENV_NAME (scratch)"
else
    echo "ERROR: cannot activate $ENV_NAME"
    exit 1
fi

pip install -q -e ".[dev]"

echo ""
echo "Running VL smoke tests..."
echo ""

python -m pytest tests/test_variational_laplace.py -m slow -v -s \
    --tb=long 2>&1 | tee "cluster/logs/pytest_vl_smoke_${SLURM_JOB_ID}.log"

PYTEST_EXIT=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "Pytest exit code: $PYTEST_EXIT"
echo "End time: $(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
echo "============================================================"

exit $PYTEST_EXIT
