#!/bin/bash
# =============================================================================
# Phase 16 v0.3.0 -- Bilinear DCM Acceptance Gate (CPU)
# =============================================================================
# Runs the slow acceptance test for the task-bilinear recovery benchmark:
#   tests/test_task_bilinear_benchmark.py::test_acceptance_gates_pass_at_10_seeds
#
# Expected wall time: ~80-150 min (10 seeds x 500 SVI steps, CPU-only).
# Wall limit: 4h (safety margin for slow nodes or GC pressure).
#
# Stack: PyTorch + Pyro only.
#
# Usage:
#   # Default invocation (actinf-py-scripts primary, no fallback, project fc37):
#   sbatch cluster/run_phase16_acceptance.slurm
#
#   # Override primary env:
#   sbatch --export=ENV_NAME=my_env cluster/run_phase16_acceptance.slurm
#
#   # Override scratch project code:
#   sbatch --export=PROJECT=ft29 cluster/run_phase16_acceptance.slurm
#
#   # Recommended: use the submit wrapper which also chains a push job:
#   bash cluster/submit_phase16.sh
#
# Env contract:
#   The script runs `pip install -e .[benchmark,dev]` into the activated env,
#   so the single source of truth for dependencies is pyproject.toml in this
#   repo. If the activated env is missing torch / pyro-ppl / torchdiffeq / zuko
#   / scipy / matplotlib, pip will pull them. Note that a missing torch will
#   pull ~2GB on first run.
#
# Outputs:
#   cluster/logs/phase16_<jobid>.out              SLURM stdout
#   cluster/logs/phase16_<jobid>.err              SLURM stderr
#   cluster/logs/pytest_phase16_<jobid>.log       Pytest tee log
#   cluster/results/phase16_acceptance_<jobid>.json  Machine-readable summary
#   cluster/results/phase16_acceptance_<jobid>.md    Human-readable report
#
# Derived from: project_utils/templates/hpc/slurm_bayesian_cpu_TEMPLATE.slurm
# =============================================================================

#SBATCH --job-name=phase16_acceptance
#SBATCH --output=cluster/logs/phase16_%j.out
#SBATCH --error=cluster/logs/phase16_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# NOTE: No `set -e` -- we want to capture the pytest exit code and still emit
# the result summary + sacct report even when the acceptance gate fails.

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Create output dirs up-front so both logging and later stages can use them.
mkdir -p cluster/logs cluster/results

START_ISO="$(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
BRANCH_NAME_GIT="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

echo "============================================================"
echo "Phase 16 v0.3.0 Bilinear Acceptance Gate -- SLURM Job"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "Project root: $PROJECT_ROOT"
echo "Branch:       $BRANCH_NAME_GIT"
echo "Commit:       $COMMIT_SHA"
echo "Start time:   $START_ISO"
echo "============================================================"

# =============================================================================
# Configuration (override via --export=KEY=VALUE)
# =============================================================================
_PROJECT="${PROJECT:-fc37}"
ENV_NAME="${ENV_NAME:-actinf-py-scripts}"
ENV_FALLBACK="${ENV_FALLBACK:-}"

# Track which env actually activated, for the result summary.
ACTIVATED_ENV="unknown"

# =============================================================================
# Conda Activation + Dep Sync (fallback on activate OR import failure)
# =============================================================================
# Previous version only fell back when `conda activate` itself failed. A live
# env with a missing package (e.g. torch) would activate cleanly and then crash
# the import check without trying the fallback. This version treats missing
# deps as a fallback trigger, and always runs `pip install -e .[benchmark,dev]`
# so pyproject.toml is the single source of truth for package versions.

activate_and_verify() {
    local env="$1"
    local project="$2"
    local activated=""

    conda deactivate 2>/dev/null || true

    if conda activate "$env" 2>/dev/null; then
        activated="$env (by name)"
    elif conda activate "/scratch/${project}/${USER}/conda/envs/${env}" 2>/dev/null; then
        activated="$env (scratch)"
    else
        echo "  cannot activate $env (tried name + /scratch/${project}/${USER}/conda/envs/${env})"
        return 1
    fi

    echo "  activated $activated"

    # Sync deps from pyproject.toml. -q to keep log small when nothing to do.
    echo "  syncing deps via pip install -e .[benchmark,dev]"
    if ! pip install -q -e ".[benchmark,dev]"; then
        echo "  pip install failed in $activated"
        local py_ver
        py_ver="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo unknown)"
        echo "  active Python: $py_ver (pyproject.toml requires >=3.11)"
        echo "  if the error was 'requires a different Python', submit with a 3.11+ env:"
        echo "    sbatch --export=ENV_NAME=<py311_env> cluster/run_phase16_acceptance.slurm"
        echo "  or re-submit with PIP_IGNORE_REQUIRES_PYTHON=1 to override (expert only)."
        return 1
    fi

    # Must-have imports (the project's runtime + benchmark stack).
    if ! python -c "import torch, pyro, torchdiffeq, zuko, scipy, matplotlib" 2>/dev/null; then
        echo "  import check failed in $activated after pip install"
        python -c "import torch, pyro, torchdiffeq, zuko, scipy, matplotlib" || true
        return 1
    fi

    ACTIVATED_ENV="$activated"
    return 0
}

echo "Trying primary env: $ENV_NAME"
if ! activate_and_verify "$ENV_NAME" "$_PROJECT"; then
    if [[ -n "$ENV_FALLBACK" ]]; then
        echo "Primary env unusable; trying fallback: $ENV_FALLBACK"
        if ! activate_and_verify "$ENV_FALLBACK" "$_PROJECT"; then
            echo "ERROR: neither $ENV_NAME nor $ENV_FALLBACK satisfies project deps"
            echo "Required packages (from pyproject.toml):"
            echo "  runtime:   torch>=2.0, torchdiffeq, pyro-ppl>=1.9, scipy, numpy, zuko>=1.2"
            echo "  benchmark: matplotlib, tabulate"
            echo "  dev:       pytest"
            echo "Fix on a login node:"
            echo "  conda activate $ENV_NAME && pip install -e .[benchmark,dev]"
            exit 1
        fi
    else
        echo "ERROR: $ENV_NAME unusable and no ENV_FALLBACK set"
        echo "Required packages (from pyproject.toml):"
        echo "  runtime:   torch>=2.0, torchdiffeq, pyro-ppl>=1.9, scipy, numpy, zuko>=1.2"
        echo "  benchmark: matplotlib, tabulate"
        echo "  dev:       pytest"
        echo "Fix on a login node:"
        echo "  conda activate $ENV_NAME && pip install -e .[benchmark,dev]"
        echo "Or submit with an override: sbatch --export=ENV_NAME=<env> ..."
        exit 1
    fi
fi

# Echo the active stack for debugging/audit.
python -c "
import torch, pyro, torchdiffeq, zuko, scipy, matplotlib
print(f'torch {torch.__version__} | pyro {pyro.__version__} | torchdiffeq {torchdiffeq.__version__}')
print(f'zuko {zuko.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Project imports (must succeed after pip install -e .).
python -c "
from pyro_dcm.models.task_dcm_model import task_dcm_model
from benchmarks.bilinear_metrics import compute_acceptance_gates
from benchmarks.plotting import plot_acceptance_gates_table
print('project imports OK')
" || {
    echo "ERROR: project import check failed"
    echo "Likely causes: pip install -e . did not register the package, or benchmarks/ module missing."
    exit 1
}
echo ""

# =============================================================================
# Run Phase 16 Acceptance Gate
# =============================================================================
echo "[$(date)] Running Phase 16 acceptance gate (10 seeds x 500 SVI steps)"
START_TIME=$(date +%s)

pytest tests/test_task_bilinear_benchmark.py \
    -m slow -k acceptance \
    -s -v --tb=short \
    2>&1 | tee "cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log"

# IMPORTANT: use PIPESTATUS[0] -- $? captures tee's exit, not pytest's.
EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
END_ISO="$(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"

echo ""
echo "[$(date)] Pytest finished (exit code: $EXIT_CODE, elapsed: ${ELAPSED}s)"

# =============================================================================
# Write Result Summaries
# =============================================================================
RESULT_JSON="cluster/results/phase16_acceptance_${SLURM_JOB_ID}.json"
RESULT_MD="cluster/results/phase16_acceptance_${SLURM_JOB_ID}.md"

cat > "$RESULT_JSON" <<EOF
{
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "node": "${SLURMD_NODENAME:-$(hostname)}",
  "branch": "${BRANCH_NAME_GIT}",
  "commit_sha": "${COMMIT_SHA}",
  "env_name": "${ACTIVATED_ENV}",
  "exit_code": ${EXIT_CODE},
  "elapsed_seconds": ${ELAPSED},
  "start_time": "${START_ISO}",
  "end_time": "${END_ISO}",
  "pytest_log": "cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log"
}
EOF

if [ $EXIT_CODE -eq 0 ]; then
    STATUS_TEXT="PASS"
else
    STATUS_TEXT="FAIL (exit $EXIT_CODE)"
fi

cat > "$RESULT_MD" <<EOF
# Phase 16 Acceptance Gate -- Job ${SLURM_JOB_ID:-local}

**Status:** ${STATUS_TEXT}

- **Node:** ${SLURMD_NODENAME:-$(hostname)}
- **Branch:** ${BRANCH_NAME_GIT}
- **Commit:** ${COMMIT_SHA}
- **Env:** ${ACTIVATED_ENV}
- **Started:** ${START_ISO}
- **Finished:** ${END_ISO}
- **Elapsed:** ${ELAPSED} s (~$((ELAPSED / 60)) min)
- **Pytest log:** \`cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log\`

Test: \`tests/test_task_bilinear_benchmark.py -m slow -k acceptance\`

See the pytest log for per-seed metrics and acceptance-gate details.
EOF

echo ""
echo "Wrote result summaries:"
echo "  $RESULT_JSON"
echo "  $RESULT_MD"

# =============================================================================
# Job Resource Report
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE REPORT"
echo "============================================================"
sacct -j "$SLURM_JOB_ID" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo ""
echo "For detailed stats: jobstats ${SLURM_JOB_ID}"
echo "============================================================"

# Propagate the pytest exit code so SLURM marks the job accordingly.
exit $EXIT_CODE
