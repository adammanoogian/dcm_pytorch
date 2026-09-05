#!/bin/bash
# =============================================================================
# ERP pytest runner (M3) — v0.8.0 phase verification / regression
# =============================================================================
# Routes multi-file pytest (suites, regression, phase verification) to M3 per
# the cluster-routing rule (laptop is only for a single <30s test file during a
# tight TDD loop). Parameterized via TEST_TARGET (space-separated pytest args);
# defaults to the full v0.8.0 ERP scope.
#
# Usage:
#   ssh mentat 'cd $DCM_CLUSTER_ROOT && \
#     sbatch --export=ALL,TEST_TARGET="tests/test_x.py tests/test_y.py" \
#     cluster/sbatch/erp_pytest.sbatch'
#
# Outputs:
#   cluster/logs/erp_pytest_<jobid>.{out,err}   SLURM logs (synced back via Mutagen)
# NEVER pip install inside this job (shared venv; concurrent resolvers
# OOM + corrupt the env). The package is found via sys.path.
# =============================================================================

#SBATCH --job-name=erp_pytest
#SBATCH --output=cluster/logs/erp_pytest_%j.out
#SBATCH --error=cluster/logs/erp_pytest_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env
verify_torch
print_job_header "ERP pytest (v0.8.0 verification)"

TEST_TARGET="${TEST_TARGET:-tests/test_local_linearization.py tests/test_cmc_forward.py tests/test_spm_erp_dcm_validation.py tests/test_erp_coupled_system.py tests/test_spm_erp_multisource_validation.py tests/test_export_erp_multisource.py tests/test_erp_leadfield.py tests/test_spm_erp_leadfield_validation.py tests/test_erp_dcm_model.py tests/test_amortized_erp.py}"

echo "TEST_TARGET=${TEST_TARGET}"
mkdir -p cluster/logs

python -m pytest ${TEST_TARGET} -q -p no:cacheprovider
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  ERP PYTEST PASSED (exit 0)"
else
    echo "  ERP PYTEST FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"
exit $EXIT_CODE
