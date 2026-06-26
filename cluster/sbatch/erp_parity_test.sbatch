#!/bin/bash
# =============================================================================
# SPM12 single-source CMC-ERP parity gate (Phase 33-03, CMC-06)
# =============================================================================
# Runs the V5 staged parity ladder + the MEASURED matrix_exp<->spm_expm floor
# against the byte-frozen fixtures generated in Plan 33-02
# (validation/data/erp_single_source_fixtures.mat). The suite is pure-torch
# vs the frozen .mat arrays -- it needs the FIXTURE, not a live MATLAB, so it
# also passes on the laptop. It is run here on M3 as the official assertion run
# (compute routing), where the Mutagen-synced fixture is present and MATLAB +
# SPM12 are available should anyone regenerate the fixture.
#
# The tests are marked @pytest.mark.spm + @pytest.mark.slow and gated on
# FIXTURE availability (skip only if the .mat is missing), so `-m "spm and slow"`
# selects exactly the parity ladder + the measured-floor test.
#
# Usage:
#   sbatch cluster/sbatch/erp_parity_test.sbatch
#
# Outputs:
#   cluster/logs/erp_parity_<jobid>.{out,err}   SLURM logs (the measured max|diff|
#                                               per rung is printed via -s)
# =============================================================================

#SBATCH --job-name=erp_parity
#SBATCH --output=cluster/logs/erp_parity_%j.out
#SBATCH --error=cluster/logs/erp_parity_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# NEVER pip install inside this job. The package is found via sys.path; there is
# no install step here. Concurrent resolvers against the shared conda env race
# -> OOM + .pth corruption (project hard rule).

verify_torch
print_job_header "SPM12 CMC-ERP single-source parity gate (Phase 33-03)"

# Verified M3 MATLAB + SPM12 locations (comp partition, valid license). The
# parity assertions do NOT call MATLAB (they compare torch to the frozen .mat),
# but these are exported for parity with the Phase-32 bridge / fixture
# regeneration.
export MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab
export SPM12_PATH=/home/aman0087/fc37/Carrick/spm12

echo "MATLAB_PATH=${MATLAB_PATH}"
echo "SPM12_PATH=${SPM12_PATH}"

mkdir -p cluster/logs

python -m pytest \
    tests/test_spm_erp_dcm_validation.py \
    tests/test_local_linearization.py \
    -m "spm and slow" -v -s 2>&1
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CMC-ERP PARITY LADDER GREEN (exit 0)"
else
    echo "  CMC-ERP PARITY LADDER FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
