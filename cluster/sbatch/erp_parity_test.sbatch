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
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env

# NEVER pip install inside this job. The package is found via sys.path; there is
# no install step here. Concurrent resolvers against the shared venv race
# -> OOM + .pth corruption (project hard rule).

verify_torch
print_job_header "SPM12 CMC-ERP single-source parity gate (Phase 33-03)"

# The parity assertions do NOT call MATLAB -- they compare pure-torch output
# against the frozen .mat fixtures, so this job needs no MATLAB at all. That is
# exactly why the v0.8.0 parity ladders still run after the M3 -> DCCN move.
# Set SPM12_PATH only if you intend to REGENERATE fixtures here.

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
