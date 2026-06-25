#!/bin/bash
# =============================================================================
# CMC-ERP Single-Source Fixture Generation (Phase 33-02, CMC-06)
# =============================================================================
# Single task (NOT an array): writes the frozen single-source DCM input .mat via
# validation.export_to_mat.export_erp_dcm, then runs run_spm_erp_dcm.m under
# MATLAB R2022a + SPM12 to produce the 5 frozen parity fixtures
# (f_field, J0, dtJ, Eexp, Q_update, y_states) + a provenance meta header
# (D == 1, x0 == 0, SPM $Id, dt, ns, ons, dur, x_test, u_test) into
# validation/data/erp_single_source_fixtures.mat. Parity is vs-SPM, never
# vs-torch (pitfall V1); the delay operator is forced to EXACT identity via the
# 2-output spm_fx_cmc_nodelay.m wrapper (nargout == 2 -> spm_int_L D = 1).
#
# Runs on M3 because the LOCAL MATLAB license server is unreachable (FlexLM
# -15). MATLAB R2022a + SPM12 are VERIFIED working on the comp partition:
#   MATLAB: /usr/local/matlab/r2022a/bin/matlab  (valid license)
#   SPM12:  /home/aman0087/fc37/Carrick/spm12     (spm_int_L, spm_fx_cmc, dcm_meeg)
# The Python orchestrator resolves the MATLAB binary from MATLAB_PATH and passes
# SPM12_PATH to the MATLAB child; run_spm_erp_dcm.m falls back to its local
# default when SPM12_PATH is unset.
#
# A soft fixture-shape / meta mismatch is RECORDED in the JSON + stdout
# (checks_pass=False) and the script exits 0 (record-don't-crash, mirroring
# 31-03-D3); only an unexpected MATLAB / IO failure exits non-zero.
#
# Usage:
#   sbatch cluster/sbatch/erp_cross_validation.sbatch
#
# Outputs:
#   validation/data/erp_single_source_fixtures.mat        the 5 arrays + meta
#   cluster/results/erp_cross_validation_<jobid>.json     provenance record
#   cluster/logs/erp_xval_<jobid>.{out,err}               SLURM logs
# =============================================================================

#SBATCH --job-name=erp_xval
#SBATCH --output=cluster/logs/erp_xval_%j.out
#SBATCH --error=cluster/logs/erp_xval_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# NEVER pip install inside this job. The package is found via sys.path in the
# Python script; there is no install step here. Concurrent resolvers against
# the shared conda env race -> OOM + .pth corruption (project hard rule).

verify_torch
print_job_header "CMC-ERP Fixture Generation (Phase 33-02)"

# Verified M3 MATLAB + SPM12 locations (comp partition, valid license).
export MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab
export SPM12_PATH=/home/aman0087/fc37/Carrick/spm12

echo "MATLAB_PATH=${MATLAB_PATH}"
echo "SPM12_PATH=${SPM12_PATH}"

mkdir -p cluster/logs cluster/results validation/data

python cluster/scripts/erp_cross_validation.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CMC-ERP FIXTURE GENERATION COMPLETE (exit 0)"
else
    echo "  CMC-ERP FIXTURE GENERATION FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
