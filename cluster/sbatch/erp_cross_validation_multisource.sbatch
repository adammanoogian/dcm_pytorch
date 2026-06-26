#!/bin/bash
# =============================================================================
# CMC-ERP Multi-Source (5-source MMN) Fixture Generation (Phase 34-02, EVOK-05/06)
# =============================================================================
# Single task (NOT an array): writes the frozen 5-source DCM input .mat via
# validation.export_to_mat.export_erp_dcm_multisource, then runs
# run_spm_erp_dcm_multisource.m under MATLAB R2022a + SPM12 to produce the
# per-condition frozen parity fixtures
#   QA{c}   {4} of (5,5)   spm_gen_Q B->all-A folding (free log space)
#   QG{c}   (5,)           spm_gen_Q diag(B)->Q.G(:,1) precision column
#   J0{c}   (40,40)        per-condition frozen Jacobian (spm_diff FD)
#   Qupd{c} (40,40)        (spm_expm(dt*dfdx)-I)/dfdx right-division
#   y{c}    (128,40)       multi-source evoked SOURCE trajectory (spm_gen_erp loop)
# plus a provenance meta (D == 1, nargout_Mf == 2, x0 == zeros(5,8), N == 5, SPM
# $Id, dt, ns, ons, dur, the locked edge list + X) into
# validation/data/erp_multisource_fixtures.mat. Parity is vs-SPM, never vs-torch
# (pitfall V1). The delay operator is forced to EXACT identity via the 2-output
# spm_fx_cmc_nodelay.m wrapper (nargout == 2 -> spm_int_L D = 1), REUSED UNCHANGED
# from 33-02 (do NOT reintroduce the unconditional-2-output recursion bug).
#
# Runs on M3 because the LOCAL MATLAB license server is unreachable (FlexLM -15).
# MATLAB R2022a + SPM12 are VERIFIED working on the comp partition:
#   MATLAB: /usr/local/matlab/r2022a/bin/matlab  (valid license)
#   SPM12:  /home/aman0087/fc37/Carrick/spm12     (spm_int_L, spm_fx_cmc,
#                                                   spm_gen_Q, dcm_meeg)
#
# A soft fixture-shape / meta mismatch is RECORDED in the JSON + stdout
# (checks_pass=False) and the script exits 0 (record-don't-crash, mirroring
# 31-03-D3); only an unexpected MATLAB / IO failure exits non-zero.
#
# Usage:
#   sbatch cluster/sbatch/erp_cross_validation_multisource.sbatch
#
# Outputs:
#   validation/data/erp_multisource_fixtures.mat                 QA/QG/J0/Qupd/y/meta
#   cluster/results/erp_cross_validation_multisource_<jobid>.json provenance record
#   cluster/logs/erp_xval_ms_<jobid>.{out,err}                   SLURM logs
# =============================================================================

#SBATCH --job-name=erp_xval_ms
#SBATCH --output=cluster/logs/erp_xval_ms_%j.out
#SBATCH --error=cluster/logs/erp_xval_ms_%j.err
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
print_job_header "CMC-ERP Multi-Source Fixture Generation (Phase 34-02)"

# Verified M3 MATLAB + SPM12 locations (comp partition, valid license).
export MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab
export SPM12_PATH=/home/aman0087/fc37/Carrick/spm12

echo "MATLAB_PATH=${MATLAB_PATH}"
echo "SPM12_PATH=${SPM12_PATH}"

mkdir -p cluster/logs cluster/results validation/data

python cluster/scripts/erp_cross_validation.py --mode multisource
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CMC-ERP MULTI-SOURCE FIXTURE GENERATION COMPLETE (exit 0)"
else
    echo "  CMC-ERP MULTI-SOURCE FIXTURE GENERATION FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
