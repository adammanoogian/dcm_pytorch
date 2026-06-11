#!/bin/bash
# =============================================================================
# VL-vs-SPM12 Cross-Validation (Phase 32-03, VLSPM-03)
# =============================================================================
# Single task (NOT an array): fits the Phase 28 Variational-Laplace engine on a
# prior-matched reciprocal-asymmetric N=2 spectral DCM problem, injects the
# IDENTICAL Python-computed CSD into SPM12 (Plan 32-01 bridge), runs
# spm_dcm_fmri_csd (spm_nlsi_GN), and compares the two engines on Ep
# (free-space 10%, S1/S2), matched free energy (strict 5% on the identical CSD,
# the headline number), and relative cross-model ranking (>=0.80, S3-safe).
#
# Runs on M3 because the LOCAL MATLAB license server is unreachable (FlexLM
# -15). MATLAB R2022a + SPM12 are VERIFIED working on the comp partition:
#   MATLAB: /usr/local/matlab/r2022a/bin/matlab  (valid license)
#   SPM12:  /home/aman0087/fc37/Carrick/spm12     (spm_nlsi_GN, spm_dcm_fmri_csd)
# The Python orchestrator resolves the MATLAB binary from MATLAB_PATH and passes
# SPM12_PATH to the MATLAB child; the injection .m falls back to its local
# default when SPM12_PATH is unset.
#
# A gate miss (incl. a real 5% matched-F miss) is RECORDED in the JSON + stdout
# and the script exits 0 (record-don't-crash, mirroring 31-03-D3); only an
# unexpected exception exits non-zero.
#
# Usage:
#   sbatch cluster/sbatch/spm_xvalms_multiseed.sbatch
#
# Outputs:
#   cluster/results/spm_xvalms_multiseed_<jobid>.json   Ep/matched-F/ranking
#   cluster/logs/spm_xvalms_<jobid>.{out,err}             SLURM logs
# =============================================================================

#SBATCH --job-name=spm_xvalms
#SBATCH --output=cluster/logs/spm_xvalms_%j.out
#SBATCH --error=cluster/logs/spm_xvalms_%j.err
#SBATCH --time=01:30:00
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
print_job_header "VL-vs-SPM12 Multi-Seed Divergence Investigation (Phase 32-03)"

# Verified M3 MATLAB + SPM12 locations (comp partition, valid license).
export MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab
export SPM12_PATH=/home/aman0087/fc37/Carrick/spm12

echo "MATLAB_PATH=${MATLAB_PATH}"
echo "SPM12_PATH=${SPM12_PATH}"

mkdir -p cluster/logs cluster/results

python cluster/scripts/spm_xval_multiseed.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  VL-VS-SPM12 CROSS-VALIDATION COMPLETE (exit 0)"
else
    echo "  VL-VS-SPM12 CROSS-VALIDATION FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
