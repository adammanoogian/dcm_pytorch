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
# MATLAB/SPM12 note (updated 2026-09-05, Monash M3 -> Radboud DCCN):
# This job originally ran on M3 because the old laptop's MATLAB licence server
# was unreachable (FlexLM -15). That is no longer true -- the DCCN workstation
# has MATLAB R2025b with a valid licence and a complete SPM12, so the SPM
# bridge is best run LOCALLY. DCCN compute nodes have MATLAB via Environment
# Modules but NO system SPM12, so any cluster run must export SPM12_PATH to a
# personal checkout first (setup_matlab enforces this and fails loudly).
# The Python orchestrator resolves the MATLAB binary from config.MATLAB_PATH and
# passes config.SPM12_PATH to the MATLAB child.
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
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env

# NEVER pip install inside this job. The package is found via sys.path in the
# Python script; there is no install step here. Concurrent resolvers against
# the shared venv race -> OOM + .pth corruption (project hard rule).

verify_torch
print_job_header "CMC-ERP Fixture Generation (Phase 33-02)"

# MATLAB comes from Environment Modules on DCCN. SPM12 is NOT installed
# cluster-wide -- export SPM12_PATH to your own checkout before submitting, or
# (preferred) run the SPM bridge on the workstation, which has R2025b + SPM12.
setup_matlab R2024b

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
