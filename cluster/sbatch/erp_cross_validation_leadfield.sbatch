#!/bin/bash
# =============================================================================
# CMC-ERP LFP Lead-Field + Scalp-ERP Fixture Generation (Phase 35-02, LEAD-05)
# =============================================================================
# Single task (NOT an array): writes the frozen LFP lead-field DCM input .mat via
# validation.export_to_mat.export_erp_dcm_leadfield, then runs
# run_spm_erp_dcm_leadfield.m under MATLAB R2022a + SPM12 to produce the frozen
# scalp-parity fixtures
#   L_full      (Nc,8N)=(5,40)   kron(P.J, L_spatial), LFP identity  spm_lx_erp.m:33
#   y_scalp{c}  (ns,Nc)=(128,5)  per-condition scalp ERP ysrc*L_full'
#   diff_wave   (ns,Nc)=(128,5)  deviant - standard difference wave
# plus a provenance meta (D == 1, nargout_Mf == 2, x0 == zeros(5,8), N == 5,
# Nc == 5, dipfit.type == 'LFP', P.J one-hot at index 2, SPM $Id for spm_lx_erp /
# spm_erp_L / spm_L_priors / spm_gen_Q / spm_int_L, dt, ns, ons, dur, the locked
# edge list + X) into validation/data/erp_leadfield_fixtures.mat. Parity is
# vs-SPM, never vs-torch (pitfall V1). LFP mode is head-model-free (no MNI coords,
# no spm_cond_units -- ECD deferred to Phase 36). The delay operator is forced to
# EXACT identity via the 2-output spm_fx_cmc_nodelay.m wrapper (nargout == 2 ->
# spm_int_L D = 1), REUSED UNCHANGED from 33-02.
#
# MATLAB/SPM12 note (updated 2026-09-05, Monash M3 -> Radboud DCCN):
# This job originally ran on M3 because the old laptop's MATLAB licence server
# was unreachable (FlexLM -15). That is no longer true -- the DCCN workstation
# has MATLAB R2025b with a valid licence and a complete SPM12, so the SPM
# bridge is best run LOCALLY. DCCN compute nodes have MATLAB via Environment
# Modules but NO system SPM12, so any cluster run must export SPM12_PATH to a
# personal checkout first (setup_matlab enforces this and fails loudly).
#
# A soft fixture-shape / meta mismatch is RECORDED in the JSON + stdout
# (checks_pass=False) and the script exits 0 (record-don't-crash, mirroring
# 31-03-D3); only an unexpected MATLAB / IO failure exits non-zero.
#
# Usage:
#   sbatch cluster/sbatch/erp_cross_validation_leadfield.sbatch
#
# Outputs:
#   validation/data/erp_leadfield_fixtures.mat                   L_full/y_scalp/diff_wave/meta
#   cluster/results/erp_cross_validation_leadfield_<jobid>.json  provenance record
#   cluster/logs/erp_xval_lf_<jobid>.{out,err}                   SLURM logs
# =============================================================================

#SBATCH --job-name=erp_xval_lf
#SBATCH --output=cluster/logs/erp_xval_lf_%j.out
#SBATCH --error=cluster/logs/erp_xval_lf_%j.err
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
print_job_header "CMC-ERP LFP Lead-Field Fixture Generation (Phase 35-02)"

# MATLAB comes from Environment Modules on DCCN. SPM12 is NOT installed
# cluster-wide -- export SPM12_PATH to your own checkout before submitting, or
# (preferred) run the SPM bridge on the workstation, which has R2025b + SPM12.
setup_matlab R2024b

mkdir -p cluster/logs cluster/results validation/data

python cluster/scripts/erp_cross_validation.py --mode leadfield
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CMC-ERP LFP LEAD-FIELD FIXTURE GENERATION COMPLETE (exit 0)"
else
    echo "  CMC-ERP LFP LEAD-FIELD FIXTURE GENERATION FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
