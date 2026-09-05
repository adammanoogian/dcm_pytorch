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
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env

# NEVER pip install inside this job. The package is found via sys.path in the
# Python script; there is no install step here. Concurrent resolvers against
# the shared venv race -> OOM + .pth corruption (project hard rule).

verify_torch
print_job_header "VL-vs-SPM12 Multi-Seed Divergence Investigation (Phase 32-03)"

# MATLAB comes from Environment Modules on DCCN. SPM12 is NOT installed
# cluster-wide -- export SPM12_PATH to your own checkout before submitting, or
# (preferred) run the SPM bridge on the workstation, which has R2025b + SPM12.
setup_matlab R2024b

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
