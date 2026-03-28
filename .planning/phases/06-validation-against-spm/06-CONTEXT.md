# Phase 6: Validation Against SPM / Reference Implementations - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Cross-validate all three DCM variants against established reference implementations. Generate identical synthetic datasets, run both our code and the reference implementation, compare posterior means element-wise. Verify ELBO/free energy model comparison ranking matches SPM free energy ranking. This produces validation scripts and a validation report, not new library modules.

</domain>

<decisions>
## Implementation Decisions

### MATLAB + SPM12 Available
- MATLAB with SPM12 is installed and accessible on this machine
- Call MATLAB via subprocess: `matlab -batch "script_name"` from Python
- Generate .mat reference files from MATLAB, load in Python via scipy.io.loadmat for comparison
- Validation scripts live in `validation/` directory

### Full SPM Cross-Validation
- Generate identical simulated datasets in Python
- Export to .mat format for SPM12 processing
- Run SPM12 DCM estimation via MATLAB batch scripts
- Load SPM results back into Python
- Compare element-wise: posterior means within 10% relative error (roadmap criterion)
- Compare model ranking: ELBO ranking matches SPM free energy ranking

### rDCM: Both Julia and MATLAB References
- Cross-validate rDCM against BOTH:
  - Julia RegressionDynamicCausalModeling.jl (already analyzed in Phase 3)
  - tapas/rDCM MATLAB toolbox
- Call Julia via subprocess or pre-generate reference data
- Call tapas via MATLAB batch alongside SPM12

### Claude's Discretion
- MATLAB script structure and SPM12 API calls
- Julia calling convention (subprocess vs pre-generated .jld2 files)
- Number and complexity of validation scenarios
- How to handle expected small discrepancies (VL vs SVI, different optimization)
- Validation report format and structure
- Whether to include figure generation (matplotlib plots)

</decisions>

<specifics>
## Specific Ideas

- Task DCM: compare against SPM12 spm_dcm_estimate on simulated BOLD
- Spectral DCM: compare against SPM12 spm_dcm_csd on simulated CSD
- rDCM: compare against tapas rDCM AND Julia rDCM on simulated frequency-domain data
- Roadmap thresholds: posterior means within 10% relative error
- Model comparison: ELBO ranking matches SPM free energy ranking on 3+ scenarios
- Discrepancies should be documented with root-cause analysis (VL vs SVI, etc.)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-validation-against-spm*
*Context gathered: 2026-03-28*
