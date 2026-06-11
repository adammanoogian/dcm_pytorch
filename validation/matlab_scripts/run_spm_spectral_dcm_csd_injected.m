% run_spm_spectral_dcm_csd_injected.m -- SPM12 spectral DCM, SAME-CSD (injected) variant
%
% Usage: matlab -batch "run_spm_spectral_dcm_csd_injected"
%   or:  matlab -batch "run('validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m')"
%
% Reads:  validation/data/spectral_dcm_csd_input.mat  (DCM struct WITH DCM.Y.csd + DCM.Y.Hz)
% Writes: validation/data/spectral_dcm_csd_spm_results.mat (results struct)
%
% Phase 32 strict-5%-F cross-validation: this is the SAME-CSD (injected) variant.
% The ONE behavioral difference from run_spm_spectral_dcm.m is that instead of
% letting spm_dcm_fmri_csd recompute the CSD from BOLD via its internal MAR model
% (spm_dcm_fmri_csd_data), we INJECT the precomputed (F, N, N) complex CSD that the
% Pyro-DCM VL engine fits. SPM then estimates on THAT identical data, so the
% Variational-Laplace free energy (VL F) and SPM free energy (SPM F) are evaluable
% on the IDENTICAL cross-spectral density -- the precondition for a meaningful
% strict 5%-matched-nat free-energy gate. When DCM.Y.csd is already populated,
% spm_dcm_fmri_csd skips the internal CSD estimation and uses the supplied CSD.
% Source: Verified against spm_dcm_fmri_csd.m (local SPM12 install).
%
% Pitfall S4: DCM.Y.csd(w, i, j) (1-based) equals the Python C-order array
% observed_csd[w-1, i-1, j-1] written by export_spectral_dcm_csd_for_spm. NO
% transpose is applied on either side.

fprintf('=== SPM12 Spectral DCM Estimation (SAME-CSD injected) ===\n');
fprintf('Start time: %s\n', datestr(now));

% --- Setup paths ---
try
    spm12_path = getenv('SPM12_PATH');
    if isempty(spm12_path)
        spm12_path = 'C:/Users/aman0087/Documents/Github/spm12';
    end
    addpath(spm12_path);
    if ~exist('spm', 'file')
        error('SPM12 not found on path. Check addpath above.');
    end
    spm('defaults', 'FMRI');
    fprintf('SPM12 loaded successfully.\n');
catch e
    fprintf('ERROR: Failed to initialize SPM12: %s\n', e.message);
    return;
end

% --- Input/output paths ---
input_path = getenv('DCM_INPUT_PATH');
if isempty(input_path)
    input_path = 'validation/data/spectral_dcm_csd_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/spectral_dcm_csd_spm_results.mat';
end

fprintf('Input:  %s\n', input_path);
fprintf('Output: %s\n', output_path);

% --- Load DCM struct ---
try
    load(input_path, 'DCM');
    fprintf('DCM loaded: %d regions, %d scans\n', DCM.n, DCM.v);
catch e
    fprintf('ERROR: Failed to load DCM from %s: %s\n', input_path, e.message);
    return;
end

% --- Force CSD analysis mode ---
DCM.options.induced = 1;
DCM.options.analysis = 'CSD';
fprintf('Set options: induced=1, analysis=CSD\n');

% --- Verify the injected CSD fields are present (loud failure if absent) ---
if ~isfield(DCM.Y, 'csd')
    fprintf('ERROR: DCM.Y.csd is missing -- this script requires an injected CSD.\n');
    return;
end
if ~isfield(DCM.Y, 'Hz')
    fprintf('ERROR: DCM.Y.Hz is missing -- this script requires an injected frequency grid.\n');
    return;
end

% --- Inject the precomputed CSD (bypass MAR recompute) ---
% spm_dcm_fmri_csd expects DCM.Y.csd as a cell array {csd} of one (Nf, n, n)
% block. When it is already populated, spm_dcm_fmri_csd_data is skipped and the
% supplied CSD is used directly. The .mat written by savemat arrives as a bare
% numeric (Nf, n, n) array, so wrap it in a cell only if it is not already one.
DCM.Y.Hz = DCM.Y.Hz(:);            % column vector of frequencies
if ~iscell(DCM.Y.csd)
    DCM.Y.csd = {squeeze(DCM.Y.csd)};
end
fprintf('Injected DCM.Y.csd (%d freqs) + DCM.Y.Hz; SPM CSD recompute bypassed.\n', ...
    numel(DCM.Y.Hz));

% --- Ensure required fields ---
if ~isfield(DCM.Y, 'Q')
    DCM.Y.Q = spm_Ce(ones(1, DCM.n) * DCM.v);
    fprintf('Added DCM.Y.Q (error precision components).\n');
end

% --- Run spectral DCM estimation on the INJECTED CSD ---
try
    fprintf('Running spm_dcm_fmri_csd on injected CSD...\n');
    DCM = spm_dcm_fmri_csd(DCM);
    fprintf('Estimation complete. Free energy F = %.4f\n', DCM.F);
catch e
    fprintf('ERROR: spm_dcm_fmri_csd failed: %s\n', e.message);
    return;
end

% --- S4 asymmetry readout (MATLAB-side) ---
% A(1,2) vs A(2,1) makes the no-transpose contract visible on the SPM side.
try
    fprintf('Ep.A(1,2)=%g Ep.A(2,1)=%g\n', DCM.Ep.A(1, 2), DCM.Ep.A(2, 1));
catch
    fprintf('(Ep.A asymmetry readout unavailable: < 2 regions)\n');
end

% --- Save results (identical block to run_spm_spectral_dcm.m) ---
try
    results.Ep_A = DCM.Ep.A;              % Posterior mean A (free params)
    results.Cp = full(DCM.Cp);            % Full posterior covariance
    results.F = DCM.F;                    % Free energy

    % Spectral-specific outputs
    if isfield(DCM.Ep, 'transit')
        results.Ep_transit = DCM.Ep.transit;
    end
    if isfield(DCM.Ep, 'decay')
        results.Ep_decay = DCM.Ep.decay;
    end
    if isfield(DCM, 'Hc')
        results.Hc = DCM.Hc;              % Predicted CSD
    end
    if isfield(DCM, 'Hz')
        results.Hz = DCM.Hz;              % Frequency vector
    end

    save(output_path, 'results');
    fprintf('Results saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save results: %s\n', e.message);
    return;
end

fprintf('=== Spectral DCM (SAME-CSD injected) estimation complete ===\n');
fprintf('End time: %s\n', datestr(now));
