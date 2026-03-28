% run_spm_task_dcm.m -- SPM12 task DCM estimation batch script
%
% Usage: matlab -batch "run_spm_task_dcm"
%   or:  matlab -batch "run('validation/matlab_scripts/run_spm_task_dcm.m')"
%
% Reads:  validation/data/task_dcm_input.mat  (DCM struct)
% Writes: validation/data/task_dcm_spm_results.mat (results struct)
%
% Source: Verified against spm_dcm_estimate.m (local SPM12 install).

fprintf('=== SPM12 Task DCM Estimation ===\n');
fprintf('Start time: %s\n', datestr(now));

% --- Setup paths ---
try
    addpath('C:/Users/aman0087/Documents/Github/spm12');
    if ~exist('spm', 'file')
        error('SPM12 not found on path. Check addpath above.');
    end
    spm('defaults', 'FMRI');
    fprintf('SPM12 loaded successfully.\n');
catch e
    fprintf('ERROR: Failed to initialize SPM12: %s\n', e.message);
    return;
end

% --- Input/output paths (can override via environment variables) ---
input_path = getenv('DCM_INPUT_PATH');
if isempty(input_path)
    input_path = 'validation/data/task_dcm_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/task_dcm_spm_results.mat';
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

% --- Ensure required fields ---
if ~isfield(DCM.Y, 'Q')
    DCM.Y.Q = spm_Ce(ones(1, DCM.n) * DCM.v);
    fprintf('Added DCM.Y.Q (error precision components).\n');
end

% --- Run estimation (Variational Laplace) ---
try
    fprintf('Running spm_dcm_estimate...\n');
    DCM = spm_dcm_estimate(DCM);
    fprintf('Estimation complete. Free energy F = %.4f\n', DCM.F);
catch e
    fprintf('ERROR: spm_dcm_estimate failed: %s\n', e.message);
    return;
end

% --- Save results ---
try
    results.Ep_A = DCM.Ep.A;           % Posterior mean A (free params)
    results.Ep_C = DCM.Ep.C;           % Posterior mean C
    results.Cp = full(DCM.Cp);         % Full posterior covariance
    results.F = DCM.F;                 % Free energy
    results.y_predicted = DCM.y;       % Predicted BOLD
    results.R = DCM.R;                 % Residuals

    save(output_path, 'results');
    fprintf('Results saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save results: %s\n', e.message);
    return;
end

fprintf('=== Task DCM estimation complete ===\n');
fprintf('End time: %s\n', datestr(now));
