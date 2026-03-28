% run_tapas_rdcm.m -- tapas rDCM estimation batch script
%
% Usage: matlab -batch "run_tapas_rdcm"
%   or:  matlab -batch "run('validation/matlab_scripts/run_tapas_rdcm.m')"
%
% Reads:  validation/data/rdcm_input.mat  (DCM struct)
% Writes: validation/data/rdcm_tapas_results.mat (results struct)
%
% Runs both rigid (methods=1) and sparse (methods=2) rDCM estimation.
% Source: tapas GitHub (translationalneuromodeling/tapas/rDCM).

fprintf('=== tapas rDCM Estimation ===\n');
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

% --- Add tapas rDCM to path ---
tapas_path = 'C:/Users/aman0087/Documents/Github/tapas/rDCM';
if exist(tapas_path, 'dir')
    addpath(genpath(tapas_path));
    fprintf('tapas rDCM path added: %s\n', tapas_path);
else
    fprintf('WARNING: tapas rDCM not found at %s\n', tapas_path);
    fprintf('Please clone: git clone https://github.com/translationalneuromodeling/tapas\n');
    fprintf('Skipping rDCM estimation.\n');
    return;
end

if ~exist('tapas_rdcm_estimate', 'file')
    fprintf('WARNING: tapas_rdcm_estimate not found on path.\n');
    fprintf('Ensure tapas/rDCM/code is on MATLAB path.\n');
    return;
end
fprintf('tapas rDCM functions found.\n');

% --- Input/output paths ---
input_path = getenv('DCM_INPUT_PATH');
if isempty(input_path)
    input_path = 'validation/data/rdcm_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/rdcm_tapas_results.mat';
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

% --- Set estimation options ---
options = [];
options.filter_str = 0;        % No temporal filtering for synthetic data
options.restrictInputs = 0;    % Don't restrict inputs
options.iter = 100;            % Permutation iterations for sparse rDCM

% --- Run rigid rDCM (methods=1) ---
try
    fprintf('Running rigid rDCM (methods=1)...\n');
    [output_rigid, ~] = tapas_rdcm_estimate(DCM, 's', options, 1);
    fprintf('Rigid rDCM complete. logF = %.4f\n', output_rigid.logF);
catch e
    fprintf('ERROR: Rigid rDCM failed: %s\n', e.message);
    output_rigid = [];
end

% --- Run sparse rDCM (methods=2) ---
try
    fprintf('Running sparse rDCM (methods=2)...\n');
    [output_sparse, ~] = tapas_rdcm_estimate(DCM, 's', options, 2);
    fprintf('Sparse rDCM complete. logF = %.4f\n', output_sparse.logF);
catch e
    fprintf('ERROR: Sparse rDCM failed: %s\n', e.message);
    output_sparse = [];
end

% --- Save results ---
try
    results = struct();

    if ~isempty(output_rigid)
        results.rigid.Ep = output_rigid.Ep;
        results.rigid.logF = output_rigid.logF;
    end

    if ~isempty(output_sparse)
        results.sparse.Ep = output_sparse.Ep;
        results.sparse.logF = output_sparse.logF;
        if isfield(output_sparse, 'Ip')
            results.sparse.Ip = output_sparse.Ip;
        end
    end

    save(output_path, 'results');
    fprintf('Results saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save results: %s\n', e.message);
    return;
end

fprintf('=== tapas rDCM estimation complete ===\n');
fprintf('End time: %s\n', datestr(now));
