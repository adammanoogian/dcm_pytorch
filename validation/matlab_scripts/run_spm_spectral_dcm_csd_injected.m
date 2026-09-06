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
        spm12_path = 'C:/Users/adaman/Documents/external/spm12';
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

% --- Inject the precomputed CSD and replicate spm_dcm_fmri_csd setup ---
% IMPORTANT: spm_dcm_fmri_csd.m calls spm_dcm_fmri_csd_data UNCONDITIONALLY
% (line ~213), which RECOMPUTES DCM.Y.csd from the BOLD timeseries (DCM.Y.y) via
% a MAR model -- silently OVERWRITING any injected CSD (our DCM.Y.y is a zeros
% placeholder, so that recompute is degenerate -> spm_nlsi_GN RCOND=NaN ->
% "Convergence failure"). To genuinely fit the injected analytic CSD we replicate
% the model setup from spm_dcm_fmri_csd.m (priors + M structure) and call
% spm_nlsi_GN directly, SKIPPING the data step. For resting-state spectral DCM
% the input is constant, so DCM.U.csd = zeros (spm_dcm_fmri_csd_data.m "else"
% branch) -- it does NOT depend on the BOLD.
% Source: spm_dcm_fmri_csd.m lines 154-243, spm_dcm_fmri_csd_data.m.
DCM.Y.Hz = DCM.Y.Hz(:);                 % column vector of frequencies
if iscell(DCM.Y.csd)
    DCM.Y.csd = DCM.Y.csd{1};
end
DCM.Y.csd = squeeze(DCM.Y.csd);         % numeric (Nf, n, n), single session
fprintf('Injected DCM.Y.csd (%d freqs); SPM CSD recompute SKIPPED.\n', ...
    numel(DCM.Y.Hz));

n = double(DCM.n);

% Spectral toolbox (spm_csd_fmri_mtf -> spm_mar2csd dependencies)
if ~isdeployed
    addpath(fullfile(spm('Dir'), 'toolbox', 'spectral'));
end

try
    % --- priors (and initial states): spm_dcm_fmri_csd.m line 154 ---
    [pE, pC, x] = spm_dcm_fmri_priors(DCM.a, DCM.b, DCM.c, DCM.d, DCM.options);

    % --- SPM12 hyperpriors (match the VL engine: hE = 8, hC = 1/128) ---
    if ~isfield(DCM, 'M')
        DCM.M = struct();
    end
    hE = 8;
    hC = 1/128;
    if isfield(DCM.M, 'hE'), hE = DCM.M.hE; end
    if isfield(DCM.M, 'hC'), hC = DCM.M.hC; end

    % --- model structure: spm_dcm_fmri_csd.m lines 188-211 ---
    DCM.M.IS = 'spm_csd_fmri_mtf';
    DCM.M.g  = @spm_gx_fmri;
    DCM.M.f  = @spm_fx_fmri;
    DCM.M.x  = x;
    DCM.M.pE = pE;
    DCM.M.pC = pC;
    DCM.M.hE = hE;
    DCM.M.hC = hC;
    DCM.M.n  = length(spm_vec(x));
    DCM.M.m  = size(DCM.U.u, 2);
    DCM.M.l  = n;
    DCM.M.p  = DCM.options.order;
    DCM.M.u  = sparse(n, 1);

    % --- spectral scaffolding normally set by spm_dcm_fmri_csd_data ---
    Nc        = size(DCM.U.u, 2);
    DCM.U.csd = zeros(numel(DCM.Y.Hz), Nc, Nc);  % constant input -> zero
    DCM.Y.p   = DCM.M.p;
    DCM.Y.pst = (1:double(DCM.v)) * DCM.Y.dt;
    DCM.M.Hz  = DCM.Y.Hz;
    DCM.M.dt  = 1/2;
    DCM.M.N   = 32;
    DCM.M.ns  = 1/DCM.Y.dt;
    % Observation precision. spm_dcm_fmri_csd.m:234-235 sets these; this
    % script bypasses spm_dcm_fmri_csd_data and so must set them itself.
    % Omitting Y.Q makes spm_nlsi_GN fall back to spm_Ce(ns*ones(1,nr))
    % (spm_nlsi_GN.m:218) -- a generic per-channel basis with N^2 = 4
    % components -- while the Python VL engine uses the spm_dcm_csd_Q port
    % with ONE component. The two engines then fit DIFFERENT noise models and
    % their free energies are not comparable, which is the origin of the
    % long-standing ~269.9-nat F offset.
    DCM.Y.Q   = spm_dcm_csd_Q(DCM.Y.csd);
    DCM.Y.X0  = sparse(size(DCM.Y.Q, 1), 0);

    % --- Variational Laplace inversion on the INJECTED CSD ---
    fprintf('Running spm_nlsi_GN on injected CSD...\n');
    Y.y = DCM.Y.csd;
    Y.Q = DCM.Y.Q;

    % --- free-energy provenance (VL-vs-SPM offset investigation) ---
    % Y.Q is now set from spm_dcm_csd_Q (matching spm_dcm_fmri_csd.m).
    % Record what spm_nlsi_GN sees: F's L(1) term is
    % logdet(iS)*nq/2 - e'*iS*e/2 - ny*log(2*pi)/2, so BOTH ny and nq must
    % match the Python side for the two free energies to be comparable.
    fe.ny = length(spm_vec(Y.y));
    % Component COUNT, not element count: Q may be a cell array of bases
    % or a single numeric matrix (which is one component).
    if ~isfield(Y, 'Q')
        fe.nq = 0;
    elseif iscell(Y.Q)
        fe.nq = numel(Y.Q);
    else
        fe.nq = 1;
    end
    fe.y_size = size(Y.y);
    fe.y_is_complex = double(~isreal(Y.y));
    fe.n_X0_cols = size(DCM.Y.X0, 2);
    fe.hE = full(DCM.M.hE); fe.hC = full(DCM.M.hC);
    fprintf('FE-PROVENANCE ny=%d nq=%d complex=%d X0cols=%d\n', ...
            fe.ny, fe.nq, fe.y_is_complex, fe.n_X0_cols);

    [Ep, Cp, Eh, F] = spm_nlsi_GN(DCM.M, DCM.U, Y);
    fe.Eh = full(Eh(:))';
    fprintf('FE-PROVENANCE Eh=%s F=%.6f\n', mat2str(fe.Eh, 8), F);
    DCM.Ep = Ep;
    DCM.Cp = Cp;
    DCM.Eh = Eh;
    DCM.F  = F;
    DCM.Hz = DCM.Y.Hz;
    fprintf('Estimation complete. Free energy F = %.4f\n', DCM.F);
catch e
    fprintf('ERROR: injected-CSD spectral DCM failed: %s\n', e.message);
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
    results.fe = fe;                      % free-energy provenance

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
