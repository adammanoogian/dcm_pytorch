% run_spm_erp_dcm.m -- SPM12 single-source CMC-ERP frozen-fixture generator
%
% Usage: matlab -batch "run_spm_erp_dcm"
%   or:  matlab -batch "run('validation/matlab_scripts/run_spm_erp_dcm.m')"
%
% Reads:  validation/data/erp_single_source_input.mat   (DCM struct from
%         validation.export_to_mat.export_erp_dcm)
% Writes: validation/data/erp_single_source_fixtures.mat (5 frozen arrays + meta)
%
% Phase 33-02 single-source parity fixtures. Generates the byte-frozen arrays the
% Wave-3 parity ladder asserts the pure-torch CMC forward / spm_int_L integrator
% against (parity is vs-SPM, never vs-torch -- pitfall V1):
%   f_field  (8,)   cmc_f at a FROZEN nonzero (x_test,u_test) -- isolates every
%                   transform/sigmoid/permutation BEFORE the integrator
%   J0       (8,8)  frozen Jacobian df/dx at x0 = 0, u0 = 0 (M1 fixed point)
%   dtJ      (8,8)  dt * (J0 - I*exp(-16)) -- exported for the matrix_exp MEASUREMENT
%   Eexp     (8,8)  spm_expm(dtJ)          -- exported for the matrix_exp MEASUREMENT
%   Q_update (8,8)  (Eexp - I) * inv(dfdx) right-division (spm_int_L.m:126-127)
%   y_states (ns,8) full spm_int_L trajectory, D = I, known Gaussian input
%
% The delay operator is forced to EXACT identity (D = 1) by setting
% M.f = 'spm_fx_cmc_nodelay' (a 2-output wrapper): nargout(M.f) == 2 sends
% spm_int_L down its :117 branch and keeps D = 1 (spm_int_L.m:112). x0 == 0 is
% the asserted CMC fixed point (M1). Both facts are recorded in meta.
% Source: spm_int_L.m:112-169, spm_fx_cmc.m:206-226, spm_dcm_delay.m:60-82.

fprintf('=== SPM12 CMC-ERP single-source fixture generation ===\n');
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
    spm('defaults', 'EEG');
    % CMC dynamics + delay live in the M/EEG DCM toolbox.
    if ~isdeployed
        addpath(fullfile(spm('Dir'), 'toolbox', 'dcm_meeg'));
        addpath(fullfile(spm('Dir'), 'toolbox', 'Neural_Models'));
    end
    if ~exist('spm_fx_cmc', 'file')
        error('spm_fx_cmc not found -- check toolbox/dcm_meeg on path.');
    end
    fprintf('SPM12 loaded successfully (EEG defaults).\n');
catch e
    fprintf('ERROR: Failed to initialize SPM12: %s\n', e.message);
    return;
end

% --- Input/output paths ---
input_path = getenv('DCM_INPUT_PATH');
if isempty(input_path)
    input_path = 'validation/data/erp_single_source_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/erp_single_source_fixtures.mat';
end

fprintf('Input:  %s\n', input_path);
fprintf('Output: %s\n', output_path);

% --- Load DCM struct ---
try
    load(input_path, 'DCM');
    P = DCM.P;
    M = DCM.M;
    fprintf('DCM loaded: input grid %dx%d (ns=%d time bins, %d input(s))\n', ...
        size(DCM.U.u, 1), size(DCM.U.u, 2), ...
        size(DCM.U.u, 1), size(DCM.U.u, 2));
catch e
    fprintf('ERROR: Failed to load DCM from %s: %s\n', input_path, e.message);
    return;
end

% --- Force D = I via the 2-output wrapper (Fact 4) + the fixed point (M1) ---
try
    M.f = 'spm_fx_cmc_nodelay';          % FORCE D = I (nargout == 2)
    M.x = zeros(1, 8);
    M.n = 8;
    M.m = size(DCM.U.u, 2);
    assert(isequal(M.x, zeros(1, 8)), 'x0 must be the zeros(1,8) fixed point (M1)');
    nout = nargout(M.f);
    assert(nout == 2, 'M.f must expose exactly 2 outputs to keep D = I; got %d', nout);
    fprintf('M.f = %s (nargout = %d -> D = 1); x0 == zeros(1,8) asserted.\n', ...
        M.f, nout);
catch e
    fprintf('ERROR: D=I / x0 setup failed: %s\n', e.message);
    return;
end

% --- Generate the 5 frozen fixture arrays (staged ladder, V5) ---
try
    x0 = spm_vec(M.x);                   % column-major flat state (8,1)
    u0 = sparse(M.m, 1);                 % Jacobian taken at u = 0 (Fact 5)

    % Frozen f-field evaluation point (same point torch evaluates -- V4).
    x_test = DCM.meta.x_test;            % (1,8)
    u_test = DCM.meta.u_test;            % scalar (peak Gaussian, P.R = 0)

    % (1) f-field at the FROZEN nonzero (x_test,u_test): isolates every
    %     transform / sigmoid / J_PERM permutation before the integrator.
    f_field = spm_vec(spm_fx_cmc(x_test, u_test, P, M));   % (8,1)

    % (2) frozen Jacobian df/dx at x0 = 0, u0 = 0.
    J0 = full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, P, M, 1)));   % (8,8)

    % (3) update operator replicating spm_int_L.m:126-127 (regulariser BEFORE
    %     both E and Q; right-division Q = (E - I) * inv(dfdx)).
    dfdx     = J0 - eye(8) * exp(-16);
    dtJ      = DCM.U.dt * dfdx;          % exported for the matrix_exp MEASUREMENT
    Eexp     = spm_expm(dtJ);            % exported for the matrix_exp MEASUREMENT
    Q_update = (Eexp - eye(8)) / dfdx;   % right-division (mldivide/mrdivide)

    % (4) full trajectory via SPM's own integrator with D = I (nargout(M.f)==2).
    y_states = spm_int_L(P, M, DCM.U);   % (ns,8)

    fprintf('Fixtures: f_field %s, J0 %s, dtJ %s, Eexp %s, Q_update %s, y_states %s\n', ...
        mat2str(size(f_field)), mat2str(size(J0)), mat2str(size(dtJ)), ...
        mat2str(size(Eexp)), mat2str(size(Q_update)), mat2str(size(y_states)));
catch e
    fprintf('ERROR: fixture generation failed: %s\n', e.message);
    return;
end

% --- Provenance metadata header (pitfall V4) ---
try
    meta = struct();
    meta.spm_ver = spm('Ver');
    % Capture the $Id headers of the load-bearing source files.
    src_files = {'spm_int_L', 'spm_fx_cmc', 'spm_cmc_priors', 'spm_erp_u'};
    for k = 1:numel(src_files)
        idstr = '';
        try
            fp = which(src_files{k});
            fid = fopen(fp, 'r');
            if fid > 0
                txt = fread(fid, 4000, '*char')';
                fclose(fid);
                tok = regexp(txt, '\$Id:[^$]*\$', 'match', 'once');
                if ~isempty(tok), idstr = tok; end
            end
        catch
            idstr = '';
        end
        meta.(['id_' src_files{k}]) = idstr;
    end
    meta.D       = 1;                     % delay operator forced to identity
    meta.nargout_Mf = nargout(M.f);       % == 2 -> the D=1 routing proof
    meta.dt      = DCM.U.dt;
    meta.ns      = size(DCM.U.u, 1);
    meta.ons     = M.ons;
    meta.dur     = M.dur;
    meta.sus     = M.sus;
    meta.x_test  = x_test;
    meta.u_test  = u_test;
    meta.x0      = x0;                    % the asserted zeros fixed point
    meta.u_grid  = DCM.U.u;               % the EXACT Gaussian drive integrated
    meta.exp_shift = exp(-16);            % the spm_int_L regulariser
    fprintf('meta: spm_ver=%s D=%d ns=%d dt=%g ons=%g dur=%g\n', ...
        meta.spm_ver, meta.D, meta.ns, meta.dt, meta.ons, meta.dur);
catch e
    fprintf('ERROR: meta assembly failed: %s\n', e.message);
    return;
end

% --- Save the frozen fixtures ---
try
    save(output_path, 'f_field', 'J0', 'dtJ', 'Eexp', 'Q_update', ...
        'y_states', 'meta');
    fprintf('Fixtures saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save fixtures: %s\n', e.message);
    return;
end

fprintf('=== CMC-ERP fixture generation complete ===\n');
fprintf('End time: %s\n', datestr(now));
