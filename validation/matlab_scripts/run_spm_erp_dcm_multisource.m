% run_spm_erp_dcm_multisource.m -- SPM12 5-source CMC-ERP frozen-fixture generator
%
% Usage: matlab -batch "run_spm_erp_dcm_multisource"
%   or:  matlab -batch "run('validation/matlab_scripts/run_spm_erp_dcm_multisource.m')"
%
% Reads:  validation/data/erp_multisource_input.mat   (DCM struct from
%         validation.export_to_mat.export_erp_dcm_multisource)
% Writes: validation/data/erp_multisource_fixtures.mat (QA, QG, J0, Qupd, y, meta)
%
% Phase 34-02 multi-source parity fixtures (EVOK-05/06). Generates the byte-frozen
% arrays the Wave-3 parity ladder asserts the pure-torch network forward
% (cmc_network_f / apply_condition_modulation / simulate_erp_dcm) against -- parity
% is vs-SPM, never vs-torch (pitfall V1). For the canonical 5-source auditory-MMN
% reference (A1L, A1R, STGL, STGR, rIFG; Cnd = 2 standard/deviant):
%
%   QA{c}   {4} of (N,N)   per-condition spm_gen_Q extrinsic blocks (B->all-A
%                          folding, free log space)            spm_gen_Q.m:41-47
%   QG{c}   (N,)           per-condition free precision column Q.G(:,1)
%                          (diag(B)->Q.G(:,1) precision path)  spm_gen_Q.m:65-67
%   J0{c}   (8N,8N)        per-condition frozen Jacobian df/dx at x0 = 0, u0 = 0
%                          via spm_diff forward differences     spm_int_L.m
%   Qupd{c} (8N,8N)        (spm_expm(dt*dfdx) - I)/dfdx right-division
%                                                              spm_int_L.m:126-127
%   y{c}    (ns,8N)        the multi-source evoked SOURCE trajectory, D = I,
%                          per condition (the spm_gen_erp loop body, lead-field
%                          deferred to Phase 35)                spm_gen_erp.m:69-86
%
% The delay operator is forced to EXACT identity (D = 1) by setting
% M.f = 'spm_fx_cmc_nodelay' (a 2-output wrapper, REUSED UNCHANGED from 33-02):
% nargout(M.f) == 2 sends spm_int_L down its :117 branch and keeps D = 1
% (spm_int_L.m:112). x0 == zeros(N,8) is the asserted CMC fixed point (M1). Both
% facts are ASSERTED here and recorded in meta (EVOK-06).
% Source: spm_int_L.m:112-169, spm_fx_cmc.m:206-226, spm_dcm_delay.m:60-82,
% spm_gen_Q.m:24-67, spm_gen_erp.m:69-86.
%
% NOTE (the spm_gen_erp trajectory): Phase 34 has no lead-field yet (Phase 35), so
% the fixture stores the per-condition SOURCE-state trajectory -- i.e. the
% spm_gen_erp.m:69-86 loop body (spm_gen_Q -> spm_int_L with D = 1) WITHOUT the
% L*x channel projection. This is exactly what the torch simulate_erp_dcm produces
% and what the Wave-3 source-state parity gate asserts against (34-02-D1).

fprintf('=== SPM12 CMC-ERP multi-source fixture generation ===\n');
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
    % CMC dynamics + condition modulation live in the M/EEG DCM toolbox.
    if ~isdeployed
        addpath(fullfile(spm('Dir'), 'toolbox', 'dcm_meeg'));
        addpath(fullfile(spm('Dir'), 'toolbox', 'Neural_Models'));
    end
    if ~exist('spm_fx_cmc', 'file')
        error('spm_fx_cmc not found -- check toolbox/dcm_meeg on path.');
    end
    if ~exist('spm_gen_Q', 'file')
        error('spm_gen_Q not found -- check toolbox/dcm_meeg on path.');
    end
    fprintf('SPM12 loaded successfully (EEG defaults).\n');
catch e
    fprintf('ERROR: Failed to initialize SPM12: %s\n', e.message);
    return;
end

% --- Input/output paths ---
input_path = getenv('DCM_INPUT_PATH');
if isempty(input_path)
    input_path = 'validation/data/erp_multisource_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/erp_multisource_fixtures.mat';
end

fprintf('Input:  %s\n', input_path);
fprintf('Output: %s\n', output_path);

% --- Load DCM struct ---
try
    load(input_path, 'DCM');
    P = DCM.P;
    M = DCM.M;
    N = size(M.x, 1);
    fprintf('DCM loaded: N=%d sources, input grid %dx%d, X %dx%d conditions.\n', ...
        N, size(DCM.U.u, 1), size(DCM.U.u, 2), ...
        size(DCM.U.X, 1), size(DCM.U.X, 2));
catch e
    fprintf('ERROR: Failed to load DCM from %s: %s\n', input_path, e.message);
    return;
end

% --- Force D = I via the 2-output wrapper (Fact 4) + the fixed point (M1) ---
try
    M.f = 'spm_fx_cmc_nodelay';          % FORCE D = I (nargout == 2)
    M.x = zeros(N, 8);
    M.n = 8 * N;
    M.m = size(DCM.U.u, 2);
    assert(isequal(M.x, zeros(N, 8)), 'x0 must be the zeros(N,8) fixed point (M1)');
    nout = nargout(M.f);
    assert(nout == 2, 'M.f must expose exactly 2 outputs to keep D = I; got %d', nout);
    fprintf('M.f = %s (nargout = %d -> D = 1); x0 == zeros(%d,8) asserted.\n', ...
        M.f, nout, N);
catch e
    fprintf('ERROR: D=I / x0 setup failed: %s\n', e.message);
    return;
end

% --- Generate the per-condition fixtures (staged ladder, V5) ---
try
    Cnd = size(DCM.U.X, 1);
    x0  = spm_vec(M.x);                  % column-major flat state (8N,1)
    u0  = sparse(M.m, 1);                % Jacobian taken at u = 0 (Fact 5)
    I8N = eye(8 * N);

    QA   = cell(1, Cnd);
    QG   = cell(1, Cnd);
    J0   = cell(1, Cnd);
    Qupd = cell(1, Cnd);
    y    = cell(1, Cnd);

    for c = 1:Cnd
        % (1) per-condition spm_gen_Q: the B-wiring guard (C4 / EVOK-05 part 1).
        Qc      = spm_gen_Q(P, DCM.U.X(c, :));
        QA{c}   = Qc.A;                  % cell of 4 (N,N) free-log blocks (spm_gen_Q:47)
        QG{c}   = Qc.G(:, 1);            % (N,) free precision column     (spm_gen_Q:66)

        % (2) per-condition frozen Jacobian + update operator (SCHEME rung, Fact 5).
        J0c     = full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, Qc, M, 1)));  % (8N,8N) FD
        dfdx    = J0c - I8N * exp(-16);
        Qupd{c} = (spm_expm(DCM.U.dt * dfdx) - I8N) / dfdx;               % right-division
        J0{c}   = J0c;

        % (3) the multi-source evoked SOURCE trajectory (EVOK-05 part 2): the
        %     spm_gen_erp.m:69-86 loop body with D = I (nargout(M.f) == 2), no
        %     lead-field (Phase 35). spm_int_L returns the (ns,8N) states.
        y{c}    = spm_int_L(Qc, M, DCM.U);                                % (ns,8N)
    end

    fprintf(['Fixtures (Cnd=%d): QA{1} %s x4, QG{1} %s, J0{1} %s, ', ...
        'Qupd{1} %s, y{1} %s\n'], Cnd, ...
        mat2str(size(QA{1}{1})), mat2str(size(QG{1})), mat2str(size(J0{1})), ...
        mat2str(size(Qupd{1})), mat2str(size(y{1})));
catch e
    fprintf('ERROR: fixture generation failed: %s\n', e.message);
    return;
end

% --- Provenance metadata header (pitfall V4) ---
try
    meta = struct();
    meta.spm_ver = spm('Ver');
    % Capture the $Id headers of the load-bearing source files.
    src_files = {'spm_int_L', 'spm_fx_cmc', 'spm_gen_Q', 'spm_gen_erp', ...
        'spm_cmc_priors', 'spm_erp_u'};
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
    meta.D          = 1;                  % delay operator forced to identity
    meta.nargout_Mf = nargout(M.f);       % == 2 -> the D=1 routing proof
    meta.N          = N;
    meta.Cnd        = size(DCM.U.X, 1);
    meta.X          = DCM.U.X;            % the between-trial design (Cnd, n_effects)
    meta.dt         = DCM.U.dt;
    meta.ns         = size(DCM.U.u, 1);
    meta.ons        = M.ons;
    meta.dur        = M.dur;
    meta.sus        = M.sus;
    meta.x0         = M.x;               % the asserted zeros(N,8) fixed point
    meta.exp_shift  = exp(-16);          % the spm_int_L regulariser
    % Carry the locked edge list forward from the exporter (V4 provenance).
    if isfield(DCM, 'meta')
        if isfield(DCM.meta, 'edges_forward')
            meta.edges_forward  = DCM.meta.edges_forward;
        end
        if isfield(DCM.meta, 'edges_backward')
            meta.edges_backward = DCM.meta.edges_backward;
        end
        if isfield(DCM.meta, 'edges_lateral')
            meta.edges_lateral  = DCM.meta.edges_lateral;
        end
        if isfield(DCM.meta, 'source_names')
            meta.source_names   = DCM.meta.source_names;
        end
    end
    fprintf('meta: spm_ver=%s D=%d N=%d Cnd=%d ns=%d dt=%g ons=%g dur=%g\n', ...
        meta.spm_ver, meta.D, meta.N, meta.Cnd, meta.ns, meta.dt, ...
        meta.ons, meta.dur);
    fprintf('  id_spm_fx_cmc = %s\n', meta.id_spm_fx_cmc);
    fprintf('  id_spm_gen_Q  = %s\n', meta.id_spm_gen_Q);
catch e
    fprintf('ERROR: meta assembly failed: %s\n', e.message);
    return;
end

% --- Save the frozen fixtures ---
try
    save(output_path, 'QA', 'QG', 'J0', 'Qupd', 'y', 'meta');
    fprintf('Fixtures saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save fixtures: %s\n', e.message);
    return;
end

fprintf('=== CMC-ERP multi-source fixture generation complete ===\n');
fprintf('End time: %s\n', datestr(now));
