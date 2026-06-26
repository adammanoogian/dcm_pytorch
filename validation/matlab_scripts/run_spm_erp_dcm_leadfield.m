% run_spm_erp_dcm_leadfield.m -- SPM12 LFP lead-field + scalp-ERP fixture generator
%
% Usage: matlab -batch "run_spm_erp_dcm_leadfield"
%   or:  matlab -batch "run('validation/matlab_scripts/run_spm_erp_dcm_leadfield.m')"
%
% Reads:  validation/data/erp_leadfield_input.mat   (DCM struct from
%         validation.export_to_mat.export_erp_dcm_leadfield)
% Writes: validation/data/erp_leadfield_fixtures.mat (L_full, y_scalp,
%         diff_wave, meta)
%
% Phase 35-02 LFP lead-field + scalp-ERP fixtures (LEAD-05). Generates the
% byte-frozen arrays the Wave-3 scalp parity ladder asserts the pure-torch lead
% field (forward_models.erp_leadfield.build_lead_field / project_to_scalp)
% against -- parity is vs-SPM, never vs-torch (pitfall V1). For the canonical
% 5-source auditory-MMN reference (A1L, A1R, STGL, STGR, rIFG; Cnd = 2
% standard/deviant) in LFP mode (head-model-free, no MNI coords, no
% spm_cond_units -- the cleanest possible target, ECD deferred to Phase 36):
%
%   L_full      (Nc, 8N) = (5, 40)   the full per-state lead field
%                                     kron(P.J, L_spatial)        spm_lx_erp.m:31-33
%   y_scalp{c}  (ns, Nc) = (128, 5)  per-condition scalp ERP
%                                     ysrc * L_full' (y = L*x)    spm_lx_erp.m header
%   diff_wave   (ns, Nc) = (128, 5)  deviant - standard difference wave
%
% LFP spatial model: P.L = ones(1,5) -> L_spatial = sparse(1:m,1:m,P.L,m,n) is
% the identity (spm_erp_L.m:112); P.J = sparse(1,3,1,1,8) -> sp-voltage (MATLAB
% column 3, 0-indexed index 2; spm_L_priors.m:108). So the LFP scalp ERP is
% literally each source's sp-voltage trace.
%
% The delay operator is forced to EXACT identity (D = 1) by setting
% M.f = 'spm_fx_cmc_nodelay' (a 2-output wrapper, REUSED UNCHANGED from 33-02):
% nargout(M.f) == 2 sends spm_int_L down its :117 branch and keeps D = 1
% (spm_int_L.m:112). x0 == zeros(N,8) is the asserted CMC fixed point (M1). Both
% facts are ASSERTED here and recorded in meta.
% Source: spm_lx_erp.m:31-33, spm_erp_L.m:105-118, spm_L_priors.m:84,106-109,
% spm_gen_Q.m:24-67, spm_gen_erp.m:69-86, spm_int_L.m:112-169.
%
% NOTE (self-contained source trajectory, Open Q6 RESOLVED): this fixture
% recomputes ysrc = spm_int_L(spm_gen_Q(P, X(c,:)), M, U) per condition -- the
% spm_gen_erp.m:69-86 loop body with D = I -- rather than coupling to the
% Phase-34 erp_multisource_fixtures.mat. One fixture, one provenance header.

fprintf('=== SPM12 CMC-ERP LFP lead-field + scalp fixture generation ===\n');
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
    spm('defaults', 'EEG');
    % CMC dynamics, condition modulation + lead fields live in the M/EEG toolbox.
    if ~isdeployed
        addpath(fullfile(spm('Dir'), 'toolbox', 'dcm_meeg'));
        addpath(fullfile(spm('Dir'), 'toolbox', 'Neural_Models'));
    end
    if ~exist('spm_fx_cmc', 'file')
        error('spm_fx_cmc not found -- check toolbox/dcm_meeg on path.');
    end
    if ~exist('spm_lx_erp', 'file')
        error('spm_lx_erp not found -- check toolbox/dcm_meeg on path.');
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
    input_path = 'validation/data/erp_leadfield_input.mat';
end
output_path = getenv('DCM_OUTPUT_PATH');
if isempty(output_path)
    output_path = 'validation/data/erp_leadfield_fixtures.mat';
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

% --- Build the LFP single-dipole lead field (head-model-free) ---
try
    dipfit.type = 'LFP';                 % spm_erp_L.m:105-118 diagonal branch
    dipfit.Ns   = N;                     % number of sources
    dipfit.Nc   = N;                     % number of channels (Nc == Ns in LFP)
    L_full = spm_lx_erp(P, dipfit);      % (Nc, 8N) = kron(P.J, L_spatial)
    L_full = full(L_full);
    assert(isequal(size(L_full), [N, 8 * N]), ...
        'L_full must be (Nc,8N) = (%d,%d); got %s', N, 8 * N, mat2str(size(L_full)));
    fprintf('L_full = spm_lx_erp(P, dipfit) %s (LFP, P.J col-3 sp-voltage).\n', ...
        mat2str(size(L_full)));
catch e
    fprintf('ERROR: lead-field construction failed: %s\n', e.message);
    return;
end

% --- Per-condition scalp ERP + difference wave (spm_gen_erp loop body) ---
try
    Cnd     = size(DCM.U.X, 1);
    ns      = size(DCM.U.u, 1);
    y_scalp = cell(1, Cnd);
    for c = 1:Cnd
        % (1) per-condition spm_gen_Q: B->all-A folding + diag(B)->Q.G(:,1).
        Qc      = spm_gen_Q(P, DCM.U.X(c, :));
        % (2) the evoked SOURCE trajectory (spm_gen_erp.m:69-86 body, D = I).
        ysrc    = spm_int_L(Qc, M, DCM.U);    % (ns, 8N)
        % (3) project to scalp: y = L*x (observer header), row-trajectory form.
        y_scalp{c} = ysrc * L_full';          % (ns, Nc)
        assert(isequal(size(y_scalp{c}), [ns, N]), ...
            'y_scalp{%d} must be (ns,Nc) = (%d,%d); got %s', ...
            c, ns, N, mat2str(size(y_scalp{c})));
    end
    diff_wave = y_scalp{2} - y_scalp{1};      % deviant - standard
    fprintf('y_scalp{1} %s, diff_wave %s (deviant - standard).\n', ...
        mat2str(size(y_scalp{1})), mat2str(size(diff_wave)));
catch e
    fprintf('ERROR: scalp-ERP generation failed: %s\n', e.message);
    return;
end

% --- Provenance metadata header (pitfall V4) ---
try
    meta = struct();
    meta.spm_ver = spm('Ver');
    % Capture the $Id headers of the load-bearing source files.
    src_files = {'spm_lx_erp', 'spm_erp_L', 'spm_L_priors', 'spm_gen_Q', ...
        'spm_int_L', 'spm_fx_cmc', 'spm_gen_erp'};
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
    meta.D           = 1;                 % delay operator forced to identity
    meta.nargout_Mf  = nargout(M.f);      % == 2 -> the D=1 routing proof
    meta.N           = N;
    meta.Nc          = N;                 % LFP: channels == sources
    meta.Cnd         = size(DCM.U.X, 1);
    meta.X           = DCM.U.X;           % the between-trial design (Cnd, n_effects)
    meta.dt          = DCM.U.dt;
    meta.ns          = size(DCM.U.u, 1);
    meta.ons         = M.ons;
    meta.dur         = M.dur;
    meta.sus         = M.sus;
    meta.x0          = M.x;               % the asserted zeros(N,8) fixed point
    meta.dipfit_type = dipfit.type;       % 'LFP' (head-model-free scope)
    meta.P_J         = P.J;               % (1,8) sp-voltage one-hot
    meta.P_L         = P.L;               % (1,N) LFP channel gains (ones -> I)
    meta.exp_shift   = exp(-16);          % the spm_int_L regulariser
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
    fprintf('meta: spm_ver=%s D=%d N=%d Nc=%d Cnd=%d ns=%d dt=%g\n', ...
        meta.spm_ver, meta.D, meta.N, meta.Nc, meta.Cnd, meta.ns, meta.dt);
    fprintf('  id_spm_lx_erp = %s\n', meta.id_spm_lx_erp);
    fprintf('  id_spm_erp_L  = %s\n', meta.id_spm_erp_L);
catch e
    fprintf('ERROR: meta assembly failed: %s\n', e.message);
    return;
end

% --- Save the frozen fixtures ---
try
    save(output_path, 'L_full', 'y_scalp', 'diff_wave', 'meta');
    fprintf('Fixtures saved to %s\n', output_path);
catch e
    fprintf('ERROR: Failed to save fixtures: %s\n', e.message);
    return;
end

fprintf('=== CMC-ERP LFP lead-field + scalp fixture generation complete ===\n');
fprintf('End time: %s\n', datestr(now));
