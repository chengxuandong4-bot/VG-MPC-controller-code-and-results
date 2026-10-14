function [candidate, history, diag] = llm_wrapper(incumbent, feedback, params, gen, role, benchmark_name, varargin)
% LLM_WRAPPER  Generate controller candidate via LLM with ParseExec validation.
%
%  Aligns with paper (CEP V5.10) Algorithm 1:
%    Line 7-9:  ParseExec(c) — structural validation
%    Line 10:   BenchmarkEval(c, E) — called by caller after this returns
%    Line 13-15: BuildFeedback + incumbent update — caller's responsibility
%
%  Prompt structure per paper Section 2.2-2.3:
%    P_sys:  fixed system prompt (built by llm_generate.py)
%    F^(t):  structured feedback = (S, H^(t-1), (c*,q*), m^(t-1), e^(t-1), F0)
%
%  Inputs:
%    incumbent     - struct with .c (config), .q (score), .g (gen), .m (metrics)
%                     May be empty struct for first generation.
%    feedback      - struct with fields:
%                     .text           — human-readable feedback string
%                     .error_diag     — error diagnostic from previous failure
%                     .stall_count    — consecutive gens without improvement
%                     .baseline_metrics — baseline metric values for comparison
%    params        - benchmark params struct (from init_params)
%    gen           - current generation number (integer)
%    role          - 'S' (single), 'A'/'B' (dual), or descriptive string
%    benchmark_name - 'cstr', 'auv', or 'rampc'
%    varargin{1}   - (optional) config override struct:
%                     .candidate_defaults — default config when no incumbent
%                     .param_spec         — custom validation spec struct
%
%  Outputs:
%    candidate     - complete controller config struct with ALL required fields.
%                    For CSTR: passes directly to run_generation_experiment(c,p.setup).
%                    For AUV:  pass through repair_auv_config + init_controller.
%    history       - updated compact history struct array (max 5 entries).
%                    Caller should stash this for next generation.
%    diag          - diagnostic struct:
%                     .ok      — true if candidate passed ParseExec
%                     .msg     — validation messages (warnings/errors)
%                     .source  — 'llm_api', 'llm_fallback', 'script_failed'
%                     .n_missing — count of fields filled from defaults
%                     .n_type_err — count of type corrections
%                     .n_clamped  — count of range-clamped values

    persistent HISTORY_CACHE;
    if isempty(HISTORY_CACHE)
        HISTORY_CACHE = struct('gen', {}, 'name', {}, 'q', {}, ...
            'accepted', {}, 'updated', {}, 'role', {});
    end

    % ── Locate Python script ──
    py_script = fullfile(fileparts(mfilename('fullpath')), 'llm_generate.py');
    tmpbase = fullfile(tempdir, sprintf('llm_%s_%s_g%03d', benchmark_name, role, gen));
    input_file  = [tmpbase '_in.json'];
    output_file = [tmpbase '_out.json'];

    % ── Extract incumbent config ──
    inc_cfg = struct();
    if ~isempty(incumbent) && isstruct(incumbent)
        if isfield(incumbent, 'c') && isstruct(incumbent.c) && ~isempty(fieldnames(incumbent.c))
            inc_cfg = incumbent.c;
        elseif ~isfield(incumbent, 'c') && ~isempty(fieldnames(incumbent))
            % incumbent IS the config struct (first gen fallback)
            inc_cfg = incumbent;
        end
    end

    % If still empty, use defaults from varargin
    if isempty(fieldnames(inc_cfg)) && nargin >= 7 && isstruct(varargin{1})
        if isfield(varargin{1}, 'candidate_defaults')
            inc_cfg = varargin{1}.candidate_defaults;
        end
    end

    % ── Build structured feedback F^(t) ──
    fb_struct = struct();
    if isstruct(feedback)
        fb_struct = feedback;
    elseif ischar(feedback) || isstring(feedback)
        fb_struct.text = char(feedback);
    end

    % Ensure stall_count is present
    if ~isfield(fb_struct, 'stall_count')
        fb_struct.stall_count = 0;
    end

    % Compute stall count from history if not provided
    if fb_struct.stall_count == 0 && ~isempty(HISTORY_CACHE)
        % Count consecutive non-improvements
        cnt = 0;
        for hi = numel(HISTORY_CACHE):-1:1
            if ~HISTORY_CACHE(hi).updated
                cnt = cnt + 1;
            else
                break;
            end
        end
        fb_struct.stall_count = cnt;
    end

    % ── Build input JSON for Python ──
    input_data = struct();
    input_data.gen = gen;
    input_data.incumbent = inc_cfg;
    input_data.feedback = fb_struct;
    input_data.error_diag = '';
    if isfield(fb_struct, 'error_diag') && ~isempty(fb_struct.error_diag)
        input_data.error_diag = fb_struct.error_diag;
    end
    input_data.stall_count = fb_struct.stall_count;

    % Metrics
    if isfield(incumbent, 'm') && ~isempty(incumbent.m)
        input_data.metrics = incumbent.m;
    else
        input_data.metrics = struct();
    end

    % Baseline metrics (for Δ% comparison in prompt)
    if isfield(fb_struct, 'baseline_metrics') && ~isempty(fb_struct.baseline_metrics)
        input_data.baseline_metrics = fb_struct.baseline_metrics;
    end

    % Scores
    input_data.q_score = 0;
    input_data.best_q = -inf;
    if isfield(incumbent, 'q') && ~isempty(incumbent.q) && isfinite(incumbent.q)
        input_data.q_score = incumbent.q;
        input_data.best_q = incumbent.q;
    end

    % Compact history H^(t-1) — force JSON array even for singleton
    if isempty(HISTORY_CACHE)
        input_data.history = {};
    else
        input_data.history = num2cell(HISTORY_CACHE);
    end

    % ── Write input JSON ──
    fid = fopen(input_file, 'w', 'n', 'UTF-8');
    if fid > 0
        fprintf(fid, '%s', jsonencode(input_data));
        fclose(fid);
    else
        diag = struct('ok', false, 'msg', 'Cannot write input file', ...
            'source', 'io_error', 'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0);
        candidate = build_fallback_from_spec(benchmark_name, inc_cfg, gen, role, varargin{:});
        history = HISTORY_CACHE;
        return;
    end

    % ── Call Python LLM (capture both stdout and stderr) ──
    cmd = sprintf('python3 "%s" "%s" "%s" "%s" "%s"', ...
        py_script, input_file, output_file, benchmark_name, role);
    [status, cmdout] = system(cmd);
    % cmdout contains stdout from Python

    % ── Parse LLM output ──
    diag = struct('ok', false, 'msg', '', 'source', 'unknown', ...
        'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0);
    candidate = struct();

    if exist(output_file, 'file')
        try
            out = jsondecode(fileread(output_file));
        catch ME
            out = struct();
            diag.msg = sprintf('jsondecode failed: %s', ME.message);
            diag.source = 'parse_failed';
        end

        if isfield(out, 'config') && isstruct(out.config)
            % ── ParseExec validation ──
            [candidate, exec_diag] = parse_exec_validate( ...
                out.config, benchmark_name, inc_cfg, varargin{:});
            diag.n_missing  = exec_diag.n_missing;
            diag.n_type_err = exec_diag.n_type_err;
            diag.n_clamped  = exec_diag.n_clamped;
            diag.ok = exec_diag.ok;
            if ~isempty(exec_diag.msg)
                diag.msg = exec_diag.msg;
            end

            % Transfer name and metadata
            if isfield(out, 'name') && ~isempty(out.name)
                candidate.name = out.name;
            end
            if isfield(out, 'label') && ~isempty(out.label)
                candidate.label = out.label;
            end

            % Track source
            if isfield(out, 'success') && out.success
                candidate.llm_source = 'api';
                diag.source = 'llm_api';
            else
                candidate.llm_source = 'fallback';
                diag.source = 'llm_fallback';
                if isfield(out, 'error') && ~isempty(out.error)
                    if isempty(diag.msg)
                        diag.msg = out.error;
                    else
                        diag.msg = sprintf('%s | API: %s', diag.msg, out.error);
                    end
                end
            end

            % Save prompt for reproducibility
            prompt_file = strrep(output_file, '.json', '_prompt.json');
            if exist(prompt_file, 'file')
                try
                    candidate.llm_prompt = fileread(prompt_file);
                catch
                end
                delete(prompt_file);
            end
        else
            diag.source = 'parse_failed';
            if ~isempty(diag.msg), diag.msg = [diag.msg '; ']; end
            diag.msg = [diag.msg 'LLM output missing config field'];
            candidate = build_fallback_from_spec(benchmark_name, inc_cfg, gen, role, varargin{:});
        end

        delete(input_file);
        delete(output_file);
    else
        % ── Python script failed completely ──
        diag.source = 'script_failed';
        diag.msg = sprintf('Python script failed (exit=%d): %s', ...
            status, strtrim(cmdout));
        candidate = build_fallback_from_spec(benchmark_name, inc_cfg, gen, role, varargin{:});
        candidate.llm_source = 'failed';
    end

    % ── Ensure candidate has required metadata fields ──
    if ~isfield(candidate, 'name') || isempty(candidate.name)
        candidate.name = sprintf('gen%03d_llm_%s_%s', gen, diag.source, role);
    end
    if ~isfield(candidate, 'label') || isempty(candidate.label)
        candidate.label = candidate.name;
    end
    if ~isfield(candidate, 'llm_source')
        candidate.llm_source = diag.source;
    end

    % Print generation summary to stdout
    if diag.ok
        src = 'LLM';
        if strcmp(diag.source, 'llm_fallback'), src = 'FALLBACK'; end
        fprintf('[LLM:%s g%03d] %s | fields=%d src=%s', ...
            benchmark_name, gen, candidate.name, length(fieldnames(candidate)), src);
        if diag.n_missing > 0 || diag.n_clamped > 0
            fprintf(' (m=%d,t=%d,c=%d)', diag.n_missing, diag.n_type_err, diag.n_clamped);
        end
        fprintf('\n');
    else
        fprintf('[LLM:%s g%03d] FAILED: %s\n', benchmark_name, gen, diag.msg);
    end

    % ── Update compact history ──
    entry = struct('gen', gen, 'name', candidate.name, 'q', 0, ...
        'accepted', false, 'updated', false, 'role', role);
    if isfield(incumbent, 'q') && ~isempty(incumbent.q)
        entry.q = incumbent.q;
    end
    HISTORY_CACHE(end+1) = entry;
    if length(HISTORY_CACHE) > 5
        HISTORY_CACHE = HISTORY_CACHE(end-4:end);
    end
    history = HISTORY_CACHE;
end


% ═══════════════════════════════════════════════════════════════════
% ParseExec: structural validation  (Algorithm 1, lines 7-9)
% ═══════════════════════════════════════════════════════════════════

function [cfg, diag] = parse_exec_validate(config, benchmark_name, template, varargin)
% PARSE_EXEC_VALIDATE  Validate LLM-generated config.
%  Checks all required fields exist, types are correct, values in safe range.
%  Missing/invalid fields are filled from template or defaults.
%
%  This is the ParseExec step from the paper — it gates execution:
%  if ParseExec fails, we still return a config (with fixes applied),
%  but the diagnostic flags the issues so feedback can be constructed.

    diag = struct('ok', true, 'msg', '', ...
        'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0);
    cfg = struct();

    % Get benchmark validation spec
    spec = get_benchmark_spec(benchmark_name, varargin{:});
    if isempty(spec) || ~isstruct(spec) || isempty(fieldnames(spec))
        % No spec — pass through as-is
        cfg = config;
        diag.msg = 'No validation spec available — passed through unchecked';
        return;
    end

    if ~isstruct(template) || isempty(fieldnames(template))
        template = struct();
    end

    fn = fieldnames(spec);
    missing_list = {};
    type_err_list = {};
    n_clamped = 0;

    for i = 1:numel(fn)
        pname = fn{i};
        pspec = spec.(pname);
        ptype = '';
        if isfield(pspec, 'type'), ptype = pspec.type; end

        % ── Field present? ──
        if ~isfield(config, pname)
            % Fill from template or default
            if isfield(template, pname)
                cfg.(pname) = template.(pname);
            elseif isfield(pspec, 'default')
                cfg.(pname) = pspec.default;
            elseif isfield(pspec, 'lo')
                cfg.(pname) = pspec.lo;
            else
                cfg.(pname) = 0;
            end
            missing_list{end+1} = pname;
            continue;
        end

        val = config.(pname);

        % ── Type validation ──
        switch ptype
            case 'bool'
                if ~islogical(val)
                    try
                        cfg.(pname) = logical(val);
                    catch
                        if isfield(template, pname)
                            cfg.(pname) = logical(template.(pname));
                        elseif isfield(pspec, 'default')
                            cfg.(pname) = logical(pspec.default);
                        else
                            cfg.(pname) = false;
                        end
                    end
                    type_err_list{end+1} = pname;
                else
                    cfg.(pname) = val;
                end

            case 'int'
                if isnumeric(val)
                    cfg.(pname) = round(double(val));
                else
                    if isfield(template, pname)
                        cfg.(pname) = template.(pname);
                    elseif isfield(pspec, 'default')
                        cfg.(pname) = pspec.default;
                    else
                        cfg.(pname) = 0;
                    end
                    type_err_list{end+1} = pname;
                end

            case 'float'
                if isnumeric(val)
                    cfg.(pname) = double(val);
                else
                    if isfield(template, pname)
                        cfg.(pname) = double(template.(pname));
                    elseif isfield(pspec, 'default')
                        cfg.(pname) = double(pspec.default);
                    else
                        cfg.(pname) = 0;
                    end
                    type_err_list{end+1} = pname;
                end

            otherwise
                cfg.(pname) = val;
        end

        % ── Range clamping ──
        if isfield(pspec, 'lo') && isfield(pspec, 'hi')
            if isnumeric(cfg.(pname)) && ~islogical(cfg.(pname))
                old = cfg.(pname);
                cfg.(pname) = max(double(pspec.lo), min(double(pspec.hi), ...
                    double(cfg.(pname))));
                if abs(old - cfg.(pname)) > 1e-15
                    n_clamped = n_clamped + 1;
                end
            end
        end
    end

    diag.n_missing  = numel(missing_list);
    diag.n_type_err = numel(type_err_list);
    diag.n_clamped  = n_clamped;

    % ── Build diagnostic message ──
    msgs = {};
    if ~isempty(missing_list)
        n_show = min(5, numel(missing_list));
        msgs{end+1} = sprintf('Missing %d: %s', numel(missing_list), ...
            strjoin(missing_list(1:n_show), ','));
    end
    if ~isempty(type_err_list)
        n_show = min(5, numel(type_err_list));
        msgs{end+1} = sprintf('TypeErr %d: %s', numel(type_err_list), ...
            strjoin(type_err_list(1:n_show), ','));
    end
    if n_clamped > 0
        msgs{end+1} = sprintf('Clamped %d', n_clamped);
    end
    diag.msg = strjoin(msgs, '; ');

    % ParseExec passes if we got a config (even with defaults filled)
    diag.ok = true;
end


% ═══════════════════════════════════════════════════════════════════
% Benchmark Validation Specs
% ═══════════════════════════════════════════════════════════════════

function spec = get_benchmark_spec(benchmark_name, varargin)
% GET_BENCHMARK_SPEC  Return parameter validation spec for each benchmark.
%  spec.(param_name) = struct('type','float'|'int'|'bool', 'lo',lo, 'hi',hi, 'default',def)

    % Allow user override via varargin
    if nargin >= 2 && isstruct(varargin{1}) && isfield(varargin{1}, 'param_spec')
        spec = varargin{1}.param_spec;
        return;
    end

    switch lower(benchmark_name)
        case 'cstr'
            spec = struct( ...
                'use_adaptive_horizon',      struct('type','bool',  'default',false), ...
                'use_disturbance_observer',  struct('type','bool',  'default',true), ...
                'use_param_adaptation',      struct('type','bool',  'default',false), ...
                'use_input_bias_adaptation', struct('type','bool',  'default',false), ...
                'horizon_max',     struct('type','int',   'lo',10,'hi',30,  'default',20), ...
                'horizon_mid',     struct('type','int',   'lo',5, 'hi',25,  'default',15), ...
                'horizon_min',     struct('type','int',   'lo',3, 'hi',15,  'default',8), ...
                'horizon_high_err',struct('type','float', 'lo',0.1,'hi',2.0, 'default',0.5), ...
                'horizon_mid_err', struct('type','float', 'lo',0.02,'hi',0.5,'default',0.1), ...
                'int_ref_gain',    struct('type','float', 'lo',0.0,'hi',2.0, 'default',0.0), ...
                'int_e_bound',     struct('type','float', 'lo',0.1,'hi',5.0, 'default',0.5), ...
                'Wy_base',   struct('type','float', 'lo',0.3,'hi',10.0, 'default',1.0), ...
                'Wy_gain',   struct('type','float', 'lo',0.0,'hi',5.0,  'default',0.3), ...
                'Wy_cap',    struct('type','float', 'lo',10.0,'hi',500.0,'default',100.0), ...
                'Wu',        struct('type','float', 'lo',1e-6,'hi',1e-2, 'default',1e-4), ...
                'Wdu_base',  struct('type','float', 'lo',1e-5,'hi',1e-2, 'default',1e-3), ...
                'Wdu_gain',  struct('type','float', 'lo',0.0,'hi',5.0,  'default',0.2), ...
                'Wdu_cap',   struct('type','float', 'lo',10.0,'hi',500.0,'default',100.0), ...
                'hessian_reg',struct('type','float', 'lo',1e-8,'hi',1e-3, 'default',1e-5), ...
                'obs_alpha', struct('type','float', 'lo',0.05,'hi',0.95,'default',0.3), ...
                'obs_bound', struct('type','float', 'lo',0.1,'hi',2.0,  'default',0.5), ...
                'gamma_k0',     struct('type','float', 'lo',0.01,'hi',1.0, 'default',0.1), ...
                'theta_k0_max', struct('type','float', 'lo',0.5,'hi',5.0,  'default',1.5), ...
                'theta_k0_min', struct('type','float', 'lo',0.1,'hi',2.0,  'default',0.5), ...
                'gamma_ua',     struct('type','float', 'lo',0.01,'hi',1.0, 'default',0.1), ...
                'theta_ua_max', struct('type','float', 'lo',0.5,'hi',5.0,  'default',1.5), ...
                'theta_ua_min', struct('type','float', 'lo',0.1,'hi',2.0,  'default',0.5), ...
                'adapt_eps',     struct('type','float', 'lo',0.001,'hi',0.1,'default',0.01), ...
                'u_bias_gain',     struct('type','float', 'lo',0.01,'hi',1.0, 'default',0.1), ...
                'u_bias_int_gain', struct('type','float', 'lo',0.001,'hi',0.1,'default',0.01), ...
                'u_bias_bound',    struct('type','float', 'lo',0.1,'hi',2.0,  'default',0.5) ...
            );

        case 'auv'
            spec = struct( ...
                'kp_vel',         struct('type','float', 'lo',0.1,'hi',5.0,   'default',0.8), ...
                'ki_vel',         struct('type','float', 'lo',0.01,'hi',1.0,  'default',0.1), ...
                'k_psi',          struct('type','float', 'lo',0.3,'hi',8.0,   'default',1.5), ...
                'ki_psi',         struct('type','float', 'lo',0.01,'hi',0.5,  'default',0.05), ...
                'k_r',            struct('type','float', 'lo',0.1,'hi',3.0,   'default',0.5), ...
                'ki_r',           struct('type','float', 'lo',0.005,'hi',0.2, 'default',0.02), ...
                'ks_tau',         struct('type','float', 'lo',0.05,'hi',2.0,  'default',0.3), ...
                'allocation_reg', struct('type','float', 'lo',1e-4,'hi',0.5,  'default',0.01), ...
                'rate_limit',     struct('type','float', 'lo',0.1,'hi',10.0,  'default',2.0), ...
                'rls_forgetting', struct('type','float', 'lo',0.90,'hi',0.9999,'default',0.98), ...
                'rls_fault_boost',struct('type','float', 'lo',0.001,'hi',0.2, 'default',0.02), ...
                'health_decay',   struct('type','float', 'lo',0.9,'hi',0.999, 'default',0.995), ...
                'health_weight_scale', struct('type','float','lo',0.5,'hi',5.0,'default',2.0), ...
                'smoothing_weight',   struct('type','float','lo',0.0,'hi',1.0,'default',0.3), ...
                'use_fault_feedforward', struct('type','bool', 'default',false), ...
                'fault_feedforward_gain',struct('type','float','lo',0.1,'hi',3.0,'default',0.5), ...
                'use_perf_boost',   struct('type','bool', 'default',false), ...
                'use_alloc_relax',  struct('type','bool', 'default',false) ...
            );

        case 'rampc'
            spec = struct( ...
                'Q_position',      struct('type','float', 'lo',0.5,'hi',20.0, 'default',1.0), ...
                'R_linear_vel',    struct('type','float', 'lo',0.05,'hi',2.0, 'default',0.1), ...
                'safety_margin',   struct('type','float', 'lo',0.05,'hi',0.80,'default',0.3), ...
                'max_linear_vel',  struct('type','float', 'lo',0.4,'hi',2.5,  'default',1.0), ...
                'obs_repulsion_w', struct('type','float', 'lo',5.0,'hi',50.0, 'default',10.0), ...
                'terminal_weight', struct('type','float', 'lo',1.0,'hi',20.0, 'default',5.0) ...
            );

        otherwise
            spec = struct();
    end
end


% ═══════════════════════════════════════════════════════════════════
% Structured Fallback Candidate
% ═══════════════════════════════════════════════════════════════════

function candidate = build_fallback_from_spec(benchmark_name, incumbent, gen, role, varargin)
% BUILD_FALLBACK_FROM_SPEC  Build fallback candidate when LLM completely fails.
%  Uses benchmark spec defaults, NOT pure random numbers.
%  If incumbent exists, applies small structured perturbation.

    spec = get_benchmark_spec(benchmark_name, varargin{:});
    candidate = struct();

    if ~isstruct(incumbent) || isempty(fieldnames(incumbent))
        % ── No incumbent: build from defaults ──
        fn = fieldnames(spec);
        for i = 1:numel(fn)
            pname = fn{i};
            pspec = spec.(pname);
            if isfield(pspec, 'default')
                candidate.(pname) = pspec.default;
            elseif isfield(pspec, 'lo')
                candidate.(pname) = pspec.lo;
            else
                candidate.(pname) = 0;
            end
        end
        candidate.name = sprintf('gen%03d_default_%s', gen, role);
        candidate.llm_source = 'default';
        return;
    end

    % ── Start from incumbent, apply small perturbation ──
    candidate = incumbent;
    fn = fieldnames(spec);
    % Scale: small perturbation for fallback (not trying to explore, just not crash)
    scale = 0.05;

    for i = 1:numel(fn)
        pname = fn{i};
        if ~isfield(candidate, pname)
            if isfield(spec.(pname), 'default')
                candidate.(pname) = spec.(pname).default;
            end
            continue;
        end

        pspec = spec.(pname);
        % Don't perturb booleans in fallback
        if isfield(pspec, 'type') && strcmp(pspec.type, 'bool')
            continue;
        end

        if isnumeric(candidate.(pname)) && ~islogical(candidate.(pname))
            old = double(candidate.(pname));
            if abs(old) > 1e-12
                candidate.(pname) = old * (1.0 + scale * randn());
            else
                candidate.(pname) = scale * abs(randn());
            end
            % Safety clamp
            if isfield(pspec, 'lo') && isfield(pspec, 'hi')
                candidate.(pname) = max(double(pspec.lo), ...
                    min(double(pspec.hi), candidate.(pname)));
            end
            % Round integer params (randn produces fractional values)
            if isfield(pspec, 'type') && strcmp(pspec.type, 'int')
                candidate.(pname) = round(candidate.(pname));
            end
        end
    end

    candidate.name = sprintf('gen%03d_fallback_%s', gen, role);
    candidate.llm_source = 'fallback';
end
