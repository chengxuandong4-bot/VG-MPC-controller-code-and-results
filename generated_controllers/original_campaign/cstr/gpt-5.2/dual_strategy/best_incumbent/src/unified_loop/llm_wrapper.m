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
%                     .source  — provider/model identifier or failure source
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
        error('GPT:InputWriteFailed', 'Cannot write LLM input file: %s', input_file);
    end

    % ── Human-in-the-loop intervention ──
    % When stall_count >= intervene_at, pause and wait for human guidance.
    % Writes an intervention file with full prompt, diagnostics, and editable params.
    intervene_at = 50;  % default
    if nargin >= 7
        for vi = 1:numel(varargin)
            if isstruct(varargin{vi}) && isfield(varargin{vi}, 'intervene_at')
                intervene_at = varargin{vi}.intervene_at;
            end
        end
    end
    if fb_struct.stall_count >= intervene_at && fb_struct.stall_count > 0
        [candidate, diag] = human_intervention(input_data, input_file, output_file, ...
            py_script, benchmark_name, role, inc_cfg, fb_struct, gen, intervene_at, varargin{:});
        history = HISTORY_CACHE;
        return;
    end

    % ── Call Python LLM ──
    py_exec = resolve_python_executable(py_script);
    cmd = sprintf('"%s" "%s" "%s" "%s" "%s" "%s"', ...
        py_exec, py_script, input_file, output_file, benchmark_name, role);
    [status, cmdout] = system(cmd);
    % cmdout contains stdout from Python

    % ── Parse LLM output ──
    diag = struct('ok', false, 'msg', '', 'source', 'unknown', ...
        'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0, ...
        'provider', '', 'transport', '', ...
        'requested_model', '', 'response_model', '', ...
        'request_id', '', 'http_status', 0, 'prompt_tokens', 0, ...
        'completion_tokens', 0, 'total_tokens', 0, 'retry_count', 0, ...
        'parse_retry_count', 0, ...
        'api_calls', 0, 'codex_exec_calls', 0, 'codex_cli_version', '');
    candidate = struct();

    if exist(output_file, 'file')
        try
            out = jsondecode(fileread(output_file));
        catch ME
            out = struct();
            diag.msg = sprintf('jsondecode failed: %s', ME.message);
            diag.source = 'parse_failed';
        end

        if isfield(out, 'success') && ~out.success
            failure_msg = 'GPT generation failed';
            if isfield(out, 'error') && ~isempty(out.error)
                failure_msg = out.error;
            end
            delete(input_file);
            delete(output_file);
            error('GPT:GenerationFailed', '%s', failure_msg);
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
                if isfield(out, 'llm_source') && ~isempty(out.llm_source)
                    candidate.llm_source = out.llm_source;
                    diag.source = out.llm_source;
                else
                    candidate.llm_source = 'model_backend';
                    diag.source = 'model_backend';
                end
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

            % Preserve upstream identity and request trace in every candidate.
            if isfield(out, 'api_metadata') && isstruct(out.api_metadata)
                am = out.api_metadata;
                trace_fields = {'provider','transport', ...
                    'requested_model','response_model', ...
                    'request_id','http_status','prompt_tokens', ...
                    'completion_tokens','total_tokens','retry_count', ...
                    'parse_retry_count','api_calls', ...
                    'codex_exec_calls','codex_cli_version'};
                candidate_fields = {'llm_provider','llm_transport', ...
                    'llm_requested_model', ...
                    'llm_response_model','llm_request_id','llm_http_status', ...
                    'llm_prompt_tokens','llm_completion_tokens','llm_total_tokens', ...
                    'llm_retry_count','llm_parse_retry_count', ...
                    'llm_api_calls','llm_codex_exec_calls','llm_codex_cli_version'};
                for ai = 1:numel(trace_fields)
                    src_field = trace_fields{ai};
                    if isfield(am, src_field)
                        diag.(src_field) = am.(src_field);
                        candidate.(candidate_fields{ai}) = am.(src_field);
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
            delete(input_file);
            delete(output_file);
            error('GPT:InvalidOutput', 'LLM output missing a valid config field');
        end

        delete(input_file);
        delete(output_file);
    else
        error('GPT:PythonFailed', 'Python script failed (exit=%d): %s', ...
            status, strtrim(cmdout));
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

    % ── Carry through template fields not in spec (partial merge mode) ──
    % For benchmarks like AUV where LLM only touches a subset of params,
    % inherit all other fields from the incumbent to avoid resetting progress.
    n_inherited = 0;
    if isstruct(template) && ~isempty(fieldnames(template))
        fn_tmpl = fieldnames(template);
        for ti = 1:numel(fn_tmpl)
            pname = fn_tmpl{ti};
            if ~isfield(cfg, pname) && ~isfield(spec, pname)
                cfg.(pname) = template.(pname);
                n_inherited = n_inherited + 1;
            end
        end
    end

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
                'kp_vel',         struct('type','float', 'lo',0.1,'hi',10.0,  'default',0.8), ...
                'ki_vel',         struct('type','float', 'lo',0.01,'hi',2.0,  'default',0.1), ...
                'k_psi',          struct('type','float', 'lo',0.3,'hi',8.0,   'default',1.5), ...
                'ki_psi',         struct('type','float', 'lo',0.01,'hi',0.5,  'default',0.05), ...
                'k_r',            struct('type','float', 'lo',0.1,'hi',10.0,   'default',0.5), ...
                'ki_r',           struct('type','float', 'lo',0.005,'hi',0.5, 'default',0.02), ...
                'ks_tau',         struct('type','float', 'lo',0.05,'hi',500.0,  'default',0.3), ...
                'allocation_reg', struct('type','float', 'lo',1e-4,'hi',0.5,  'default',0.01), ...
                'rate_limit',     struct('type','float', 'lo',0.05,'hi',10.0, 'default',2.0), ...
                'rls_forgetting', struct('type','float', 'lo',0.90,'hi',0.9999,'default',0.98), ...
                'rls_fault_boost',struct('type','float', 'lo',0.001,'hi',0.2, 'default',0.02), ...
                'health_decay',   struct('type','float', 'lo',0.001,'hi',0.999,'default',0.995), ...
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


% ═══════════════════════════════════════════════════════════════════
% Human-in-the-Loop Intervention
% ═══════════════════════════════════════════════════════════════════

function [candidate, diag] = human_intervention(input_data, input_file, output_file, ...
    py_script, benchmark_name, role, inc_cfg, fb_struct, gen, intervene_at, varargin)
% HUMAN_INTERVENTION  Pause and request human guidance when stall threshold reached.
%
%  Writes an intervention file with:
%    1. Full prompt text (by calling Python in --dry-run mode, or reading last prompt)
%    2. Current incumbent config (all parameters, editable)
%    3. Stall diagnostics (how long, best so far, metric trends)
%    4. Suggested intervention directions
%
%  User edits the file, deletes it, and the loop resumes with the overrides.

    intervene_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'HOPEAI_GPT52', 'interventions');
    if ~exist(intervene_dir, 'dir'), mkdir(intervene_dir); end
    intervene_file = fullfile(intervene_dir, ...
        sprintf('intervene_%s_%s_g%03d.json', benchmark_name, role, gen));

    % ── 1. Try to get the full LLM prompt ──
    prompt_text = build_prompt_dry_run(input_file, output_file, py_script, benchmark_name, role);

    % ── 2. Build diagnostic summary ──
    diag_summary = struct();
    diag_summary.stall_count = fb_struct.stall_count;
    diag_summary.intervene_at = intervene_at;
    diag_summary.current_gen = gen;
    diag_summary.best_q = input_data.best_q;
    diag_summary.best_gen = 0;
    diag_summary.role = role;
    diag_summary.benchmark = benchmark_name;

    % ── 3. Metric analysis ──
    if isfield(input_data, 'metrics') && isstruct(input_data.metrics)
        m = input_data.metrics;
        diag_summary.metrics = struct();
        fn_m = fieldnames(m);
        for i = 1:min(numel(fn_m), 15)
            if isnumeric(m.(fn_m{i})) && isscalar(m.(fn_m{i}))
                diag_summary.metrics.(fn_m{i}) = m.(fn_m{i});
            end
        end
    end

    % Baseline comparison
    if isfield(fb_struct, 'baseline_metrics') && isstruct(fb_struct.baseline_metrics)
        diag_summary.baseline_metrics = struct();
        fn_b = fieldnames(fb_struct.baseline_metrics);
        for i = 1:min(numel(fn_b), 15)
            if isnumeric(fb_struct.baseline_metrics.(fn_b{i}))
                diag_summary.baseline_metrics.(fn_b{i}) = fb_struct.baseline_metrics.(fn_b{i});
            end
        end
    end

    % ── 4. Editable parameters (incumbent config) ──
    diag_summary.params = struct();
    if isstruct(inc_cfg) && ~isempty(fieldnames(inc_cfg))
        fn_p = fieldnames(inc_cfg);
        for i = 1:min(numel(fn_p), 60)
            if isnumeric(inc_cfg.(fn_p{i})) || islogical(inc_cfg.(fn_p{i}))
                diag_summary.params.(fn_p{i}) = inc_cfg.(fn_p{i});
            end
        end
    end

    % ── 5. Suggested directions ──
    suggestions = {};
    if fb_struct.stall_count >= 50
        suggestions{end+1} = 'CRITICAL: Very long stall. Consider radical structural changes.';
        suggestions{end+1} = '  - Toggle binary flags (observer, adaptation, adaptive horizon)';
        suggestions{end+1} = '  - Make >50% changes to key parameters';
    end
    if isfield(input_data, 'metrics') && isstruct(input_data.metrics)
        m = input_data.metrics;
        if isfield(m, 'integral_error') && isfield(fb_struct, 'baseline_metrics')
            bl = fb_struct.baseline_metrics;
            if isfield(bl, 'integral_error') && m.integral_error > bl.integral_error * 0.8
                suggestions{end+1} = 'Integral error still close to baseline — increase tracking gains.';
            end
        end
        if isfield(m, 'fault1_peak_error') && m.fault1_peak_error > 0.01
            suggestions{end+1} = 'Fault peak error elevated — increase RLS fault boost and reduce health decay.';
        end
    end
    diag_summary.suggestions = suggestions;

    % ── 6. Prompt override field ──
    diag_summary.prompt_override = '';
    diag_summary.prompt_override_instruction = ...
        'Set this to override the NEXT user prompt sent to the LLM. Leave empty to use default.';

    % ── 7. Instruction ──
    diag_summary.instruction = struct(...
        'to_edit', 'Edit the ''params'' fields above to set new parameter values. Set ''prompt_override'' to guide the next LLM call. Then DELETE THIS FILE to resume.', ...
        'to_skip', 'Delete this file without editing to continue with unmodified iteration.', ...
        'to_force_params', 'Changes to ''params'' are applied directly to the candidate (bypass LLM for this gen).', ...
        'to_guide_llm', 'Changes to ''prompt_override'' will replace the next user prompt.', ...
        'to_continue_normal', 'Delete this file to resume normal LLM-driven iteration.');

    % ── 8. Full prompt text ──
    diag_summary.llm_prompt = prompt_text;

    % ── Write intervention file ──
    fid = fopen(intervene_file, 'w', 'n', 'UTF-8');
    if fid > 0
        fprintf(fid, '%s', jsonencode(diag_summary, 'PrettyPrint', true));
        fclose(fid);
    end

    % ── Print to console ──
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  HUMAN INTERVENTION REQUIRED                                 ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Benchmark: %-16s  Role: %-4s  Gen: %-4d                   ║\n', ...
        benchmark_name, role, gen);
    fprintf('║  Stall count: %-3d  Best q: %-12.6f                          ║\n', ...
        fb_struct.stall_count, input_data.best_q);
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  File: %-52s ║\n', intervene_file);
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    for si = 1:numel(suggestions)
        fprintf('║  %-60s ║\n', suggestions{si});
    end
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Actions:                                                    ║\n');
    fprintf('║  1. Edit params in the JSON file                             ║\n');
    fprintf('║  2. Optionally set prompt_override for next LLM call         ║\n');
    fprintf('║  3. DELETE the file to resume                                ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\nWaiting for intervention file deletion: %s\n', intervene_file);

    % ── Wait for user ──
    pause_time = 0;
    while exist(intervene_file, 'file')
        pause(5);
        pause_time = pause_time + 5;
        if mod(pause_time, 60) == 0
            fprintf('  Still waiting... (%.0f min elapsed)\n', pause_time/60);
        end
    end
    fprintf('Intervention file deleted. Resuming...\n');

    % ── The file was deleted. Try reading user edits from .edited.json ──
    edited_file = strrep(intervene_file, '.json', '.edited.json');
    has_user_edits = exist(edited_file, 'file');

    if ~has_user_edits
        % User deleted without editing — do normal LLM call
        fprintf('No edits found. Calling LLM directly...\n');
        if exist(intervene_file, 'file'), delete(intervene_file); end
        % Fall through to normal LLM call below
    else
        % ── Read user edits ──
        try
            user_edit = jsondecode(fileread(edited_file));
        catch
            user_edit = struct();
        end

        % ── Build candidate from user edits ──
        candidate = inc_cfg;
        diag = struct('ok', true, 'msg', 'Human intervention', ...
            'source', 'human_intervention', ...
            'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0);

        if isfield(user_edit, 'params') && isstruct(user_edit.params)
            fn = fieldnames(user_edit.params);
            for i = 1:numel(fn)
                candidate.(fn{i}) = user_edit.params.(fn{i});
            end
            fprintf('Applied %d parameter overrides from human intervention.\n', numel(fn));
        end

        if isfield(user_edit, 'prompt_override') && ~isempty(user_edit.prompt_override)
            candidate.llm_intervention_prompt = user_edit.prompt_override;
            fprintf('Prompt override saved for next LLM call.\n');
        end

        candidate.name = sprintf('gen%03d_human_%s', gen, role);
        candidate.label = candidate.name;
        candidate.llm_source = 'human';

        % Clean up
        delete(edited_file);
        if exist(intervene_file, 'file'), delete(intervene_file); end
        return;
    end

    % ── Fallback: normal LLM call (user skipped intervention) ──
    py_exec = resolve_python_executable(py_script);
    cmd = sprintf('"%s" "%s" "%s" "%s" "%s" "%s"', ...
        py_exec, py_script, input_file, output_file, benchmark_name, role);
    [status, cmdout] = system(cmd);

    candidate = struct();
    diag = struct('ok', false, 'msg', '', 'source', 'post_intervention_fallback', ...
        'n_missing', 0, 'n_type_err', 0, 'n_clamped', 0);

    if exist(output_file, 'file')
        try
            out = jsondecode(fileread(output_file));
        catch
            out = struct();
            diag.msg = 'jsondecode failed after intervention';
            error('GPT:InvalidOutput', '%s', diag.msg);
        end
        if isfield(out, 'success') && ~out.success
            failure_msg = 'GPT generation failed after intervention';
            if isfield(out, 'error') && ~isempty(out.error)
                failure_msg = out.error;
            end
            error('GPT:GenerationFailed', '%s', failure_msg);
        end
        if isfield(out, 'config') && isstruct(out.config)
            spec = get_benchmark_spec(benchmark_name, varargin{:});
            [candidate, exec_diag] = parse_exec_validate(out.config, benchmark_name, inc_cfg, varargin{:});
            diag.n_missing=exec_diag.n_missing; diag.n_type_err=exec_diag.n_type_err;
            diag.n_clamped=exec_diag.n_clamped; diag.ok=exec_diag.ok;
            if ~isempty(exec_diag.msg), diag.msg=exec_diag.msg; end
            if isfield(out,'name'), candidate.name=out.name; end
            if isfield(out,'label'), candidate.label=out.label; end
            if isfield(out, 'llm_source') && ~isempty(out.llm_source)
                candidate.llm_source = out.llm_source;
                diag.source = [out.llm_source '_post_intervention'];
            else
                candidate.llm_source = 'model_backend';
                diag.source = 'model_backend_post_intervention';
            end
            if isfield(out, 'api_metadata') && isstruct(out.api_metadata)
                am = out.api_metadata;
                trace_fields = {'provider','transport', ...
                    'requested_model','response_model','request_id', ...
                    'http_status','prompt_tokens','completion_tokens', ...
                    'total_tokens','retry_count','parse_retry_count', ...
                    'api_calls','codex_exec_calls', ...
                    'codex_cli_version'};
                candidate_fields = {'llm_provider','llm_transport', ...
                    'llm_requested_model','llm_response_model', ...
                    'llm_request_id','llm_http_status','llm_prompt_tokens', ...
                    'llm_completion_tokens','llm_total_tokens','llm_retry_count', ...
                    'llm_parse_retry_count', ...
                    'llm_api_calls','llm_codex_exec_calls', ...
                    'llm_codex_cli_version'};
                for ai = 1:numel(trace_fields)
                    if isfield(am, trace_fields{ai})
                        diag.(trace_fields{ai}) = am.(trace_fields{ai});
                        candidate.(candidate_fields{ai}) = am.(trace_fields{ai});
                    end
                end
            end
            delete(input_file); delete(output_file);
            return;
        end
    end
    error('GPT:PythonFailed', 'GPT generation produced no valid output (exit=%d): %s', ...
        status, strtrim(cmdout));
end


function prompt_text = build_prompt_dry_run(input_file, output_file, py_script, benchmark_name, role)
% BUILD_PROMPT_DRY_RUN  Call llm_generate.py with --dry-run flag to get prompt text.
%  Returns the full system + user prompt without making an API call.

    dry_out = [output_file '_dry.json'];
    py_exec = resolve_python_executable(py_script);
    cmd = sprintf('"%s" "%s" "%s" "%s" "%s" "%s" --dry-run 2>/dev/null', ...
        py_exec, py_script, input_file, dry_out, benchmark_name, role);
    system(cmd);

    prompt_text = '';
    if exist(dry_out, 'file')
        try
            dry_data = jsondecode(fileread(dry_out));
            if isfield(dry_data, 'system_prompt')
                prompt_text = sprintf('=== SYSTEM PROMPT (P_sys) ===\n%s\n\n=== USER PROMPT (F^(t)) ===\n%s', ...
                    dry_data.system_prompt, dry_data.user_prompt);
            end
        catch
        end
        delete(dry_out);
    end

    if isempty(prompt_text)
        % Fallback: read from the input JSON directly
        try
            in_data = jsondecode(fileread(input_file));
            prompt_text = jsonencode(in_data, 'PrettyPrint', true);
        catch
            prompt_text = '(Could not reconstruct prompt)';
        end
    end
end


function py_exec = resolve_python_executable(py_script)
% Prefer the project-local virtual environment without modifying global Python.
    configured = getenv('GPT_PYTHON');
    if ~isempty(configured)
        py_exec = configured;
        return;
    end

    project_root = fileparts(fileparts(py_script));
    local_python = fullfile(project_root, '.venv', 'bin', 'python');
    if isfile(local_python)
        py_exec = local_python;
    else
        py_exec = 'python3';
    end
end
