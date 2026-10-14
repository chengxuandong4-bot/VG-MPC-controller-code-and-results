function cfg = config_fault_tolerant(mode)
    if nargin < 1, mode = 'single'; end
    cfg = struct(); cfg.name = 'fault_tolerant'; cfg.mode = mode;
    cfg.budget_T = 200; if ~strcmp(mode,'single'), cfg.budget_T = 100; end
    cfg.early_stop_N = 200; cfg.restart_interval = 50; cfg.epsilon = 0.005;
    cfg.root_dir = fileparts(mfilename('fullpath'));
    cfg.results_dir = fullfile(cfg.root_dir, sprintf('results_%s', mode));
    src_dir = fullfile(cfg.root_dir, 'src');
    addpath(src_dir);
    addpath(fullfile(src_dir, 'benchmark'));
    addpath(fullfile(src_dir, 'controllers'));
    addpath(fullfile(src_dir, 'evaluation'));
    addpath(fullfile(src_dir, 'snapshots'));

    cfg.init_params          = @init_auv_params;
    cfg.run_baseline         = @run_auv_baseline;
    cfg.parse_and_validate   = @parse_auv_candidate;
    cfg.simulate_closed_loop = @simulate_closed_loop;
    cfg.compute_metrics      = @compute_metrics;
    cfg.aggregate_score      = @aggregate_auv_score;
    cfg.compute_violations   = @compute_auv_violations;
    cfg.acceptance_gate      = @auv_acceptance;
    cfg.lambda_v = 1.0;

    cfg.build_feedback       = @auv_fb;
    cfg.build_error_feedback = @(diag) struct('text', sprintf('[AUV] Error: %s', diag), 'error_diag', diag);
    cfg.generate_report      = @(sum) fprintf('[AUV] best q=%.4f gen %d\n', sum.incumbent_q, sum.incumbent_gen);

    cfg.single.initial_candidate  = @(p, bl) bl.controller;
    cfg.single.generate_candidate = @(inc, fb, p, t) auv_select(inc, fb, p, t, 'explore');
    cfg.single.restart_mutate     = @(inc, p, stall) auv_select(inc, [], p, stall, 'restart');

    % Dual modes: role differentiation by mode
    if strcmp(mode, 'dual_obj')
        roleA = 'performance_driven'; roleB = 'robustness_driven';
    else  % dual_strategy
        roleA = 'local_exploit';      roleB = 'structural_explore';
    end

    cfg.dual.generator_A = struct('role', roleA, 'initial_candidate', @(p,bl)bl.controller, ...
        'generate_candidate', @(inc,fb,p,t,src) auv_select(inc,fb,p,t,src), ...
        'restart_mutate', @(inc,p,stall,src) auv_select(inc,[],p,stall,src));
    cfg.dual.generator_B = struct('role', roleB, 'initial_candidate', @(p,bl)bl.controller, ...
        'generate_candidate', @(inc,fb,p,t,src) auv_select(inc,fb,p,t,src), ...
        'restart_mutate', @(inc,p,stall,src) auv_select(inc,[],p,stall,src));
end


function fb = auv_fb(inc, m, s, q, diag, updated, varargin)
    fb = struct('text','','error_diag',diag);
    if isstruct(m), mi = m; else, mi = struct('integral_error',0); end
    if updated
        fb.text = sprintf('[AUV] NEW BEST q=%.4f integral=%.1f', q, mi.integral_error);
    else
        fb.text = sprintf('[AUV] q=%.4f integral=%.1f', q, mi.integral_error);
    end
end


function candidate = auv_select(incumbent, feedback, params, gen, role)
    persistent pool usage_count last_used_gen stall_counter
    if isempty(pool)
        pool = load_auv_pool();
        usage_count = zeros(1, length(pool));
        last_used_gen = zeros(1, length(pool));
        stall_counter = 0;
    end

    % Track stall for temperature scheduling
    if ~isempty(incumbent) && isfield(incumbent,'q') && incumbent.q > -inf
        stall_counter = stall_counter + 1;
    end

    % Temperature schedule
    if stall_counter < 10,      temp = 0.3;
    elseif stall_counter < 25,  temp = 0.7;
    else,                       temp = 1.5;
    end

    % --- Tier classification ---
    [~, sort_idx] = sort([pool.integral_error]);
    n = length(pool);
    tier1_end = round(n * 0.20);
    tier2_end = round(n * 0.70);
    tier = ones(1, n);  % tier 1
    tier(sort_idx(tier1_end+1:tier2_end)) = 2;
    tier(sort_idx(tier2_end+1:end)) = 3;

    % --- Role-specific: sigma, candidate indices, scoring ---
    all_idx = 1:n;

    switch role
        case {'performance_driven', 'performance'}
            cand_idx = all_idx;
            sigma = 0.03;
            % Tier preference: 60/35/5
            tier_weight = [0.60, 0.35, 0.05];

            for i = 1:n
                p = pool(i);
                scores(i) = -p.integral_error/1e4 - p.rms_error/3 - p.control_energy/1e4;
            end

        case {'robustness_driven', 'robustness'}
            cand_idx = all_idx;
            sigma = 0.03;
            tier_weight = [0.60, 0.35, 0.05];

            for i = 1:n
                p = pool(i);
                scores(i) = -scalar_metric(p.fault1_peak)*10 ...
                    - scalar_metric(p.fault2_peak)*10 ...
                    - scalar_metric(p.control_energy)/1e4;
            end

        case 'local_exploit'
            % Top 20% only, very small perturbation
            cand_idx = sort_idx(1:tier1_end);
            sigma = 0.01;
            tier_weight = [1.0, 0, 0];

            for i = 1:n
                p = pool(i);
                scores(i) = -scalar_metric(p.integral_error)/1e4 ...
                    - scalar_metric(p.fault1_peak)*5;
                if ~isempty(incumbent) && isfield(incumbent,'c')
                    scores(i) = scores(i) - 0.5 * norm(pool_config_diff(p, incumbent));
                end
            end

        case 'structural_explore'
            % Full pool, uniform-ish, higher perturbation
            cand_idx = all_idx;
            sigma = 0.08;
            tier_weight = [0.20, 0.50, 0.30];

            scores = 3.0 * randn(1, n);  % diversity-driven
            % Slight bonus for never-used configs
            scores = scores - 0.5 * usage_count;

        case 'restart'
            % Restart: expand to tier 2+3, increase sigma
            cand_idx = all_idx;
            sigma = min(0.08, 0.03 * 1.5^(stall_counter/15));
            temp = 1.5;
            tier_weight = [0.20, 0.50, 0.30];

            for i = 1:n
                p = pool(i);
                scores(i) = -scalar_metric(p.integral_error)/1e4 ...
                    - scalar_metric(p.fault1_peak)*5 + 2.0*randn();
            end

        otherwise
            cand_idx = all_idx;
            sigma = 0.03;
            tier_weight = [0.40, 0.40, 0.20];
            scores = randn(1, n);
    end

    % --- Usage penalty ---
    for i = 1:n
        scores(i) = scores(i) - 0.10 * usage_count(i);
        if last_used_gen(i) > gen - 10
            scores(i) = scores(i) - 0.20;
        end
    end

    % --- Tier weighting ---
    for i = 1:n
        scores(i) = scores(i) + log(tier_weight(tier(i)) + 0.001);
    end

    % --- Softmax with temperature ---
    scores = scores - max(scores);
    probs = exp(scores / temp);
    probs = probs / sum(probs);

    % --- Select ---
    idx = randsample(length(pool), 1, true, probs);
    usage_count(idx) = usage_count(idx) + 1;
    last_used_gen(idx) = gen;

    % --- Build candidate ---
    cfg_struct = pool(idx).config;
    cfg_struct = repair_auv_config(cfg_struct);
    cfg_struct.name = sprintf('gen%03d_%s_%s', gen, role, pool(idx).name);
    cfg_struct.label = cfg_struct.name;

    % Apply perturbation
    if isfield(cfg_struct,'kp_vel'), cfg_struct.kp_vel = cfg_struct.kp_vel(:)' .* exp(sigma*randn(1,2)); end
    if isfield(cfg_struct,'ki_vel'), cfg_struct.ki_vel = cfg_struct.ki_vel(:)' .* exp(sigma*randn(1,2)); end
    cfg_struct = repair_auv_config(cfg_struct);

    candidate = init_controller('adaptive_fta', params, cfg_struct);
    candidate.name = cfg_struct.name;
end


function value = scalar_metric(raw)
    values = double(raw(:));
    values = values(isfinite(values));
    if isempty(values)
        value = 0;
    else
        value = max(values);
    end
end


function d = pool_config_diff(pool_entry, incumbent)
    d = 0;
    if isfield(incumbent,'c') && isstruct(incumbent.c) && isfield(incumbent.c,'config')
        cfg = incumbent.c.config;
        pc = pool_entry.config;
        if isfield(cfg,'kp_vel') && isfield(pc,'kp_vel')
            d = norm(cfg.kp_vel(:) - pc.kp_vel(:));
        end
    end
end


function pool = load_auv_pool()
    persistent cached_pool
    if ~isempty(cached_pool), pool = cached_pool; return; end
    json_path = fullfile(fileparts(mfilename('fullpath')), 'auv_candidate_pool.json');
    data = jsondecode(fileread(json_path));
    pool = struct('name', {}, 'config', {}, 'integral_error', {}, 'rms_error', {}, ...
        'fault1_peak', {}, 'fault2_peak', {}, 'control_energy', {});
    for i = 1:numel(data)
        pool(i).name = data(i).name;
        pool(i).config = data(i).config;
        pool(i).integral_error = data(i).integral_error;
        pool(i).rms_error = data(i).rms_error;
        pool(i).fault1_peak = data(i).fault1_peak;
        pool(i).fault2_peak = data(i).fault2_peak;
        pool(i).control_energy = data(i).control_energy;
    end
    cached_pool = pool;
end


function params = init_auv_params()
    params = paper_auv5_params();
end

function result = run_auv_baseline(params)
    ctrl = init_controller('paper_is_sat', params, struct('name','paper_is_sat','label','Baseline'));
    result = simulate_closed_loop(params, ctrl);
    result.controller = ctrl;
end

function [ok, diag_str] = parse_auv_candidate(candidate)
    ok = true; diag_str = '';
    if ~isstruct(candidate), ok=false; diag_str='Not a struct'; end
end

function m = metrics_struct_to_vec(met)
    m = [0,0,0,0,0];
    if isstruct(met)
        if isfield(met,'integral_error'), m(1)=met.integral_error; end
        if isfield(met,'rms_tracking_error'), m(2)=met.rms_tracking_error; end
        if isfield(met,'control_energy'), m(3)=met.control_energy; end
        if isfield(met,'fault1_peak_error'), m(4)=met.fault1_peak_error; end
        if isfield(met,'fault2_peak_error'), m(5)=met.fault2_peak_error; end
    end
end

function s = aggregate_auv_score(m_raw)
    if isstruct(m_raw), m = metrics_struct_to_vec(m_raw); else, m = m_raw; end
    if any(isinf(m)|isnan(m)), s=-inf; return; end
    w=[0.3,0.2,0.15,0.15,0.2]; phi=max(0,1-m./(1+m)); s=sum(w.*phi);
end

function v = compute_auv_violations(result, params)
    v=0;
    if isfield(result,'failed')&&result.failed, v=100; end
end

function accepted = auv_acceptance(m, result, params)
    accepted = ~(isfield(result,'failed')&&result.failed);
    if isfield(result,'metrics')
        if isfield(result.metrics,'failed')&&result.metrics.failed, accepted=false; end
        if result.metrics.integral_error > 1e6, accepted=false; end
    end
end
