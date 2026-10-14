function cfg = config_adaptive(mode)
    if nargin < 1, mode = 'single'; end
    cfg = struct(); cfg.name = 'adaptive'; cfg.mode = mode;
    cfg.budget_T = 200; if ~strcmp(mode,'single'), cfg.budget_T = 100; end
    cfg.early_stop_N = 200; cfg.restart_interval = 50; cfg.epsilon = 0.005;
    cfg.root_dir = fileparts(mfilename('fullpath'));
    cfg.results_dir = fullfile(cfg.root_dir, sprintf('results_%s', mode));
    addpath(fullfile(cfg.root_dir, 'src'));

    cfg.init_params          = @init_rampc_params;
    cfg.run_baseline         = @run_rampc_baseline;
    cfg.parse_and_validate   = @parse_rampc_candidate;
    cfg.simulate_closed_loop = @simulate_rampc;
    cfg.compute_metrics      = @compute_rampc_metrics;
    cfg.aggregate_score      = @aggregate_rampc_score;
    cfg.compute_violations   = @compute_rampc_violations;
    cfg.acceptance_gate      = @rampc_acceptance;
    cfg.lambda_v = 1.0;

    cfg.build_feedback       = @(inc, m, s, q, diag, updated, varargin) ...
        struct('text',sprintf('[RAMPC] q=%.4f',q),'error_diag',diag);
    cfg.build_error_feedback = @(diag) struct('text', sprintf('[RAMPC] Error: %s', diag), 'error_diag', diag);
    cfg.generate_report      = @(sum) fprintf('[RAMPC] best q=%.4f gen %d\n', sum.incumbent_q, sum.incumbent_gen);

    % Build parameter-sweep pool
    cfg.rampc_pool = build_rampc_pool();

    cfg.single.initial_candidate  = @(p, bl) cfg.rampc_pool{1};
    cfg.single.generate_candidate = @(inc, fb, p, t) rampc_select(cfg.rampc_pool, inc, fb, t, 'explore');
    cfg.single.restart_mutate     = @(inc, p, stall) rampc_select(cfg.rampc_pool, inc, [], stall, 'restart');

    if strcmp(mode, 'dual_obj')
        roleA = 'performance_driven'; roleB = 'robustness_driven';
    else
        roleA = 'local_exploit';      roleB = 'structural_explore';
    end

    cfg.dual.generator_A = struct('role', roleA, 'initial_candidate', @(p,bl) cfg.rampc_pool{1}, ...
        'generate_candidate', @(inc,fb,p,t,src) rampc_select(cfg.rampc_pool,inc,fb,t,src), ...
        'restart_mutate', @(inc,p,stall,src) rampc_select(cfg.rampc_pool,inc,[],stall,src));
    cfg.dual.generator_B = struct('role', roleB, 'initial_candidate', @(p,bl) cfg.rampc_pool{1}, ...
        'generate_candidate', @(inc,fb,p,t,src) rampc_select(cfg.rampc_pool,inc,fb,t,src), ...
        'restart_mutate', @(inc,p,stall,src) rampc_select(cfg.rampc_pool,inc,[],stall,src));
end


function pool = build_rampc_pool()
    pool = {};
    idx = 0;
    % Sweep: Q_pos, R_vel, safety_margin, v_max
    Q_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    R_vals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0];
    S_vals = [0.10, 0.20, 0.30, 0.50, 0.70];
    V_vals = [0.6, 0.9, 1.2, 1.5, 2.0];

    for q = Q_vals
        for r = R_vals
            for s = S_vals
                for v = V_vals
                    idx = idx + 1;
                    cfg = struct();
                    cfg.name = sprintf('r%03d_q%.0f_r%.2f_s%.2f_v%.1f', idx, q, r, s, v);
                    cfg.Q_position = q; cfg.Q_heading = q/3; cfg.Q_velocity = q/6;
                    cfg.R_linear_vel = r; cfg.R_angular_vel = r/2;
                    cfg.terminal_weight = q;
                    cfg.safety_margin = s;
                    cfg.obs_repulsion_w = 10 + 30*s;
                    cfg.slack_penalty = 20 + 80*s;
                    cfg.max_linear_vel = v; cfg.max_angular_vel = min(v, 1.5);
                    cfg.tube_radius = 0.05 + 0.15*s;
                    cfg.adaptive_gain = 0.5 + 2.0*s;
                    cfg.disturbance_bound = 0.05;
                    cfg.warm_start_on = (mod(idx,3) ~= 0);
                    cfg.constraint_soft_on = (mod(idx,2) == 0);
                    pool{end+1} = cfg;
                end
            end
        end
    end
    fprintf('[RAMPC] Built pool: %d configs\n', length(pool));
end


function candidate = rampc_select(pool, incumbent, feedback, gen, role)
    persistent usage last_used stall_count
    if isempty(usage), usage = zeros(1,length(pool)); last_used = zeros(1,length(pool)); stall_count = 0; end

    n = length(pool);
    scores = zeros(1, n);

    switch role
        case 'performance_driven'
            for i = 1:n
                c = pool{i};
                scores(i) = c.Q_position/10 - c.R_linear_vel/2 + c.max_linear_vel/3;
            end
        case 'robustness_driven'
            for i = 1:n
                c = pool{i};
                scores(i) = c.safety_margin*3 + c.obs_repulsion_w/30 - c.max_linear_vel/3;
            end
        case 'local_exploit'
            for i = 1:n
                c = pool{i};
                scores(i) = -rand()*0.5;
            end
        case 'structural_explore'
            scores = randn(1, n) * 3;
        case 'restart'
            scores = randn(1, n) * 2;
        otherwise
            scores = randn(1, n);
    end

    % Usage penalty
    for i = 1:n
        scores(i) = scores(i) - 0.1*usage(i);
        if last_used(i) > gen-10, scores(i) = scores(i) - 0.2; end
    end

    scores = scores - max(scores);
    probs = exp(scores) / sum(exp(scores));
    idx = randsample(n, 1, true, probs);
    usage(idx) = usage(idx) + 1; last_used(idx) = gen;
    candidate = pool{idx};
    candidate.name = sprintf('gen%03d_%s_%s', gen, role, candidate.name);
end


function params = init_rampc_params()
    params.goal = [3.0; 0.0]; params.x0 = [0;0;0;0;0];
    params.goal_tol = 0.5; params.d_safe_min = 0.05;
    params.N_sim = 200; params.Ts = 0.05; params.N_pred = 10;
    params.obstacles = struct('center', [2.0; 1.0], 'radius', 0.5);
end

function result = run_rampc_baseline(params)
    base = struct('Q_position',3,'Q_heading',1,'Q_velocity',0.5,'R_linear_vel',0.3,...
        'R_angular_vel',0.15,'terminal_weight',5,'safety_margin',0.3,...
        'obs_repulsion_w',15,'max_linear_vel',1.2,'max_angular_vel',1.0);
    result = simulate_rampc_casadi(params, base);
end

function [ok, diag_str] = parse_rampc_candidate(candidate)
    ok = isstruct(candidate); diag_str = '';
    if ~ok, diag_str = 'Not a struct'; end
end

function result = simulate_rampc(params, candidate)
    result = simulate_rampc_casadi(params, candidate);
end

function result = simulate_rampc_casadi(params, candidate)
    % Write candidate config to tmp
    tmp_dir = fileparts(mfilename('fullpath'));
    cfg_file = fullfile(tmp_dir, 'tmp_cfg.json');
    out_file = fullfile(tmp_dir, 'tmp_res.json');

    % Build config for Python: candidate + embedded params
    c = candidate;
    c.N_sim_ = params.N_sim;
    c.Ts_ = params.Ts;
    c.N_pred_ = params.N_pred;
    c.goal_ = params.goal;
    c.x0_ = params.x0;
    c.obs_center_ = params.obstacles.center;
    c.obs_radius_ = params.obstacles.radius;

    write_json_rampc(cfg_file, c);

    % Call Python CasADi simulator
    py_script = fullfile(fileparts(mfilename('fullpath')), 'src', 'rampc_casadi_sim.py');
    project_root = fileparts(fileparts(mfilename('fullpath')));
    py_exec = fullfile(project_root, '.venv', 'bin', 'python');
    if ~isfile(py_exec), py_exec = 'python3'; end
    cmd = sprintf('"%s" "%s" "%s" "%s"', py_exec, py_script, cfg_file, out_file);
    [status, cmdout] = system(cmd);
    if status ~= 0
        error('GPT:RAMPCPythonFailed', 'RAMPC Python failed (exit=%d): %s', ...
            status, strtrim(cmdout));
    end

    % Read result
    result = struct();
    if exist(out_file, 'file')
        r = jsondecode(fileread(out_file));
        result.metrics = [r.cost, r.state_energy, r.control_energy, r.min_clearance, r.plan_width];
        result.collision = r.collision;
        result.reached_goal = r.goal_reached;
        result.solver_ok = r.solver_ok;
        result.max_saturation = r.max_saturation;
        delete(cfg_file); delete(out_file);
    else
        result.metrics = [inf, inf, inf, 0, inf];
        result.collision = true; result.reached_goal = false;
        result.solver_ok = false; result.max_saturation = 1.0;
    end
end

function write_json_rampc(path, data)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    if fid < 1, return; end
    try, fprintf(fid, '%s', jsonencode(data)); catch, end
    fclose(fid);
end

function m = compute_rampc_metrics(result, params)
    m = result.metrics; if m(4)<0, m(4)=0; end
end

function s = aggregate_rampc_score(m)
    if any(isinf(m)|isnan(m)), s=-inf; return; end
    w=[0.30,0.15,0.15,0.25,0.15];
    phi1=max(0,1-m(1)/(m(1)+10));
    phi2=max(0,1-m(2)/(m(2)+20));
    phi3=max(0,1-m(3)/(m(3)+10));
    phi4=min(1,m(4)/(m(4)+0.5));
    phi5=max(0,1-m(5));
    s=sum(w.*[phi1,phi2,phi3,phi4,phi5]);
end

function v = compute_rampc_violations(result, params)
    v = 0;
    if isfield(result,'collision')&&result.collision, v=v+500; end
    if isfield(result,'reached_goal')&&~result.reached_goal, v=v+500; end
end

function accepted = rampc_acceptance(m, result, params)
    accepted = true;
    if isfield(result,'collision')&&result.collision, accepted=false; end
    if m(4) < params.d_safe_min, accepted=false; end
end
