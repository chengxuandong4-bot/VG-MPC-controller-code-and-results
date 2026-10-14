function cfg = config_lightweight_drone(mode)
%CONFIG_LIGHTWEIGHT_DRONE Drone TinyMPC benchmark config for unified loop.
%  mode: 'single' | 'dual_obj' | 'dual_strategy'
    if nargin < 1, mode = 'single'; end

    cfg = struct();
    cfg.name = 'lightweight_drone';
    cfg.benchmark_name = 'drone';
    cfg.mode = mode;

    cfg.budget_T = 200;
    if ~strcmp(mode, 'single'), cfg.budget_T = 100; end
    cfg.early_stop_N = 200;
    cfg.restart_interval = 50;
    cfg.epsilon = 0.005;

    cfg.root_dir = fileparts(mfilename('fullpath'));
    cfg.results_dir = fullfile(cfg.root_dir, sprintf('results_%s', mode));
    addpath(genpath(fullfile(cfg.root_dir, 'src')));
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'unified_loop')));

    % --- Benchmark function handles ---
    setup = init_drone_params();
    cfg.init_params          = @init_drone_params;
    cfg.run_baseline         = @(p) run_drone_sim(setup.baseline, setup);
    cfg.parse_and_validate   = @parse_drone_candidate;
    cfg.simulate_closed_loop = @(p, c) run_drone_sim(c, setup);
    cfg.compute_metrics      = @compute_drone_metrics;
    cfg.aggregate_score      = @aggregate_drone_score;
    cfg.compute_violations   = @compute_drone_violations;
    cfg.acceptance_gate      = @drone_acceptance;
    cfg.lambda_v = 0.5;

    % --- LLM-tuneable parameter list ---
    cfg.parameter_list = {'Q_diag', 'R_diag', 'horizon_N'};
    cfg.metric_names = {'IAE_pos', 'RMS_pos_error', 'max_pos_error', ...
        'final_pos_error', 'IAE_att', 'control_energy', ...
        'avg_solve_time_ms', 'max_solve_time_ms', 'saturation_rate'};
    cfg.metric_directions = {'lower','lower','lower','lower','lower', ...
        'lower','lower','lower','lower'};

    % --- Baseline candidate ---
    cfg.single.initial_candidate  = @(p, bl) setup.baseline;
    cfg.single.generate_candidate = @(inc, fb, p, t) drone_mutate(inc.c, fb, t);

    cfg.dual.generator_A = struct('role', 'performance_driven', ...
        'initial_candidate',  @(p, bl) setup.baseline, ...
        'generate_candidate', @(inc, fb, p, t, src) drone_mutate(inc.c, fb, t));
    cfg.dual.generator_B = struct('role', 'robustness_driven', ...
        'initial_candidate',  @(p, bl) setup.baseline, ...
        'generate_candidate', @(inc, fb, p, t, src) drone_mutate(inc.c, fb, t));

    % --- Default candidate ---
    cfg.get_default_candidate = @() setup.baseline;
end

function cfg = parse_drone_candidate(llm_output, role, template)
    cfg = template;
    if isfield(llm_output, 'config')
        raw = llm_output.config;
    else
        raw = llm_output;
    end
    if isfield(raw, 'Q_diag'),  cfg.Q_diag   = raw.Q_diag(:); end
    if isfield(raw, 'R_diag'),  cfg.R_diag   = raw.R_diag(:); end
    if isfield(raw, 'horizon_N'), cfg.horizon_N = raw.horizon_N; end
    cfg.Q_diag = max(0.01, min(100, cfg.Q_diag));
    cfg.R_diag = max(0.001, min(100, cfg.R_diag));
    cfg.horizon_N = max(5, min(50, round(cfg.horizon_N)));
end

function cand = drone_mutate(incumbent, feedback, gen)
    cand = incumbent;
    scale = 0.1;
    if isfield(feedback, 'stall_count') && feedback.stall_count >= 5
        scale = 0.3;
    end
    cand.Q_diag = incumbent.Q_diag .* (1 + scale * randn(12,1));
    cand.R_diag = incumbent.R_diag .* (1 + scale * randn(4,1));
    cand.horizon_N = max(5, min(50, round(incumbent.horizon_N + scale*5*randn)));
    cand.Q_diag = max(0.01, min(100, cand.Q_diag));
    cand.R_diag = max(0.001, min(100, cand.R_diag));
    cand.name = sprintf('drone_gen%03d_mutate', gen);
end
