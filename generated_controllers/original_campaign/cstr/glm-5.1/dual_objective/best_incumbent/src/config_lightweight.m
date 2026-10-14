function cfg = config_lightweight(mode)
    if nargin < 1, mode = 'single'; end
    cfg = struct(); cfg.name = 'lightweight'; cfg.mode = mode;
    cfg.benchmark_name = 'cstr';

    cfg.budget_T = 200; if ~strcmp(mode,'single'), cfg.budget_T = 100; end
    cfg.early_stop_N = 200; cfg.restart_interval = 50; cfg.epsilon = 0.005;
    cfg.root_dir = fileparts(mfilename('fullpath'));
    cfg.results_dir = fullfile(cfg.root_dir, sprintf('results_%s', mode));
    addpath(genpath(fullfile(cfg.root_dir, 'src')));
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'unified_loop')));

    % === First-round generation pool ===
    % These are the hand-crafted configs from the first round's generation_configs()
    gen_pool = build_cstr_generation_pool();

    % === Benchmark function handles ===
    cfg.init_params          = @init_cstr_params;
    cfg.run_baseline         = @(p) run_generation_experiment(gen_pool{1}, p.setup);
    cfg.parse_and_validate   = @parse_cstr_candidate;
    cfg.simulate_closed_loop = @(p, c) run_generation_experiment(c, p.setup);
    cfg.compute_metrics      = @compute_cstr_metrics;
    cfg.aggregate_score      = @aggregate_cstr_score;
    cfg.compute_violations   = @compute_cstr_violations;
    cfg.acceptance_gate      = @cstr_acceptance;
    cfg.lambda_v = 1.0;

    % --- Parameter lists for role-conditioned generation ---
    cfg.tracking_params = {'Wy_base','Wy_gain','Wu_smooth_gain'};
    cfg.safety_params   = {'hessian_reg','Wdu_base','Wdu_gain'};
    cfg.penalty_params  = {'Wu','Wdu_base'};
    cfg.structural_params = {'use_adaptive_horizon','use_disturbance_observer','use_param_adaptation'};

    % --- Feedback mapping: metric→{params, directions, gains} ---
    cfg.feedback_mapping = struct(...
        'IAE',        struct('params',{{'Wy_base','horizon_max'}},     'directions',[1,1],  'gains',[5,3]),...
        'RMS_error',  struct('params',{{'Wy_base','Wy_gain'}},         'directions',[1,1],  'gains',[4,3]),...
        'ctrl_energy',struct('params',{{'Wu','Wdu_base'}},             'directions',[1,1],  'gains',[3,2]),...
        'cond_num',   struct('params',{{'hessian_reg'}},               'directions',[1],    'gains',[8]),...
        'KKT_resid',  struct('params',{{'hessian_reg','Wu'}},          'directions',[1,1],  'gains',[10,2])...
    );
    cfg.metric_names = {'IAE','RMS_error','max_error','final_error','ctrl_energy','solve_time','cond_num','KKT_resid'};
    cfg.metric_directions = {'lower','lower','lower','lower','lower','lower','lower','lower'};

    % --- Scenarios ---
    cfg.scenario_set_train = @(p) generate_cstr_scenarios(1:5, p);
    cfg.scenario_set_test  = @(p) generate_cstr_scenarios(101:108, p);

    cfg.get_parameter_list = @(role) get_cstr_param_list(role);
    cfg.get_structural_params = @() cfg.structural_params;
    cfg.get_safety_params = @() cfg.safety_params;
    cfg.get_tracking_params = @() cfg.tracking_params;
    cfg.get_penalty_params = @() cfg.penalty_params;
    cfg.get_default_candidate = @() gen_pool{1};
    cfg.get_baseline_param = @(pn) getfield(gen_pool{1}, pn);

    % === Feedback ===
    cfg.build_feedback       = @cstr_feedback;
    cfg.build_error_feedback = @(diag) struct('text', sprintf('[CSTR] Error: %s', diag), 'error_diag', diag);
    cfg.generate_report      = @(sum) fprintf('[CSTR] Report: best q=%.4f at gen %d\n', sum.incumbent_q, sum.incumbent_gen);

    % === Candidate generators: pick from first-round pool with feedback bias ===
    cfg.single.initial_candidate  = @(p, bl) gen_pool{1};
    cfg.single.generate_candidate = @(inc, fb, p, t) pick_from_pool(gen_pool, inc, fb, t, 'explore');
    cfg.single.restart_mutate     = @(inc, p, stall) pick_from_pool(gen_pool, inc, [], stall, 'restart');

    cfg.dual.generator_A = struct('role','performance_driven', ...
        'initial_candidate',@(p,bl) gen_pool{1}, ...
        'generate_candidate',@(inc,fb,p,t,src) pick_from_pool(gen_pool,inc,fb,t,'performance'), ...
        'restart_mutate',@(inc,p,stall,src) pick_from_pool(gen_pool,inc,[],stall,'restart'), ...
        'objective_profile',struct('tracking',0.45,'energy',0.15,'violation',0.10,'safety',0.10,'numerical',0.10,'smoothness',0.10), ...
        'search_strategy',struct('mutation_scale',0.05,'structure_freeze',true,'samples',5,'internal_select',true));

    cfg.dual.generator_B = struct('role','robustness_driven', ...
        'initial_candidate',@(p,bl) gen_pool{1}, ...
        'generate_candidate',@(inc,fb,p,t,src) pick_from_pool(gen_pool,inc,fb,t,'robustness'), ...
        'restart_mutate',@(inc,p,stall,src) pick_from_pool(gen_pool,inc,[],stall,'restart'), ...
        'objective_profile',struct('tracking',0.20,'energy',0.10,'violation',0.25,'safety',0.25,'numerical',0.10,'smoothness',0.10), ...
        'search_strategy',struct('mutation_scale',0.20,'structure_freeze',false,'samples',1,'internal_select',false));
end

% ===================================================================
% FIRST-ROUND GENERATION POOL
% ===================================================================

function pool = build_cstr_generation_pool()
    % Build from the first-round generation_configs() logic.
    % Each entry is a controller config that run_generation_experiment accepts.
    pool = {};

    % Baseline
    pool{end+1} = make_cfg('baseline', 'QR-AMPC Baseline', false, false, false, 20, 1.0, 0, 1e-4, 1e-3, 0, 1e-6);

    % Disturbance observer variants (innovation from first-round mechanism A)
    for obs_gain = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]
        pool{end+1} = make_cfg(sprintf('obs_g%.1f', obs_gain), 'Disturbance Observer', ...
            false, true, false, 20, 2.0, 0.3, 1e-4, 5e-4, 0.2, 1e-4);
    end

    % Adaptive horizon variants (mechanism B)
    for hmax = [15, 18, 22, 25, 28, 30]
        pool{end+1} = make_cfg(sprintf('ahor%d', hmax), 'Adaptive Horizon', ...
            true, false, false, hmax, 1.5, 0.4, 1e-4, 1e-3, 0.3, 1e-5);
    end

    % Parameter adaptation variants
    for ada = [true, false]
        pool{end+1} = make_cfg(sprintf('param%d', ada), 'Parameter Adaptation', ...
            false, true, ada, 20, 2.0, 0.5, 1e-4, 8e-4, 0.3, 5e-5);
    end

    % Regularization variants (mechanism C)
    for reg = [1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4]
        pool{end+1} = make_cfg(sprintf('reg%.0e', reg), 'Regularization', ...
            false, true, true, 20, 2.0, 0.5, 1e-4, 5e-4, 0.3, reg);
    end

    % Weight sweep
    for wy = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        for wdu = [1e-4, 5e-4, 1e-3, 2e-3]
            pool{end+1} = make_cfg(sprintf('w%.0f_d%.0e', wy, wdu), 'Weight Sweep', ...
                false, true, false, 20, wy, 0.3, 1e-4, wdu, 0.2, 1e-5);
        end
    end

    % Combined innovations (mechanism A+B+C together)
    pool{end+1} = make_cfg('final_innov', 'Final Innovative', true, true, true, 25, 3.0, 0.6, 5e-5, 3e-4, 0.5, 5e-6);
    pool{end+1} = make_cfg('final_v2', 'Final v2', true, true, true, 22, 2.5, 0.5, 5e-5, 5e-4, 0.4, 1e-6);

    % Fine-tuned combos around best
    for i = 1:30
        h = randi([15, 28]);
        wy = 1.0 + 3.0 * rand();
        wdu = 1e-4 * 10^(3*rand());
        reg = 1e-8 * 10^(5*rand());
        pool{end+1} = make_cfg(sprintf('tune%02d', i), 'Fine-tuned', ...
            rand()>0.5, rand()>0.3, rand()>0.5, h, wy, 0.4*rand(), 5e-5, wdu, 0.3*rand(), reg);
    end

    fprintf('[CSTR] Generation pool: %d configs\n', length(pool));
end


function cfg = make_cfg(name, label, use_ah, use_obs, use_ada, hmax, wy, wy_gain, wu, wdu, wdu_gain, reg)
    cfg.name = name; cfg.label = label;
    cfg.use_adaptive_horizon = use_ah; cfg.use_disturbance_observer = use_obs;
    cfg.use_param_adaptation = use_ada; cfg.use_input_bias_adaptation = false;
    cfg.horizon_max = hmax; cfg.horizon_mid = max(10, hmax-5); cfg.horizon_min = max(5, hmax-10);
    cfg.horizon_high_err = 0.5; cfg.horizon_mid_err = 0.1;
    cfg.int_ref_gain = 0.0; cfg.int_e_bound = 0.5;
    cfg.Wy_base = wy; cfg.Wy_gain = wy_gain; cfg.Wy_cap = 100;
    cfg.Wu = wu; cfg.Wdu_base = wdu; cfg.Wdu_gain = wdu_gain; cfg.Wdu_cap = 100;
    cfg.hessian_reg = reg;
    cfg.obs_alpha = 0.3; cfg.obs_bound = 0.5;
    cfg.gamma_k0 = 0.1; cfg.theta_k0_max = 1.5; cfg.theta_k0_min = 0.5;
    cfg.gamma_ua = 0.1; cfg.theta_ua_max = 1.5; cfg.theta_ua_min = 0.5;
    cfg.adapt_eps = 0.01;
    cfg.u_bias_gain = 0.1; cfg.u_bias_int_gain = 0.01; cfg.u_bias_bound = 0.5;
end

% ===================================================================
% FEEDBACK-DIRECTED POOL SELECTION
% ===================================================================

function candidate = pick_from_pool(pool, incumbent, feedback, gen, role)
    % Pick from the pre-built generation pool, biased by feedback.
    % This replaces random perturbation with intelligent selection
    % from a hand-crafted config space.

    n = length(pool);

    if gen <= 2 || isempty(feedback) || ~isstruct(feedback) || ~isfield(feedback,'text')
        % Early generations: diverse exploration
        idx = mod(gen-1, n) + 1;
        candidate = pool{idx};
        candidate.name = sprintf('gen%03d_%s_%s', gen, role, pool{idx}.name);
        return;
    end

    % Feedback-directed: prefer configs matching the feedback direction
    scores = zeros(1, n);
    for i = 1:n
        cfg = pool{i};
        % Baseline: prefer configs similar to incumbent for exploit roles
        scores(i) = 0.01 * rand();  % small random component

        if strcmp(role, 'performance')
            if cfg.Wy_base > 1.5, scores(i) = scores(i) + 0.3; end
            if cfg.use_adaptive_horizon, scores(i) = scores(i) + 0.2; end
            if cfg.horizon_max >= 22, scores(i) = scores(i) + 0.1; end
        elseif strcmp(role, 'robustness')
            if cfg.hessian_reg >= 1e-5, scores(i) = scores(i) + 0.3; end
            if cfg.Wdu_base >= 5e-4, scores(i) = scores(i) + 0.2; end
            if cfg.Wy_base < 3.0, scores(i) = scores(i) + 0.1; end
        elseif strcmp(role, 'local_exploit')
            scores(i) = scores(i) + similarity_score(cfg, pool{1}) + 0.05*rand();
        elseif strcmp(role, 'structural_explore')
            scores(i) = scores(i) + 1.0*rand();
        elseif strcmp(role, 'explore')
            scores(i) = scores(i) + 0.5 * rand();
        else
            scores(i) = scores(i) + similarity_score(cfg, pool{1});
        end

        % If feedback says "LOST", boost configs that address the weakness
        if contains(feedback.text, 'LOST')
            if contains(feedback.text, 'tracking') && cfg.Wy_base > 2.0
                scores(i) = scores(i) + 0.4;
            end
            if contains(feedback.text, 'numerical') && cfg.hessian_reg >= 1e-4
                scores(i) = scores(i) + 0.4;
            end
            if contains(feedback.text, 'energy') && cfg.Wu >= 5e-4
                scores(i) = scores(i) + 0.4;
            end
        end
    end

    % Weighted random selection (softmax-like)
    scores = exp(scores - max(scores));
    probs = scores / sum(scores);
    idx = randsample(n, 1, true, probs);
    candidate = pool{idx};
    candidate.name = sprintf('gen%03d_%s_%s', gen, role, pool{idx}.name);
end


function s = similarity_score(a, b)
    s = 0;
    if a.use_adaptive_horizon == b.use_adaptive_horizon, s = s + 0.2; end
    if a.use_disturbance_observer == b.use_disturbance_observer, s = s + 0.2; end
    if a.use_param_adaptation == b.use_param_adaptation, s = s + 0.2; end
    if a.horizon_max == b.horizon_max, s = s + 0.1; end
    s = s + 0.3 * exp(-abs(a.Wy_base - b.Wy_base)/2);
end


% ===================================================================
% BENCHMARK-SPECIFIC FUNCTIONS
% ===================================================================

function params = init_cstr_params()
    setup = struct();
    setup.nominal_params.F=1; setup.nominal_params.V=1; setup.nominal_params.R=1.985875;
    setup.nominal_params.dH=-5960; setup.nominal_params.E=11843; setup.nominal_params.k0=34393800;
    setup.nominal_params.rhoCp=500; setup.nominal_params.UA=150; setup.nominal_params.CAf=10;
    setup.nominal_params.Tf=298.15;
    setup.Ts=0.5; setup.preview=5; setup.t_end=60; setup.Nsim=round(setup.t_end/setup.Ts);
    setup.cons.u_min=285.15; setup.cons.u_max=312.15; setup.cons.du_min=-2; setup.cons.du_max=2;
    t_ref=(0:(setup.Nsim+setup.preview))*setup.Ts;
    ca_ref=8*ones(size(t_ref)); ca_ref(t_ref>=10 & t_ref<30)=2; ca_ref(t_ref>=30 & t_ref<50)=8; ca_ref(t_ref>=50)=5;
    setup.ca_ref=ca_ref;
    setup.k0_scale_amp=0.20; setup.k0_step_1=0.15; setup.k0_step_2=-0.10;
    setup.ua_scale_amp=0.15; setup.caf_amp=0.40; setup.caf_step=0.80; setup.tf_amp=5.0; setup.tf_step=-3.5;
    params.setup = setup;
end

function [ok, diag_str] = parse_cstr_candidate(candidate)
    ok = true; diag_str = '';
    try
        if ~isstruct(candidate), ok=false; diag_str='Not a struct'; return; end
        if ~isfield(candidate,'use_adaptive_horizon'), ok=false; diag_str='Missing config fields'; return; end
    catch ME, ok=false; diag_str=getReport(ME,'basic'); end
end

function m = compute_cstr_metrics(result, params)
    if isfield(result,'failed') && result.failed, m=inf(1,8); return; end
    r=result.metrics;
    m=[r.integral_error, r.rms_tracking_error, r.max_tracking_error, r.final_tracking_error, ...
       r.control_energy, r.avg_controller_time_ms, r.median_condition_number, r.max_kkt_residual];
end

function s = aggregate_cstr_score(m)
    if any(isinf(m)|isnan(m)), s=-inf; return; end
    phi=max(0,1-m./(1+m)); w=ones(1,length(m))/length(m); s=sum(w.*phi);
end

function v = compute_cstr_violations(result, params)
    v=0; if isfield(result,'failed')&&result.failed, v=100; end
end

function accepted = cstr_acceptance(m, result, params)
    accepted = ~(isfield(result,'failed')&&result.failed);
    if isfield(result,'metrics')
        if result.metrics.max_kkt_residual > 1e8, accepted=false; end
        if result.metrics.median_condition_number > 1e6, accepted=false; end
    end
end

function fb = cstr_feedback(inc, m, s, q, diag, updated, varargin)
    fb=struct("text","","error_diag",diag);
    if isempty(inc)||~isstruct(inc), return; end
    if updated, fb.text=sprintf("[CSTR] NEW BEST q=%.4f gen %d",q,inc.g);
    else, fb.text=sprintf("[CSTR] q=%.4f vs best=%.4f",q,inc.q); end
end

function scenarios = generate_cstr_scenarios(ids, params)
    scenarios=struct();  % Placeholder for multi-profile scenarios
end

function params = get_cstr_param_list(role)
    if strcmp(role,'performance'), params={'Wy_base','Wy_gain','horizon_max','Wu'};
    else, params={'hessian_reg','Wdu_base','Wdu_gain','Wu','Wy_base','horizon_max'}; end
end
