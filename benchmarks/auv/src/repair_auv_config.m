function cfg = repair_auv_config(cfg)
% REPAIR_AUV_CONFIG  Complete schema repair for all adaptive_fta fields.
%  Ensures no field is missing before init_controller is called.

    D = auv_full_defaults();

    % Fill ALL missing fields
    for fn = fieldnames(D)'
        if ~isfield(cfg, fn{1})
            cfg.(fn{1}) = D.(fn{1});
        end
    end

    % Force vector dimensions
    cfg.kp_vel = force_vec2(cfg.kp_vel, D.kp_vel);
    cfg.ki_vel = force_vec2(cfg.ki_vel, D.ki_vel);
    cfg.int_vel_limit = force_vec2(cfg.int_vel_limit, D.int_vel_limit);
    cfg.ks_tau = force_vec3(cfg.ks_tau, D.ks_tau);

    % Clamp critical scalars
    cfg.rls_cov0 = clamp(cfg.rls_cov0, 0.01, 1e6);
    cfg.rls_forgetting = clamp(cfg.rls_forgetting, 0.90, 0.9999);
    cfg.rls_regularization = clamp(cfg.rls_regularization, 1e-9, 1e-1);
    cfg.health_decay = clamp(cfg.health_decay, 0.001, 0.999);
    cfg.health_min = clamp(cfg.health_min, 0.01, 0.8);
    cfg.health_max = clamp(cfg.health_max, cfg.health_min+0.05, 1.5);
    cfg.allocation_reg = clamp(cfg.allocation_reg, 1e-8, 0.5);
    cfg.estimation_input_floor = clamp(cfg.estimation_input_floor, 1e-6, 0.5);
    cfg.smoothing_weight = clamp(cfg.smoothing_weight, 0, 1);
    cfg.measurement_filter_alpha = clamp(cfg.measurement_filter_alpha, 0.1, 0.99);
    cfg.rate_limit = clamp(cfg.rate_limit, 0.05, 50);
end


function D = auv_full_defaults()
    D = struct();

    % Velocity control
    D.kp_vel = [0.8, 0.8]; D.ki_vel = [0.1, 0.1]; D.int_vel_limit = [2.0, 2.0];
    D.int_leak_vel = 0.001; D.int_leak_psi = 0.001; D.int_leak_r = 0.001;

    % Heading/rate control
    D.k_psi = 1.5; D.ki_psi = 0.05; D.int_psi_limit = 1.0;
    D.k_r = 0.5; D.ki_r = 0.02; D.int_r_limit = 0.5;
    D.r_cmd_limit = 1.0;

    % Thrust allocation
    D.ks_tau = [0.3, 0.3, 0.3]; D.rate_limit = 2.0;
    D.allocation_reg = 0.01; D.allocation_reg_min = 0.001;

    % RLS estimation
    D.rls_cov0 = 10.0; D.rls_forgetting = 0.98; D.rls_regularization = 1e-4;
    D.rls_forgetting_min = 0.95; D.rls_forgetting_max = 0.999;
    D.rls_fault_boost = 0.02; D.use_adaptive_forgetting = true;

    % Health monitoring
    D.health_decay = 0.995; D.health_max = 1.2; D.health_min = 0.3;
    D.health_weight_scale = 2.0;

    % Measurement & estimation
    D.measurement_filter_alpha = 0.7; D.estimation_input_floor = 0.05;
    D.smoothing_weight = 0.3; D.innov_norm_ref = 0.1;

    % Event trigger
    D.use_event_trigger = true; D.event_innov_threshold = 0.15;
    D.event_perf_threshold = 0.1; D.max_hold_steps = 30; D.force_refresh_steps = 120;

    % Performance boost
    D.use_perf_boost = false; D.perf_rho0 = 0.9; D.perf_rho_inf = 0.3;
    D.perf_decay = 0.02; D.perf_boost_threshold = 0.3;
    D.perf_boost_slope = 5.0; D.perf_boost_max = 3.0;

    % Allocation relax
    D.use_alloc_relax = false; D.alloc_relax_gain = 0.5;
    D.rate_limit_boost = 3.0; D.rate_limit_shock = 1.5;
    D.shock_rate_gain = 0.5; D.shock_threshold = 0.8; D.shock_relax_gain = 0.2;

    % Residual compensation
    D.use_residual_comp = false; D.residual_comp_gain = 0.3;
    D.residual_filter = 0.5;

    % Fault feedforward
    D.use_fault_feedforward = false; D.fault_feedforward_gain = 0.5;
end


function v = force_vec2(v, def)
    if isempty(v) || ~isnumeric(v), v = def; end
    v = v(:); if isscalar(v), v = [v; v]; elseif numel(v) > 2, v = v(1:2); end
end

function v = force_vec3(v, def)
    if isempty(v) || ~isnumeric(v), v = def; end
    v = v(:); if isscalar(v), v = repmat(v,3,1); elseif numel(v) < 3, v = [v(1);v(1);v(1)]; elseif numel(v) > 3, v = v(1:3); end
end

function x = clamp(x, lo, hi)
    if isempty(x) || ~isfinite(x), x = lo; end; x = min(max(x, lo), hi);
end
