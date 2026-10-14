function [u, controller, debugInfo] = controller_step(controller, x, ref, refDot, actualFault, params)
switch lower(controller.type)
    case 'paper_is_sat'
        errorState = x - ref;
        u = controller.K * errorState;
        u = min(max(u, params.actuator.u_min), params.actuator.u_max);

        debugInfo = default_debug_info();
        debugInfo.tau_cmd = params.actuator.G * u;

    case 'adaptive_fta'
        cfg = controller.config;
        st = controller.state;
        st.step_count = st.step_count + 1;

        Ts = params.sim.Ts;
        innovation = zeros(3, 1);
        innovationNorm = 0.0;
        lambdaEff = cfg.rls_forgetting;
        shockLevel = 0.0;

        if norm(st.prev_u, 2) >= cfg.estimation_input_floor
            xDotMeasured = (x(1:3) - st.prev_x(1:3)) / Ts;
            tauMeasured = inverse_generalized_force(x, xDotMeasured, params);
            st.tau_meas_hat = (1.0 - cfg.measurement_filter_alpha) * st.tau_meas_hat + cfg.measurement_filter_alpha * tauMeasured;

            Phi = params.actuator.G * diag(st.prev_u);
            innovation = st.tau_meas_hat - Phi * st.h_hat;
            innovationNorm = norm(innovation, 2);
            if get_option(cfg, 'use_adaptive_forgetting', true)
                lambdaMin = get_option(cfg, 'rls_forgetting_min', 0.955);
                lambdaMax = get_option(cfg, 'rls_forgetting_max', 0.998);
                shockNormRef = get_option(cfg, 'innov_norm_ref', 1.8);
                shockRatio = max(0.0, innovationNorm / max(shockNormRef, 1e-6) - 1.0);
                lambdaEff = cfg.rls_forgetting - get_option(cfg, 'rls_fault_boost', 0.02) * tanh(shockRatio);
                lambdaEff = min(max(lambdaEff, lambdaMin), lambdaMax);
            end
            S = Phi * st.P * Phi' + ((1.0 - lambdaEff) + cfg.rls_regularization) * eye(3);
            if rcond(S) > 1e-11
                gainRls = st.P * Phi' / S;
                st.h_hat = st.h_hat + gainRls * innovation;
                st.h_hat = min(max(st.h_hat, cfg.health_min), cfg.health_max);
                st.P = (st.P - gainRls * Phi * st.P) / max(lambdaEff, 1e-6);
                st.P = 0.5 * (st.P + st.P');
            end
        else
            st.h_hat = (1.0 - cfg.health_decay) * st.h_hat + cfg.health_decay * ones(4, 1);
        end

        eVel = x(1:2) - ref(1:2);
        st.int_vel = clip_symmetric((1.0 - get_option(cfg, 'int_leak_vel', 0.0)) * st.int_vel + Ts * eVel, cfg.int_vel_limit);

        ePsi = x(4) - ref(4);
        st.int_psi = clip_symmetric((1.0 - get_option(cfg, 'int_leak_psi', 0.0)) * st.int_psi + Ts * ePsi, cfg.int_psi_limit);

        timeNow = (st.step_count - 1) * Ts;
        usePerfBoost = get_option(cfg, 'use_perf_boost', false);
        if usePerfBoost
            rho0 = get_option(cfg, 'perf_rho0', 3.0);
            rhoInf = get_option(cfg, 'perf_rho_inf', 0.10);
            rhoDecay = get_option(cfg, 'perf_decay', 0.08);
            rhoNow = rhoInf + (rho0 - rhoInf) * exp(-rhoDecay * timeNow);
            perfRatio = norm([eVel; x(3)], 2) / max(rhoNow, 1e-6);
            gainScale = 1.0 + get_option(cfg, 'perf_boost_slope', 1.6) * max(0.0, perfRatio - get_option(cfg, 'perf_boost_threshold', 1.0));
            gainScale = min(gainScale, get_option(cfg, 'perf_boost_max', 2.8));
        else
            perfRatio = NaN;
            gainScale = 1.0;
        end

        kPsiEff = cfg.k_psi * min(gainScale, 1.8);
        kiPsiEff = cfg.ki_psi * min(gainScale, 1.3);
        rCmd = -kPsiEff * ePsi - kiPsiEff * st.int_psi;

        rCmdDot = (rCmd - st.prev_r_cmd) / Ts;
        st.prev_r_cmd = rCmd;

        eR = x(3) - rCmd;
        st.int_r = clip_symmetric((1.0 - get_option(cfg, 'int_leak_r', 0.0)) * st.int_r + Ts * eR, cfg.int_r_limit);

        kpVelEff = cfg.kp_vel * gainScale;
        kiVelEff = cfg.ki_vel * min(gainScale, 1.4);
        kREff = cfg.k_r * min(gainScale, 1.8);
        kiREff = cfg.ki_r * min(gainScale, 1.3);
        ksEff = cfg.ks_tau * gainScale;

        tauCmd = zeros(3, 1);
        tauCmd(1) = params.model.m * (refDot(1) - kpVelEff(1) * eVel(1) - kiVelEff(1) * st.int_vel(1)) ...
            + params.model.Xu * x(1) + params.model.Xuu * x(1)^2 - params.model.m * x(2) * x(3) ...
            - ksEff(1) * tanh(2.0 * eVel(1));
        tauCmd(2) = params.model.m * (refDot(2) - kpVelEff(2) * eVel(2) - kiVelEff(2) * st.int_vel(2)) ...
            + params.model.Yv * x(2) + params.model.Yvv * x(2)^2 + params.model.m * x(1) * x(3) ...
            - ksEff(2) * tanh(2.0 * eVel(2));
        tauCmd(3) = params.model.Jz * (rCmdDot - kREff * eR - kiREff * st.int_r) ...
            + params.model.Nr * x(3) + params.model.Nrr * x(3)^2 ...
            - ksEff(3) * tanh(1.5 * eR);

        hEff = max(st.h_hat, cfg.health_min);
        shockLevel = max(0.0, innovationNorm - get_option(cfg, 'shock_threshold', 2.0));
        if get_option(cfg, 'use_fault_feedforward', false)
            ffGain = get_option(cfg, 'fault_feedforward_gain', 0.15);
            tauRef = reference_generalized_force(ref, refDot, params);
            tauNom = params.actuator.G * (hEff .* st.prev_u);
            tauCmd = tauCmd + ffGain * (tauRef - tauNom);
        end

        if get_option(cfg, 'use_residual_comp', false)
            residualFilter = get_option(cfg, 'residual_filter', 0.90);
            residualCompGain = get_option(cfg, 'residual_comp_gain', 0.20);
            st.residual_hat = residualFilter * st.residual_hat + (1.0 - residualFilter) * innovation;
            tauCmd = tauCmd - residualCompGain * st.residual_hat;
        end

        Ghat = params.actuator.G * diag(hEff);
        healthWeight = diag(1.0 + cfg.health_weight_scale * ((1.0 ./ (hEff .^ 2)) - 1.0));

        useEventTrigger = get_option(cfg, 'use_event_trigger', false);
        innovationNorm = norm(innovation, 2);
        if useEventTrigger
            forceRefreshSteps = get_option(cfg, 'force_refresh_steps', [1 2 3 5 8 12]);
            maxHoldSteps = get_option(cfg, 'max_hold_steps', 4);
            eventInnovThreshold = get_option(cfg, 'event_innov_threshold', 1.8);
            eventPerfThreshold = get_option(cfg, 'event_perf_threshold', 1.0);
            perfGate = isfinite(perfRatio) && perfRatio >= eventPerfThreshold;
            needRefresh = ismember(st.step_count, forceRefreshSteps) || ...
                (st.event_hold_steps >= maxHoldSteps) || ...
                (innovationNorm >= eventInnovThreshold) || perfGate;
        else
            needRefresh = true;
        end

        if needRefresh || st.step_count == 1
            allocRegEff = cfg.allocation_reg;
            if get_option(cfg, 'use_alloc_relax', false) && isfinite(perfRatio)
                allocRelaxGain = get_option(cfg, 'alloc_relax_gain', 0.8);
                allocRegMin = get_option(cfg, 'allocation_reg_min', 0.02);
                allocRegEff = max(allocRegMin, cfg.allocation_reg / (1.0 + allocRelaxGain * max(0.0, perfRatio - 1.0)));
            end
            if shockLevel > 0.0
                allocRegMin = get_option(cfg, 'allocation_reg_min', 0.02);
                shockRelaxGain = get_option(cfg, 'shock_relax_gain', 0.55);
                allocRegEff = max(allocRegMin, allocRegEff / (1.0 + shockRelaxGain * shockLevel));
            end

            A = Ghat' * Ghat + allocRegEff * healthWeight + cfg.smoothing_weight * eye(4);
            b = Ghat' * tauCmd + cfg.smoothing_weight * st.prev_u;
            uTarget = A \ b;
            if any(~isfinite(uTarget))
                uTarget = st.prev_u;
            end
            st.last_u_target = uTarget;
            st.event_hold_steps = 0;
            st.event_refresh_count = st.event_refresh_count + 1;
            eventRefreshed = 1;
            eventReused = 0;
        else
            uTarget = st.last_u_target;
            st.event_hold_steps = st.event_hold_steps + 1;
            st.event_hold_count = st.event_hold_count + 1;
            eventRefreshed = 0;
            eventReused = 1;
        end

        uTarget = min(max(uTarget, params.actuator.u_min), params.actuator.u_max);
        if usePerfBoost && isfinite(perfRatio) && perfRatio >= get_option(cfg, 'perf_boost_threshold', 1.0)
            rateLimit = get_option(cfg, 'rate_limit_boost', cfg.rate_limit);
        else
            rateLimit = cfg.rate_limit;
        end
        if shockLevel > 0.0
            rateLimit = max(rateLimit, min(get_option(cfg, 'rate_limit_shock', cfg.rate_limit), ...
                rateLimit + get_option(cfg, 'shock_rate_gain', 0.08) * shockLevel));
        end
        deltaU = clip_symmetric(uTarget - st.prev_u, rateLimit);
        u = st.prev_u + deltaU;
        u = min(max(u, params.actuator.u_min), params.actuator.u_max);
        faultConfidence = min(max(1.0 - min(hEff), 0.0), 1.0);

        st.prev_x = x;
        st.prev_u = u;
        controller.state = st;

        debugInfo = default_debug_info();
        debugInfo.h_hat = st.h_hat;
        debugInfo.innovation_norm = innovationNorm;
        debugInfo.fault_confidence = faultConfidence;
        debugInfo.perf_ratio = perfRatio;
        debugInfo.gain_scale = gainScale;
        debugInfo.lambda_eff = lambdaEff;
        debugInfo.shock_level = shockLevel;
        debugInfo.tau_cmd = tauCmd;
        debugInfo.mpc_refreshed = eventRefreshed;
        debugInfo.mpc_reused = eventReused;
        debugInfo.mpc_refresh_count = st.event_refresh_count;
        debugInfo.mpc_hold_steps = st.event_hold_steps;

    otherwise
        error('Unknown controller type: %s', controller.type);
end

if nargin >= 5 && ~isempty(actualFault)
    debugInfo.actual_fault = actualFault(:);
else
    debugInfo.actual_fault = nan(4, 1);
end
end

function tau = inverse_generalized_force(x, xDot, params)
tau = zeros(3, 1);
tau(1) = params.model.m * xDot(1) + params.model.Xu * x(1) + params.model.Xuu * x(1)^2 - params.model.m * x(2) * x(3);
tau(2) = params.model.m * xDot(2) + params.model.Yv * x(2) + params.model.Yvv * x(2)^2 + params.model.m * x(1) * x(3);
tau(3) = params.model.Jz * xDot(3) + params.model.Nr * x(3) + params.model.Nrr * x(3)^2;
end

function tau = reference_generalized_force(ref, refDot, params)
tau = zeros(3, 1);
tau(1) = params.model.m * refDot(1) + params.model.Xu * ref(1) + params.model.Xuu * ref(1)^2 - params.model.m * ref(2) * ref(3);
tau(2) = params.model.m * refDot(2) + params.model.Yv * ref(2) + params.model.Yvv * ref(2)^2 + params.model.m * ref(1) * ref(3);
tau(3) = params.model.Jz * refDot(3) + params.model.Nr * ref(3) + params.model.Nrr * ref(3)^2;
end

function y = clip_symmetric(x, limit)
if isscalar(limit)
    y = min(max(x, -limit), limit);
else
    limitVec = limit(:);
    y = min(max(x, -limitVec), limitVec);
end
end

function value = get_option(cfg, fieldName, defaultValue)
if isfield(cfg, fieldName) && ~isempty(cfg.(fieldName))
    value = cfg.(fieldName);
else
    value = defaultValue;
end
end

function debugInfo = default_debug_info()
debugInfo = struct( ...
    'h_hat', nan(4, 1), ...
    'innovation_norm', nan, ...
    'fault_confidence', nan, ...
    'perf_ratio', nan, ...
    'gain_scale', nan, ...
    'lambda_eff', nan, ...
    'shock_level', nan, ...
    'tau_cmd', nan(3, 1), ...
    'actual_fault', nan(4, 1), ...
    'mpc_refreshed', 0, ...
    'mpc_reused', 0, ...
    'mpc_refresh_count', 0, ...
    'mpc_hold_steps', nan);
end
