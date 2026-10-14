function result = run_generation_experiment(cfg, setup)
%RUN_GENERATION_EXPERIMENT Simulate one generation on the CSTR LPV-MPC benchmark.

if nargin < 2 || isempty(setup)
    setup = cstr_default_setup();
end

if ~exist('quadprog', 'file')
    error('quadprog is required to run this benchmark.');
end

Ts = setup.Ts;
Nsim = setup.Nsim;

[x0, u_prev] = cstr_steady_state_from_ca(setup.ca_ref(1), setup.nominal_params);

x_hist = zeros(2, Nsim + 1);
u_hist = zeros(1, Nsim);
err_hist = zeros(1, Nsim);
horizon_hist = zeros(1, Nsim);
cond_hist = zeros(1, Nsim);
solve_ms_hist = zeros(1, Nsim);
kkt_hist = zeros(1, Nsim);
status_hist = zeros(1, Nsim);
theta_k0_hist = zeros(1, Nsim);
theta_ua_hist = zeros(1, Nsim);
dhat_hist = zeros(1, Nsim);
ubias_hist = zeros(1, Nsim);

x_hist(:, 1) = x0;

ctrl.theta_k0 = 1.0;
ctrl.theta_ua = 1.0;
ctrl.dhat_ca = 0.0;
ctrl.int_e = 0.0;
ctrl.u_bias = 0.0;

for k = 1:Nsim
    t_now = (k - 1) * Ts;
    xk = x_hist(:, k);
    r_now = setup.ca_ref(k);
    e_now = xk(1) - r_now;

    if cfg.use_adaptive_horizon
        p_cur = select_horizon(abs(e_now), cfg);
    else
        p_cur = cfg.horizon_max;
    end

    model_params = setup.nominal_params;
    if cfg.use_param_adaptation
        model_params.k0 = setup.nominal_params.k0 * ctrl.theta_k0;
        model_params.UA = setup.nominal_params.UA * ctrl.theta_ua;
    end

    x_for_model = xk;

    x_for_model(1) = max(x_for_model(1), 1e-4);
    x_for_model(2) = min(max(x_for_model(2), 250), 500);

    [A_d, B_d, c_d] = cstr_linearize(x_for_model, u_prev, model_params, Ts);
    if cfg.use_disturbance_observer
        c_d(1) = c_d(1) + ctrl.dhat_ca;
    end

    wy = cfg.Wy_base * (1 + cfg.Wy_gain * min(abs(e_now), cfg.Wy_cap));
    wdu = cfg.Wdu_base * (1 + cfg.Wdu_gain * min(abs(e_now), cfg.Wdu_cap));

    weights.Wu = cfg.Wu;
    weights.Wdu = wdu;
    weights.Wy = wy;
    weights.C = [1 0];

    yref = build_yref(setup.ca_ref, k, p_cur, setup.preview, ctrl, cfg);
    u_ref = build_uref(setup.ca_ref, k, p_cur, setup.preview, model_params);

    qp = build_mpc_qp(A_d, B_d, c_d, x_for_model, u_prev, u_ref, yref, weights, setup.cons, p_cur, cfg.hessian_reg);
    [Zr, sr, Hr, hr, Gr, gr] = condense_qr(qp.H, qp.h, qp.Aeq, qp.beq, qp.G, qp.g);

    cond_hist(k) = cond_number(Hr);

    tic;
    [zbar, exitflag, lambda_red] = solve_qp(Hr, hr, Gr, gr);
    solve_ms_hist(k) = toc * 1000;

    if isempty(zbar)
        zbar = zeros(size(Hr, 1), 1);
        status_hist(k) = 0;
    else
        status_hist(k) = 1;
    end

    z = Zr * zbar + sr;
    u = z(qp.idx_u{1});
    if cfg.use_input_bias_adaptation
        u = u + ctrl.u_bias;
    end
    u = min(max(u, setup.cons.u_min), setup.cons.u_max);

    % Conservative anti-windup like clamp for numerical safety.
    du = u - u_prev;
    du = min(max(du, setup.cons.du_min), setup.cons.du_max);
    u = u_prev + du;

    u_hist(k) = u;
    err_hist(k) = e_now;
    horizon_hist(k) = p_cur;
    theta_k0_hist(k) = ctrl.theta_k0;
    theta_ua_hist(k) = ctrl.theta_ua;
    dhat_hist(k) = ctrl.dhat_ca;
    ubias_hist(k) = ctrl.u_bias;

    true_params = cstr_true_params(setup.nominal_params, t_now, setup);
    x_next = xk + Ts * cstr_rhs(xk, u, true_params);

    % Keep states in physically meaningful region.
    x_next(1) = max(x_next(1), 1e-5);
    x_next(2) = min(max(x_next(2), 250), 500);
    x_hist(:, k + 1) = x_next;

    % KKT residual in reduced coordinates.
    kkt_hist(k) = reduced_kkt_residual(Hr, hr, Gr, gr, zbar, lambda_red);

    if cfg.use_disturbance_observer || cfg.use_param_adaptation
        x_nom_next = xk + Ts * cstr_rhs(xk, u, model_params);
        residual = x_next - x_nom_next;

        if cfg.use_disturbance_observer
            ctrl.dhat_ca = (1 - cfg.obs_alpha) * ctrl.dhat_ca + cfg.obs_alpha * residual(1);
            ctrl.dhat_ca = min(max(ctrl.dhat_ca, -cfg.obs_bound), cfg.obs_bound);
        end

        if cfg.use_param_adaptation
            CA = max(xk(1), 1e-5);
            Tr = min(max(xk(2), 250), 500);
            expo = exp(-setup.nominal_params.E / (setup.nominal_params.R * Tr));

            % Sensitivity wrt theta_k0 when k0 = theta_k0 * k0_nom.
            phi_k0 = -Ts * setup.nominal_params.k0 * expo * CA;
            denom_k0 = phi_k0 * phi_k0 + cfg.adapt_eps;
            ctrl.theta_k0 = ctrl.theta_k0 + cfg.gamma_k0 * residual(1) * phi_k0 / denom_k0;
            ctrl.theta_k0 = min(max(ctrl.theta_k0, cfg.theta_k0_min), cfg.theta_k0_max);

            % Sensitivity wrt theta_ua when UA = theta_ua * UA_nom.
            phi_ua = Ts * (-setup.nominal_params.UA / (setup.nominal_params.rhoCp * setup.nominal_params.V)) * (Tr - u);
            denom_ua = phi_ua * phi_ua + cfg.adapt_eps;
            ctrl.theta_ua = ctrl.theta_ua + cfg.gamma_ua * residual(2) * phi_ua / denom_ua;
            ctrl.theta_ua = min(max(ctrl.theta_ua, cfg.theta_ua_min), cfg.theta_ua_max);
        end
    end

    if cfg.use_input_bias_adaptation
        ctrl.int_e = ctrl.int_e + Ts * e_now;
        ctrl.int_e = min(max(ctrl.int_e, -cfg.int_e_bound), cfg.int_e_bound);
    end

    if cfg.use_input_bias_adaptation
        ctrl.u_bias = ctrl.u_bias - cfg.u_bias_gain * e_now - cfg.u_bias_int_gain * ctrl.int_e;
        ctrl.u_bias = min(max(ctrl.u_bias, -cfg.u_bias_bound), cfg.u_bias_bound);
    end

    u_prev = u;
end

t = (0:(Nsim - 1)) * Ts;
ref = setup.ca_ref(1:Nsim);
err = x_hist(1, 1:Nsim) - ref;

metrics = struct();
metrics.integral_error = sum(abs(err)) * Ts;
metrics.rms_tracking_error = sqrt(mean(err.^2));
metrics.mean_abs_error = mean(abs(err));
metrics.max_tracking_error = max(abs(err));
metrics.final_tracking_error = abs(err(end));
metrics.control_energy = sum((u_hist - mean(u_hist)).^2) * Ts;
metrics.avg_controller_time_ms = mean(solve_ms_hist);
metrics.max_controller_time_ms = max(solve_ms_hist);
metrics.median_condition_number = median(cond_hist);
metrics.max_condition_number = max(cond_hist);
metrics.median_kkt_residual = median(kkt_hist);
metrics.max_kkt_residual = max(kkt_hist);
metrics.success_ratio = mean(status_hist);
metrics.avg_horizon = mean(horizon_hist);
metrics.min_horizon = min(horizon_hist);
metrics.max_horizon = max(horizon_hist);
metrics.mean_u_bias = mean(ubias_hist);
metrics.final_u_bias = ubias_hist(end);

result = struct();
result.config = cfg;
result.setup = setup;
result.metrics = metrics;
result.logs = struct();
result.logs.t = t;
result.logs.ref = ref;
result.logs.x = x_hist;
result.logs.u = u_hist;
result.logs.err = err;
result.logs.horizon = horizon_hist;
result.logs.cond = cond_hist;
result.logs.solve_ms = solve_ms_hist;
result.logs.kkt = kkt_hist;
result.logs.status = status_hist;
result.logs.theta_k0 = theta_k0_hist;
result.logs.theta_ua = theta_ua_hist;
result.logs.dhat_ca = dhat_hist;
result.logs.u_bias = ubias_hist;
end

function setup = cstr_default_setup()
setup = struct();
setup.nominal_params.F = 1;
setup.nominal_params.V = 1;
setup.nominal_params.R = 1.985875;
setup.nominal_params.dH = -5960;
setup.nominal_params.E = 11843;
setup.nominal_params.k0 = 34393800;
setup.nominal_params.rhoCp = 500;
setup.nominal_params.UA = 150;
setup.nominal_params.CAf = 10;
setup.nominal_params.Tf = 298.15;

setup.Ts = 0.5;
setup.preview = 5;
setup.t_end = 60;
setup.Nsim = round(setup.t_end / setup.Ts);

setup.cons.u_min = 285.15;
setup.cons.u_max = 312.15;
setup.cons.du_min = -2;
setup.cons.du_max = 2;

% Use the same set-point profile as the TAC CSTR case.
t_ref = (0:(setup.Nsim + setup.preview)) * setup.Ts;
ca_ref = 8 * ones(size(t_ref));
ca_ref(t_ref >= 10 & t_ref < 30) = 2;
ca_ref(t_ref >= 30 & t_ref < 50) = 8;
ca_ref(t_ref >= 50) = 5;
setup.ca_ref = ca_ref;

% Disturbance and mismatch profile in the true plant.
setup.k0_scale_amp = 0.20;
setup.k0_step_1 = 0.15;
setup.k0_step_2 = -0.10;
setup.ua_scale_amp = 0.15;
setup.caf_amp = 0.40;
setup.caf_step = 0.80;
setup.tf_amp = 5.0;
setup.tf_step = -3.5;
end

function p = select_horizon(abs_error, cfg)
if abs_error >= cfg.horizon_high_err
    p = cfg.horizon_max;
elseif abs_error >= cfg.horizon_mid_err
    p = cfg.horizon_mid;
else
    p = cfg.horizon_min;
end
end

function yref = build_yref(ca_ref, k, p, preview, ctrl, cfg)
ca_seq = zeros(1, p);
for i = 1:p
    if i <= preview
        idx = min(k + i, numel(ca_ref));
    else
        idx = min(k + preview, numel(ca_ref));
    end
    ca_seq(i) = ca_ref(idx);
end
if cfg.use_input_bias_adaptation
    ca_seq = ca_seq - cfg.int_ref_gain * ctrl.int_e;
end
yref = ca_seq;
end

function uref = build_uref(ca_ref, k, p, preview, params)
u_seq = zeros(1, p);
for i = 1:p
    if i <= preview
        idx = min(k + i, numel(ca_ref));
    else
        idx = min(k + preview, numel(ca_ref));
    end
    [~, u_ss] = cstr_steady_state_from_ca(ca_ref(idx), params);
    u_seq(i) = u_ss;
end
uref = u_seq;
end

function [A_d, B_d, c] = cstr_linearize(x, u, params, Ts)
CA = max(x(1), 1e-6);
Tr = min(max(x(2), 250), 500);
k = params.k0 * exp(-params.E / (params.R * Tr));
dr_dCA = k;
dr_dTr = k * CA * (params.E / (params.R * Tr^2));

dCA_dCA = -(params.F / params.V) - dr_dCA;
dCA_dTr = -dr_dTr;
dTr_dCA = -(params.dH / params.rhoCp) * dr_dCA;
dTr_dTr = -(params.F / params.V) ...
    - (params.dH / params.rhoCp) * dr_dTr ...
    - (params.UA / (params.rhoCp * params.V));

A_c = [dCA_dCA, dCA_dTr; dTr_dCA, dTr_dTr];
B_c = [0; params.UA / (params.rhoCp * params.V)];
f = cstr_rhs(x, u, params);

A_d = eye(2) + Ts * A_c;
B_d = Ts * B_c;
c = Ts * (f - A_c * x - B_c * u);
end

function qp = build_mpc_qp(A_d, B_d, c, x0, u_prev, u_ref, yref, weights, cons, p, hessian_reg)
nx = size(A_d, 1);
nu = size(B_d, 2);
m = p * (nx + nu);
n = p * nx;

H = zeros(m, m);
h = zeros(m, 1);

idx_u = cell(p, 1);
idx_x = cell(p, 1);
for i = 1:p
    base = (i - 1) * (nx + nu);
    idx_u{i} = base + (1:nu);
    idx_x{i} = base + nu + (1:nx);
end

Wu = weights.Wu.^2;
Wdu = weights.Wdu.^2;
Wy = weights.Wy' * weights.Wy;
C = weights.C;

if isscalar(u_ref)
    u_ref = repmat(u_ref, 1, p);
end

for i = 1:p
    u_idx = idx_u{i};
    x_idx = idx_x{i};

    H(u_idx, u_idx) = H(u_idx, u_idx) + 2 * Wu;
    h(u_idx) = h(u_idx) - 2 * Wu * u_ref(i);

    if i == 1
        H(u_idx, u_idx) = H(u_idx, u_idx) + 2 * Wdu;
        h(u_idx) = h(u_idx) - 2 * Wdu * u_prev;
    else
        u_prev_idx = idx_u{i - 1};
        H(u_idx, u_idx) = H(u_idx, u_idx) + 2 * Wdu;
        H(u_prev_idx, u_prev_idx) = H(u_prev_idx, u_prev_idx) + 2 * Wdu;
        H(u_idx, u_prev_idx) = H(u_idx, u_prev_idx) - 2 * Wdu;
        H(u_prev_idx, u_idx) = H(u_prev_idx, u_idx) - 2 * Wdu;
    end

    y_i = yref(:, i);
    H(x_idx, x_idx) = H(x_idx, x_idx) + 2 * (C' * Wy * C);
    h(x_idx) = h(x_idx) - 2 * C' * Wy * y_i;
end

Aeq = zeros(n, m);
beq = zeros(n, 1);
for i = 1:p
    row = (i - 1) * nx + (1:nx);
    u_idx = idx_u{i};
    x_idx = idx_x{i};
    Aeq(row, x_idx) = eye(nx);
    Aeq(row, u_idx) = -B_d;
    if i == 1
        beq(row) = A_d * x0 + c;
    else
        x_prev_idx = idx_x{i - 1};
        Aeq(row, x_prev_idx) = -A_d;
        beq(row) = c;
    end
end

m_c = 4 * p * nu;
G = zeros(m_c, m);
g = zeros(m_c, 1);
row = 0;
for i = 1:p
    u_idx = idx_u{i};

    row = row + 1;
    G(row, u_idx) = 1;
    g(row) = cons.u_max;

    row = row + 1;
    G(row, u_idx) = -1;
    g(row) = -cons.u_min;

    if i == 1
        row = row + 1;
        G(row, u_idx) = 1;
        g(row) = cons.du_max + u_prev;

        row = row + 1;
        G(row, u_idx) = -1;
        g(row) = -cons.du_min + u_prev;
    else
        u_prev_idx = idx_u{i - 1};
        row = row + 1;
        G(row, u_idx) = 1;
        G(row, u_prev_idx) = -1;
        g(row) = cons.du_max;

        row = row + 1;
        G(row, u_idx) = -1;
        G(row, u_prev_idx) = 1;
        g(row) = -cons.du_min;
    end
end

qp.H = (H + H') / 2 + hessian_reg * eye(m);
qp.h = h;
qp.Aeq = Aeq;
qp.beq = beq;
qp.G = G;
qp.g = g;
qp.idx_u = idx_u;
qp.idx_x = idx_x;
end

function [Z, s, Hr, hr, Gr, gr] = condense_qr(H, h, Aeq, beq, G, g)
A = Aeq';
[Q, R] = qr(A, 0);
n = size(A, 2);
Q1 = Q(:, 1:n);
Q2 = null(Aeq);
R1 = R(1:n, :);

if any(~isfinite(R1(:))) || any(~isfinite(beq)) || rcond(R1) < 1e-12
    s = Aeq' * ((Aeq * Aeq') \ beq);
else
    s = Q1 * (R1' \ beq);
end

if isempty(Q2)
    Q2 = zeros(size(Aeq, 2), 0);
end

Z = Q2;

if isempty(Z)
    Hr = zeros(0, 0);
    hr = zeros(0, 1);
else
    Hr = Z' * H * Z;
    hr = Z' * (H * s + h);
end

if isempty(G)
    Gr = [];
    gr = [];
else
    Gr = G * Z;
    gr = g - G * s;
end
end

function [zbar, exitflag, lambda] = solve_qp(H, h, G, g)
lambda = struct('ineqlin', []);
if isempty(H)
    zbar = zeros(0, 1);
    exitflag = 1;
    return;
end

H = (H + H') / 2;
opts = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
[zbar, ~, exitflag, ~, lambda] = quadprog(H, h, G, g, [], [], [], [], [], opts);
if isempty(zbar)
    zbar = [];
end
end

function res = reduced_kkt_residual(H, h, G, g, z, lambda)
if isempty(z)
    res = NaN;
    return;
end
res_ineq = 0;
if ~isempty(G)
    res_ineq = norm(max(G * z - g, 0), inf);
end
grad = H * z + h;
if ~isempty(lambda) && isfield(lambda, 'ineqlin') && ~isempty(lambda.ineqlin)
    grad = grad + G' * lambda.ineqlin;
end
res_stat = norm(grad, inf);
res = max(res_ineq, res_stat);
end

function k = cond_number(M)
if isempty(M)
    k = NaN;
    return;
end
M = (M + M') / 2;
r = rcond(M);
if r <= eps
    k = 1 / eps;
else
    k = 1 / r;
end
end

function [x_ss, u_ss] = cstr_steady_state_from_ca(CA, params)
r = (params.F / params.V) * (params.CAf - CA);
k = max(r / max(CA, 1e-6), 1e-12);
ratio = min(max(k / params.k0, 1e-12), 1 - 1e-12);
Tr = -params.E / (params.R * log(ratio));
energy = (params.F / params.V) * (params.Tf - Tr) ...
    - (params.dH / params.rhoCp) * r;
u_ss = Tr - energy / (params.UA / (params.rhoCp * params.V));
x_ss = [CA; Tr];
end

function dx = cstr_rhs(x, u, params)
CA = max(x(1), 1e-6);
Tr = min(max(x(2), 250), 500);
r = params.k0 * exp(-params.E / (params.R * Tr)) * CA;
dCA = (params.F / params.V) * (params.CAf - CA) - r;
dTr = (params.F / params.V) * (params.Tf - Tr) ...
    - (params.dH / params.rhoCp) * r ...
    - (params.UA / (params.rhoCp * params.V)) * (Tr - u);
dx = [dCA; dTr];
end

function params = cstr_true_params(nominal_params, t_now, setup)
params = nominal_params;

k0_scale = 1 ...
    + setup.k0_scale_amp * sin(2 * pi * t_now / 45) ...
    + setup.k0_step_1 * double(t_now >= 20) ...
    + setup.k0_step_2 * double(t_now >= 42);
params.k0 = nominal_params.k0 * max(0.45, k0_scale);

ua_scale = 1 + setup.ua_scale_amp * sin(2 * pi * t_now / 28);
params.UA = nominal_params.UA * max(0.5, ua_scale);

params.CAf = nominal_params.CAf ...
    + setup.caf_amp * sin(2 * pi * t_now / 22) ...
    + setup.caf_step * double(t_now >= 32);

params.Tf = nominal_params.Tf ...
    + setup.tf_amp * sin(2 * pi * t_now / 30) ...
    + setup.tf_step * double(t_now >= 38);
end
