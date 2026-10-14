function result = run_drone_sim(candidate, setup)
%RUN_DRONE_SIM Closed-loop MPC simulation for quadrotor benchmark.
%  candidate: struct with Q_diag[12], R_diag[4], horizon_N, u_min, u_max
%  setup:     struct with A_d, B_d, reference, T_sim, x0

    % Get model
    [~, ~, A_d, B_d, model_params] = drone_model();
    dt = model_params.dt;  % 0.01 s

    % Extract config
    Q_diag = candidate.Q_diag(:);
    R_diag = candidate.R_diag(:);
    N = max(5, min(50, round(candidate.horizon_N)));

    u_min = model_params.u_min(:);
    u_max = model_params.u_max(:);

    % Simulation settings
    T_sim = setup.T_sim;
    N_steps = round(T_sim / dt);
    x = setup.x0(:);

    % Pre-allocate
    x_log   = zeros(12, N_steps+1);
    u_log   = zeros(4, N_steps);
    ref_log = zeros(12, N_steps+1);
    solve_t = zeros(1, N_steps);
    x_log(:,1) = x;

    % Hover reference input
    u_hover = [model_params.mass * model_params.gravity; 0; 0; 0];

    for k = 1:N_steps
        t_now = (k-1) * dt;

        % Reference over horizon
        x_ref_N = zeros(12, N+1);
        u_ref_N = zeros(4, N);
        for i = 0:N
            x_ref_N(:, i+1) = setup.reference(t_now + i*dt);
        end
        for i = 1:N
            u_ref_N(:, i) = u_hover;
        end

        ref_log(:, k) = x_ref_N(:, 1);

        % Solve MPC
        [u_opt, info] = drone_mpc_solve(x, x_ref_N, u_ref_N, ...
            Q_diag, R_diag, N, A_d, B_d, u_min, u_max);
        solve_t(k) = info.solve_time_ms;

        % Apply + step
        u_k = u_opt;
        x = drone_step(x, u_k, dt, model_params);
        x = drone_clamp_state(x, model_params);

        u_log(:, k) = u_k;
        x_log(:, k+1) = x;
    end

    result.x_log = x_log;
    result.u_log = u_log;
    result.ref_log = ref_log;
    result.solve_time_ms = solve_t;
    result.dt = dt;
    result.T_sim = T_sim;
    result.setup = setup;
end

function x_next = drone_step(x, u, dt, params)
    % RK4 integration of continuous linearized dynamics
    [A, B] = drone_model_cont();
    m = params.mass; g = params.gravity;
    % Add gravity compensation to linear dynamics
    % vz_dot = F/m - g, so net = A*x + B*u + [0;0;0;0;0;-g;0;0;0;0;0;0]
    gravity_vec = [0;0;0;0;0;-g;0;0;0;0;0;0];
    f = @(xx, uu) A * xx + B * uu + gravity_vec;

    k1 = f(x, u);
    k2 = f(x + 0.5*dt*k1, u);
    k3 = f(x + 0.5*dt*k2, u);
    k4 = f(x + dt*k3, u);
    x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function [A_cont, B_cont] = drone_model_cont()
    [A_cont, B_cont] = drone_model();
end

function x = drone_clamp_state(x, params)
    x(3) = max(params.z_min, min(params.z_max, x(3)));
    x(7) = max(-params.max_tilt_rad, min(params.max_tilt_rad, x(7)));
    x(8) = max(-params.max_tilt_rad, min(params.max_tilt_rad, x(8)));
end
