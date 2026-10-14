function setup = init_drone_params()
%INIT_DRONE_PARAMS Setup for quadrotor benchmark simulation.
    [~, ~, A_d, B_d, mp] = drone_model();

    setup = struct();
    setup.A_d = A_d;
    setup.B_d = B_d;
    setup.model_params = mp;
    setup.dt = mp.dt;
    setup.T_sim = 10.0;  % 10 second simulation

    % Hover as initial state
    setup.x0 = zeros(12, 1);

    % Reference trajectory: step changes in position + constant yaw
    setup.reference = @(t) drone_reference(t);

    % Baseline candidate
    setup.baseline = drone_default_candidate();
end

function x_ref = drone_reference(t)
    % Position reference with step changes
    if t < 2.0
        p_ref = [0.5; 0.0; 1.0];
    elseif t < 5.0
        p_ref = [1.0; 0.5; 1.5];
    elseif t < 8.0
        p_ref = [0.5; 1.0; 2.0];
    else
        p_ref = [0.0; 0.5; 1.5];
    end
    x_ref = [p_ref; zeros(9,1)];  % position, zero velocity/attitude/rates
end

function cfg = drone_default_candidate()
    cfg = struct();
    cfg.Q_diag   = [10;10;20; 1;1;2; 0.1;0.1;0.1; 0.05;0.05;0.05];  % pos/vel/att/rate
    cfg.R_diag   = [0.1; 1; 1; 1];   % thrust / tau_x / tau_y / tau_z
    cfg.horizon_N = 15;
    cfg.u_min    = [0.0; -0.05; -0.05; -0.02];
    cfg.u_max    = [1.2;  0.05;  0.05;  0.02];
    cfg.name     = 'baseline_hover';
end
