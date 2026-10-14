function [A, B, A_d, B_d, params] = drone_model()
%DRONE_MODEL 12-state quadrotor linearized model around hover.
%  States:  [px py pz vx vy vz roll pitch yaw p q r]
%  Inputs:  [F_total tau_x tau_y tau_z]  (thrust + body moments)
%  Discrete-time: dt = 0.01 s (100 Hz)
%
%  Returns continuous (A,B) and discrete (A_d,B_d) plus physical params.

    % --- Physical parameters (Crazyflie-like) ---
    params = struct();
    params.mass    = 0.030;   % kg
    params.gravity = 9.81;    % m/s^2
    params.Ixx     = 1.4e-5;  % kg·m^2
    params.Iyy     = 1.4e-5;
    params.Izz     = 2.2e-5;
    params.dt      = 0.01;    % 100 Hz
    params.arm_length = 0.0397;  % m
    params.kf      = 3.16e-10;   % thrust coefficient
    params.km      = 7.94e-12;   % moment coefficient

    % Input limits (total thrust, moments)
    params.u_min = [0.0;   -0.05; -0.05; -0.02];
    params.u_max = [1.2;    0.05;  0.05;  0.02];

    % State limits
    params.z_min = 0.0;
    params.z_max = 5.0;
    params.max_tilt_rad = deg2rad(40);

    % --- Continuous A, B (hover linearization) ---
    g = params.gravity;
    m = params.mass;
    Ix = params.Ixx; Iy = params.Iyy; Iz = params.Izz;

    A = zeros(12, 12);
    % Position ← velocity (simple integrator)
    A(1,4) = 1;  % px_dot = vx
    A(2,5) = 1;  % py_dot = vy
    A(3,6) = 1;  % pz_dot = vz
    % Velocity ← attitude (small angle: g*theta, -g*phi)
    A(4,8) =  g;  % vx_dot = g*theta
    A(5,7) = -g;  % vy_dot = -g*phi
    % Attitude ← body rates (Euler kinematics, small angles)
    A(7,10) = 1;  % phi_dot = p
    A(8,11) = 1;  % theta_dot = q
    A(9,12) = 1;  % psi_dot = r
    % Body rates — no damping (pure integrator)
    % p,q,r are directly controlled by moments

    B = zeros(12, 4);
    B(6,1)  = 1/m;     % vz_dot = F/m - g (gravity subtracted in ref)
    B(10,2) = 1/Ix;    % p_dot = tau_x / Ixx
    B(11,3) = 1/Iy;    % q_dot = tau_y / Iyy
    B(12,4) = 1/Iz;    % r_dot = tau_z / Izz

    % --- Discrete-time (ZOH) ---
    dt = params.dt;
    sys_c = ss(A, B, eye(12), zeros(12,4));
    sys_d = c2d(sys_c, dt, 'zoh');
    A_d = sys_d.A;
    B_d = sys_d.B;
end
