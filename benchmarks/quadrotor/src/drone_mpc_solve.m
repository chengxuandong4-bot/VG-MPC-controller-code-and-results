function [u_opt, info] = drone_mpc_solve(x0, x_ref, u_ref, Q_diag, R_diag, N, A_d, B_d, u_min, u_max)
%DRONE_MPC_SOLVE Linear MPC solver (dense QP via quadprog).
%  Minimizes: sum_k (x_k-x_ref)'*Q*(x_k-x_ref) + (u_k-u_ref)'*R*(u_k-u_ref)
%            + (x_N-x_ref)'*Q_f*(x_N-x_ref)
%  s.t. x_{k+1} = A_d*x_k + B_d*u_k,  u_min <= u_k <= u_max
%
%  Inputs:
%    x0     : current state [12x1]
%    x_ref  : reference trajectory [12 x (N+1)]
%    u_ref  : reference input [4 x N]
%    Q_diag : state weight diagonal [12x1]  (LLM-tuneable)
%    R_diag : input weight diagonal [4x1]   (LLM-tuneable)
%    N      : prediction horizon (LLM-tuneable)
%    A_d, B_d : discrete dynamics
%    u_min, u_max : input bounds [4x1]
%
%  Outputs:
%    u_opt  : first optimal control [4x1]
%    info   : solve info (status, time, cost)

    nx = 12; nu = 4;
    Q = diag(Q_diag);
    R = diag(R_diag);
    Q_f = 2 * Q;  % terminal cost

    % --- Build dense QP: min 0.5*z'*H*z + f'*z  s.t. lb <= z <= ub ---
    % Decision variable z = [u_0; u_1; ...; u_{N-1}]  (nu*N x 1)

    nz = nu * N;
    H = zeros(nz, nz);
    f = zeros(nz, 1);

    % Precompute state propagation matrices
    % x_k = A_d^k * x0 + sum_{i=0}^{k-1} A_d^{k-1-i} * B_d * u_i
    % Build as: X = S_x * x0 + S_u * U

    S_x = zeros(nx * (N+1), nx);
    S_u = zeros(nx * (N+1), nu * N);

    S_x(1:nx, :) = eye(nx);
    for k = 1:N
        S_x(k*nx+1:(k+1)*nx, :) = A_d * S_x((k-1)*nx+1:k*nx, :);
        S_u(k*nx+1:(k+1)*nx, :) = A_d * S_u((k-1)*nx+1:k*nx, :);
        col_start = (k-1)*nu + 1;
        col_end = k*nu;
        S_u(k*nx+1:(k+1)*nx, col_start:col_end) = B_d;
    end

    % --- Build H and f ---
    for k = 1:N
        row_x = k*nx+1:(k+1)*nx;  % x_k
        col_u_start = (k-1)*nu + 1;
        col_u_end = k*nu;
        for j = 0:k-1
            u_col = j*nu+1:(j+1)*nu;
            % x_k contribution from u_j: A_d^{k-1-j} * B_d
            A_pow = A_d^(k-1-j);
            B_contrib = A_pow * B_d;
            % Add to H: 2 * B_contrib' * Q * B_contrib
            H(u_col, u_col) = H(u_col, u_col) + 2 * B_contrib' * Q * B_contrib;
        end
        % f contribution from x_ref
        x_pred_no_u = S_x(row_x, :) * x0;
        for i_col = 1:k
            u_col_range = (i_col-1)*nu+1:i_col*nu;
            x_pred_no_u = x_pred_no_u + S_u(row_x, u_col_range) * zeros(nu,1);  % placeholder
        end
        % Actually, compute f directly
        x_free = S_x(row_x, :) * x0;
        for j = 0:k-1
            u_col_j = j*nu+1:(j+1)*nu;
            A_pow_j = A_d^(k-1-j);
            B_j = A_pow_j * B_d;
            x_ref_k = x_ref(:, k);
            f(u_col_j) = f(u_col_j) + B_j' * Q * (x_free - x_ref_k);
        end
    end

    % Terminal cost contributions
    x_N_free = S_x(N*nx+1:(N+1)*nx, :) * x0;
    for j = 0:N-1
        u_col_j = j*nu+1:(j+1)*nu;
        A_pow_j = A_d^(N-1-j);
        B_j = A_pow_j * B_d;
        H(u_col_j, u_col_j) = H(u_col_j, u_col_j) + B_j' * Q_f * B_j;
        f(u_col_j) = f(u_col_j) + B_j' * Q_f * (x_N_free - x_ref(:, N+1));
    end

    % Input stage costs
    for k = 1:N
        u_col_k = (k-1)*nu+1:k*nu;
        H(u_col_k, u_col_k) = H(u_col_k, u_col_k) + R;
        if k <= size(u_ref, 2)
            f(u_col_k) = f(u_col_k) - R * u_ref(:, k);
        end
    end

    % --- Box constraints ---
    lb = repmat(u_min, N, 1);
    ub = repmat(u_max, N, 1);

    % --- Solve QP ---
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
    t_start = tic;
    [z_opt, ~, exitflag] = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    solve_time = toc(t_start);

    if exitflag >= 0 && ~isempty(z_opt)
        u_opt = z_opt(1:nu);
        info.status = 'solved';
    else
        % Fallback: zero-order hold of reference or zero input
        u_opt = max(u_min, min(u_max, zeros(nu,1)));
        info.status = 'failed';
    end
    info.solve_time_ms = solve_time * 1000;
    info.exitflag = exitflag;
    info.cost = 0.5 * z_opt' * H * z_opt + f' * z_opt;
end
