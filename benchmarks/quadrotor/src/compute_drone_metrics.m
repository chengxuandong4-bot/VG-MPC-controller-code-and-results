function m = compute_drone_metrics(result, setup)
%COMPUTE_DRONE_METRICS Compute tracking and control metrics from simulation.
%  Returns struct with: IAE_pos, RMS_pos_error, max_pos_error, final_pos_error,
%                       IAE_att, control_energy, avg_solve_time_ms, max_solve_time_ms

    x = result.x_log;
    u = result.u_log;
    ref = result.ref_log;
    st = result.solve_time_ms;

    % Position tracking (first 3 states)
    pos_error = vecnorm(x(1:3, :) - ref(1:3, :), 2, 1);  % 2-norm per column

    m.IAE_pos          = sum(pos_error) * result.dt;
    m.RMS_pos_error    = sqrt(mean(pos_error.^2));
    m.max_pos_error    = max(pos_error);
    m.final_pos_error  = pos_error(end);

    % Attitude error (states 7-9: roll, pitch, yaw)
    att_error = vecnorm(x(7:9, :) - ref(7:9, :), 2, 1);
    m.IAE_att          = sum(att_error) * result.dt;

    % Control effort
    m.control_energy   = sum(vecnorm(u, 2, 1).^2) * result.dt;

    % Solve time (skip first cold start)
    st_warm = st(2:end);
    m.avg_solve_time_ms = mean(st_warm);
    m.max_solve_time_ms = max(st_warm);

    % Saturation rate
    [~, ~, ~, ~, mp] = drone_model();
    u_min = mp.u_min; u_max = mp.u_max;
    saturated = any(u < u_min + 1e-6, 1) | any(u > u_max - 1e-6, 1);
    m.saturation_rate = mean(saturated);
end
