function v = compute_drone_violations(result, setup)
%COMPUTE_DRONE_VIOLATIONS Constraint violation penalty.
    x = result.x_log;
    [~, ~, ~, ~, mp] = drone_model();

    % Altitude violations
    z_violation_lo = sum(max(0, mp.z_min - x(3,:))) * result.dt;
    z_violation_hi = sum(max(0, x(3,:) - mp.z_max)) * result.dt;

    % Tilt violations
    tilt = sqrt(x(7,:).^2 + x(8,:).^2);
    tilt_violation = sum(max(0, tilt - mp.max_tilt_rad)) * result.dt;

    v = z_violation_lo + z_violation_hi + tilt_violation;
end
