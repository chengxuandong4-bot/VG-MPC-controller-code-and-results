function q = aggregate_drone_score(m)
%AGGREGATE_DRONE_SCORE Weighted combination of metrics into scalar q.
%  Higher q = better. All metrics are "lower is better" except where noted.
%  Weights: position tracking (dominant), attitude, control, solve, saturation.

    w_iae    = 0.35;
    w_rms    = 0.20;
    w_max    = 0.10;
    w_final  = 0.05;
    w_att    = 0.10;
    w_ctrl   = 0.05;
    w_solve  = 0.10;
    w_sat    = 0.05;

    % Normalize by baseline (use sensible defaults)
    iae_n    = m.IAE_pos / 5.0;           % ~5 m·s baseline
    rms_n    = m.RMS_pos_error / 0.5;     % ~0.5 m baseline
    max_n    = m.max_pos_error / 1.5;      % ~1.5 m baseline
    final_n  = m.final_pos_error / 0.5;
    att_n    = m.IAE_att / 2.0;           % ~2 rad·s baseline
    ctrl_n   = m.control_energy / 50.0;   % ~50 baseline
    solve_n  = m.avg_solve_time_ms / 50.0; % ~50 ms baseline
    sat_n    = m.saturation_rate / 0.1;    % 10% baseline

    score = w_iae*iae_n + w_rms*rms_n + w_max*max_n + w_final*final_n ...
          + w_att*att_n + w_ctrl*ctrl_n + w_solve*solve_n + w_sat*sat_n;

    % Invert so higher q = better (q = 1/(1+score), ranges [0,1])
    q = 1 / (1 + score);
end
