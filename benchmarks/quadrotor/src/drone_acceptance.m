function [acc, reasons] = drone_acceptance(m, result, setup)
%DRONE_ACCEPTANCE Non-degradation gate for drone candidates.
    acc = true;
    reasons = {};

    % Feasibility checks
    if m.max_pos_error > 3.0
        acc = false;
        reasons{end+1} = sprintf('max_pos_error %.2f > 3.0 m', m.max_pos_error);
    end
    if m.final_pos_error > 2.5
        acc = false;
        reasons{end+1} = sprintf('final_pos_error %.2f > 2.5 m', m.final_pos_error);
    end
    if m.saturation_rate > 0.2
        acc = false;
        reasons{end+1} = sprintf('saturation_rate %.1f%% > 20%%', m.saturation_rate*100);
    end
    if m.max_solve_time_ms > 200
        acc = false;
        reasons{end+1} = sprintf('max_solve_time %.0f > 200 ms', m.max_solve_time_ms);
    end
end
