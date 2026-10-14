function fig = plot_paper_style_comparison(params, baselineResult, bestResult, bestConfig)
time = baselineResult.time;

baselineError = baselineResult.error_norm;
bestError = bestResult.error_norm;

baselineError(~isfinite(baselineError)) = NaN;
bestError(~isfinite(bestError)) = NaN;

baselineCum = cumsum(fillmissing(abs(baselineError), 'constant', 0.0));
bestCum = cumsum(fillmissing(abs(bestError), 'constant', 0.0));

fig = figure('Color', 'w', 'Position', [80 80 1080 1120]);
tiledlayout(4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
hold(ax1, 'on');
semilogy(ax1, time, baselineError, 'LineWidth', 1.6, 'Color', [0.10 0.30 0.75], 'DisplayName', baselineResult.controller.label);
semilogy(ax1, time, bestError, '--', 'LineWidth', 1.6, 'Color', [0.85 0.33 0.10], 'DisplayName', bestConfig.label);
apply_fault_shading(ax1, params);
ylabel(ax1, '||e||_2');
title(ax1, 'Original Paper Metric: Tracking Error Norm');
legend(ax1, 'Location', 'southwest');
grid(ax1, 'on');

ax2 = nexttile;
hold(ax2, 'on');
plot(ax2, time, baselineCum, 'LineWidth', 1.6, 'Color', [0.10 0.30 0.75], 'DisplayName', baselineResult.controller.label);
plot(ax2, time, bestCum, '--', 'LineWidth', 1.6, 'Color', [0.85 0.33 0.10], 'DisplayName', bestConfig.label);
apply_fault_shading(ax2, params);
ylabel(ax2, '\int ||e|| dt (discrete)');
title(ax2, 'Integrated Error Accumulation (Lower is Better)');
grid(ax2, 'on');

ax3 = nexttile;
hold(ax3, 'on');
plot(ax3, time, vecnorm(baselineResult.u, 2, 1), 'LineWidth', 1.4, 'Color', [0.10 0.30 0.75], 'DisplayName', baselineResult.controller.label);
plot(ax3, time, vecnorm(bestResult.u, 2, 1), '--', 'LineWidth', 1.4, 'Color', [0.85 0.33 0.10], 'DisplayName', bestConfig.label);
apply_fault_shading(ax3, params);
ylabel(ax3, '||u||_2');
title(ax3, 'Control Effort Norm');
grid(ax3, 'on');

ax4 = nexttile;
hold(ax4, 'on');
plot(ax4, time, bestResult.fault(2, :), 'LineWidth', 1.2, 'Color', [0.15 0.55 0.15], 'DisplayName', 'True h_2');
plot(ax4, time, bestResult.fault(3, :), 'LineWidth', 1.2, 'Color', [0.00 0.45 0.75], 'DisplayName', 'True h_3');
plot(ax4, time, bestResult.health_hat(2, :), '--', 'LineWidth', 1.4, 'Color', [0.40 0.80 0.20], 'DisplayName', 'Estimated h_2');
plot(ax4, time, bestResult.health_hat(3, :), '--', 'LineWidth', 1.4, 'Color', [0.20 0.70 0.95], 'DisplayName', 'Estimated h_3');
apply_fault_shading(ax4, params);
xlabel(ax4, 'Time [s]');
ylabel(ax4, 'Actuator health');
title(ax4, 'Adaptive Fault Estimation');
legend(ax4, 'Location', 'southwest');
grid(ax4, 'on');
end

function apply_fault_shading(ax, params)
currentY = ylim(ax);
xl = xlim(ax);

if isfinite(currentY(1)) && isfinite(currentY(2))
    patch(ax, [params.sim.fault_time_1 params.sim.fault_time_2 params.sim.fault_time_2 params.sim.fault_time_1], ...
        [currentY(1) currentY(1) currentY(2) currentY(2)], ...
        [1.00 0.95 0.75], 'EdgeColor', 'none', 'FaceAlpha', 0.26, 'HandleVisibility', 'off');
    patch(ax, [params.sim.fault_time_2 xl(2) xl(2) params.sim.fault_time_2], ...
        [currentY(1) currentY(1) currentY(2) currentY(2)], ...
        [0.82 1.00 0.82], 'EdgeColor', 'none', 'FaceAlpha', 0.24, 'HandleVisibility', 'off');
    uistack(findobj(ax, 'Type', 'line'), 'top');
    ylim(ax, currentY);
end
end
