%TEST_DRONE_SMOKE Quick validation of drone benchmark.
fprintf('=== Drone Benchmark Smoke Test ===\n');

addpath(genpath(fileparts(mfilename('fullpath'))));

% 1. Model
[A, B, A_d, B_d, mp] = drone_model();
fprintf('Model: A(%dx%d) B(%dx%d) A_d(%dx%d) B_d(%dx%d) dt=%.3f\n', ...
    size(A,1),size(A,2), size(B,1),size(B,2), size(A_d,1),size(A_d,2), size(B_d,1),size(B_d,2), mp.dt);

% 2. Init params
setup = init_drone_params();
fprintf('Setup: T_sim=%.0fs x0=[%dx1]\n', setup.T_sim, length(setup.x0));

% 3. Baseline sim
fprintf('Running baseline simulation... ');
tic;
result = run_drone_sim(setup.baseline, setup);
fprintf('done (%.1fs, %d steps)\n', toc, size(result.x_log,2));

% 4. Metrics
m = compute_drone_metrics(result, setup);
fprintf('Metrics: IAE_pos=%.2f RMS_pos=%.3f max_pos=%.3f final_pos=%.3f\n', ...
    m.IAE_pos, m.RMS_pos_error, m.max_pos_error, m.final_pos_error);
fprintf('         IAE_att=%.3f ctrl_energy=%.1f solve=%.1fms sat=%.1f%%\n', ...
    m.IAE_att, m.control_energy, m.avg_solve_time_ms, m.saturation_rate*100);

% 5. Score
q = aggregate_drone_score(m);
v = compute_drone_violations(result, setup);
fprintf('Score: q=%.4f violations=%.4f\n', q, v);

% 6. Acceptance
[acc, reasons] = drone_acceptance(m, result, setup);
fprintf('Acceptance: %s\n', ternary(acc, 'PASS', 'FAIL'));
if ~acc, for i=1:numel(reasons), fprintf('  - %s\n', reasons{i}); end, end

% 7. Quick MPC solve test
fprintf('Testing single MPC solve... ');
x0 = setup.x0;
x_ref = zeros(12, setup.baseline.horizon_N+1);
for i = 1:setup.baseline.horizon_N+1
    x_ref(:,i) = setup.reference((i-1)*mp.dt);
end
u_ref = repmat([mp.mass*mp.gravity;0;0;0], 1, setup.baseline.horizon_N);
tic;
[u_opt, info] = drone_mpc_solve(x0, x_ref, u_ref, ...
    setup.baseline.Q_diag, setup.baseline.R_diag, setup.baseline.horizon_N, ...
    A_d, B_d, mp.u_min, mp.u_max);
t_solve = toc*1000;
fprintf('done (%.1fms, status=%s)\n', t_solve, info.status);
fprintf('  u_opt = [%.3f %.4f %.4f %.4f]\n', u_opt(1), u_opt(2), u_opt(3), u_opt(4));

fprintf('\n=== SMOKE TEST COMPLETE ===\n');

function v = ternary(c, a, b)
    if c, v = a; else, v = b; end
end
