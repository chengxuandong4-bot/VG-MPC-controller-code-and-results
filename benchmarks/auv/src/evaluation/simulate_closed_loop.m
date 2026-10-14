function result = simulate_closed_loop(params, controller)
numSteps = params.sim.num_steps;
trackedCount = params.sim.tracked_state_count;

xHist = nan(5, numSteps);
uHist = nan(4, numSteps);
faultHist = nan(4, numSteps);
refHist = nan(5, numSteps);
healthHatHist = nan(4, numSteps);
innovationNormHist = nan(1, numSteps);
faultConfidenceHist = nan(1, numSteps);
perfRatioHist = nan(1, numSteps);
gainScaleHist = nan(1, numSteps);
tauCmdHist = nan(3, numSteps);
errorNormHist = nan(1, numSteps);
mpcRefreshHist = zeros(1, numSteps);
mpcReuseHist = zeros(1, numSteps);
mpcHoldStepsHist = nan(1, numSteps);

x = params.sim.x0;
controllerCpu = 0.0;
failed = false;
failureStep = numSteps;

simTimer = tic;
for k = 1:numSteps
    stepIndex = k - 1;
    [ref, refDot] = paper_reference(stepIndex, params);
    actualFault = paper_fault(stepIndex, params);

    tController = tic;
    [u, controller, debugInfo] = controller_step(controller, x, ref, refDot, actualFault, params);
    controllerCpu = controllerCpu + toc(tController);

    xHist(:, k) = x;
    uHist(:, k) = u;
    faultHist(:, k) = actualFault;
    refHist(:, k) = ref;
    healthHatHist(:, k) = debugInfo.h_hat;
    innovationNormHist(k) = debugInfo.innovation_norm;
    faultConfidenceHist(k) = get_or_default(debugInfo, 'fault_confidence', NaN);
    perfRatioHist(k) = get_or_default(debugInfo, 'perf_ratio', NaN);
    gainScaleHist(k) = get_or_default(debugInfo, 'gain_scale', NaN);
    tauCmdHist(:, k) = get_or_default(debugInfo, 'tau_cmd', nan(3, 1));
    errorNormHist(k) = norm(x(1:trackedCount) - ref(1:trackedCount), 2);
    mpcRefreshHist(k) = get_or_default(debugInfo, 'mpc_refreshed', 0);
    mpcReuseHist(k) = get_or_default(debugInfo, 'mpc_reused', 0);
    mpcHoldStepsHist(k) = get_or_default(debugInfo, 'mpc_hold_steps', NaN);

    xNext = auv5_dynamics(x, u, actualFault, ref(4), params);
    if any(~isfinite(xNext)) || any(~isfinite(u))
        failed = true;
        failureStep = k;
        break;
    end
    x = xNext;
end

result = struct();
result.controller = controller;
result.time = (0:numSteps-1) * params.sim.Ts;
result.x = xHist;
result.u = uHist;
result.fault = faultHist;
result.ref = refHist;
result.health_hat = healthHatHist;
result.innovation_norm = innovationNormHist;
result.fault_confidence = faultConfidenceHist;
result.perf_ratio = perfRatioHist;
result.gain_scale = gainScaleHist;
result.tau_cmd = tauCmdHist;
result.error_norm = errorNormHist;
result.mpc_refresh = mpcRefreshHist;
result.mpc_reuse = mpcReuseHist;
result.mpc_hold_steps = mpcHoldStepsHist;
result.failed = failed;
result.failure_step = failureStep;
result.simulation_runtime = toc(simTimer);
result.controller_runtime = controllerCpu;
end

function value = get_or_default(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
