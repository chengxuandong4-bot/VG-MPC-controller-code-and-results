function metrics = compute_metrics(result, params)
trackedCount = params.sim.tracked_state_count;
stepCount = numel(result.time);

error = result.x(1:trackedCount, :) - result.ref(1:trackedCount, :);
errorNorm = vecnorm(error, 2, 1);
controlNorm = vecnorm(result.u, 2, 1);

validError = isfinite(errorNorm);
validControl = isfinite(controlNorm);

if any(validError)
    integralErrorRaw = sum(abs(errorNorm(validError)));
    rmsTracking = sqrt(mean(errorNorm(validError) .^ 2));
    maxTracking = max(errorNorm(validError));
    finalTracking = errorNorm(find(validError, 1, 'last'));
else
    integralErrorRaw = inf;
    rmsTracking = inf;
    maxTracking = inf;
    finalTracking = inf;
end

missingRatio = 1.0 - (nnz(validError) / max(1, numel(errorNorm)));
failurePenalty = 1e6 * max(0.0, missingRatio);
if isfield(result, 'failed') && result.failed
    failurePenalty = failurePenalty + 1e6;
end

mpcRefreshMask = result.mpc_refresh > 0;
mpcReuseMask = result.mpc_reuse > 0;
holdStepsSeries = result.mpc_hold_steps;

if ~any(mpcRefreshMask) && ~any(mpcReuseMask)
    % For controllers without event-trigger logs, treat each step as refresh.
    mpcRefreshMask = true(1, stepCount);
    mpcReuseMask = false(1, stepCount);
    holdStepsSeries = zeros(1, stepCount);
end

refreshCount = sum(mpcRefreshMask);
reuseCount = sum(mpcReuseMask);
totalTimeSeconds = result.time(end) - result.time(1) + params.sim.Ts;

validHoldSteps = holdStepsSeries(isfinite(holdStepsSeries));
if isempty(validHoldSteps)
    meanHoldSteps = 0.0;
else
    meanHoldSteps = mean(validHoldSteps);
end

[meanRefreshIntervalSteps, refreshIntervalStdSteps] = event_interval_stats(mpcRefreshMask);
if ~isfinite(meanRefreshIntervalSteps) || meanRefreshIntervalSteps <= 0
    meanRefreshIntervalSteps = stepCount / max(1, refreshCount);
end
if ~isfinite(refreshIntervalStdSteps)
    refreshIntervalStdSteps = 0.0;
end

deltaU = diff(result.u, 1, 2);
if isempty(deltaU)
    deltaUNorm = zeros(1, 1);
else
    deltaUNorm = vecnorm(deltaU, 2, 1);
end

metrics = struct();
metrics.name = result.controller.name;
metrics.integral_error_raw = integralErrorRaw;
metrics.integral_error = integralErrorRaw + failurePenalty;
metrics.rms_tracking_error = rmsTracking;
metrics.max_tracking_error = maxTracking;
metrics.final_tracking_error = finalTracking;
metrics.control_energy = safe_sum(result.u(:) .^ 2);
metrics.mean_control_norm = safe_mean(controlNorm(validControl));
metrics.peak_control_norm = safe_max(controlNorm(validControl));
metrics.peak_input = safe_max(abs(result.u(:)));
metrics.controller_runtime = result.controller_runtime;
metrics.avg_controller_time_ms = 1000.0 * result.controller_runtime / max(1, stepCount);
metrics.simulation_runtime = result.simulation_runtime;
metrics.fault1_peak_error = window_peak(errorNorm, result.time, params.sim.fault_time_1, params.sim.fault_time_2);
metrics.fault2_peak_error = window_peak(errorNorm, result.time, params.sim.fault_time_2, result.time(end));
metrics.fault1_recovery_time = recovery_time(errorNorm, result.time, params.sim.fault_time_1);
metrics.fault2_recovery_time = recovery_time(errorNorm, result.time, params.sim.fault_time_2);
metrics.mean_health_estimate_act2 = safe_mean(result.health_hat(2, :));
metrics.mean_health_estimate_act3 = safe_mean(result.health_hat(3, :));
metrics.health_estimation_mae_act2 = safe_mae(result.health_hat(2, :), result.fault(2, :));
metrics.health_estimation_mae_act3 = safe_mae(result.health_hat(3, :), result.fault(3, :));
metrics.mean_innovation_norm = safe_mean(result.innovation_norm);
metrics.mean_fault_confidence = safe_mean(result.fault_confidence);
metrics.mean_perf_ratio = safe_mean(result.perf_ratio);
metrics.mean_gain_scale = safe_mean(result.gain_scale);
metrics.failed = isfield(result, 'failed') && result.failed;
metrics.failure_step = get_or_default(result, 'failure_step', NaN);
metrics.missing_ratio = missingRatio;

metrics.mpc_refresh_count = refreshCount;
metrics.mpc_refresh_ratio = refreshCount / max(1, stepCount);
metrics.mpc_reuse_count = reuseCount;
metrics.mpc_reuse_ratio = reuseCount / max(1, stepCount);
metrics.refresh_duty_cycle = metrics.mpc_refresh_ratio;
metrics.hold_duty_cycle = metrics.mpc_reuse_ratio;
metrics.reuse_to_refresh_ratio = reuseCount / max(1, refreshCount);
metrics.mean_refresh_interval_steps = meanRefreshIntervalSteps;
metrics.refresh_interval_std_steps = refreshIntervalStdSteps;
metrics.refresh_interval_cv = refreshIntervalStdSteps / max(meanRefreshIntervalSteps, 1e-9);
metrics.mean_refresh_interval_seconds = meanRefreshIntervalSteps * params.sim.Ts;
metrics.refresh_interval_std_seconds = refreshIntervalStdSteps * params.sim.Ts;
metrics.mean_hold_steps = meanHoldSteps;
metrics.lightweight_refresh_reduction_pct = 100.0 * (1.0 - metrics.mpc_refresh_ratio);
metrics.total_time_seconds = totalTimeSeconds;
metrics.refresh_frequency_hz = refreshCount / max(totalTimeSeconds, params.sim.Ts);
metrics.reuse_frequency_hz = reuseCount / max(totalTimeSeconds, params.sim.Ts);
metrics.mean_controller_time_per_refresh_ms = 1000.0 * result.controller_runtime / max(1, refreshCount);
metrics.mean_controller_time_per_reuse_ms = 1000.0 * result.controller_runtime / max(1, reuseCount);
metrics.compute_load_proxy = metrics.avg_controller_time_ms * metrics.mpc_refresh_ratio;
metrics.error_per_refresh = integralErrorRaw / max(1, refreshCount);
metrics.error_per_second = integralErrorRaw / max(totalTimeSeconds, params.sim.Ts);

metrics.mean_delta_u_norm = safe_mean(deltaUNorm);
metrics.max_delta_u_norm = safe_max(deltaUNorm);
metrics.control_variation_energy = safe_sum(deltaUNorm .^ 2);
metrics.quiet_update_ratio = safe_mean(double(deltaUNorm < 1e-3));
metrics.medium_update_ratio = safe_mean(double(deltaUNorm < 1e-2));
metrics.large_update_ratio = safe_mean(double(deltaUNorm > 0.10));
metrics.high_update_ratio = safe_mean(double(deltaUNorm > 0.25));
metrics.active_update_ratio = safe_mean(double(deltaUNorm > 1e-4));

[metrics.hold_segment_count, metrics.mean_hold_segment_len, metrics.max_hold_segment_len] = ...
    hold_segment_stats(mpcReuseMask);
metrics.mean_hold_segment_seconds = metrics.mean_hold_segment_len * params.sim.Ts;
metrics.max_hold_segment_seconds = metrics.max_hold_segment_len * params.sim.Ts;
metrics.refresh_segment_count = sum(mpcRefreshMask);
end

function peakValue = window_peak(errorNorm, time, tStart, tEnd)
mask = time >= tStart & time < tEnd & isfinite(errorNorm);
if ~any(mask)
    peakValue = NaN;
else
    peakValue = max(errorNorm(mask));
end
end

function tRecovery = recovery_time(errorNorm, time, tFault)
postMask = find(time >= tFault & isfinite(errorNorm), 1, 'first');
if isempty(postMask)
    tRecovery = NaN;
    return;
end

tail = errorNorm(postMask:end);
tail = tail(isfinite(tail));
if isempty(tail)
    tRecovery = NaN;
    return;
end

threshold = max(0.20, 0.25 * max(tail));
settleWindow = 100;
tRecovery = NaN;
for idx = postMask:max(postMask, numel(errorNorm) - settleWindow)
    segment = errorNorm(idx:min(numel(errorNorm), idx + settleWindow - 1));
    if all(isfinite(segment)) && all(segment < threshold)
        tRecovery = time(idx) - tFault;
        return;
    end
end
end

function value = safe_sum(samples)
valid = samples(isfinite(samples));
if isempty(valid)
    value = NaN;
else
    value = sum(valid);
end
end

function value = safe_mean(samples)
valid = samples(isfinite(samples));
if isempty(valid)
    value = NaN;
else
    value = mean(valid);
end
end

function value = safe_max(samples)
valid = samples(isfinite(samples));
if isempty(valid)
    value = NaN;
else
    value = max(valid);
end
end

function value = safe_mae(estimate, truth)
mask = isfinite(estimate) & isfinite(truth);
if ~any(mask)
    value = NaN;
else
    value = mean(abs(estimate(mask) - truth(mask)));
end
end

function value = get_or_default(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function [count, meanLen, maxLen] = hold_segment_stats(reuseMask)
if isempty(reuseMask)
    count = 0;
    meanLen = 0;
    maxLen = 0;
    return;
end

d = diff([false, reuseMask(:).', false]);
startIdx = find(d == 1);
endIdx = find(d == -1) - 1;
if isempty(startIdx)
    count = 0;
    meanLen = 0;
    maxLen = 0;
    return;
end

lens = endIdx - startIdx + 1;
count = numel(lens);
meanLen = mean(lens);
maxLen = max(lens);
end

function [meanInterval, stdInterval] = event_interval_stats(eventMask)
idx = find(eventMask(:).' > 0);
if numel(idx) <= 1
    meanInterval = NaN;
    stdInterval = NaN;
    return;
end

intervals = diff(idx);
meanInterval = mean(intervals);
stdInterval = std(intervals);
end
