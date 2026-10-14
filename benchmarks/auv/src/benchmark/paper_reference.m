function [ref, refDot] = paper_reference(stepIndex, params)
timeNow = stepIndex * params.sim.Ts;
omega = 0.1;

ref = zeros(5, 1);
ref(1) = 0.5 + 0.2 * cos(omega * timeNow);
ref(2) = 0.2 * cos(omega * timeNow + 0.6);
ref(4) = params.sim.paper_ref_x4;

refDot = zeros(5, 1);
refDot(1) = -0.02 * sin(omega * timeNow);
refDot(2) = -0.02 * sin(omega * timeNow + 0.6);
end
