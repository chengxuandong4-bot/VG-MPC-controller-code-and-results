function h = paper_fault(stepIndex, params)
timeNow = stepIndex * params.sim.Ts;
h = ones(4, 1);

if timeNow >= params.sim.fault_time_1 && timeNow < params.sim.fault_time_2
    h(3) = params.sim.fault_level;
elseif timeNow >= params.sim.fault_time_2
    h(2) = params.sim.fault_level;
    h(3) = params.sim.fault_level;
end
end
