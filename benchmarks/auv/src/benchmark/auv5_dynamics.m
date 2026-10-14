function xNext = auv5_dynamics(x, u, h, refX4, params)
m = params.model.m;
Jz = params.model.Jz;
Xu = params.model.Xu;
Xuu = params.model.Xuu;
Yv = params.model.Yv;
Yvv = params.model.Yvv;
Nr = params.model.Nr;
Nrr = params.model.Nrr;
Ts = params.sim.Ts;

u = min(max(u, params.actuator.u_min), params.actuator.u_max);
tau = params.actuator.G * (h(:) .* u(:));

x1dot = (-Xu * x(1) - Xuu * x(1)^2 + m * x(2) * x(3) + tau(1)) / m;
x2dot = (-Yv * x(2) - Yvv * x(2)^2 - m * x(1) * x(3) + tau(2)) / m;
x3dot = (-Nr * x(3) - Nrr * x(3)^2 + tau(3)) / Jz;
x4dot = x(3);
x5dot = x(4) - refX4;

xNext = x + Ts * [x1dot; x2dot; x3dot; x4dot; x5dot];
end
