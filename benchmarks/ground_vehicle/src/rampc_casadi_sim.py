#!/usr/bin/env python3
"""RAMPC CasADi NMPC simulation engine.
   Reads candidate_config.json, runs closed-loop NMPC, writes metrics_result.json.
   Called by MATLAB via: system('python3 rampc_casadi_sim.py candidate.json result.json')
"""
import json, sys, os, time as tmod
import numpy as np
import casadi as ca

def simulate(candidate_config, params):
    """Run CasADi NMPC closed-loop simulation."""

    # Unpack params
    N_sim = params.get('N_sim', 200)
    Ts = params.get('Ts', 0.05)
    N_pred = min(params.get('N_pred', 15), 20)
    goal = np.array(params.get('goal', [4.0, 0.0]))
    x0 = np.array(params.get('x0', [0.0, 0.0, 0.0, 0.0, 0.0]))
    obstacles = params.get('obstacles', [{'center': [3.0, 0.0], 'radius': 0.5}])

    # Candidate params (with defaults)
    Q_pos = candidate_config.get('Q_position', 3.0)
    Q_head = candidate_config.get('Q_heading', 1.0)
    Q_vel = candidate_config.get('Q_velocity', 0.5)
    R_lin = candidate_config.get('R_linear_vel', 0.3)
    R_ang = candidate_config.get('R_angular_vel', 0.15)
    Q_term = candidate_config.get('terminal_weight', 5.0)
    safety_margin = candidate_config.get('safety_margin', 0.3)
    v_max = candidate_config.get('max_linear_vel', 1.2)
    omega_max = candidate_config.get('max_angular_vel', 1.0)

    # Build NMPC problem with CasADi
    opti = ca.Opti()

    # Decision variables: states and controls over prediction horizon
    X = opti.variable(5, N_pred + 1)
    U = opti.variable(2, N_pred)

    # Initial condition
    opti.subject_to(X[:, 0] == x0)

    # Cost
    cost = 0
    for k in range(N_pred):
        xk = X[:, k]
        uk = U[:, k]
        # Tracking cost
        pos_err = xk[:2] - goal
        cost += Q_pos * ca.sumsqr(pos_err)
        cost += Q_head * xk[2]**2
        cost += Q_vel * (xk[3]**2 + xk[4]**2)
        # Control cost
        cost += R_lin * uk[0]**2 + R_ang * uk[1]**2
        # Dynamics constraint
        x_next = ca.vertcat(
            xk[0] + Ts * uk[0] * ca.cos(xk[2]),
            xk[1] + Ts * uk[0] * ca.sin(xk[2]),
            xk[2] + Ts * uk[1],
            uk[0],
            uk[1]
        )
        opti.subject_to(X[:, k+1] == x_next)
        # Input limits
        opti.subject_to(opti.bounded(-v_max, uk[0], v_max))
        opti.subject_to(opti.bounded(-omega_max, uk[1], omega_max))
        # Obstacle avoidance (soft constraint)
        for obs in obstacles:
            oc = np.array(obs['center'])
            or_rad = obs['radius']
            dist = ca.sqrt(ca.sumsqr(xk[:2] - oc))
            opti.subject_to(dist >= or_rad + safety_margin - 0.05)  # slightly softened

    # Terminal cost
    opti.subject_to(ca.sumsqr(X[:2, N_pred] - goal) <= 0.1)  # soft terminal constraint
    cost += Q_term * ca.sumsqr(X[:2, N_pred] - goal)

    opti.minimize(cost)

    # Solver options
    opts = {'print_time': False, 'ipopt.print_level': 0, 'print_level': 0}
    opti.solver('ipopt', opts)

    # Closed-loop simulation
    x = x0.copy()
    x_hist = np.zeros((5, N_sim))
    u_hist = np.zeros((2, N_sim))
    total_cost = 0.0
    state_energy = 0.0
    ctrl_energy = 0.0
    min_clearance = np.inf
    collision = False
    solver_ok = True
    solve_times = []

    actual_steps = N_sim
    for k in range(N_sim):
        # Early stop if goal reached
        if np.linalg.norm(x[:2] - goal) < 0.3:
            actual_steps = k
            break
        # Set initial condition for this MPC step
        opti.set_initial(X[:, 0], x)

        try:
            t0 = tmod.time()
            sol = opti.solve()
            solve_times.append(tmod.time() - t0)
            u_opt = sol.value(U[:, 0])
        except Exception as e:
            solver_ok = False
            # Fallback: proportional control
            e_pos = goal - x[:2]
            dist = np.linalg.norm(e_pos)
            u_opt = np.array([
                min(v_max, Q_pos * dist + 0.1),
                max(-omega_max, min(omega_max, 2.0 * (np.arctan2(e_pos[1], e_pos[0]) - x[2])))
            ])

        u_hist[:, k] = u_opt

        # Dynamics
        x_next = np.array([
            x[0] + Ts * u_opt[0] * np.cos(x[2]),
            x[1] + Ts * u_opt[0] * np.sin(x[2]),
            x[2] + Ts * u_opt[1],
            u_opt[0],
            u_opt[1]
        ])
        x = x_next
        x_hist[:, k] = x

        # Metrics
        dist_to_goal = np.linalg.norm(x[:2] - goal)
        total_cost += dist_to_goal + R_lin * abs(u_opt[0])
        state_energy += np.linalg.norm(x)
        ctrl_energy += np.linalg.norm(u_opt)**2

        # Safety check
        for obs in obstacles:
            oc = np.array(obs['center'])
            d = np.linalg.norm(x[:2] - oc) - obs['radius']
            min_clearance = min(min_clearance, d)
            if d < 0:
                collision = True

    result = {
        'cost': float(total_cost),
        'state_energy': float(state_energy),
        'control_energy': float(ctrl_energy),
        'min_clearance': float(min_clearance),
        'plan_width': float(np.mean(np.abs(u_hist[1, :]))),
        'collision': collision,
        'goal_reached': bool(np.linalg.norm(x[:2] - goal) < 0.5),
        'actual_steps': actual_steps,
        'solver_ok': solver_ok,
        'mean_solve_time': float(np.mean(solve_times)) if solve_times else 0.0,
        'max_saturation': float(max(abs(u_hist.flatten())) / max(v_max, omega_max))
    }
    return result


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        cfg_file = sys.argv[1]
        out_file = sys.argv[2]
    else:
        cfg_file = '/tmp/candidate_config.json'
        out_file = '/tmp/metrics_result.json'

    with open(cfg_file) as f:
        cfg = json.load(f)

    # Default params (can be overridden in cfg)
    params = {
        'N_sim': cfg.get('N_sim_', 200),
        'Ts': cfg.get('Ts_', 0.05),
        'N_pred': cfg.get('N_pred_', 10),
        'goal': cfg.get('goal_', [4.0, 0.0]),
        'x0': cfg.get('x0_', [0.0, 0.0, 0.0, 0.0, 0.0]),
        'obstacles': [{'center': cfg.get('obs_center_', [2.0, 0.3]),
                        'radius': cfg.get('obs_radius_', 0.5)}]
    }

    result = simulate(cfg, params)

    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Sim: cost={result['cost']:.1f} clear={result['min_clearance']:.3f} goal={result['goal_reached']} coll={result['collision']}")
