#!/usr/bin/env python3
"""RAMPC full experiment runner (replaces MATLAB loop for RAMPC).
   Usage: python3 run_rampc_python.py <mode> <output_dir>
   mode: single | dual_obj | dual_strategy
"""
import json, os, sys, time as tmod, copy, random
import numpy as np
from rampc_casadi_sim import simulate

def build_pool():
    """Same 1200-config pool as MATLAB build_rampc_pool."""
    pool = []
    Q_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    R_vals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    S_vals = [0.10, 0.20, 0.30, 0.50, 0.70]
    V_vals = [0.6, 0.9, 1.2, 1.5, 2.0]
    idx = 0
    for q in Q_vals:
        for r in R_vals:
            for s in S_vals:
                for v in V_vals:
                    idx += 1
                    cfg = {
                        'name': f'r{idx:03d}_q{q:.0f}_r{r:.2f}_s{s:.2f}_v{v:.1f}',
                        'Q_position': q, 'Q_heading': q/3, 'Q_velocity': q/6,
                        'R_linear_vel': r, 'R_angular_vel': r/2,
                        'terminal_weight': q,
                        'safety_margin': s, 'obs_repulsion_w': 10 + 30*s,
                        'slack_penalty': 20 + 80*s,
                        'max_linear_vel': v, 'max_angular_vel': min(v, 1.5),
                        'tube_radius': 0.05 + 0.15*s,
                        'adaptive_gain': 0.5 + 2.0*s,
                        'disturbance_bound': 0.05,
                        'warm_start_on': (idx % 3 != 0),
                        'constraint_soft_on': (idx % 2 == 0),
                    }
                    pool.append(cfg)
    return pool


def select_from_pool(pool, incumbent, usage, last_used, gen, role):
    """Weighted selection from pool."""
    n = len(pool)
    scores = np.zeros(n)
    for i in range(n):
        c = pool[i]
        if role in ('performance_driven', 'performance'):
            scores[i] = c['Q_position']/10 - c['R_linear_vel']/2 + c['max_linear_vel']/3
        elif role in ('robustness_driven', 'robustness'):
            scores[i] = c['safety_margin']*3 + c['obs_repulsion_w']/30 - c['max_linear_vel']/3
        elif role == 'local_exploit':
            scores[i] = -random.random()*0.5
        elif role == 'structural_explore':
            scores[i] = random.gauss(0, 3)
        else:
            scores[i] = random.gauss(0, 1)
        scores[i] -= 0.1 * usage[i]
        if last_used[i] > gen - 10:
            scores[i] -= 0.2
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()
    idx = np.random.choice(n, p=probs)
    usage[idx] += 1
    last_used[idx] = gen
    return copy.deepcopy(pool[idx]), idx


def compute_metrics(result):
    m = result.get('metrics', [0,0,0,0,0])
    return [result['cost'], result['state_energy'], result['control_energy'],
            result['min_clearance'], result['plan_width']]


def aggregate_score(m):
    if any(np.isinf(m)) or any(np.isnan(m)):
        return -float('inf')
    w = [0.30, 0.15, 0.15, 0.25, 0.15]
    phi = [max(0, 1-m[0]/(m[0]+10)), max(0, 1-m[1]/(m[1]+20)),
           max(0, 1-m[2]/(m[2]+10)), min(1, m[3]/(m[3]+0.5)), max(0, 1-m[4])]
    return sum(w[i]*phi[i] for i in range(5))


def compute_violations(result):
    v = 0
    if result.get('collision', False): v += 500
    if not result.get('goal_reached', False): v += 500
    return v


def acceptance_gate(result, metrics):
    if result.get('collision', False): return False
    if metrics[3] < 0.05: return False  # min_clearance
    return True


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run_experiment(mode, output_dir):
    pool = build_pool()
    print(f"Pool: {len(pool)} configs, mode={mode}")

    T = 200 if mode == 'single' else 100
    params = {
        'N_sim': 200, 'Ts': 0.05, 'N_pred': 10,
        'goal': [3.0, 0.0], 'x0': [0.0, 0.0, 0.0, 0.0, 0.0],
        'obstacles': [{'center': [2.0, 1.0], 'radius': 0.5}]
    }

    # Baseline
    baseline_cfg = {'Q_position': 3, 'Q_heading': 1, 'Q_velocity': 0.5,
                    'R_linear_vel': 0.3, 'R_angular_vel': 0.15,
                    'terminal_weight': 5, 'safety_margin': 0.3,
                    'obs_repulsion_w': 15, 'max_linear_vel': 1.2, 'max_angular_vel': 1.0}
    base_result = simulate(baseline_cfg, params)
    base_m = compute_metrics(base_result)
    base_q = aggregate_score(base_m)
    print(f"Baseline q={base_q:.4f} cost={base_result['cost']:.1f} goal={base_result['goal_reached']} coll={base_result['collision']}")

    os.makedirs(f"{output_dir}/generation_snapshots", exist_ok=True)
    incumbent = {'cfg': baseline_cfg, 'q': base_q, 'gen': 0}
    usage = np.zeros(len(pool), dtype=int)
    last_used = np.zeros(len(pool), dtype=int)

    def save_snap(gen, cand, result, role='S'):
        tag = f"gen_{gen:03d}_{role}"
        sd = f"{output_dir}/generation_snapshots/{tag}"
        os.makedirs(sd, exist_ok=True)
        m = compute_metrics(result)
        s = aggregate_score(m)
        v = compute_violations(result)
        q = s - v
        acc = acceptance_gate(result, m)
        write_json(f"{sd}/config_snapshot.json", cand)
        write_json(f"{sd}/metrics_snapshot.json", {
            'm_vec': m, 'q_score': q, 's_score': s, 'v_penalty': v,
            'accepted': acc, 'updated': False})
        write_json(f"{sd}/meta_snapshot.json", {
            'generation': gen, 'name': cand.get('name',''), 'q_score': q,
            'accepted': acc, 'source': role, 'timestamp': tmod.strftime('%Y-%m-%d %H:%M:%S')})

    def run_single():
        inc = dict(cfg=baseline_cfg, q=base_q, gen=0)
        save_snap(0, baseline_cfg, base_result, 'baseline')
        for t in range(1, T+1):
            cand, idx = select_from_pool(pool, inc, usage, last_used, t, 'explore')
            cand['name'] = f"gen{t:03d}_S_{cand['name']}"
            result = simulate(cand, params)
            m = compute_metrics(result)
            s = aggregate_score(m)
            v = compute_violations(result)
            q = s - v
            acc = acceptance_gate(result, m)
            upd = acc and q > inc['q']
            if upd:
                inc = dict(cfg=cand, q=q, gen=t)
            if upd or t%20==0 or t==T:
                print(f"[{t:03d}/{T}] q={q:.4f} cost={result['cost']:.1f} {'NEW BEST' if upd else ''}")
            save_snap(t, cand, result, 'S')
        print(f"Single DONE: best gen {inc['gen']} q={inc['q']:.4f}")
        return inc

    def run_dual():
        inc = dict(cfg=None, q=-float('inf'), gen=0)
        sA, sB = 0, 0
        for t in range(1, T+1):
            candA, _ = select_from_pool(pool, inc, usage, last_used, t, 'performance_driven' if mode=='dual_obj' else 'local_exploit')
            candB, _ = select_from_pool(pool, inc, usage, last_used, t, 'robustness_driven' if mode=='dual_obj' else 'structural_explore')
            candA['name'] = f"gen{t:03d}_A_{candA['name']}"
            candB['name'] = f"gen{t:03d}_B_{candB['name']}"

            rA = simulate(candA, params); rB = simulate(candB, params)
            mA = compute_metrics(rA); mB = compute_metrics(rB)
            qA = aggregate_score(mA) - compute_violations(rA)
            qB = aggregate_score(mB) - compute_violations(rB)
            accA = acceptance_gate(rA, mA); accB = acceptance_gate(rB, mB)

            dq = qA - qB
            if not accA and accB: w = 'B'
            elif accA and not accB: w = 'A'
            elif not accA and not accB: w = 'N'
            elif dq > 0.005: w = 'A'
            elif dq < -0.005: w = 'B'
            else: w = 'T'

            upd = False
            if w == 'A' and qA > inc['q']:
                inc = dict(cfg=candA, q=qA, gen=t); upd = True; sA = 0; sB += 1
            elif w == 'B' and qB > inc['q']:
                inc = dict(cfg=candB, q=qB, gen=t); upd = True; sB = 0; sA += 1
            else:
                sA += 1; sB += 1

            if t%20==0 or t==1:
                print(f"[{t:03d}/{T}] A:q={qA:.4f} B:q={qB:.4f} win={w} best={inc['q']:.4f}")
            save_snap(t, candA, rA, 'A')
            save_snap(t, candB, rB, 'B')
        print(f"Dual DONE: best gen {inc['gen']} q={inc['q']:.4f}")
        return inc

    if mode == 'single':
        inc = run_single()
    else:
        inc = run_dual()

    # Final outputs
    summary = {
        'mode': mode, 'benchmark': 'rampc',
        'T_actual': T, 'best_gen': inc['gen'], 'best_q': inc['q'],
        'baseline_q': base_q, 'timestamp': tmod.strftime('%Y-%m-%d %H:%M:%S')
    }
    write_json(f"{output_dir}/summary_metrics.json", summary)
    with open(f"{output_dir}/指标分析报告.txt", 'w') as f:
        f.write(f"RAMPC {mode} — 指标分析报告\n")
        f.write(f"基线 q={base_q:.4f}, 最佳 q={inc['q']:.4f} (gen {inc['gen']})\n")
        if base_q > 0:
            f.write(f"提升: {((inc['q']/base_q)-1)*100:+.1f}%\n")
    print(f"Outputs written to {output_dir}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/rampc_results'
    run_experiment(mode, out_dir)
