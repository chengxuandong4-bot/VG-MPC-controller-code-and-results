#!/usr/bin/env python3
"""
Verification-Guided LLM-Assisted Controller Synthesis — Python backend V2.
Aligns with paper (CEP revision V5.10) Section 2 prompt structure.

V2 enhancements over the baseline llm_generate.py:
  Point 1 — Few-shot CoT exemplars for controller tuning trajectories
  Point 3 — Module-level parameter patches with dual-generator search branches

Design:
  - RAMPC: full V2 treatment (exemplars + modules + diagnostics + patch output)
  - CSTR / AUV: module-ready enhanced prompts (exemplars to be added)
  - Dual mode: Generator A assigned to "performance" module, B to "robustness" module
  - Output: patch format {"module": "<name>", "changes": {"param": value}} for RAMPC;
            full config format for CSTR/AUV (until V2 extended)

Input:  JSON with incumbent config, feedback, metrics, history
Output: JSON with complete controller config + metadata
"""

import json, os, sys, requests, re, random, copy

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

def load_config():
    """Load LLM API config from file or environment."""
    config = {
        'provider': 'deepseek',
        'api_key': '',
        'model': 'deepseek-chat',
        'temperature': 0.3,
        'max_tokens': 2048,  # V2: raised from 512 to accommodate few-shot exemplars
        'base_url': 'https://api.deepseek.com/v1/chat/completions',
    }
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'llm_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            file_cfg = json.load(f)
            # Merge file config, but prefer environment for api_key if set
            for k, v in file_cfg.items():
                if k == 'api_key':
                    env_key = os.environ.get('DEEPSEEK_API_KEY', '')
                    if env_key:
                        config[k] = env_key
                    else:
                        config[k] = v
                else:
                    config[k] = v
    # Environment override takes highest priority
    env_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if env_key:
        config['api_key'] = env_key
    return config


# ═══════════════════════════════════════════════════════════════════
# Benchmark Specifications (unified with V1 BENCHMARKS)
# ═══════════════════════════════════════════════════════════════════

BENCHMARKS = {
    'cstr': {
        'name': 'CSTR set-point tracking (Lightweight QR-AMPC)',
        'task_desc': (
            'A Continuous Stirred Tank Reactor under parameter uncertainty '
            '(k0 ±20%, UA ±15%, CAf ±40%, Tf ±5K). The controller must track '
            'concentration setpoint changes (8→2→8→5 mol/L) while satisfying '
            'input constraints (285.15–312.15 K) and rate limits (±2 K/s).\n\n'
            'Controller architecture: QR-AMPC with optional adaptive horizon '
            'scheduling, disturbance observer, and parameter adaptation. '
            'The QP solver uses numerical regularization for conditioning.'
        ),
        'param_mode': 'full',
        'params': {
            # ── Structural flags ──
            'use_adaptive_horizon': {
                'type': 'bool', 'default': False,
                'desc': 'Enable horizon scheduling (longer horizon when tracking error is large)',
            },
            'use_disturbance_observer': {
                'type': 'bool', 'default': True,
                'desc': 'Enable disturbance observer for offset-free tracking',
            },
            'use_param_adaptation': {
                'type': 'bool', 'default': False,
                'desc': 'Enable online parameter adaptation (k0, UA)',
            },
            'use_input_bias_adaptation': {
                'type': 'bool', 'default': False,
                'desc': 'Enable input bias adaptation for persistent disturbances',
            },
            # ── Horizon scheduling ──
            'horizon_max': {
                'type': 'int', 'lo': 10, 'hi': 30, 'default': 20,
                'desc': 'Prediction horizon when tracking error is large',
            },
            'horizon_mid': {
                'type': 'int', 'lo': 5, 'hi': 25, 'default': 15,
                'desc': 'Prediction horizon at medium tracking error',
            },
            'horizon_min': {
                'type': 'int', 'lo': 3, 'hi': 15, 'default': 8,
                'desc': 'Prediction horizon when tracking error is small',
            },
            'horizon_high_err': {
                'type': 'float', 'lo': 0.1, 'hi': 2.0, 'default': 0.5,
                'desc': 'Tracking error threshold to switch to max horizon',
            },
            'horizon_mid_err': {
                'type': 'float', 'lo': 0.02, 'hi': 0.5, 'default': 0.1,
                'desc': 'Tracking error threshold to switch to mid horizon',
            },
            # ── Integral action ──
            'int_ref_gain': {
                'type': 'float', 'lo': 0.0, 'hi': 2.0, 'default': 0.0,
                'desc': 'Integral reference gain (0 = no integral action)',
            },
            'int_e_bound': {
                'type': 'float', 'lo': 0.1, 'hi': 5.0, 'default': 0.5,
                'desc': 'Bound on integral error accumulation',
            },
            # ── Tracking weight family ──
            'Wy_base': {
                'type': 'float', 'lo': 0.3, 'hi': 10.0, 'default': 1.0,
                'desc': 'Base output tracking weight. HIGHER → tighter tracking but may oscillate',
            },
            'Wy_gain': {
                'type': 'float', 'lo': 0.0, 'hi': 5.0, 'default': 0.3,
                'desc': 'Tracking weight gain factor for error-dependent scaling',
            },
            'Wy_cap': {
                'type': 'float', 'lo': 10.0, 'hi': 500.0, 'default': 100.0,
                'desc': 'Upper cap on tracking weight to prevent ill-conditioning',
            },
            # ── Input penalty family ──
            'Wu': {
                'type': 'float', 'lo': 1e-6, 'hi': 1e-2, 'default': 1e-4,
                'desc': 'Input move penalty. HIGHER → smoother control, slower response',
            },
            'Wdu_base': {
                'type': 'float', 'lo': 1e-5, 'hi': 1e-2, 'default': 1e-3,
                'desc': 'Base input rate penalty. Key for robustness',
            },
            'Wdu_gain': {
                'type': 'float', 'lo': 0.0, 'hi': 5.0, 'default': 0.2,
                'desc': 'Input rate penalty gain factor',
            },
            'Wdu_cap': {
                'type': 'float', 'lo': 10.0, 'hi': 500.0, 'default': 100.0,
                'desc': 'Upper cap on input rate penalty',
            },
            # ── Numerical regularization ──
            'hessian_reg': {
                'type': 'float', 'lo': 1e-8, 'hi': 1e-3, 'default': 1e-5,
                'desc': 'Hessian regularization. HIGHER → better conditioning, slower convergence',
            },
            # ── Observer ──
            'obs_alpha': {
                'type': 'float', 'lo': 0.05, 'hi': 0.95, 'default': 0.3,
                'desc': 'Observer forgetting factor (0=slow memory, 1=fast adaptation)',
            },
            'obs_bound': {
                'type': 'float', 'lo': 0.1, 'hi': 2.0, 'default': 0.5,
                'desc': 'Bound on observer state estimate',
            },
            # ── Parameter adaptation ──
            'gamma_k0': {
                'type': 'float', 'lo': 0.01, 'hi': 1.0, 'default': 0.1,
                'desc': 'Adaptation rate for reaction rate k0',
            },
            'theta_k0_max': {
                'type': 'float', 'lo': 0.5, 'hi': 5.0, 'default': 1.5,
                'desc': 'Max multiplier for k0 adaptation',
            },
            'theta_k0_min': {
                'type': 'float', 'lo': 0.1, 'hi': 2.0, 'default': 0.5,
                'desc': 'Min multiplier for k0 adaptation',
            },
            'gamma_ua': {
                'type': 'float', 'lo': 0.01, 'hi': 1.0, 'default': 0.1,
                'desc': 'Adaptation rate for heat transfer UA',
            },
            'theta_ua_max': {
                'type': 'float', 'lo': 0.5, 'hi': 5.0, 'default': 1.5,
                'desc': 'Max multiplier for UA adaptation',
            },
            'theta_ua_min': {
                'type': 'float', 'lo': 0.1, 'hi': 2.0, 'default': 0.5,
                'desc': 'Min multiplier for UA adaptation',
            },
            'adapt_eps': {
                'type': 'float', 'lo': 0.001, 'hi': 0.1, 'default': 0.01,
                'desc': 'Adaptation deadzone — no update when error < eps',
            },
            # ── Input bias adaptation ──
            'u_bias_gain': {
                'type': 'float', 'lo': 0.01, 'hi': 1.0, 'default': 0.1,
                'desc': 'Input bias estimation gain',
            },
            'u_bias_int_gain': {
                'type': 'float', 'lo': 0.001, 'hi': 0.1, 'default': 0.01,
                'desc': 'Input bias integral gain',
            },
            'u_bias_bound': {
                'type': 'float', 'lo': 0.1, 'hi': 2.0, 'default': 0.5,
                'desc': 'Bound on input bias estimate',
            },
        },
        'metrics': {
            'IAE':                     ('Integral absolute tracking error', 'lower'),
            'RMS_tracking_error':      ('RMS tracking error', 'lower'),
            'max_tracking_error':      ('Max instantaneous tracking error', 'lower'),
            'final_tracking_error':    ('Final tracking error at end of horizon', 'lower'),
            'control_energy':          ('Total control energy (ΣΔu²)', 'lower'),
            'avg_solve_time_ms':       ('Average QP solve time in ms', 'lower'),
            'median_condition_number': ('Median Hessian condition number', 'lower'),
            'max_KKT_residual':        ('Max KKT residual — numerical quality indicator', 'lower'),
        },
        'acceptance': (
            'Non-degradation gate: all 8 metrics must be no worse than baseline.\n'
            'Also: KKT residual < 1e8, condition number < 1e6.'
        ),
        # ═══════════════════════════════════════════════════════════════
        # V2: CSTR module definitions and search branch mapping
        # ═══════════════════════════════════════════════════════════════
        'modules': {
            'structural': {
                'params': ['use_adaptive_horizon', 'use_disturbance_observer',
                          'use_param_adaptation', 'use_input_bias_adaptation'],
                'desc': 'Architecture flags — toggle controller structure',
            },
            'horizon': {
                'params': ['horizon_max', 'horizon_mid', 'horizon_min',
                          'horizon_high_err', 'horizon_mid_err'],
                'desc': 'Prediction horizon scheduling — longer horizon = more lookahead but slower',
            },
            'tracking': {
                'params': ['Wy_base', 'Wy_gain', 'Wy_cap'],
                'desc': 'Output tracking weights — HIGHER Wy_base = tighter setpoint tracking',
            },
            'smoothness': {
                'params': ['Wu', 'Wdu_base', 'Wdu_gain', 'Wdu_cap'],
                'desc': 'Input penalty weights — HIGHER Wdu_base = smoother control, better robustness',
            },
            'adaptation': {
                'params': ['gamma_k0', 'theta_k0_max', 'theta_k0_min',
                          'gamma_ua', 'theta_ua_max', 'theta_ua_min', 'adapt_eps'],
                'desc': 'Parameter adaptation rates for k0 and UA uncertainty',
            },
            'observer': {
                'params': ['obs_alpha', 'obs_bound', 'int_ref_gain', 'int_e_bound'],
                'desc': 'Observer and integral action — disturbance rejection',
            },
            'numerical': {
                'params': ['hessian_reg', 'u_bias_gain', 'u_bias_int_gain', 'u_bias_bound'],
                'desc': 'Numerical regularization and bias estimation',
            },
        },
        # Dual-generator module assignments for CSTR
        'dual_role_modules': {
            # Generator A (performance-driven): tracking + horizon
            'A': 'tracking',
            # Generator B (robustness-driven): smoothness + numerical
            'B': 'smoothness',
            # Single-generator mode: can modify any module
            'S': 'all',
            'performance': 'tracking',
            'robustness': 'smoothness',
        },
    },

    'auv': {
        'name': 'AUV fault-tolerant control (AFTC)',
        'task_desc': (
            'An Autonomous Underwater Vehicle executing trajectory tracking '
            'under sequential actuator faults. Two fault windows are injected '
            'during the mission, each degrading specific thruster effectiveness.\n\n'
            'Controller architecture: Adaptive FTC with RLS-based online '
            'effectiveness estimation, health monitoring with decay/recovery, '
            'reconfigurable thrust allocation with regularization, event-triggered '
            'refresh logic, and optional performance boosting.\n\n'
            'The controller must maintain tracking accuracy during and after '
            'fault windows while keeping allocation smooth and avoiding saturation.'
        ),
        'param_mode': 'partial',
        'params': {
            # ── Velocity tracking ──
            'kp_vel': {
                'type': 'float', 'lo': 0.1, 'hi': 10.0, 'default': 0.8,
                'desc': 'Velocity proportional gain (surge+sway). HIGHER → tighter tracking, range 0.5-8 typical',
            },
            'ki_vel': {
                'type': 'float', 'lo': 0.01, 'hi': 2.0, 'default': 0.1,
                'desc': 'Velocity integral gain. Removes steady-state error, range 0.05-1.5 typical',
            },
            # ── Heading/rate control ──
            'k_psi': {
                'type': 'float', 'lo': 0.3, 'hi': 8.0, 'default': 1.5,
                'desc': 'Heading proportional gain',
            },
            'ki_psi': {
                'type': 'float', 'lo': 0.01, 'hi': 0.5, 'default': 0.05,
                'desc': 'Heading integral gain',
            },
            'k_r': {
                'type': 'float', 'lo': 0.1, 'hi': 10.0, 'default': 0.5,
                'desc': 'Yaw rate proportional gain. Values up to 8 typical',
            },
            'ki_r': {
                'type': 'float', 'lo': 0.005, 'hi': 0.5, 'default': 0.02,
                'desc': 'Yaw rate integral gain. Values up to 0.5 typical',
            },
            # ── Thrust allocation ──
            'ks_tau': {
                'type': 'float', 'lo': 0.05, 'hi': 500.0, 'default': 0.3,
                'desc': 'Thrust allocation gain. HIGHER → more aggressive allocation. Optimal values can be 50-300.',
            },
            'allocation_reg': {
                'type': 'float', 'lo': 1e-4, 'hi': 0.5, 'default': 0.01,
                'desc': 'Allocation regularization. Typical range 0.05-0.25',
            },
            'rate_limit': {
                'type': 'float', 'lo': 0.05, 'hi': 10.0, 'default': 2.0,
                'desc': 'Actuator rate limit. LOWER → smoother, typical 0.1-2.0',
            },
            # ── RLS estimation ──
            'rls_forgetting': {
                'type': 'float', 'lo': 0.90, 'hi': 0.9999, 'default': 0.98,
                'desc': 'RLS forgetting factor. 0.96-0.98 typical for fault adaptation',
            },
            'rls_fault_boost': {
                'type': 'float', 'lo': 0.001, 'hi': 0.2, 'default': 0.02,
                'desc': 'RLS forgetting boost on fault detection. 0.01-0.05 typical',
            },
            # ── Health monitoring ──
            'health_decay': {
                'type': 'float', 'lo': 0.001, 'hi': 0.999, 'default': 0.995,
                'desc': 'Health decay per step. Low values (<0.1) = rapid health drop after fault; high (>0.95) = slow decay. Optimal is near 0.001 — effectively disable health-based degradation.',
            },
            'health_weight_scale': {
                'type': 'float', 'lo': 0.5, 'hi': 5.0, 'default': 2.0,
                'desc': 'Health influence on allocation. Typical 0.5-2.0',
            },
            # ── Smoothing ──
            'smoothing_weight': {
                'type': 'float', 'lo': 0.0, 'hi': 1.0, 'default': 0.3,
                'desc': 'Control smoothing. HIGHER → more filtered, slower response',
            },
            # ── Fault feedforward ──
            'use_fault_feedforward': {
                'type': 'bool', 'default': False,
                'desc': 'Enable fault feedforward compensation',
            },
            'fault_feedforward_gain': {
                'type': 'float', 'lo': 0.1, 'hi': 3.0, 'default': 0.5,
                'desc': 'Fault feedforward gain. HIGHER → stronger pre-compensation',
            },
            # ── Performance boost ──
            'use_perf_boost': {
                'type': 'bool', 'default': False,
                'desc': 'Enable performance boosting after fault recovery',
            },
            # ── Allocation relax ──
            'use_alloc_relax': {
                'type': 'bool', 'default': False,
                'desc': 'Enable allocation constraint relaxation during saturation',
            },
        },
        'metrics': {
            'integral_error':    ('Integral absolute tracking error', 'lower'),
            'rms_tracking_error':('RMS tracking error', 'lower'),
            'control_energy':    ('Total control energy', 'lower'),
            'fault1_peak_error': ('Peak tracking error during fault-1 window', 'lower'),
            'fault2_peak_error': ('Peak tracking error during fault-2 window', 'lower'),
        },
        'acceptance': (
            'Simulation success + integral_error < 1e6 + no divergence.\n'
            'Fault peak errors and recovery speed are key quality indicators.'
        ),
        # V2: AUV module definitions
        'modules': {
            'tracking': {
                'params': ['kp_vel', 'ki_vel', 'k_psi', 'ki_psi', 'k_r', 'ki_r'],
                'desc': 'Velocity and heading tracking gains — tighter tracking vs stability',
            },
            'allocation': {
                'params': ['ks_tau', 'allocation_reg', 'rate_limit'],
                'desc': 'Thrust allocation — aggression vs smoothness of thruster commands',
            },
            'estimation': {
                'params': ['rls_forgetting', 'rls_fault_boost'],
                'desc': 'RLS fault estimation — speed vs stability of fault detection',
            },
            'health': {
                'params': ['health_decay', 'health_weight_scale'],
                'desc': 'Health monitoring — how quickly thrusters are trusted after faults',
            },
            'smoothing': {
                'params': ['smoothing_weight'],
                'desc': 'Control smoothing filter',
            },
            'fault_comp': {
                'params': ['use_fault_feedforward', 'fault_feedforward_gain',
                          'use_perf_boost', 'use_alloc_relax'],
                'desc': 'Fault compensation flags — structural toggles for fault response',
            },
        },
        'dual_role_modules': {
            'A': 'all',
            'B': 'all',
            'S': 'all',
            'performance': 'all',
            'robustness': 'all',
        },
    },

    'rampc': {
        'name': 'Obstacle-aware adaptive RAMPC (CSTAR-RAMPC)',
        'task_desc': (
            'A mobile robot navigating to a goal position while avoiding '
            'obstacles in the environment. The controller uses CasADi IPOPT '
            'for nonlinear MPC with obstacle avoidance constraints.\n\n'
            'Controller architecture: CSTAR-RAMPC with adaptive safety margin, '
            'obstacle repulsion weighting, and terminal cost for convergence.'
        ),
        'param_mode': 'full',
        'params': {
            'Q_position': {
                'type': 'float', 'lo': 0.5, 'hi': 20.0, 'default': 1.0,
                'desc': 'Position tracking weight. HIGHER → more aggressive toward goal',
            },
            'R_linear_vel': {
                'type': 'float', 'lo': 0.05, 'hi': 2.0, 'default': 0.1,
                'desc': 'Linear velocity penalty. HIGHER → slower, smoother motion',
            },
            'safety_margin': {
                'type': 'float', 'lo': 0.05, 'hi': 0.80, 'default': 0.3,
                'desc': 'Minimum clearance to obstacles in meters',
            },
            'max_linear_vel': {
                'type': 'float', 'lo': 0.4, 'hi': 2.5, 'default': 1.0,
                'desc': 'Maximum allowed linear velocity',
            },
            'obs_repulsion_w': {
                'type': 'float', 'lo': 5.0, 'hi': 50.0, 'default': 10.0,
                'desc': 'Obstacle repulsion weight. HIGHER → stronger avoidance',
            },
            'terminal_weight': {
                'type': 'float', 'lo': 1.0, 'hi': 20.0, 'default': 5.0,
                'desc': 'Terminal cost weight for convergence guarantee',
            },
        },
        'metrics': {
            'cost':              ('Realized total cost', 'lower'),
            'state_energy':      ('State trajectory energy', 'lower'),
            'control_energy':    ('Control input energy', 'lower'),
            'min_clearance':     ('Minimum obstacle clearance', 'higher'),
            'plan_theta_width':  ('Planning trajectory width', 'lower'),
        },
        'acceptance': (
            'No collision + goal reached + min_clearance >= safety_margin.'
        ),
        # ═══════════════════════════════════════════════════════════════
        # V2: RAMPC module definitions and search branch mapping
        # ═══════════════════════════════════════════════════════════════
        'modules': {
            'tracking': {
                'params': ['Q_position', 'terminal_weight'],
                'desc': 'Goal-reaching speed — balance Q_position for aggressiveness, terminal_weight for convergence',
            },
            'safety': {
                'params': ['safety_margin', 'obs_repulsion_w'],
                'desc': 'Obstacle avoidance — tune safety_margin for clearance, obs_repulsion_w for repulsion strength',
            },
            'smoothness': {
                'params': ['R_linear_vel', 'max_linear_vel'],
                'desc': 'Motion smoothness — tune R_linear_vel for velocity penalty, max_linear_vel for speed cap',
            },
        },
        # Dual-generator module assignments
        'dual_role_modules': {
            # Both A and B can access all params; system prompt uses
            # PRIMARY/SECONDARY language to guide search direction
            'A': 'all',
            'B': 'all',
            # Single-generator mode
            'S': 'all',
            # Legacy role labels from prompt_override
            'performance': 'all',
            'robustness': 'all',
        },
    },

    'drone': {
        'name': 'Quadrotor TinyMPC (linear QP-MPC)',
        'task_desc': (
            'A 12-state quadrotor hovering and tracking position setpoints '
            'under a linear MPC controller. The controller solves a QP at '
            '100 Hz with box constraints on motor thrust and body moments.\n\n'
            'Controller architecture: linear MPC with quadratic state/input '
            'costs, prediction horizon N, and dense QP solver (quadprog). '
            'The 12-element Q_diag vector weights position, velocity, attitude, '
            'and angular rate errors. The 4-element R_diag vector penalizes '
            'thrust and moment commands.'
        ),
        'param_mode': 'full',
        'params': {
            'Q_pos_xy': {
                'type': 'float', 'lo': 1.0, 'hi': 50.0, 'default': 10.0,
                'desc': 'Position (x,y) tracking weight. HIGHER → tighter horizontal tracking',
            },
            'Q_pos_z': {
                'type': 'float', 'lo': 2.0, 'hi': 100.0, 'default': 20.0,
                'desc': 'Altitude (z) tracking weight. HIGHER → tighter altitude hold',
            },
            'Q_vel_xy': {
                'type': 'float', 'lo': 0.1, 'hi': 10.0, 'default': 1.0,
                'desc': 'Horizontal velocity penalty. HIGHER → smoother horizontal motion',
            },
            'Q_vel_z': {
                'type': 'float', 'lo': 0.2, 'hi': 20.0, 'default': 2.0,
                'desc': 'Vertical velocity penalty. HIGHER → smoother climb/descent',
            },
            'Q_att_rp': {
                'type': 'float', 'lo': 0.01, 'hi': 5.0, 'default': 0.1,
                'desc': 'Roll/pitch attitude penalty. HIGHER → more level flight',
            },
            'Q_att_yaw': {
                'type': 'float', 'lo': 0.01, 'hi': 5.0, 'default': 0.1,
                'desc': 'Yaw attitude penalty',
            },
            'Q_rates': {
                'type': 'float', 'lo': 0.005, 'hi': 2.0, 'default': 0.05,
                'desc': 'Body angular rate penalty. HIGHER → smoother rotation',
            },
            'R_thrust': {
                'type': 'float', 'lo': 0.01, 'hi': 10.0, 'default': 0.1,
                'desc': 'Thrust input penalty. HIGHER → less aggressive altitude changes',
            },
            'R_moment': {
                'type': 'float', 'lo': 0.1, 'hi': 50.0, 'default': 1.0,
                'desc': 'Moment (tau_x, tau_y, tau_z) penalty. HIGHER → smoother attitude control',
            },
            'horizon_N': {
                'type': 'int', 'lo': 5, 'hi': 40, 'default': 15,
                'desc': 'MPC prediction horizon. Longer → more lookahead but slower solves',
            },
        },
        'metrics': {
            'IAE_pos':          ('Integral absolute position error', 'lower'),
            'RMS_pos_error':    ('RMS position tracking error', 'lower'),
            'max_pos_error':    ('Max instantaneous position error', 'lower'),
            'final_pos_error':  ('Final position error', 'lower'),
            'IAE_att':          ('Integral attitude error', 'lower'),
            'control_energy':   ('Total control energy', 'lower'),
            'avg_solve_time_ms':('Average MPC solve time', 'lower'),
            'saturation_rate':  ('Input saturation rate', 'lower'),
        },
        'acceptance': (
            'max_pos_error < 3.0 m, final_pos_error < 2.5 m, '
            'saturation_rate < 20%, max_solve_time < 200 ms.'
        ),
        'modules': None,
    },
}


# ═══════════════════════════════════════════════════════════════════
# RAMPC Few-Shot CoT Exemplars (Point 1)
# ═══════════════════════════════════════════════════════════════════

RAMPC_EXEMPLARS = [
    {
        'scenario': 'Tracking Direction Rejected by Evidence',
        'observation': (
            'Incumbent has q=0.384 with Q_position=12, terminal_weight=12. '
            'A candidate increased both to Q=18, T=18 and produced q=0.381 — '
            'a 0.003 degradation. cost rose from 37.1 to 39.9.'
        ),
        'diagnosis': (
            'The evidence REJECTS further simultaneous increases of Q_position '
            'and terminal_weight in this local region. The degradation is not '
            'explained by measurement noise — it is a consistent direction-level '
            'failure. Do NOT repeat this direction.'
        ),
        'module': 'tracking',
        'action': (
            'Either test a CONTROLLED DECREASE of one tracking parameter '
            '(e.g., Q=9.5 with T=12 held), or switch to an untested module '
            '(smoothness or safety). The key is to isolate which parameter '
            'change caused the degradation.'
        ),
        'tradeoff': (
            'If Q alone is reduced without changing T, cost may improve if '
            'excessive tracking aggressiveness was the limiting factor. '
            'control_energy may decrease as a side effect.'
        ),
        'output': '{"module":"tracking","changes":{"Q_position":9.5},"expected":"cost -3% to -8%; q improves only if over-aggressive tracking was limiting"}',
    },
    {
        'scenario': 'Feasibility Is Not Maximization',
        'observation': (
            'min_clearance = safety_margin = 0.50. No collision. q = 0.384 '
            '(incumbent best). A safety-driven candidate increased '
            'safety_margin from 0.5 to 0.7 and obs_repulsion_w from 35 to 50, '
            'producing q=0.365 — a 0.019 degradation.'
        ),
        'diagnosis': (
            'Safety is FEASIBLE (clearance ≥ margin) but increasing safety '
            'parameters further actively HARMED q. Safety is a constraint, '
            'not a maximization objective. When the constraint is satisfied, '
            'further safety increases are not justified by evidence.'
        ),
        'module': 'safety',
        'action': (
            'Do NOT modify the safety module solely because you are the safety '
            'generator. When safety is feasible and q has not improved from '
            'safety changes, select an untested hypothesis in tracking or '
            'smoothness instead.'
        ),
        'tradeoff': (
            'Excessive safety_margin or obs_repulsion_w can force longer paths '
            'that increase cost without benefit. Respect the feasibility gate.'
        ),
        'output': '{"module":"smoothness","changes":{"R_linear_vel":0.08},"expected":"isolate damping effect on cost; q neutral or slight improvement if damping is not currently limiting"}',
    },
    {
        'scenario': 'Diagnostic Label Does Not Mandate Action',
        'observation': (
            'control_energy is flagged HIGH (137.5) but the incumbent q=0.384 '
            'is the best known solution. Several smoothness candidates tried '
            'R_linear_vel=0.10→0.15 and max_linear_vel=2.5→2.0, all producing '
            'q ≤ 0.381 — none improved upon the incumbent.'
        ),
        'diagnosis': (
            'HIGH control_energy in the incumbent is an ACCEPTABLE TRADE-OFF '
            'for achieving q=0.384. The diagnostic label identifies a property, '
            'not a defect requiring repair. Multiple attempts to "fix" it '
            'have been rejected by evidence.'
        ),
        'module': 'smoothness',
        'action': (
            'Stop modifying smoothness parameters in response to this label. '
            'The smoothness-up and smoothness-down directions have both been '
            'tested and rejected. Switch to a genuinely untested direction in '
            'tracking or safety, or accept that the current configuration may '
            'be locally optimal.'
        ),
        'tradeoff': (
            'Repeatedly testing rejected directions wastes generations. '
            'The Directional Evidence Ledger tracks what has been tried.'
        ),
        'output': '{"module":"tracking","changes":{"terminal_weight":8.0},"expected":"cost may change ±3%; testing whether terminal_weight decoupling from Q_position affects q"}',
    },
    {
        'scenario': 'Boundary-Aware Exploration',
        'observation': (
            'Current Q_position=12 is at 60% of its range [0.5, 20.0]. '
            'max_linear_vel=2.5 is at the UPPER BOUND of its range [0.4, 2.5]. '
            'Both are near parameter space boundaries.'
        ),
        'diagnosis': (
            'When parameters are near range boundaries, large increases are '
            'either impossible (max_linear_vel) or have already been tested '
            '(Q_position). Exploration should favor the INTERIOR of the range '
            'or test decreases from boundary values.'
        ),
        'module': 'smoothness',
        'action': (
            'Test a decrease of max_linear_vel (e.g., 2.0) to probe whether '
            'the boundary value is genuinely optimal. For Q_position, test '
            'moderate decreases (8–10) rather than further increases.'
        ),
        'tradeoff': (
            'Boundary values are often local optima. Testing interior values '
            'can reveal whether the boundary itself is limiting q.'
        ),
        'output': '{"module":"smoothness","changes":{"max_linear_vel":2.0},"expected":"cost may increase slightly; tests whether speed boundary is necessary for current q"}',
    },
]
# ═══════════════════════════════════════════════════════════════════

CSTR_EXEMPLARS = [
    {
        'scenario': 'Tracking Improvement Achieved',
        'observation': (
            'Baseline q=0.082 with Wy_base=1.0 (default). IAE=2.1, control_energy=40. '
            'The controller is sluggish — tracking error dominates the score.'
        ),
        'diagnosis': (
            'Tracking weight Wy_base is too low at default 1.0. The controller '
            'prioritizes input smoothness over setpoint tracking. Increasing Wy_base '
            'should reduce IAE at the cost of higher control_energy.'
        ),
        'module': 'tracking',
        'action': (
            'Increase Wy_base from 1.0 to 2.0–4.0 range. Increase Wy_gain from '
            '0.3 to 0.8–1.5 for error-dependent scaling. Keep Wy_cap at 200–400 '
            'to avoid clipping. This is a bounded, attributable change.'
        ),
        'tradeoff': (
            'IAE should decrease. control_energy may increase — this is expected '
            'and acceptable as long as q improves. Monitor KKT residual for '
            'signs of ill-conditioning from higher weights.'
        ),
        'output': '{"module":"tracking","changes":{"Wy_base":3.0,"Wy_gain":1.0},"expected":"IAE -30% to -50%; q improves if tracking was the bottleneck"}',
    },
    {
        'scenario': 'Smoothness Improvement Achieved',
        'observation': (
            'q improved from 0.082 to 0.102 via Generator B (smoothness module). '
            'control_energy was reduced by increasing Wdu_base. This confirms '
            'that input penalty tuning can unlock q improvements.'
        ),
        'diagnosis': (
            'When control_energy is high and tracking is acceptable, increasing '
            'input rate penalty Wdu_base reduces oscillations without harming IAE. '
            'The Wy_base/Wdu_base ratio should stay below 3000 for stability.'
        ),
        'module': 'smoothness',
        'action': (
            'Increase Wdu_base from 1e-3 to 2e-3–8e-3. Increase Wdu_gain from '
            '0.2 to 0.5–1.5. This penalizes rapid input changes without affecting '
            'steady-state tracking. Keep Wu unchanged to isolate the effect.'
        ),
        'tradeoff': (
            'control_energy should decrease. IAE may increase slightly if '
            'Wdu_base is set too high. The key is finding the crossing point '
            'where q peaks.'
        ),
        'output': '{"module":"smoothness","changes":{"Wdu_base":0.005,"Wdu_gain":1.0},"expected":"control_energy -20% to -40%; q improves if oscillations were limiting"}',
    },
    {
        'scenario': 'Numerical Issue Interacting with Tracking',
        'observation': (
            'After increasing Wy_base, median_condition_number rose above 1e4. '
            'KKT residual increased. QP solver struggles — solve times doubled. '
            'The tracking improvement is undermined by numerical instability.'
        ),
        'diagnosis': (
            'Higher tracking weights amplify Hessian ill-conditioning. hessian_reg '
            'at default 1e-5 is insufficient. Wy_cap may be too high, allowing '
            'extreme weights. This is a TRACKING-NUMERICAL coupling issue.'
        ),
        'module': 'numerical',
        'action': (
            'Increase hessian_reg from 1e-5 to 1e-4–5e-4. Reduce Wy_cap from '
            '100 to 30–80. These changes improve QP conditioning without directly '
            'changing the tracking weights. Isolate numerical from tracking changes.'
        ),
        'tradeoff': (
            'Condition number should drop below 1e4. Tracking may degrade slightly '
            'due to increased regularization. The net effect on q depends on whether '
            'numerical issues or tracking error was the dominant bottleneck.'
        ),
        'output': '{"module":"numerical","changes":{"hessian_reg":0.0003,"Wy_cap":50.0},"expected":"condition_number -50%+; q improves only if numerics were the active bottleneck"}',
    },
    {
        'scenario': 'When Action Direction Is Rejected',
        'observation': (
            'Wy_base was increased from 1.0 to 5.0. q DECREASED from 0.102 to '
            '0.094 — a degradation despite the tracking metric improving. '
            'control_energy spiked, dominating the q calculation.'
        ),
        'diagnosis': (
            'The tracking improvement was REAL (IAE decreased), but the control_energy '
            'penalty outweighed it in the aggregate q. This direction — aggressive '
            'tracking weight increase — is REJECTED at the current operating point.'
        ),
        'module': 'tracking',
        'action': (
            'Do NOT repeat further Wy_base increases. The evidence shows this '
            'direction is counterproductive. Instead, try a DIFFERENT module '
            '(smoothness, numerical, or observer) or test a controlled DECREASE '
            'of Wy_base back toward 2.0–3.0.'
        ),
        'tradeoff': (
            'When a direction is rejected, the correct response is module switching, '
            'not further escalation in the same direction. Check the Evidence Ledger.'
        ),
        'output': '{"module":"smoothness","changes":{"Wdu_base":0.004},"expected":"control_energy -15% to -25%; tests whether input penalty can offset tracking aggressiveness"}',
    },
]


# ═══════════════════════════════════════════════════════════════════
# AUV Few-Shot CoT Exemplars
# ═══════════════════════════════════════════════════════════════════

AUV_EXEMPLARS = [
    {
        'scenario': 'Fault Response Too Slow',
        'observation': (
            'fault1_peak_error is elevated after the first fault window. '
            'integral_error accumulates during faults. RLS forgetting factor '
            'rls_forgetting=0.98 is too slow to adapt to sudden thruster faults.'
        ),
        'diagnosis': (
            'The RLS estimator adapts too slowly to abrupt effectiveness changes. '
            'rls_fault_boost should provide a temporary forgetting factor drop '
            'upon fault detection. If rls_fault_boost is too low, the estimator '
            'lags behind the true fault magnitude.'
        ),
        'module': 'estimation',
        'action': (
            'Increase rls_fault_boost from 0.02 to 0.03–0.08. Consider lowering '
            'rls_forgetting slightly to 0.95–0.97 for faster baseline adaptation. '
            'Enable use_fault_feedforward if not already active.'
        ),
        'tradeoff': (
            'fault1/2 peak errors should decrease. integral_error may improve. '
            'Too aggressive RLS can cause oscillation in fault-free periods.'
        ),
        'output': '{"name":"faster_fault_response","config":{"rls_fault_boost":0.05,"rls_forgetting":0.96,"use_fault_feedforward":true}}',
    },
    {
        'scenario': 'Thrust Allocation Too Aggressive',
        'observation': (
            'control_energy is high. ks_tau=50 is in the upper range. '
            'The allocator responds aggressively to tracking errors, causing '
            'rapid thruster saturation and high energy consumption.'
        ),
        'diagnosis': (
            'ks_tau (thrust allocation gain) trades tracking for smoothness. '
            'Values above 100 can cause oscillation. allocation_reg should be '
            'increased proportionally to dampen allocation commands. '
            'rate_limit can also cap sudden thruster changes.'
        ),
        'module': 'allocation',
        'action': (
            'Reduce ks_tau from ~50 to 20–40 range. Increase allocation_reg '
            'from ~0.01 to 0.05–0.15. Reduce rate_limit from ~2.0 to 0.5–1.5 '
            'for smoother actuator commands.'
        ),
        'tradeoff': (
            'control_energy should decrease. Tracking may degrade slightly. '
            'The ks_tau/allocation_reg ratio determines the aggressiveness-smoothness tradeoff.'
        ),
        'output': '{"name":"smoother_allocation","config":{"ks_tau":30,"allocation_reg":0.1,"rate_limit":1.0}}',
    },
    {
        'scenario': 'Health Monitoring Overly Conservative',
        'observation': (
            'After fault windows, recovery is slow. health_decay=0.995 causes '
            'thrusters to be distrusted for many steps after effectiveness '
            'returns. health_weight_scale=2.0 amplifies health influence on allocation.'
        ),
        'diagnosis': (
            'health_decay near 1.0 means trust recovers very slowly. For fault '
            'tolerance, you want LOW health_decay (0.001–0.01) to effectively '
            'disable health-based degradation, OR moderate values (0.5–0.8) '
            'for temporary distrust after faults. health_weight_scale should be '
            'reduced if health is dominating allocation decisions.'
        ),
        'module': 'health',
        'action': (
            'Set health_decay to 0.001 (essentially disable) for baseline. '
            'Reduce health_weight_scale to 0.5–1.5. Enable use_perf_boost '
            'to actively restore performance after fault recovery.'
        ),
        'tradeoff': (
            'Post-fault recovery should improve. Fault-peak errors should not '
            'worsen if RLS estimation is adequate. Disabling health degradation '
            'is the V1 optimal strategy.'
        ),
        'output': '{"name":"aggressive_recovery","config":{"health_decay":0.001,"health_weight_scale":1.0,"use_perf_boost":true}}',
    },
    {
        'scenario': 'q Is The Only Objective',
        'observation': (
            'A candidate improved integral_error but control_energy worsened '
            'more, causing q to decrease. The candidate was rejected despite '
            'improving a visible metric.'
        ),
        'diagnosis': (
            'Single-metric optimization is misleading. q aggregates all metrics '
            'with tradeoff weights. An improvement in tracking does not guarantee '
            'an improvement in q. Always check the directional evidence before '
            'repeating a parameter direction.'
        ),
        'module': 'tracking',
        'action': (
            'When a direction improves one metric but degrades q, do NOT repeat '
            'that direction. The Directional Evidence Ledger marks it as rejected. '
            'Switch to a different module or test the opposite direction.'
        ),
        'tradeoff': (
            'Parameter semantics are hypotheses, not action rules. Evidence takes priority.'
        ),
        'output': '{"name":"evidence_driven_tracking","config":{"kp_vel":2.5,"ki_vel":0.15}}',
    },
]


# ═══════════════════════════════════════════════════════════════════
# V2 System Prompt Builder (Point 1 + Point 3)
# ═══════════════════════════════════════════════════════════════════

def _build_drone_system_prompt(benchmark, role, bm, modules):
    """Dedicated drone fullconfig system prompt (V1 output format, V3 quality)."""
    param_lines_d = []
    for pname, pmeta in bm['params'].items():
        range_str = ''
        if 'lo' in pmeta:
            range_str = f'  Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
        pt = pmeta['type']
        param_lines_d.append(f'  {pname} ({pt}): {pmeta["desc"]}{range_str}')
    param_list_d = '\n'.join(param_lines_d)

    role_descriptions = {
        'A': ('You are Generator A (performance-driven). Preferred order: tracking weights '
              '(Q_pos, Q_vel) → horizon → attitude/input. Minimize position error. '
              'This is a tie-breaker, not a direction command.'),
        'B': ('You are Generator B (robustness-driven). Preferred order: input smoothness '
              '(R_thrust, R_moment) → attitude → tracking. Prioritize low saturation '
              'and smooth control. Smoothness is a means, not an end.'),
        'S': 'Single generator. Explore all parameters. Evidence-first.',
    }
    role_text = role_descriptions.get(str(role), role_descriptions['S'])

    return f"""You are an expert in Model Predictive Control (MPC) design and tuning for quadrotor UAVs.

## Task
Synthesize a complete controller configuration for a 12-state quadrotor tracking position setpoints under linear MPC at 100 Hz.

{bm['task_desc']}

## Configurable Parameters
You must assign a specific value to EVERY parameter listed below.

{param_list_d}

## Your Role
{role_text}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: max_pos_error < 3.0 m, final_pos_error < 2.5 m, saturation < 20%.

Among feasible candidates, **q is the ONLY ranking objective.** Do not treat improvement in any single metric as success unless q also improves. Parameter semantics are hypotheses, not action rules.

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must contain EVERY parameter name listed above with its value.
3. Do NOT invent new parameter names.

## Output Format
{{"name": "<descriptive_name>", "config": {{"Q_pos_xy": 15.0, "Q_pos_z": 25.0, ...}}}}"""


def build_system_prompt_v2(benchmark, role):
    """Build enhanced system prompt with few-shot exemplars and module-level
    search branch assignment.

    Supports: RAMPC, CSTR (full V2). AUV -> enhanced V1.
    """
    bm = BENCHMARKS[benchmark]
    modules = bm.get('modules')

    # Drone handled separately (fullconfig V3 with legacy parser)
    if benchmark == 'drone':
        return _build_drone_system_prompt(benchmark, role, bm, modules)

    if modules is None:
        # ── AUV: enhanced V1 prompt ──
        return build_system_prompt_legacy(benchmark, role)

    # ── Benchmark-specific exemplars and context ──
    exemplar_map = {
        'rampc': RAMPC_EXEMPLARS,
        'cstr': CSTR_EXEMPLARS,
        'auv': AUV_EXEMPLARS,
        'drone': [],  # no few-shot yet — baseline run first
    }
    exemplars = exemplar_map.get(benchmark, [])

    # ── Benchmark-specific expert intro ──
    expert_intro = {
        'rampc': 'You are an expert in Model Predictive Control (MPC) design and tuning, specializing in nonlinear MPC for mobile robot navigation.',
        'cstr': 'You are an expert in Model Predictive Control (MPC) design and tuning, specializing in chemical process control and quadratic-programming-based MPC.',
    }

    # ── Benchmark-specific role contexts ──
    if benchmark == 'cstr':
        role_context = {
            'A': (
                'You are Generator A (performance-driven). Your goal is to minimize '
                'tracking error (IAE) and improve setpoint response speed. You focus '
                'on the `tracking` module (Wy_base, Wy_gain, Wy_cap). Do NOT modify '
                'smoothness or numerical parameters — Generator B handles those.'
            ),
            'B': (
                'You are Generator B (robustness-driven). Your goal is to ensure '
                'numerical stability and smooth control action. You focus on the '
                '`smoothness` module (Wu, Wdu_base, Wdu_gain, Wdu_cap). Do NOT '
                'modify tracking or horizon parameters — Generator A handles those.'
            ),
            'S': (
                'You are in single-generator mode. Balance tracking performance, '
                'numerical stability, and control smoothness. You may modify any '
                'parameters across all modules.'
            ),
            'performance': (
                'You are the performance specialist. Focus on minimizing tracking '
                'error via the `tracking` module.'
            ),
            'robustness': (
                'You are the robustness specialist. Focus on numerical stability '
                'and smooth control via the `smoothness` module.'
            ),
        }
    elif benchmark == 'drone':
        role_context = {
            'A': (
                'You are Generator A (performance-driven).\n\n'
                'Your preferred order: 1) tracking (Q_pos, Q_vel), 2) horizon, '
                '3) attitude/input. Minimize position tracking error. '
                'This is a tie-breaker, not a direction command.'
            ),
            'B': (
                'You are Generator B (robustness-driven).\n\n'
                'Your preferred order: 1) input smoothness (R_thrust, R_moment), '
                '2) attitude stability, 3) tracking. '
                'Prioritize smooth control and low saturation. '
                'Smoothness is a means to robustness, not an end in itself.'
            ),
            'S': 'Single generator. Explore all modules. Evidence-first.',
            'performance': 'Performance: tracking → horizon.',
            'robustness': 'Robustness: input → attitude.',
        }
    elif benchmark == 'auv':
        role_context = {
            'A': (
                'You are Generator A (performance-driven).\n\n'
                'Your preferred investigation order is: 1) tracking (velocity/heading gains), '
                '2) estimation (RLS fault detection), 3) fault compensation flags. '
                'This order is a TIE-BREAKER, not a direction command. '
                'Modify tracking gains ONLY when evidence shows tracking is the bottleneck. '
                'If the Evidence Ledger rejects a direction, do NOT repeat it.'
            ),
            'B': (
                'You are Generator B (robustness-driven).\n\n'
                'Your preferred investigation order is: 1) allocation (thrust smoothing), '
                '2) health monitoring, 3) control smoothing. '
                'Health is a FEASIBILITY requirement (fault recovery), not a maximand. '
                'Do NOT reduce health_decay or increase ks_tau merely because these are '
                'your primary modules. Modify only when evidence supports the change.'
            ),
            'S': (
                'Single-generator mode. Investigate all modules. Prioritize untested '
                'directions over repeating rejected ones.'
            ),
            'performance': 'Performance specialist. Preferred: tracking → estimation. Use evidence.',
            'robustness': 'Robustness specialist. Preferred: allocation → health. Safety is a constraint.',
        }
    else:
        role_context = {
            'A': (
                'You are Generator A (performance-driven).\n\n'
                'Your preferred investigation order is: 1) tracking, 2) smoothness, '
                '3) safety. This order is a TIE-BREAKER for choosing among untested '
                'hypotheses — it is NOT an instruction to increase Q_position or '
                'terminal_weight.\n\n'
                'Modify the tracking module only when evidence suggests tracking is '
                'the current bottleneck or when a tracking direction remains untested. '
                'If recent history REJECTS a tracking direction, do NOT repeat that '
                'direction unless a new and explicit causal hypothesis distinguishes '
                'the new candidate from the rejected one.'
            ),
            'B': (
                'You are Generator B (robustness-driven).\n\n'
                'Your preferred investigation order is: 1) safety feasibility, '
                '2) smoothness, 3) tracking. Safety is a FEASIBILITY REQUIREMENT '
                '(min_clearance ≥ safety_margin), not an objective to maximize '
                'indefinitely.\n\n'
                'Do NOT increase safety_margin or obs_repulsion_w merely because '
                'safety is your primary module. Modify safety only when collision '
                'risk, clearance violation, or a tracking-safety conflict is '
                'supported by observed results.'
            ),
            'S': (
                'You are in single-generator mode. Investigate tracking, safety, '
                'and smoothness modules. Prioritize untested directions over '
                'repeating rejected ones.'
            ),
            'performance': (
                'Performance specialist. Preferred order: tracking → smoothness → '
                'safety. Use evidence, not defaults, to choose direction.'
            ),
            'robustness': (
                'Robustness specialist. Preferred order: safety feasibility → '
                'smoothness → tracking. Safety is a constraint, not a maximand.'
            ),
        }

    # Determine module assignment for this role
    role_module_map = bm.get('dual_role_modules', {})
    assigned_module = role_module_map.get(str(role), 'all')

    # Build exemplar section
    exemplar_lines = []
    for i, ex in enumerate(exemplars):
        exemplar_lines.append(f"### Example {i+1}: {ex['scenario']}")
        exemplar_lines.append(f"**Observation:** {ex['observation']}")
        exemplar_lines.append(f"**Diagnosis:** {ex['diagnosis']}")
        exemplar_lines.append(f"**Module:** `{ex['module']}`")
        exemplar_lines.append(f"**Action:** {ex['action']}")
        exemplar_lines.append(f"**Trade-off:** {ex['tradeoff']}")
        exemplar_lines.append(f"**Valid output:** `{ex['output']}`")
        exemplar_lines.append("")
    exemplars_text = '\n'.join(exemplar_lines)

    # Module descriptions
    module_desc_lines = []
    for mname, minfo in modules.items():
        params_str = ', '.join(minfo['params'])
        module_desc_lines.append(
            f"| `{mname}` | {params_str} | {minfo['desc']} |"
        )
    module_table = '\n'.join(module_desc_lines)

    # Allowed parameters for this role
    if assigned_module == 'all':
        allowed_params = list(bm['params'].keys())
        allowed_desc = 'All parameters — balance all objectives'
    else:
        allowed_params = modules[assigned_module]['params']
        allowed_desc = modules[assigned_module]['desc']

    # Build parameter spec table
    param_lines = []
    for pname in allowed_params:
        pmeta = bm['params'][pname]
        range_str = ''
        if 'lo' in pmeta:
            range_str = f'Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
        if pmeta['type'] == 'bool':
            param_lines.append(
                f'  - `{pname}` (bool): {pmeta["desc"]}. Default: {pmeta.get("default", False)}'
            )
        else:
            param_lines.append(
                f'  - `{pname}` ({pmeta["type"]}): {pmeta["desc"]}. {range_str}'
            )
    param_spec = '\n'.join(param_lines)

    role_text = role_context.get(str(role), role_context.get('S', str(role)))

    task_intro = {
        'rampc': 'Optimize a CSTAR-RAMPC controller for obstacle-aware mobile robot navigation.\nThe controller uses CasADi IPOPT for nonlinear MPC with obstacle avoidance constraints.',
        'cstr': 'Synthesize an executable controller configuration for CSTR set-point tracking.\nThe controller uses QR-AMPC with QP-based optimization.',
    }

    if benchmark == 'rampc':
        # ── V3 RAMPC System Prompt ──
        return f"""You are an expert in Model Predictive Control (MPC) design and tuning for mobile robot navigation.

## Task
Synthesize controller configurations for obstacle-aware RAMPC. The controller uses CasADi IPOPT for nonlinear MPC.

{bm['task_desc']}

## Parameter Modules

| Module | Parameters | Purpose |
|--------|-----------|---------|
{module_table}

All six parameters are eligible for exploration.

## Search Scope

For each candidate:
1. Select exactly ONE active module: `tracking`, `safety`, or `smoothness`.
2. Change one or two parameters within that active module only.
3. Do NOT mix modules in one candidate.
4. The "module" field must identify the active module. Do NOT output "all".

This ensures each candidate's effect is causally attributable to a single module change.

## Your Role
{role_text}

## Parameter Specifications
{param_spec}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: no collision, goal reached, min_clearance >= safety_margin.

Among feasible candidates, **q is the ONLY ranking objective.**

Do not treat an improvement in cost, control_energy, clearance, or any single diagnostic metric as success unless q also improves.

Parameter semantics are hypotheses, not action rules:
- Higher Q_position does NOT automatically imply a better candidate.
- Higher terminal_weight does NOT automatically imply better convergence.
- Higher R_linear_vel does NOT automatically imply a better smoothness trade-off.
- Higher safety_margin or obs_repulsion_w does NOT automatically imply a more desirable solution.

Use observed q changes from history to decide direction, not parameter semantics.

## Diagnostic Interpretation

Diagnostic labels (HIGH, LOW, etc.) identify possible mechanisms. They are NOT mandatory actions.

Modify a parameter only when at least one condition holds:
- the associated metric worsened together with q degradation;
- a previous change indicates a specific problem (oscillation, overshoot, collision risk);
- the relevant direction is genuinely untested;
- the candidate is designed as an explicit counterfactual test.

If the incumbent has the best q despite a diagnostic label, treat the label as a possibly acceptable trade-off rather than an automatic defect.

## Few-Shot Examples (from this benchmark)

{exemplars_text}

## Output Rules

Return exactly ONE JSON object. No markdown, no explanation, no code fences.

- "module": one of "tracking", "safety", or "smoothness" (never "all").
- "changes": ONE or TWO parameters from that module only.
- "expected": a concise, falsifiable prediction (e.g., "cost -5% to -10%; q improves only if X is limiting").

Do NOT output the full configuration. Do NOT invent parameter names.

## Output Format
{{"module":"tracking","changes":{{"Q_position":9.5}},"expected":"cost -3% to -8%; q improves if over-aggressive tracking was limiting"}}"""

    if benchmark == 'cstr':
        # ── V3 CSTR System Prompt ──
        use_fullconfig = '--fullconfig' in sys.argv
        if use_fullconfig:
            # V3 quality + V1 full-config output format
            param_lines_full = []
            for pname, pmeta in bm['params'].items():
                range_str = ''
                if 'lo' in pmeta:
                    range_str = f'  Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
                pt = pmeta['type']
                param_lines_full.append(f'  {pname} ({pt}): {pmeta["desc"]}{range_str}')
            param_list_full = '\n'.join(param_lines_full)

            return f"""You are an expert in Model Predictive Control (MPC) design and tuning for chemical process control.

## Task
Synthesize a complete controller configuration for CSTR set-point tracking under parameter uncertainty.
The controller uses QR-AMPC with QP-based optimization.

{bm['task_desc']}

## Configurable Parameters
You must assign a specific value to EVERY parameter listed below.

{param_list_full}

## Your Role
{role_text}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: all 8 metrics must be no worse than baseline, KKT residual < 1e8, condition number < 1e6.

Among feasible candidates, **q is the ONLY ranking objective.**

Do not treat improvement in any single metric (IAE, control_energy, etc.) as success unless q also improves.

Parameter semantics are hypotheses, not action rules — use observed q changes from history to decide direction. Higher Wy_base does NOT automatically imply better tracking. Enabling structural flags does NOT automatically improve performance.

## Diagnostic Interpretation

Diagnostic labels identify possible mechanisms, NOT mandatory actions. If the incumbent has the best q despite a label (e.g., HIGH control_energy), treat it as a possibly acceptable trade-off.

## Few-Shot Examples

{exemplars_text}

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must contain EVERY parameter name listed above with its value.
3. Boolean parameters must be true or false (not 0/1).
4. Float parameters are numerical; integer parameters must be whole numbers.
5. Do NOT invent new parameter names.

## Output Format
{{"name": "<descriptive_controller_name>", "config": {{"param1": value1, "param2": value2, ...}}}}"""

        # V3 module-patch format (default)
        return f"""You are an expert in Model Predictive Control (MPC) design and tuning for chemical process control.

## Task
Synthesize controller configurations for CSTR set-point tracking under parameter uncertainty.
The controller uses QR-AMPC with QP-based optimization.

{bm['task_desc']}

## Parameter Modules

| Module | Parameters | Purpose |
|--------|-----------|---------|
{module_table}

All {len(bm['params'])} parameters are eligible for exploration.

## Search Scope

For each candidate:
1. Select exactly ONE active module.
2. Change 1-3 parameters within that active module only.
3. Do NOT mix modules in one candidate.
4. Boolean (bool) parameters: use true or false, not string values.
5. The "module" field must identify the active module. Do NOT output "all".

This ensures each candidate's effect is causally attributable.

## Your Role
{role_text}

## Parameter Specifications (Your Module Only)
{param_spec}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: all 8 metrics must be no worse than baseline, KKT residual < 1e8, condition number < 1e6.

Among feasible candidates, **q is the ONLY ranking objective.**

Do not treat improvement in any single metric (IAE, control_energy, etc.) as success unless q also improves.

Parameter semantics are hypotheses, not action rules:
- Higher Wy_base does NOT automatically imply better tracking.
- Higher Wdu_base does NOT automatically imply better robustness.
- Enabling structural flags (adaptive_horizon, param_adaptation) does NOT automatically improve performance.

Use observed q changes from history to decide direction.

## Diagnostic Interpretation

Diagnostic labels identify possible mechanisms. They are NOT mandatory actions.

Modify a parameter only when at least one condition holds:
- the associated metric worsened together with q degradation;
- a previous change indicates a specific coupling problem;
- the relevant direction is genuinely untested;
- the candidate is an explicit counterfactual test.

If the incumbent has the best q despite a diagnostic label, treat it as a possibly acceptable trade-off.

## Few-Shot Examples (from this benchmark)

{exemplars_text}

## Output Rules

Return exactly ONE JSON object. No markdown, no explanation, no code fences.

- "module": one of the seven module names (never "all").
- "changes": 1-3 parameters from that module only.
- "expected": a concise, falsifiable prediction.

Bool parameters must be true/false. Float parameters are numerical. Do NOT invent new parameter names.

## Output Format
{{"module":"tracking","changes":{{"Wy_base":3.0,"Wy_gain":1.0}},"expected":"IAE -30% to -50%; q improves if tracking was bottleneck"}}"""

    if benchmark == 'auv':
        # ── V3 AUV Fullconfig System Prompt ──
        param_lines_auv = []
        for pname, pmeta in bm['params'].items():
            range_str = ''
            if 'lo' in pmeta:
                range_str = f'  Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
            pt = pmeta['type']
            param_lines_auv.append(f'  {pname} ({pt}): {pmeta["desc"]}{range_str}')
        param_list_auv = '\n'.join(param_lines_auv)

        return f"""You are an expert in fault-tolerant control (FTC) design for autonomous underwater vehicles.

## Task
Synthesize a complete controller configuration for AUV trajectory tracking under sequential actuator faults.
The controller uses adaptive FTC with RLS-based online effectiveness estimation.

{bm['task_desc']}

## Configurable Parameters
You must assign a specific value to EVERY parameter listed below.

{param_list_auv}

## Parameter Modules (for reference)

| Module | Parameters | Purpose |
|--------|-----------|---------|
{module_table}

## Your Role
{role_text}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: simulation success, integral_error < 1e6, no divergence.

Among feasible candidates, **q is the ONLY ranking objective.** Do not treat improvement in any single metric (integral_error, control_energy, fault peak errors) as success unless q also improves.

Parameter semantics are hypotheses, not action rules:
- Higher ks_tau does NOT automatically imply better tracking.
- Lower health_decay does NOT automatically imply better fault recovery.
- Enabling use_fault_feedforward does NOT automatically improve fault response.

Use observed q changes from history to decide direction.

## Diagnostic Interpretation

Diagnostic labels identify possible mechanisms, NOT mandatory actions. If the incumbent has the best q despite a label (e.g., HIGH control_energy), treat it as a possibly acceptable trade-off.

## Few-Shot Examples (from this benchmark)

{exemplars_text}

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must contain EVERY parameter name listed above with its value.
3. Boolean parameters must be true or false (not 0/1).
4. Float parameters are numerical; integer parameters must be whole numbers.
5. Do NOT invent new parameter names.

## Output Format
{{"name": "<descriptive_controller_name>", "config": {{"param1": value1, "param2": value2, ...}}}}"""

    if benchmark == 'drone':
        # ── V3 Drone Fullconfig System Prompt ──
        param_lines_d = []
        for pname, pmeta in bm['params'].items():
            range_str = ''
            if 'lo' in pmeta:
                range_str = f'  Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
            pt = pmeta['type']
            param_lines_d.append(f'  {pname} ({pt}): {pmeta["desc"]}{range_str}')
        param_list_d = '\n'.join(param_lines_d)

        return f"""You are an expert in Model Predictive Control (MPC) design and tuning for quadrotor UAVs.

## Task
Synthesize a complete controller configuration for a 12-state quadrotor tracking position setpoints.
The controller uses linear MPC with QP-based optimization at 100 Hz.

{bm['task_desc']}

## Configurable Parameters
You must assign a specific value to EVERY parameter listed below.

{param_list_d}

## Parameter Modules (for reference)

| Module | Parameters | Purpose |
|--------|-----------|---------|
{module_table}

## Your Role
{role_text}

## Acceptance Criteria
{bm['acceptance']}

## Selection Principle

Feasibility is a hard gate: max_pos_error < 3.0 m, final_pos_error < 2.5 m, saturation < 20%.

Among feasible candidates, **q is the ONLY ranking objective.** Do not treat improvement in any single metric (IAE, control_energy, etc.) as success unless q also improves.

Parameter semantics are hypotheses, not action rules — use observed q changes from history to decide direction. Higher Q_pos does NOT automatically imply better tracking. Lower R does NOT automatically imply better control.

## Diagnostic Interpretation

Diagnostic labels identify possible mechanisms, NOT mandatory actions. If the incumbent has the best q despite a label, treat it as a possibly acceptable trade-off.

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must contain EVERY parameter name listed above with its value.
3. Do NOT invent new parameter names.

## Output Format
{{"name": "<descriptive_name>", "config": {{"param1": value1, "param2": value2, ...}}}}"""

    # ── Fallback for unknown benchmarks ──
    return f"""{expert_intro.get(benchmark, expert_intro['rampc'])}

## Task
{task_intro.get(benchmark, task_intro['rampc'])}

{bm['task_desc']}

## Tuning Approach
You will output a **parameter patch** — small, targeted changes to parameters within a single module. This enables controlled, traceable improvements. Do NOT rewrite all parameters.

## Parameter Modules

| Module | Parameters | Purpose |
|--------|-----------|---------|
{module_table}

## Your Role
{role_text}

## Your Allowed Module: `{assigned_module}`
Allowed parameters: {', '.join(allowed_params)}
{allowed_desc}

## Parameter Specifications (Your Module Only)
{param_spec}

## Acceptance Criteria
{bm['acceptance']}

## Few-Shot Tuning Examples
Study these examples carefully. They demonstrate the correct diagnostic reasoning and output format.

{exemplars_text}

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must include: "module", "changes" (1-2 params), and "expected" (predicted impact).
3. Prioritize your assigned module `{assigned_module}` — cross into other modules ONLY if you have a specific, justified hypothesis.
4. Keep changes bounded: typically 10–50% from the parent values.
5. For bool parameters, use true or false (not string values).
6. Do NOT invent new parameter names.
7. Do NOT output the full configuration — only the changes.
8. In "expected": predict direction and rough magnitude for 1-2 key metrics.

## Output Format
{{"module": "{assigned_module}", "changes": {{"param1": value1}}, "expected": "metric1 ±X%, metric2 ±Y%"}}"""


# ═══════════════════════════════════════════════════════════════════
# Legacy System Prompt (CSTR / AUV — enhanced)
# ═══════════════════════════════════════════════════════════════════

def build_system_prompt_legacy(benchmark, role):
    """Original system prompt for CSTR/AUV (unchanged from V1 for now).
    Enhancement planned: add module definitions and exemplars for these benchmarks.
    """
    bm = BENCHMARKS[benchmark]

    param_lines = []
    for pname, pmeta in bm['params'].items():
        range_str = ''
        if 'lo' in pmeta:
            range_str = f'  Range: [{pmeta["lo"]}, {pmeta["hi"]}]'
        pt = pmeta['type']
        param_lines.append(
            f'  {pname} ({pt}): {pmeta["desc"]}{range_str}'
        )
    param_list = '\n'.join(param_lines)

    return f"""You are an expert in Model Predictive Control (MPC) design and tuning.

## Task
Synthesize an executable controller configuration for:
**{bm['name']}**

{bm['task_desc']}

## Configurable Parameters
You must assign a specific value to EVERY parameter listed below.
{param_list}

## Acceptance Criteria
{bm['acceptance']}

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation text, no code fences.
2. The JSON must contain EVERY parameter name listed above with its value.
3. Boolean parameters must be true or false (not 0/1).
4. Float parameters are numerical; integer parameters must be whole numbers.
5. Do NOT invent new parameter names.
6. Do NOT output parameter change percentages — output actual values.

## Output Format
{{"name": "<descriptive_controller_name>", "config": {{"param1": value1, "param2": value2, ...}}}}"""


# ═══════════════════════════════════════════════════════════════════
# V2 User Prompt Builder (Point 1 + Point 3)
# ═══════════════════════════════════════════════════════════════════

def build_user_prompt_v2(benchmark, incumbent, feedback, metrics, q_score, best_q,
                          history, error_diag, gen, role, baseline_metrics=None):
    """Build enhanced user prompt with score decomposition, metric-based
    diagnostics, module-level parameter freezing, and verified search patterns.
    """
    bm = BENCHMARKS[benchmark]
    modules = bm.get('modules')

    # Check for prompt_override from caller
    prompt_override = None
    if isinstance(feedback, dict):
        prompt_override = feedback.get('prompt_override', None)
    if isinstance(prompt_override, str) and len(prompt_override.strip()) < 10:
        prompt_override = None

    if prompt_override:
        # Caller-supplied guidance mode — but still use enhanced structure
        return _build_override_prompt_v2(benchmark, incumbent, feedback, metrics,
                                          q_score, best_q, history, error_diag,
                                          gen, role, baseline_metrics, prompt_override)

    if modules is None:
        # ── CSTR/AUV: use enhanced legacy prompt ──
        return build_user_prompt_legacy(benchmark, incumbent, feedback, metrics,
                                         q_score, best_q, history, error_diag,
                                         gen, role, baseline_metrics)

    # ── RAMPC V2: full enhanced user prompt ──

    # Determine module assignment
    role_module_map = bm.get('dual_role_modules', {})
    assigned_module = role_module_map.get(str(role), 'all')

    if assigned_module == 'all':
        allowed_params = list(bm['params'].keys())
    else:
        allowed_params = modules[assigned_module]['params']

    frozen_params = [p for p in bm['params'].keys() if p not in allowed_params]

    # Normalize history
    if isinstance(history, dict):
        history = [history]
    elif history is None:
        history = []

    parts = []

    # ── Section 1: Search State (Point 4: phase-aware) ──
    parts.append("## Current Search State")
    parts.append(f"Generation: {gen}")

    # Point 4: Phase-aware strategy guidance (V3: revamped stall policy)
    if isinstance(feedback, dict):
        stall = feedback.get('stall_count', 0)
    else:
        stall = 0

    if stall >= 10:
        phase = 'STALLED'
        phase_guidance = (
            f'STALLED for {stall} generations. A stall means recent hypotheses have '
            f'failed — it does NOT automatically justify larger parameter jumps. '
            f'1) Identify the most recently rejected direction from the Evidence Ledger. '
            f'2) Do NOT repeat it. '
            f'3) Prefer an UNTESTED direction within the same active module. '
            f'4) If both directions of that module have been tested, switch modules. '
            f'5) Use 15-30% changes. Use 30-50% only for an explicit boundary probe.'
        )
    elif stall >= 5:
        phase = 'STALLED'
        phase_guidance = (
            f'STALLED for {stall} generations. Prefer untested directions over '
            f'larger changes. Check the Evidence Ledger for rejected directions '
            f'to avoid. 15-30% change magnitude.'
        )
    elif gen <= 10:
        phase = 'EXPLORE'
        phase_guidance = (
            'EXPLORATION phase. Test diverse MODULE-DIRECTION hypotheses. '
            'Prefer untested module-direction combinations. 15-30% changes.'
        )
    else:
        phase = 'FOCUS'
        phase_guidance = (
            'FOCUS phase. Concentrate on the most promising module based on '
            'evidence. 10-25% changes. Avoid repeating rejected directions.'
        )

    parts.append(f"## Search Phase: {phase}")
    parts.append(phase_guidance)

    role_labels = {
        'A': 'Generator A (performance-driven). Preferred order: tracking → smoothness → safety. Use evidence, not defaults.',
        'B': 'Generator B (robustness-driven). Preferred order: safety feasibility → smoothness → tracking. Safety is a constraint, not a maximand.',
        'S': 'Single generator. Prioritize untested directions over repeating rejected ones.',
        'performance': 'Performance specialist. Use evidence to choose direction.',
        'robustness': 'Robustness specialist. Safety is a constraint, not a maximand.',
    }
    parts.append(f"Role: {role_labels.get(str(role), str(role))}")
    parts.append(f"Global best q*: {best_q:.6f}")

    # ── Section 2: Incumbent Anchor ──
    parts.append("\n## Incumbent (Current Best)")
    if incumbent and any(v is not None and v != 0 for v in incumbent.values()
                         if not isinstance(v, bool)):
        parts.append(f"Best selection score q* = {best_q:.6f}")
        # Point 3: Show delta vs current q
        if q_score != 0 and best_q != -float('inf'):
            delta_q = q_score - best_q
            if delta_q > 0:
                parts.append(f"Current vs Best: q = {q_score:.6f} (Δ +{delta_q:.4f} BETTER ✓)")
            elif delta_q < 0:
                parts.append(f"Current vs Best: q = {q_score:.6f} (Δ {delta_q:.4f} WORSE ✗)")
            else:
                parts.append(f"Current vs Best: q = {q_score:.6f} (unchanged)")
        parts.append("\nCurrent parameter values:")
        for pname in list(bm['params'].keys()):
            if pname in incumbent:
                parts.append(f"  {pname}: {incumbent[pname]:.6g}")
            else:
                parts.append(f"  {pname}: {bm['params'][pname]['default']} (default)")
    else:
        parts.append("No incumbent yet — this is the first generation.")
        parts.append(f"\nAll parameters available for exploration: {', '.join(list(bm['params'].keys()))}")
        parts.append("\nStart from sensible defaults. EXPLORE broadly across different parameter regions.")
        parts.append("See the Few-Shot Examples in the system prompt for guidance.")

    # ── Section 3: Directional Evidence Ledger (Point 5) ──
    if history:
        # Analyze history for direction-level patterns
        rejected_dirs = {}  # module_direction -> list of (from_val, to_val, q_delta)
        for h in history[-10:]:
            if not h.get('accepted') and not h.get('updated'):
                mod = h.get('module', '?')
                qv = h.get('q', 0)
                dq = qv - best_q
                # Infer direction from config changes if available
                cfg_changes = h.get('changes', {})
                if cfg_changes:
                    for pname, new_val in cfg_changes.items():
                        if pname in incumbent:
                            old_val = incumbent[pname]
                            if new_val > old_val:
                                dir_key = f"{mod}_{pname}_up"
                            elif new_val < old_val:
                                dir_key = f"{mod}_{pname}_down"
                            else:
                                continue
                            if dir_key not in rejected_dirs:
                                rejected_dirs[dir_key] = []
                            rejected_dirs[dir_key].append({
                                'from': old_val, 'to': new_val, 'dq': dq
                            })

        if rejected_dirs:
            parts.append("\n## Directional Evidence Ledger")
            parts.append("REJECTED DIRECTIONS — do NOT repeat:")
            for dir_key, entries in rejected_dirs.items():
                e = entries[-1]  # latest entry
                parts.append(f"  - {dir_key}: {e['from']:.3g}→{e['to']:.3g}, Δq={e['dq']:.4f}")
            parts.append("")
            parts.append("Decision rule: A rejected direction may be revisited ONLY if:")
            parts.append("  1) the changed parameter set is materially different; AND")
            parts.append("  2) the new candidate tests a new causal hypothesis.")

    # ── Section 4: Score Decomposition with Δ% (Point 3: relative metrics) ──
    if metrics:
        mdict = {}
        if isinstance(metrics, dict):
            mdict = metrics
        elif isinstance(metrics, (list, tuple)):
            # MATLAB jsonencode may produce list of [key, value] pairs
            for item in metrics:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    mdict[item[0]] = item[1]
                elif isinstance(item, dict) and 'key' in item:
                    mdict[item['key']] = item.get('value', 0)
        elif hasattr(metrics, 'items'):
            try:
                mdict = dict(metrics)
            except (TypeError, ValueError):
                pass
        numeric_metrics = {k: v for k, v in mdict.items()
                          if isinstance(v, (int, float)) and v != 0}
        if numeric_metrics:
            parts.append("\n## Score Decomposition & Diagnostics")
            parts.append(f"{'Metric':<22} | {'Value':>10} | {'Direction':<10} | {'Δ vs Best':>12} | Status")
            parts.append("-" * 85)

            # Domain-specific diagnostic thresholds
            if benchmark == 'cstr':
                thresholds = {
                    'IAE':                ('lower', 1.5, 'HIGH — poor tracking'),
                    'RMS_tracking_error': ('lower', 0.5, 'HIGH — large oscillations'),
                    'max_tracking_error': ('lower', 3.0, 'HIGH — large overshoot'),
                    'control_energy':     ('lower', 50.0, 'HIGH — excessive input moves'),
                    'median_condition_number': ('lower', 1e4, 'HIGH — ill-conditioned QP'),
                    'max_KKT_residual':   ('lower', 1e-6, 'HIGH — numerical issues'),
                }
            else:
                thresholds = {
                    'cost':              ('lower', 300, 'HIGH — robot slow to reach goal'),
                    'min_clearance':     ('higher', 0.25, 'LOW — risk of collision'),
                    'control_energy':    ('lower', 80, 'HIGH — jerky motion'),
                    'state_energy':      ('lower', 100, 'HIGH — large state excursions'),
                    'plan_theta_width':  ('lower', 1.0, 'WIDE — uncertain trajectory'),
                }

            issues = []
            for mname, (mdesc, direction) in bm['metrics'].items():
                if mname in numeric_metrics:
                    val = numeric_metrics[mname]
                    arrow = '↓ good' if direction == 'lower' else '↑ good'
                    status = ''
                    # Point 3: Compute Δ% vs baseline
                    delta_str = ''
                    if baseline_metrics and mname in baseline_metrics:
                        bl = baseline_metrics[mname]
                        if isinstance(bl, (int, float)) and bl != 0:
                            pct = (val - bl) / abs(bl) * 100
                            if direction == 'lower':
                                # Lower is better: negative Δ = improvement
                                if pct < -1:
                                    delta_str = f'{pct:+.1f}% ✓'
                                elif pct > 1:
                                    delta_str = f'{pct:+.1f}% ✗'
                                else:
                                    delta_str = f'{pct:+.1f}%'
                            else:
                                # Higher is better: positive Δ = improvement
                                if pct > 1:
                                    delta_str = f'{pct:+.1f}% ✓'
                                elif pct < -1:
                                    delta_str = f'{pct:+.1f}% ✗'
                                else:
                                    delta_str = f'{pct:+.1f}%'
                    if mname in thresholds:
                        _, thresh, msg = thresholds[mname]
                        if direction == 'lower' and val > thresh:
                            status = f'⚠️ {msg}'
                            issues.append((mname, msg))
                        elif direction == 'higher' and val < thresh:
                            status = f'⚠️ {msg}'
                            issues.append((mname, msg))
                    parts.append(f"{mname:<22} | {val:>10.4f} | {arrow:<10} | {delta_str:>12} | {status}")

            # Synthesize diagnostic guidance (benchmark-aware)
            if issues:
                parts.append("\n### Diagnostic Guidance")
                for mname, msg in issues:
                    if benchmark == 'cstr':
                        if any(kw in mname.lower() for kw in ['iae', 'tracking', 'rms', 'max_tracking']) and assigned_module in ('tracking', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase Wy_base (2.0–5.0) to strengthen tracking")
                            parts.append(f"    → Also consider increasing Wy_gain for error-dependent scaling")
                        elif 'energy' in mname.lower() and assigned_module in ('smoothness', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase Wdu_base (2e-3–8e-3) to penalize input moves")
                            parts.append(f"    → Also increase Wdu_gain to 0.5–1.5 for rate-dependent damping")
                        elif any(kw in mname.lower() for kw in ['condition', 'kkt']) and assigned_module in ('smoothness', 'numerical', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase hessian_reg (1e-4–5e-4) to improve conditioning")
                            parts.append(f"    → Also reduce Wy_cap (30–80) to bound tracking weights")
                    else:
                        if 'cost' in mname.lower() and assigned_module in ('tracking', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase Q_position to accelerate goal-reaching")
                            parts.append(f"    → Also consider increasing terminal_weight for convergence")
                        elif 'clearance' in mname.lower() and assigned_module in ('safety', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase obs_repulsion_w to strengthen obstacle avoidance")
                            parts.append(f"    → Also consider increasing safety_margin for explicit clearance buffer")
                        elif 'energy' in mname.lower() and assigned_module in ('smoothness', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase R_linear_vel to penalize velocity changes")
                            parts.append(f"    → Also consider reducing max_linear_vel to cap peak speeds")
                        elif 'plan_theta_width' in mname.lower() and assigned_module in ('tracking', 'smoothness', 'all'):
                            parts.append(f"  - {msg}")
                            parts.append(f"    → Action: increase terminal_weight for tighter convergence")
            else:
                if assigned_module != 'all':
                    parts.append(f"\n### Diagnostic Guidance")
                    parts.append(f"  No critical issues detected. Consider fine-tuning within `{assigned_module}` with small changes (5–15%).")

    # ── Section 4: Recent Generation History ──
    if history:
        parts.append("\n## Recent Generation History")
        parts.append(f"{'Gen':>4} | {'Module':<12} | {'q_score':>10} | Status")
        parts.append("-" * 50)
        for h in history[-5:]:
            g = h.get('gen', '?')
            mod = str(h.get('module', '?'))[:11]
            qv = h.get('q', 0)
            status = []
            if h.get('accepted'):
                status.append('accepted')
            else:
                status.append('rejected')
            if h.get('updated'):
                status.append('★INCUMBENT')
            parts.append(f"{g:>4} | {mod:<12} | {qv:>10.6f} | {', '.join(status)}")

    # ── Section 5: Error Diagnostics ──
    if error_diag:
        parts.append("\n## Previous Execution Error")
        parts.append(f"ERROR: {error_diag}")
        parts.append("Fix this error before proceeding.")

    # ── Section 6: Change Magnitude Guard (V3: no automatic escalation) ──
    parts.append("\n## Change Magnitude Rule")
    parts.append("Default: 10-25% changes.")
    parts.append("Use 25-50% ONLY for an explicit boundary probe with no comparable historical evidence.")
    parts.append("Diversity means testing different causal hypotheses, not larger numerical changes.")
    parts.append("Select exactly ONE module. Change 1-2 parameters only.")

    # ── Section 7: Instruction (V3: evidence-first, benchmark-aware) ──
    parts.append("\n## Instruction")
    if benchmark == 'cstr':
        # CSTR V3: evidence-first instruction
        role_str = str(role)
        if role_str in ('A', 'performance'):
            parts.append("Preferred investigation order: tracking → horizon → adaptation → observer.")
            parts.append("This is a TIE-BREAKER, not a direction command.")
            parts.append("Modify tracking weights ONLY when IAE or RMS error is the active bottleneck.")
            parts.append("If the Evidence Ledger rejects a tracking direction, do NOT repeat it.")
        elif role_str in ('B', 'robustness'):
            parts.append("Preferred investigation order: smoothness → numerical → structural.")
            parts.append("Your goal is stability and smooth control — not to maximize Wy_base.")
            parts.append("Modify Wdu/hessian_reg ONLY when control_energy or KKT indicates a specific issue.")
            parts.append("When numerical conditioning is acceptable, prefer untested directions.")
        else:
            parts.append("Prioritize untested module-direction combinations.")
            parts.append("Do not repeat directions that the Evidence Ledger marks as rejected.")
    else:
        # RAMPC V3: evidence-first instruction
        role_str = str(role)
        if role_str in ('A', 'performance'):
            parts.append("Preferred investigation order: tracking → smoothness → safety.")
            parts.append("This is a TIE-BREAKER, not a direction command.")
            parts.append("Modify tracking ONLY when evidence suggests it is the bottleneck.")
            parts.append("If the Evidence Ledger shows a rejected tracking direction, do NOT repeat it.")
        elif role_str in ('B', 'robustness'):
            parts.append("Preferred investigation order: safety feasibility → smoothness → tracking.")
            parts.append("Safety is a CONSTRAINT (min_clearance ≥ safety_margin), not a maximand.")
            parts.append("Do NOT increase safety params unless clearance is threatened.")
            parts.append("When safety is feasible, prefer untested directions in other modules.")
        else:
            parts.append("Prioritize untested module-direction combinations.")
            parts.append("Do not repeat directions that the Evidence Ledger marks as rejected.")

    parts.append("\nSelect exactly ONE module. Output ONLY:")
    if benchmark == 'rampc':
        parts.append('{"module":"<tracking|safety|smoothness>","changes":{"param":value},"expected":"falsifiable prediction"}')
    else:
        parts.append('{"module":"<module_name>","changes":{"param":value},"expected":"falsifiable prediction"}')

    return '\n'.join(parts)


def _build_override_prompt_v2(benchmark, incumbent, feedback, metrics, q_score,
                                best_q, history, error_diag, gen, role,
                                baseline_metrics, prompt_override):
    """Build prompt when caller supplies guidance via prompt_override.
    Still enforces module-level constraints for RAMPC.
    """
    bm = BENCHMARKS[benchmark]
    modules = bm.get('modules')

    parts = []
    parts.append(f"## Task: {bm['name']} — Generation {gen} — Role: {role}")

    # Module assignment for override mode
    if modules:
        role_module_map = bm.get('dual_role_modules', {})
        assigned_module = role_module_map.get(str(role), 'all')
        if assigned_module == 'all':
            allowed_params = list(bm['params'].keys())
        else:
            allowed_params = modules[assigned_module]['params']
        frozen_params = [p for p in bm['params'].keys() if p not in allowed_params]

        parts.append(f"\nModule constraint: `{assigned_module}` only")
        parts.append(f"Allowed parameters: {', '.join(allowed_params)}")
        parts.append(f"Frozen parameters (DO NOT modify): {', '.join(frozen_params)}")

    parts.append("")
    parts.append("## Controller Expert Guidance (Follow These Priorities)")
    parts.append(prompt_override)
    parts.append("")
    parts.append("## Incumbent Status")
    parts.append(f"Best q* = {best_q:.6f}")
    if incumbent:
        # Show only allowed params if module-constrained
        if modules:
            role_module_map = bm.get('dual_role_modules', {})
            assigned_module = role_module_map.get(str(role), 'all')
            if assigned_module != 'all':
                allowed = modules[assigned_module]['params']
                inc_filtered = {k: v for k, v in incumbent.items() if k in allowed}
            else:
                inc_filtered = {k: v for k, v in incumbent.items() if k in bm['params']}
        else:
            inc_filtered = {k: v for k, v in incumbent.items() if k in bm['params']}
        parts.append("Current best parameters (allowed module only):")
        parts.append(json.dumps(inc_filtered, indent=2, default=str))

    if modules:
        role_module_map = bm.get('dual_role_modules', {})
        assigned_module = role_module_map.get(str(role), 'all')
        parts.append(f"\nOutput format: {{'module': '{assigned_module}', 'changes': {{...}}}}")
    else:
        parts.append("\nOutput ONLY a JSON object with ALL parameters. Set EVERY field to a specific value.")

    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════════
# Legacy User Prompt (CSTR / AUV — enhanced with diagnostics)
# ═══════════════════════════════════════════════════════════════════

def build_user_prompt_legacy(benchmark, incumbent, feedback, metrics, q_score, best_q,
                              history, error_diag, gen, role, baseline_metrics=None):
    """Enhanced legacy user prompt for CSTR/AUV.
    Same structure as V1 but adds metric-based diagnostic guidance.
    """
    bm = BENCHMARKS[benchmark]

    if isinstance(history, dict):
        history = [history]
    elif history is None:
        history = []

    parts = []

    # Check for prompt override
    prompt_override = None
    if isinstance(feedback, dict):
        prompt_override = feedback.get('prompt_override', None)
    if isinstance(prompt_override, str) and len(prompt_override.strip()) < 10:
        prompt_override = None

    if prompt_override:
        parts.append(f"## Task: {bm['name']} — Generation {gen} — Role: {role}")
        parts.append("")
        parts.append("## Controller Expert Guidance (Follow These Priorities)")
        parts.append(prompt_override)
        parts.append("")
        parts.append("## Incumbent Status")
        parts.append(f"Best q* = {best_q:.6f}")
        if incumbent:
            inc_filtered = {k: v for k, v in incumbent.items() if k in bm['params']}
            parts.append("Current best parameters:")
            parts.append(json.dumps(inc_filtered, indent=2, default=str))
        parts.append("")
        parts.append("Output ONLY a JSON object with ALL parameters. Set EVERY field to a specific value.")
        return '\n'.join(parts)

    # ── Section S: task context ──
    parts.append(f"## Task: {bm['name']} — Generation {gen}")
    role_desc = {
        'S': 'Single-generator mode: balance tracking and robustness.',
        'A': 'Generator A (performance-driven): prioritize tracking accuracy '
             'and low IAE. Accept slightly higher control energy if tracking '
             'improves significantly.',
        'B': 'Generator B (robustness-driven): prioritize numerical stability, '
             'constraint satisfaction, and fault tolerance. Prefer conservative '
             'tuning with good regularization.',
        'performance': 'Performance-driven: minimize tracking error.',
        'robustness': 'Robustness-driven: maximize numerical stability.',
    }
    parts.append(f"Role: {role_desc.get(str(role), str(role))}")

    # ── Incumbent anchor (c*, q*) ──
    parts.append("\n## Current Best (Incumbent)")
    if incumbent and any(v is not None and v != 0 for v in incumbent.values()
                         if not isinstance(v, bool)):
        parts.append(f"Best selection score q* = {best_q:.6f}")
        parts.append(f"Best raw benchmark score s* = {q_score:.6f}")
        inc_filtered = {k: v for k, v in incumbent.items() if k in bm['params']}
        parts.append("Incumbent parameters:")
        parts.append(json.dumps(inc_filtered, indent=2, default=str))
    else:
        parts.append("No incumbent yet — this is the first generation.")
        parts.append("Start from sensible defaults and explore the parameter space.")

    # ── Compact history H^(t-1) ──
    if history:
        parts.append("\n## Recent Generation History")
        parts.append(f"{'Gen':>4} | {'Name':<28} | {'q_score':>10} | Status")
        parts.append("-" * 60)
        for h in history[-5:]:
            g = h.get('gen', '?')
            nm = str(h.get('name', '?'))[:27]
            qv = h.get('q', 0)
            status = []
            if h.get('accepted'):
                status.append('accepted')
            else:
                status.append('rejected')
            if h.get('updated'):
                status.append('★INCUMBENT')
            parts.append(f"{g:>4} | {nm:<28} | {qv:>10.6f} | {', '.join(status)}")

    # ── Annotated metric summary m^(t-1) ──
    if metrics:
        mdict = {}
        if isinstance(metrics, dict):
            mdict = metrics
        elif hasattr(metrics, 'items'):
            mdict = dict(metrics)

        numeric_metrics = {k: v for k, v in mdict.items()
                          if isinstance(v, (int, float)) and v != 0}
        if numeric_metrics:
            parts.append("\n## Last Evaluation Metrics")
            parts.append(f"{'Metric':<28} | {'Value':>12} | Direction | Baseline | Δ%")
            parts.append("-" * 85)
            for mname, (mdesc, direction) in bm['metrics'].items():
                if mname in numeric_metrics:
                    val = numeric_metrics[mname]
                    arrow = '↑ good' if direction == 'higher' else '↓ good'
                    bl_str = ''
                    delta_str = ''
                    if baseline_metrics and mname in baseline_metrics:
                        bl = baseline_metrics[mname]
                        if bl and bl != 0:
                            bl_str = f'{bl:>10.4f}'
                            pct = (bl - val) / abs(bl) * 100
                            delta_str = f'{pct:>+7.1f}%'
                    parts.append(
                        f"{mname:<28} | {val:>12.4f} | {arrow:<8} | {bl_str:<10} | {delta_str}"
                    )

    # ── Error diagnostics e^(t-1) ──
    if error_diag:
        parts.append("\n## Previous Execution Error")
        parts.append(f"ERROR: {error_diag}")
        parts.append("You MUST fix this error. Check for missing fields or invalid values.")

    # ── Stall warning ──
    if isinstance(feedback, dict):
        stall = feedback.get('stall_count', 0)
    else:
        stall = 0

    if stall >= 10:
        parts.append(f"\n## ⚠️ CRITICAL: Stalled for {stall} generations")
        parts.append("The search has made no progress for many generations.")
        parts.append("You MUST try a RADICALLY different configuration:")
        parts.append("  - Toggle structural flags (observer, adaptation, adaptive horizon)")
        parts.append("  - Make large parameter changes (>50%)")
        parts.append("  - Explore a completely different region of the parameter space")
    elif stall >= 5:
        parts.append(f"\n## ⚠️ Stalled for {stall} generations without improvement")
        parts.append("Try significantly different parameters or structural changes.")
    elif stall >= 3:
        parts.append(f"\n## ⚠️ No improvement for {stall} generations")
        parts.append("Consider increasing exploration magnitude.")

    # ── F0: Refinement instruction ──
    parts.append("\n## Instruction (F0)")
    parts.append("Either refine the current best configuration or propose a "
                 "structurally different design.")
    parts.append("Output a COMPLETE configuration with ALL parameters set to "
                 "specific numerical values.")
    parts.append("Output ONLY the JSON object. No explanation, no markdown.")

    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════════
# LLM API Call
# ═══════════════════════════════════════════════════════════════════

def call_llm(config, system_prompt, user_prompt):
    """Call DeepSeek API. Returns raw response text."""
    headers = {
        'Authorization': f'Bearer {config["api_key"]}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': config['model'],
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': config['temperature'],
        'max_tokens': config['max_tokens'],
    }
    # DeepSeek supports json_object natively
    payload['response_format'] = {'type': 'json_object'}

    resp = requests.post(config['base_url'], headers=headers, json=payload,
                         timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


# ═══════════════════════════════════════════════════════════════════
# V2 Response Parsing (patch format for RAMPC)
# ═══════════════════════════════════════════════════════════════════

def parse_llm_response_v2(response_text, benchmark, template_config):
    """Parse V2 response (patch format) and merge with parent configuration.
    For RAMPC: expects {"module": "...", "changes": {...}}.
    For CSTR/AUV: falls back to legacy full-config parsing.
    """
    bm = BENCHMARKS[benchmark]
    modules = bm.get('modules')

    if modules is None:
        # CSTR/AUV: use legacy parser
        return parse_llm_response_legacy(response_text, benchmark, template_config)

    # ── RAMPC V2: patch parsing ──

    # Extract JSON
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return dict(template_config), False, \
                    f"JSON parse failed. Response preview: {response_text[:200]}"
        else:
            return dict(template_config), False, \
                f"No JSON object found. Response preview: {response_text[:200]}"

    # Handle both patch format and legacy full-config format
    if 'changes' in data and isinstance(data['changes'], dict):
        # Patch format: {"module": "...", "changes": {...}}
        changes = data['changes']
        module = data.get('module', 'all')
    else:
        # Legacy full-config format or raw params: extract only known RAMPC params
        changes = {}
        module = 'all'
        for pname in bm['params'].keys():
            if pname in data and data[pname] is not None:
                changes[pname] = data[pname]
        if not changes:
            return dict(template_config), False, \
                f"No valid changes found. Response preview: {response_text[:200]}"

    # Start from parent/template config
    result = dict(template_config) if template_config else {}

    # Apply validated changes
    validated_count = 0
    warnings = []

    for pname, val in changes.items():
        if pname not in bm['params']:
            warnings.append(f"Ignored unknown param: {pname}")
            continue
        pmeta = bm['params'][pname]
        try:
            if pmeta['type'] == 'bool':
                if isinstance(val, bool):
                    result[pname] = val
                elif isinstance(val, (int, float)):
                    result[pname] = bool(val)
                elif isinstance(val, str):
                    result[pname] = val.lower() in ('true', '1', 'yes')
                else:
                    result[pname] = bool(val)
            elif pmeta['type'] == 'int':
                result[pname] = int(float(val))
            else:
                result[pname] = float(val)

            # Clamp to safe range
            if 'lo' in pmeta and 'hi' in pmeta and not isinstance(result[pname], bool):
                old = result[pname]
                result[pname] = max(pmeta['lo'], min(pmeta['hi'], result[pname]))
                if old != result[pname]:
                    warnings.append(f"Clamped {pname}: {old} → {result[pname]}")

            validated_count += 1
        except (ValueError, TypeError):
            # Use template value on conversion failure
            result[pname] = template_config.get(pname, pmeta.get('default', 0))
            warnings.append(f"Type error on {pname}, used template value")

    # Fill in missing params from template or defaults
    n_filled = 0
    for pname, pmeta in bm['params'].items():
        if pname not in result:
            result[pname] = template_config.get(pname, pmeta.get('default', 0))
            n_filled += 1

    # Build status message
    msgs = []
    if validated_count:
        msgs.append(f"Module: {module}, Applied {validated_count} changes")
    if n_filled:
        msgs.append(f"Filled {n_filled} params from template")
    if warnings:
        msgs.append(f"Warnings: {'; '.join(warnings[:3])}")
    msg = '; '.join(msgs) if msgs else ''

    return result, True, msg


def parse_llm_response_legacy(response_text, benchmark, template_config):
    """Legacy parser for CSTR/AUV full-config responses (unchanged from V1)."""
    bm = BENCHMARKS[benchmark]
    template = template_config or {}

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return dict(template), False, \
                    f"JSON parse failed. Response preview: {response_text[:200]}"
        else:
            return dict(template), False, \
                f"No JSON object found. Response preview: {response_text[:200]}"

    if 'config' in data and isinstance(data['config'], dict):
        raw = data['config']
    else:
        raw = data

    validated = {}
    missing = []
    type_warnings = []

    for pname, pmeta in bm['params'].items():
        if pname in raw and raw[pname] is not None:
            val = raw[pname]
            try:
                if pmeta['type'] == 'bool':
                    if isinstance(val, bool):
                        validated[pname] = val
                    elif isinstance(val, (int, float)):
                        validated[pname] = bool(val)
                    elif isinstance(val, str):
                        validated[pname] = val.lower() in ('true', '1', 'yes')
                    else:
                        validated[pname] = bool(val)
                elif pmeta['type'] == 'int':
                    validated[pname] = int(float(val))
                else:
                    validated[pname] = float(val)
            except (ValueError, TypeError):
                validated[pname] = template.get(pname, pmeta.get('default', 0))
                type_warnings.append(pname)
        else:
            if pname in template:
                validated[pname] = template[pname]
            else:
                validated[pname] = pmeta.get('default', 0)
            missing.append(pname)

    # Safety clamp
    n_clamped = 0
    for pname, pmeta in bm['params'].items():
        if 'lo' in pmeta and 'hi' in pmeta and isinstance(validated.get(pname), (int, float)):
            if not isinstance(validated[pname], bool):
                old = validated[pname]
                validated[pname] = max(pmeta['lo'], min(pmeta['hi'], validated[pname]))
                if old != validated[pname]:
                    n_clamped += 1

    # Carry through non-LLM params from template (partial merge mode)
    n_inherited = 0
    if template:
        for k, v in template.items():
            if k not in validated:
                validated[k] = v
                n_inherited += 1

    msgs = []
    if missing:
        msgs.append(f"Missing {len(missing)} fields (used template): {', '.join(missing[:8])}")
    if type_warnings:
        msgs.append(f"Type errors on {len(type_warnings)} fields (used template): {', '.join(type_warnings[:5])}")
    if n_clamped:
        msgs.append(f"Clamped {n_clamped} out-of-range values")
    if n_inherited:
        msgs.append(f"Inherited {n_inherited} non-target fields from incumbent")
    msg = '; '.join(msgs) if msgs else ''

    return validated, True, msg


# ═══════════════════════════════════════════════════════════════════
# Fallback Generation (structured perturbation)
# ═══════════════════════════════════════════════════════════════════

def generate_fallback(template_config, benchmark, stall_count=0):
    """Generate fallback config when LLM call fails.
    Uses structured perturbation scaled by stall count.
    """
    bm = BENCHMARKS[benchmark]
    result = {}

    if stall_count < 5:
        scale = 0.05
    elif stall_count < 10:
        scale = 0.15
    else:
        scale = 0.35

    for pname, pmeta in bm['params'].items():
        if pname in template_config and template_config[pname] is not None:
            old = template_config[pname]
        else:
            result[pname] = pmeta.get('default', 0)
            continue

        if pmeta['type'] == 'bool':
            if stall_count >= 10 and random.random() < 0.3:
                result[pname] = not bool(old)
            else:
                result[pname] = bool(old)
            continue

        try:
            old_f = float(old)
        except (ValueError, TypeError):
            result[pname] = pmeta.get('default', 0)
            continue

        if abs(old_f) > 1e-12:
            new = old_f * (1.0 + scale * random.gauss(0, 1))
        else:
            new = scale * abs(random.gauss(0, 1))

        if 'lo' in pmeta and 'hi' in pmeta:
            new = max(pmeta['lo'], min(pmeta['hi'], new))
        result[pname] = new

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='V2 LLM-Assisted Controller Synthesis (few-shot + modules)')
    parser.add_argument('input_file', nargs='?', default='/tmp/llm_input.json',
                        help='JSON input file')
    parser.add_argument('output_file', nargs='?', default='/tmp/llm_output.json',
                        help='JSON output file')
    parser.add_argument('benchmark', nargs='?', default='cstr',
                        choices=['cstr', 'auv', 'rampc', 'drone'],
                        help='Benchmark name')
    parser.add_argument('role', nargs='?', default='S',
                        help='Generator role: S, A, B, performance, robustness')
    parser.add_argument('--v2', action='store_true', default=True,
                        help='Use V2 enhanced prompts (default: on)')
    parser.add_argument('--v1', action='store_true',
                        help='Force V1 legacy prompts')
    parser.add_argument('--fullconfig', action='store_true',
                        help='Use V1-style full config output (for fair V1 vs V3 comparison)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Save prompts without calling API')
    args = parser.parse_args()

    # Determine V1 vs V2
    use_v2 = not args.v1  # default to V2 unless --v1 is explicitly passed

    # Read input
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    config = load_config()
    bm = BENCHMARKS.get(args.benchmark, BENCHMARKS['cstr'])

    # Extract input fields
    incumbent = input_data.get('incumbent', {})
    feedback = input_data.get('feedback', {})
    if isinstance(feedback, str):
        feedback = {'text': feedback}
    metrics = input_data.get('metrics', {})
    q_score = input_data.get('q_score', 0)
    best_q = input_data.get('best_q', -float('inf'))
    history = input_data.get('history', [])
    error_diag = input_data.get('error_diag', '')
    gen = input_data.get('gen', 1)
    stall_count = input_data.get('stall_count', 0)
    baseline_metrics = input_data.get('baseline_metrics', None)

    # Build prompts
    if use_v2:
        system_prompt = build_system_prompt_v2(args.benchmark, args.role)
        user_prompt = build_user_prompt_v2(
            args.benchmark, incumbent, feedback, metrics, q_score, best_q,
            history, error_diag, gen, args.role, baseline_metrics
        )
    else:
        # Fall back to original V1 prompts (delegated to legacy builders)
        bm_modules = bm.get('modules')
        if bm_modules:
            system_prompt = build_system_prompt_v2(args.benchmark, args.role)
        else:
            system_prompt = build_system_prompt_legacy(args.benchmark, args.role)
        user_prompt = build_user_prompt_legacy(
            args.benchmark, incumbent, feedback, metrics, q_score, best_q,
            history, error_diag, gen, args.role, baseline_metrics
        )

    # ── Dry-run mode ──
    if args.dry_run:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'dry_run': True,
                'v2': use_v2,
                'benchmark': args.benchmark,
                'role': args.role,
            }, f, indent=2, ensure_ascii=False)
        print(f"DRY-RUN V2: sys={len(system_prompt)} chars, usr={len(user_prompt)} chars")
        sys.exit(0)

    # Call LLM
    success = False
    llm_response = ''
    error_msg = ''
    parsed_config = {}

    try:
        llm_response = call_llm(config, system_prompt, user_prompt)
        if use_v2:
            parsed_config, parse_ok, parse_msg = parse_llm_response_v2(
                llm_response, args.benchmark, incumbent
            )
        else:
            parsed_config, parse_ok, parse_msg = parse_llm_response_legacy(
                llm_response, args.benchmark, incumbent
            )

        if parse_ok:
            success = True
            error_msg = parse_msg
        else:
            error_msg = parse_msg
            parsed_config = generate_fallback(incumbent, args.benchmark, stall_count)
    except Exception as e:
        error_msg = f"API call failed: {e}"
        parsed_config = generate_fallback(incumbent, args.benchmark, stall_count)

    # Build candidate name
    src_tag = 'api' if success else 'fallback'
    v_tag = 'v2' if use_v2 else 'v1'
    parsed_config['name'] = f"gen{gen:03d}_llm_{src_tag}_{args.role}_{v_tag}"
    parsed_config['label'] = parsed_config['name']

    # Attach module info for V2 RAMPC
    bm_modules = bm.get('modules')
    if bm_modules and use_v2:
        role_module_map = bm.get('dual_role_modules', {})
        assigned = role_module_map.get(str(args.role), 'all')
        parsed_config['module'] = assigned

    # Build output
    output = {
        'success': success,
        'error': error_msg,
        'name': parsed_config.get('name', f'gen{gen:03d}_llm_{args.role}'),
        'label': parsed_config.get('label', ''),
        'config': parsed_config,
        'llm_source': 'api' if success else 'fallback',
        'n_params': len(parsed_config),
        'v2': use_v2,
        'module': parsed_config.get('module', 'legacy'),
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save prompt artifact for reproducibility
    prompt_file = args.output_file.replace('.json', '_prompt.json')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump({
            'benchmark': args.benchmark,
            'role': args.role,
            'gen': gen,
            'v2': use_v2,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'llm_response': llm_response[:2000] if llm_response else error_msg,
            'success': success,
            'error': error_msg,
        }, f, indent=2, ensure_ascii=False)

    status = 'OK' if success else f'FALLBACK ({error_msg[:80]})'
    n = len(parsed_config)
    module_info = f", module={parsed_config.get('module', 'N/A')}" if bm_modules else ''
    print(f"LLM V2 {status}: {n} params{module_info}")
