#!/usr/bin/env python3
"""
Verification-Guided LLM-Assisted Controller Synthesis — Python backend.
Aligns with paper (CEP revision V5.10) Section 2 prompt structure:

  P_sys:  fixed system prompt with task description, interface constraints,
          parameter specs, acceptance criteria (paper Section 2.2 promptbox)
  F^(t):  structured iteration feedback with incumbent anchor (c*,q*),
          compact history H^(t-1), annotated metric summary m^(t-1),
          execution error diagnostics e^(t-1), and F0 refinement instruction
          (paper Section 2.3 feedback promptbox)

Input:  JSON with incumbent config, feedback, metrics, history
Output: JSON with complete controller config + metadata
Supports: CSTR (31-field full config), AUV (partial merge mode)
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
        'max_tokens': 2048,
        'base_url': 'https://api.deepseek.com/v1/chat/completions',
    }
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'llm_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
    return config


# ═══════════════════════════════════════════════════════════════════
# Benchmark Specifications
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
        'param_mode': 'full',  # LLM outputs complete config
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
        'param_mode': 'partial',  # LLM outputs key params, merged into full template
        'params': {
            # ── Velocity tracking ──
            'kp_vel': {
                'type': 'float', 'lo': 0.1, 'hi': 5.0, 'default': 0.8,
                'desc': 'Velocity proportional gain (surge+sway). HIGHER → tighter tracking, may oscillate',
            },
            'ki_vel': {
                'type': 'float', 'lo': 0.01, 'hi': 1.0, 'default': 0.1,
                'desc': 'Velocity integral gain. Removes steady-state error',
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
                'type': 'float', 'lo': 0.1, 'hi': 3.0, 'default': 0.5,
                'desc': 'Yaw rate proportional gain',
            },
            'ki_r': {
                'type': 'float', 'lo': 0.005, 'hi': 0.2, 'default': 0.02,
                'desc': 'Yaw rate integral gain',
            },
            # ── Thrust allocation ──
            'ks_tau': {
                'type': 'float', 'lo': 0.05, 'hi': 2.0, 'default': 0.3,
                'desc': 'Thrust allocation gain. HIGHER → more aggressive allocation',
            },
            'allocation_reg': {
                'type': 'float', 'lo': 1e-4, 'hi': 0.5, 'default': 0.01,
                'desc': 'Allocation regularization. HIGHER → smoother, less aggressive allocation',
            },
            'rate_limit': {
                'type': 'float', 'lo': 0.1, 'hi': 10.0, 'default': 2.0,
                'desc': 'Actuator rate limit. LOWER → smoother but slower response',
            },
            # ── RLS estimation ──
            'rls_forgetting': {
                'type': 'float', 'lo': 0.90, 'hi': 0.9999, 'default': 0.98,
                'desc': 'RLS forgetting factor. HIGHER → longer memory, slower fault detection',
            },
            'rls_fault_boost': {
                'type': 'float', 'lo': 0.001, 'hi': 0.2, 'default': 0.02,
                'desc': 'RLS forgetting boost on fault detection. HIGHER → faster fault adaptation',
            },
            # ── Health monitoring ──
            'health_decay': {
                'type': 'float', 'lo': 0.9, 'hi': 0.999, 'default': 0.995,
                'desc': 'Health decay factor per step. LOWER → faster health degradation after fault',
            },
            'health_weight_scale': {
                'type': 'float', 'lo': 0.5, 'hi': 5.0, 'default': 2.0,
                'desc': 'Health influence on allocation. HIGHER → stronger health-weighting',
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
    },
}


# ═══════════════════════════════════════════════════════════════════
# Prompt Construction  (paper Section 2.2–2.3)
# ═══════════════════════════════════════════════════════════════════

def build_system_prompt(benchmark, role):
    """Build P_sys: fixed system prompt per benchmark.
    Corresponds to paper Section 2.2 promptbox.
    """
    bm = BENCHMARKS[benchmark]

    # Format parameter list
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


def build_user_prompt(benchmark, incumbent, feedback, metrics, q_score, best_q,
                      history, error_diag, gen, role, baseline_metrics=None):
    """Build F^(t): structured iteration feedback.
    Corresponds to paper Section 2.3 feedback promptbox:
      F^(t) = (S, H^(t-1), (c*, q*), m^(t-1), e^(t-1), F0)
    """
    bm = BENCHMARKS[benchmark]

    # Normalize history to list-of-dicts (MATLAB jsonencode may send
    # singleton struct arrays as a single dict)
    if isinstance(history, dict):
        history = [history]
    elif history is None:
        history = []

    parts = []

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
        inc_filtered = {k: v for k, v in incumbent.items()
                        if k in bm['params']}
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
        # Flatten struct metrics to dict
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
    """Call LLM API. Returns raw response text."""
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
    # DeepSeek supports json_object; doesn't hurt for others
    payload['response_format'] = {'type': 'json_object'}

    resp = requests.post(config['base_url'], headers=headers, json=payload,
                         timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


# ═══════════════════════════════════════════════════════════════════
# Response Parsing and Validation
# ═══════════════════════════════════════════════════════════════════

def parse_llm_response(response_text, benchmark, template_config):
    """Parse LLM JSON response into a validated config dict.

    Returns: (config_dict, success, message)
      config_dict: validated config with all required fields
      success: bool — True if JSON was parseable at all
      message: str — warnings or error description
    """
    bm = BENCHMARKS[benchmark]
    template = template_config or {}

    # ── Extract JSON ──
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON block from response
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

    # Extract config — accept both {"config": {...}} and raw {...}
    if 'config' in data and isinstance(data['config'], dict):
        raw = data['config']
    else:
        raw = data

    # ── Validate each field ──
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
                else:  # float
                    validated[pname] = float(val)
            except (ValueError, TypeError):
                # Use template value
                validated[pname] = template.get(pname, pmeta.get('default', 0))
                type_warnings.append(pname)
        else:
            # Field missing from LLM output — use template or default
            if pname in template:
                validated[pname] = template[pname]
            else:
                validated[pname] = pmeta.get('default', 0)
            missing.append(pname)

    # ── Safety clamp ──
    n_clamped = 0
    for pname, pmeta in bm['params'].items():
        if 'lo' in pmeta and 'hi' in pmeta and isinstance(validated.get(pname), (int, float)):
            if not isinstance(validated[pname], bool):
                old = validated[pname]
                validated[pname] = max(pmeta['lo'], min(pmeta['hi'], validated[pname]))
                if old != validated[pname]:
                    n_clamped += 1

    # ── Build status message ──
    msgs = []
    if missing:
        msgs.append(f"Missing {len(missing)} fields (used template): "
                    f"{', '.join(missing[:8])}")
    if type_warnings:
        msgs.append(f"Type errors on {len(type_warnings)} fields (used template): "
                    f"{', '.join(type_warnings[:5])}")
    if n_clamped:
        msgs.append(f"Clamped {n_clamped} out-of-range values")
    msg = '; '.join(msgs) if msgs else ''

    return validated, True, msg


# ═══════════════════════════════════════════════════════════════════
# Fallback Generation  (structured, not pure random walk)
# ═══════════════════════════════════════════════════════════════════

def generate_fallback(template_config, benchmark, stall_count=0):
    """Generate fallback config when LLM call fails.
    Uses structured perturbation scaled by stall count.
    """
    bm = BENCHMARKS[benchmark]
    result = {}

    # Determine perturbation scale
    if stall_count < 5:
        scale = 0.05   # local search around template
    elif stall_count < 10:
        scale = 0.15   # moderate exploration
    else:
        scale = 0.35   # aggressive exploration

    for pname, pmeta in bm['params'].items():
        if pname in template_config and template_config[pname] is not None:
            old = template_config[pname]
        else:
            result[pname] = pmeta.get('default', 0)
            continue

        if pmeta['type'] == 'bool':
            # Occasionally flip boolean when stalled
            if stall_count >= 10 and random.random() < 0.3:
                result[pname] = not bool(old)
            else:
                result[pname] = bool(old)
            continue

        # Continuous params: multiplicative perturbation
        try:
            old_f = float(old)
        except (ValueError, TypeError):
            result[pname] = pmeta.get('default', 0)
            continue

        if abs(old_f) > 1e-12:
            new = old_f * (1.0 + scale * random.gauss(0, 1))
        else:
            new = scale * abs(random.gauss(0, 1))

        # Safety clamp
        if 'lo' in pmeta and 'hi' in pmeta:
            new = max(pmeta['lo'], min(pmeta['hi'], new))
        result[pname] = new

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/llm_input.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/llm_output.json'
    benchmark = sys.argv[3] if len(sys.argv) > 3 else 'cstr'
    role = sys.argv[4] if len(sys.argv) > 4 else 'S'

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    config = load_config()
    bm = BENCHMARKS.get(benchmark, BENCHMARKS['cstr'])

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
    system_prompt = build_system_prompt(benchmark, role)
    user_prompt = build_user_prompt(
        benchmark, incumbent, feedback, metrics, q_score, best_q,
        history, error_diag, gen, role, baseline_metrics
    )

    # Call LLM
    success = False
    llm_response = ''
    error_msg = ''
    parsed_config = {}

    try:
        llm_response = call_llm(config, system_prompt, user_prompt)
        parsed_config, parse_ok, parse_msg = parse_llm_response(
            llm_response, benchmark, incumbent
        )
        if parse_ok:
            success = True
            error_msg = parse_msg  # may contain warnings
        else:
            error_msg = parse_msg
            parsed_config = generate_fallback(incumbent, benchmark, stall_count)
    except Exception as e:
        error_msg = f"API call failed: {e}"
        parsed_config = generate_fallback(incumbent, benchmark, stall_count)

    # Build candidate name
    src_tag = 'api' if success else 'fallback'
    parsed_config['name'] = f"gen{gen:03d}_llm_{src_tag}_{role}"
    parsed_config['label'] = parsed_config['name']

    # Build output
    output = {
        'success': success,
        'error': error_msg,
        'name': parsed_config.get('name', f'gen{gen:03d}_llm_{role}'),
        'label': parsed_config.get('label', ''),
        'config': parsed_config,
        'llm_source': 'api' if success else 'fallback',
        'n_params': len(parsed_config),
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save prompt artifact for reproducibility
    prompt_file = output_file.replace('.json', '_prompt.json')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump({
            'benchmark': benchmark,
            'role': role,
            'gen': gen,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'llm_response': llm_response[:2000] if llm_response else error_msg,
            'success': success,
            'error': error_msg,
        }, f, indent=2, ensure_ascii=False)

    status = 'OK' if success else f'FALLBACK ({error_msg[:80]})'
    n = len(parsed_config)
    print(f"LLM {status}: {n} params in config")
