#!/usr/bin/env python3
"""
LLM Generate V2 - Enhanced prompt with CoT/ToT principles.
Implements:
  1. Few-shot exemplars for controller tuning trajectory
  2. Module-level parameter patches (not full config rewrite)
  3. Verified search memory instead of raw history table
  4. A/B as different parameter module branches
"""

import json, os, sys, requests, re, random, copy

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

def load_config():
    config = {
        'provider': 'openai-compatible',
        'api_key': '',
        'model': 'glm-5.1',
        'temperature': 0.3,
        'max_tokens': 2048,
        'base_url': 'https://api.0x7e.vip/v1/chat/completions',
        'api_key_env': 'HOPEAI_API_KEY',
    }
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            file_cfg = json.load(f)
            for k, v in file_cfg.items():
                if k != 'api_key':
                    config[k] = v
    api_key_env = config.get('api_key_env', 'HOPEAI_API_KEY')
    config['api_key'] = os.environ.get(api_key_env, '')
    return config


# ═══════════════════════════════════════════════════════════════════
# RAMPC-specific configuration and few-shot exemplars
# ═══════════════════════════════════════════════════════════════════

RAMPC_PARAMS = {
    'Q_position': {'type': 'float', 'lo': 0.5, 'hi': 20.0, 'default': 1.0,
                   'desc': 'Position tracking weight', 'module': 'tracking'},
    'R_linear_vel': {'type': 'float', 'lo': 0.05, 'hi': 2.0, 'default': 0.1,
                     'desc': 'Linear velocity penalty', 'module': 'smoothness'},
    'safety_margin': {'type': 'float', 'lo': 0.05, 'hi': 0.80, 'default': 0.3,
                      'desc': 'Minimum clearance to obstacles', 'module': 'safety'},
    'max_linear_vel': {'type': 'float', 'lo': 0.4, 'hi': 2.5, 'default': 1.0,
                       'desc': 'Maximum allowed linear velocity', 'module': 'smoothness'},
    'obs_repulsion_w': {'type': 'float', 'lo': 5.0, 'hi': 50.0, 'default': 10.0,
                        'desc': 'Obstacle repulsion weight', 'module': 'safety'},
    'terminal_weight': {'type': 'float', 'lo': 1.0, 'hi': 20.0, 'default': 5.0,
                        'desc': 'Terminal cost weight', 'module': 'tracking'},
}

# Parameter modules for RAMPC
RAMPC_MODULES = {
    'tracking': ['Q_position', 'terminal_weight'],
    'safety': ['safety_margin', 'obs_repulsion_w'],
    'smoothness': ['R_linear_vel', 'max_linear_vel'],
}

# Few-shot exemplars for RAMPC (CoT-style)
RAMPC_EXEMPLARS = [
    """## Tuning Example 1: Tracking Performance Issue

Observed issue:
Total cost is high (312.4), but min_clearance is safe (0.35). The robot reaches goal slowly.

Diagnosis:
Position tracking is too conservative. Q_position may be too low.

Allowed module:
tracking block only (Q_position, terminal_weight).

Bounded change:
Increase Q_position moderately (e.g., from 1.0 to 3.0-5.0).
Keep terminal_weight proportional (≈ Q_position/3).

Expected trade-off:
Cost should decrease; control_energy may increase slightly due to more aggressive motion.

Valid output:
{"module":"tracking","changes":{"Q_position":4.0,"terminal_weight":4.0}}""",

    """## Tuning Example 2: Safety Margin Issue

Observed issue:
min_clearance is 0.18, below safety_margin target of 0.25. Near-collision detected.

Diagnosis:
Obstacle repulsion is insufficient. obs_repulsion_w may be too low relative to Q_position.

Allowed module:
safety block only (safety_margin, obs_repulsion_w).

Bounded change:
Increase obs_repulsion_w (e.g., from 10 to 15-25).
Optionally increase safety_margin slightly (e.g., from 0.25 to 0.30).

Expected trade-off:
min_clearance should increase; cost may increase due to longer path around obstacles.

Valid output:
{"module":"safety","changes":{"obs_repulsion_w":20.0,"safety_margin":0.30}}""",

    """## Tuning Example 3: Motion Smoothness Issue

Observed issue:
control_energy is high (89.2), trajectory is jerky. RMS velocity changes are large.

Diagnosis:
Velocity penalty is too low. Robot makes abrupt speed changes.

Allowed module:
smoothness block only (R_linear_vel, max_linear_vel).

Bounded change:
Increase R_linear_vel (e.g., from 0.1 to 0.2-0.4).
Consider reducing max_linear_vel if overshoot occurs.

Expected trade-off:
control_energy should decrease; cost may increase slightly due to slower motion.

Valid output:
{"module":"smoothness","changes":{"R_linear_vel":0.25,"max_linear_vel":0.8}}""",
]


# ═══════════════════════════════════════════════════════════════════
# Enhanced Prompt Construction (V2)
# ═══════════════════════════════════════════════════════════════════

def build_system_prompt_v2(benchmark, role):
    """Build enhanced system prompt with few-shot exemplars."""

    if benchmark != 'rampc':
        # For other benchmarks, use original prompt (can be extended later)
        return build_system_prompt_legacy(benchmark, role)

    # RAMPC-specific enhanced prompt
    exemplars_text = '\n\n'.join(RAMPC_EXEMPLARS)

    # Determine which module this role is assigned to
    module_assignment = {
        'A': 'tracking',  # A: tracking optimization (cost reduction)
        'B': 'safety',    # B: safety optimization (clearance improvement)
        'S': 'all',       # S: single mode, can modify any module
    }
    assigned_module = module_assignment.get(str(role), 'all')

    module_desc = {
        'tracking': 'Q_position and terminal_weight — focus on reaching goal faster',
        'safety': 'safety_margin and obs_repulsion_w — focus on obstacle avoidance',
        'smoothness': 'R_linear_vel and max_linear_vel — focus on motion smoothness',
        'all': 'any parameters — balance all objectives',
    }

    allowed_params = RAMPC_MODULES.get(assigned_module, list(RAMPC_PARAMS.keys()))

    return f"""You are an expert in Model Predictive Control (MPC) design and tuning.

## Task
Optimize a CSTAR-RAMPC controller for obstacle-aware mobile robot navigation.
The controller uses CasADi IPOPT for nonlinear MPC with obstacle avoidance.

## Tuning Approach
You will output a **parameter patch** (small changes to specific parameters),
NOT a complete configuration rewrite. This enables controlled, traceable improvements.

## Available Parameter Modules

| Module | Parameters | Purpose |
|--------|-----------|---------|
| tracking | Q_position, terminal_weight | Goal-reaching speed |
| safety | safety_margin, obs_repulsion_w | Obstacle avoidance |
| smoothness | R_linear_vel, max_linear_vel | Motion smoothness |

## Parameter Details
- Q_position: Range [0.5, 20.0]. Higher → more aggressive toward goal
- R_linear_vel: Range [0.05, 2.0]. Higher → slower, smoother motion
- safety_margin: Range [0.05, 0.80]. Minimum clearance to obstacles (meters)
- max_linear_vel: Range [0.4, 2.5]. Maximum allowed velocity
- obs_repulsion_w: Range [5.0, 50.0]. Higher → stronger obstacle avoidance
- terminal_weight: Range [1.0, 20.0]. Terminal cost for convergence

## Few-Shot Tuning Examples

{exemplars_text}

## Your Assigned Role
Role: {role}
Assigned module: {assigned_module}
Allowed parameters: {', '.join(allowed_params)}
Module description: {module_desc[assigned_module]}

## Output Rules
1. Output ONLY a single JSON object. No markdown, no explanation, no code fences.
2. The JSON must contain: {"module": "<module_name>", "changes": {"param": value}}
3. Only modify parameters within your assigned module.
4. Keep changes bounded: typically 10-50% from the parent values.
5. Do NOT invent new parameter names.

## Output Format
{"module": "{assigned_module}", "changes": {"param1": value1}}"""


def build_user_prompt_v2(benchmark, incumbent, feedback, metrics, q_score, best_q,
                         history, error_diag, gen, role, baseline_metrics=None,
                         verified_patterns=None):
    """Build enhanced user prompt with verified search memory."""

    if benchmark != 'rampc':
        return build_user_prompt_legacy(benchmark, incumbent, feedback, metrics,
                                        q_score, best_q, history, error_diag,
                                        gen, role, baseline_metrics)

    # RAMPC-specific enhanced prompt
    parts = []

    # Determine module assignment
    module_assignment = {'A': 'tracking', 'B': 'safety', 'S': 'all'}
    assigned_module = module_assignment.get(str(role), 'all')
    allowed_params = RAMPC_MODULES.get(assigned_module, list(RAMPC_PARAMS.keys()))
    frozen_params = [p for p in RAMPC_PARAMS.keys() if p not in allowed_params]

    # ── Current Search State ──
    parts.append("## Current Search State")
    parts.append(f"Generation: {gen}")
    parts.append(f"Role: {role} ({assigned_module} module)")
    parts.append(f"Best q* (global): {best_q:.6f}")

    # ── Score Decomposition ──
    if metrics:
        mdict = metrics if isinstance(metrics, dict) else dict(metrics)
        parts.append("\n## Score Decomposition")
        parts.append(f"{'Metric':<20} | {'Value':>10} | {'Direction':<10}")
        parts.append("-" * 45)
        metric_info = {
            'cost': ('lower', 'Total trajectory cost'),
            'min_clearance': ('higher', 'Minimum obstacle clearance'),
            'control_energy': ('lower', 'Control input energy'),
            'state_energy': ('lower', 'State trajectory energy'),
        }
        dominant = None
        dominant_contrib = 0
        for mname, (direction, desc) in metric_info.items():
            if mname in mdict and isinstance(mdict[mname], (int, float)):
                val = mdict[mname]
                arrow = '↓ good' if direction == 'lower' else '↑ good'
                parts.append(f"{mname:<20} | {val:>10.4f} | {arrow:<10}")
                # Determine dominant deficit
                if direction == 'lower' and val > 50:
                    if val > dominant_contrib:
                        dominant = mname
                        dominant_contrib = val
                elif direction == 'higher' and val < 0.25:
                    if -val > dominant_contrib:
                        dominant = mname
                        dominant_contrib = -val
        if dominant:
            parts.append(f"\nDominant deficit: {dominant} needs improvement")

    # ── Verified Diagnostics ──
    parts.append("\n## Verified Diagnostics")
    if error_diag:
        parts.append(f"Previous error: {error_diag}")
        parts.append("Fix this before proceeding.")
    else:
        # Generate specific diagnostics based on metrics
        if metrics:
            mdict = metrics if isinstance(metrics, dict) else dict(metrics)
            if 'min_clearance' in mdict and mdict['min_clearance'] < 0.25:
                parts.append("- min_clearance is LOW (below 0.25)")
                parts.append("  → Consider increasing safety_margin or obs_repulsion_w")
            if 'cost' in mdict and mdict['cost'] > 300:
                parts.append("- cost is HIGH (above 300)")
                parts.append("  → Consider increasing Q_position for faster goal-reaching")
            if 'control_energy' in mdict and mdict['control_energy'] > 80:
                parts.append("- control_energy is HIGH")
                parts.append("  → Consider increasing R_linear_vel for smoother motion")

    # ── Verified Search Memory ──
    parts.append("\n## Verified Search Memory")
    if verified_patterns:
        parts.append("Beneficial patterns observed:")
        for p in verified_patterns.get('beneficial', [])[:3]:
            parts.append(f"  - {p}")
        parts.append("Harmful patterns observed:")
        for p in verified_patterns.get('harmful', [])[:3]:
            parts.append(f"  - {p}")
    else:
        # Extract from history if available
        if history and len(history) > 0:
            parts.append("Recent observations:")
            # Analyze last few generations
            for h in history[-3:]:
                if h.get('updated'):
                    parts.append(f"  - Gen {h.get('gen')}: improvement achieved (q={h.get('q', 0):.4f})")
                else:
                    parts.append(f"  - Gen {h.get('gen')}: no improvement (rejected)")
        else:
            parts.append("No prior search history available.")

    # ── Parent Configuration ──
    parts.append("\n## Parent Configuration")
    parts.append(f"Parent parameters (modify ONLY allowed ones):")
    if incumbent:
        for pname in allowed_params:
            if pname in incumbent:
                parts.append(f"  {pname}: {incumbent[pname]:.4f}")
            else:
                parts.append(f"  {pname}: {RAMPC_PARAMS[pname]['default']} (default)")
    else:
        for pname in allowed_params:
            parts.append(f"  {pname}: {RAMPC_PARAMS[pname]['default']} (default)")

    parts.append(f"\nFrozen parameters (DO NOT modify):")
    for pname in frozen_params:
        if incumbent and pname in incumbent:
            parts.append(f"  {pname}: {incumbent[pname]:.4f} (LOCKED)")

    # ── Allowed Modification Rule ──
    parts.append("\n## Allowed Modification Rule")
    parts.append(f"Module: {assigned_module}")
    parts.append(f"Allowed parameters: {', '.join(allowed_params)}")
    parts.append(f"Max changed parameters: 1-2 (bounded changes)")
    parts.append(f"Change magnitude: 10-50% from parent values")

    # ── Instruction ──
    parts.append("\n## Instruction")
    if assigned_module == 'tracking':
        parts.append("Focus on reducing cost by tuning Q_position and terminal_weight.")
        parts.append("Higher Q_position → faster goal-reaching, but may increase control_energy.")
    elif assigned_module == 'safety':
        parts.append("Focus on improving min_clearance by tuning safety_margin and obs_repulsion_w.")
        parts.append("Higher obs_repulsion_w → stronger obstacle avoidance, but may increase cost.")
    else:
        parts.append("Balance all objectives. Choose the module most needed based on diagnostics.")

    parts.append("\nOutput ONLY:")
    parts.append(f'{"module": "{assigned_module}", "changes": {"param": value}}')

    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════════
# Legacy prompt builders (for non-RAMPC benchmarks)
# ═══════════════════════════════════════════════════════════════════

def build_system_prompt_legacy(benchmark, role):
    """Original system prompt for other benchmarks."""
    return """You are an expert in Model Predictive Control (MPC) design and tuning.

## Task
Synthesize an executable controller configuration for the given benchmark.

## Output Rules
1. Output ONLY a single JSON object.
2. Include all required parameters.
3. Do NOT invent new parameter names.

## Output Format
{"name": "<controller_name>", "config": {"param1": value1, ...}}"""


def build_user_prompt_legacy(benchmark, incumbent, feedback, metrics, q_score, best_q,
                              history, error_diag, gen, role, baseline_metrics=None):
    """Original user prompt for other benchmarks."""
    parts = []
    parts.append(f"## Generation {gen}")
    parts.append(f"Role: {role}")
    parts.append(f"Best q: {best_q:.6f}")
    if incumbent:
        parts.append(f"\nCurrent best config:")
        parts.append(json.dumps(incumbent, indent=2, default=str))
    parts.append("\nOutput ONLY a JSON object with all parameters.")
    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════════
# LLM API Call
# ═══════════════════════════════════════════════════════════════════

def call_llm(config, system_prompt, user_prompt):
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
    payload['response_format'] = {'type': 'json_object'}

    resp = requests.post(config['base_url'], headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


# ═══════════════════════════════════════════════════════════════════
# Response Parsing (V2 - handles patch format)
# ═══════════════════════════════════════════════════════════════════

def parse_llm_response_v2(response_text, benchmark, template_config):
    """Parse V2 response (patch format) and merge with parent."""

    bm_params = RAMPC_PARAMS if benchmark == 'rampc' else {}

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return template_config, False, f"JSON parse failed"
        else:
            return template_config, False, "No JSON found"

    # Extract changes
    changes = data.get('changes', {})
    module = data.get('module', 'all')

    # Start from parent/template config
    result = dict(template_config) if template_config else {}

    # Apply changes with validation
    validated_changes = {}
    for pname, val in changes.items():
        if pname in bm_params:
            pspec = bm_params[pname]
            try:
                validated_val = float(val)
                # Clamp to range
                if 'lo' in pspec and 'hi' in pspec:
                    validated_val = max(pspec['lo'], min(pspec['hi'], validated_val))
                validated_changes[pname] = validated_val
                result[pname] = validated_val
            except (ValueError, TypeError):
                pass

    # Fill in missing params from template or defaults
    for pname, pspec in bm_params.items():
        if pname not in result:
            result[pname] = template_config.get(pname, pspec['default'])

    msg = f"Module: {module}, Changes: {len(validated_changes)} params"
    return result, True, msg


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/llm_input.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/llm_output.json'
    benchmark = sys.argv[3] if len(sys.argv) > 3 else 'rampc'
    role = sys.argv[4] if len(sys.argv) > 4 else 'S'
    use_v2 = '--v2' in sys.argv

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    config = load_config()

    incumbent = input_data.get('incumbent', {})
    feedback = input_data.get('feedback', {})
    metrics = input_data.get('metrics', {})
    q_score = input_data.get('q_score', 0)
    best_q = input_data.get('best_q', -float('inf'))
    history = input_data.get('history', [])
    error_diag = input_data.get('error_diag', '')
    gen = input_data.get('gen', 1)
    stall_count = input_data.get('stall_count', 0)
    baseline_metrics = input_data.get('baseline_metrics', None)
    verified_patterns = input_data.get('verified_patterns', None)

    # Build prompts
    if use_v2 and benchmark == 'rampc':
        system_prompt = build_system_prompt_v2(benchmark, role)
        user_prompt = build_user_prompt_v2(benchmark, incumbent, feedback, metrics,
                                           q_score, best_q, history, error_diag,
                                           gen, role, baseline_metrics, verified_patterns)
    else:
        system_prompt = build_system_prompt_legacy(benchmark, role)
        user_prompt = build_user_prompt_legacy(benchmark, incumbent, feedback, metrics,
                                               q_score, best_q, history, error_diag,
                                               gen, role, baseline_metrics)

    # Dry-run mode
    if '--dry-run' in sys.argv:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'dry_run': True,
                'v2': use_v2,
            }, f, indent=2, ensure_ascii=False)
        print(f"DRY-RUN V2: {len(system_prompt)} + {len(user_prompt)} chars")
        sys.exit(0)

    # Call LLM
    success = False
    llm_response = ''
    error_msg = ''
    parsed_config = {}

    try:
        llm_response = call_llm(config, system_prompt, user_prompt)
        if use_v2 and benchmark == 'rampc':
            parsed_config, success, error_msg = parse_llm_response_v2(
                llm_response, benchmark, incumbent)
        else:
            parsed_config = incumbent
            success = True
            error_msg = "Legacy mode - pass through"
    except Exception as e:
        error_msg = f"API call failed: {e}"
        parsed_config = dict(incumbent) if incumbent else {}

    # Build output
    parsed_config['name'] = f"gen{gen:03d}_v2_{role}"
    parsed_config['llm_source'] = 'api' if success else 'fallback'
    parsed_config['module'] = RAMPC_MODULES.get(role, 'all') if use_v2 else 'legacy'

    output = {
        'success': success,
        'error': error_msg,
        'name': parsed_config.get('name', f'gen{gen:03d}_v2_{role}'),
        'config': parsed_config,
        'llm_source': 'api' if success else 'fallback',
        'v2': use_v2,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save prompt
    prompt_file = output_file.replace('.json', '_prompt.json')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump({
            'benchmark': benchmark,
            'role': role,
            'gen': gen,
            'v2': use_v2,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'llm_response': llm_response[:2000] if llm_response else error_msg,
        }, f, indent=2, ensure_ascii=False)

    print(f"LLM V2: {success}, {len(parsed_config)} params, module={parsed_config.get('module', 'N/A')}")