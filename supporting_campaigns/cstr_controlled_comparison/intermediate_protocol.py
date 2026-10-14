#!/usr/bin/env python3
"""Frozen protocol primitives for the additive intermediate validation stage."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import score_only_vgmpc as score_only


PROTOCOL_VERSION = "cstr_four_method_intermediate_validation_v1"
SCORE_PROTOCOL_VERSION = "cstr_nonruntime_score_v1_minimal"
ACCEPTANCE_PROTOCOL_VERSION = "cstr_acceptance_v1_runtime_independent"
DUPLICATE_POLICY_VERSION = "cstr_block_global_deterministic_cache_v1"
TPE_OBJECTIVE_VERSION = "tpe_feasibility_objective_v1"
SCORE_PROMPT_VERSION = "score_only_nonruntime_v1"
FULL_PROMPT_VERSION = "full_feedback_nonruntime_v1"
FULL_FEEDBACK_SCHEMA_VERSION = "full_feedback_object_nonruntime_v1"
PARSER_VERSION = "strict_json_v1"
VALIDATION_NAME = "cstr_four_method_intermediate_validation"

VALIDATION_ONLY = True
PILOT_ONLY = False
PUBLICATION_ELIGIBLE = False
FORMAL_EXPERIMENT = False

METHODS = ("random_search", "tpe", "score_only_vgmpc", "full_feedback_vgmpc")
SLOTS_PER_METHOD = 40
ROUNDS_PER_BLOCK = 40
TOTAL_SLOTS_PER_BLOCK = 160
PAIRED_BLOCKS = 2
MASTER_SEED = 20260901
METHOD_SEED_OFFSETS = {
    "random_search": 11,
    "tpe": 22,
    "score_only_vgmpc": 33,
    "full_feedback_vgmpc": 44,
}
SCORE_METRICS = (
    "integral_error",
    "rms_tracking_error",
    "max_tracking_error",
    "final_tracking_error",
    "control_energy",
    "median_condition_number",
    "max_kkt_residual",
)
RUNTIME_METRIC = "avg_controller_time_ms"
ALL_METRICS = (
    "integral_error", "rms_tracking_error", "max_tracking_error",
    "final_tracking_error", "control_energy", RUNTIME_METRIC,
    "median_condition_number", "max_kkt_residual",
)
SCORE_TOLERANCE = 0.0
FEASIBLE_OBJECTIVE_OFFSET = 1.0
INFEASIBLE_OBJECTIVE = 0.0


class IntermediateIntegrityError(RuntimeError):
    """A block-level protocol, provenance, or artifact failure."""


def block_seed(block_id: int) -> int:
    if block_id not in (1, 2):
        raise ValueError("block_id must be 1 or 2")
    return MASTER_SEED + 1000 * (block_id - 1)


def method_seeds(block_id: int) -> Dict[str, int]:
    seed = block_seed(block_id)
    return {method: seed + offset for method, offset in METHOD_SEED_OFFSETS.items()}


def build_schedule(block_id: int) -> List[Dict[str, Any]]:
    cycles = (
        (METHODS[0], METHODS[1], METHODS[2], METHODS[3]),
        (METHODS[1], METHODS[2], METHODS[3], METHODS[0]),
        (METHODS[2], METHODS[3], METHODS[0], METHODS[1]),
        (METHODS[3], METHODS[0], METHODS[1], METHODS[2]),
    )
    offset = (block_id - 1) % 4
    schedule: List[Dict[str, Any]] = []
    counters = {method: 0 for method in METHODS}
    for round_index in range(1, ROUNDS_PER_BLOCK + 1):
        cycle = cycles[(round_index - 1 + offset) % 4]
        for position, method in enumerate(cycle, 1):
            counters[method] += 1
            schedule.append({
                "validation_name": VALIDATION_NAME,
                "block_id": block_id,
                "block_seed": block_seed(block_id),
                "round_index": round_index,
                "order_cycle": "ABCD"[(round_index - 1 + offset) % 4],
                "order_position": position,
                "global_slot_index": len(schedule) + 1,
                "method": method,
                "method_slot_index": counters[method],
            })
    validate_schedule(schedule, block_id)
    return schedule


def validate_schedule(schedule: List[Dict[str, Any]], block_id: int) -> None:
    if len(schedule) != TOTAL_SLOTS_PER_BLOCK:
        raise IntermediateIntegrityError("schedule length mismatch")
    for expected, slot in enumerate(schedule, 1):
        if slot["global_slot_index"] != expected or slot["block_id"] != block_id:
            raise IntermediateIntegrityError("schedule index/block mismatch")
    counts = {method: 0 for method in METHODS}
    positions = {method: [0, 0, 0, 0] for method in METHODS}
    for slot in schedule:
        method = slot["method"]
        if method not in counts:
            raise IntermediateIntegrityError("unknown method in schedule")
        counts[method] += 1
        positions[method][slot["order_position"] - 1] += 1
    if any(value != SLOTS_PER_METHOD for value in counts.values()):
        raise IntermediateIntegrityError("method slot count mismatch")
    if any(value != [10, 10, 10, 10] for value in positions.values()):
        raise IntermediateIntegrityError("position balance mismatch")


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metrics_dict(values: Any) -> Optional[Dict[str, Optional[float]]]:
    if isinstance(values, dict):
        return {name: _finite(values.get(name)) for name in ALL_METRICS}
    if not isinstance(values, (list, tuple)) or len(values) != len(ALL_METRICS):
        return None
    return {name: _finite(value) for name, value in zip(ALL_METRICS, values)}


def _authoritative_phi(metric: Any) -> float:
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        return float("-inf")
    metric = float(metric)
    if not math.isfinite(metric):
        return float("-inf")
    try:
        return max(0.0, 1.0 - metric / (1.0 + metric))
    except ZeroDivisionError:
        return float("-inf")


def minimal_raw_score(metrics: Any) -> float:
    """Apply the existing transform to exactly the seven non-runtime terms."""
    values = metrics_dict(metrics)
    if values is None:
        return float("-inf")
    terms = [_authoritative_phi(values[name]) for name in SCORE_METRICS]
    if any(not math.isfinite(term) for term in terms):
        return float("-inf")
    return sum(terms) / len(terms)


def violation_penalty_from_result(result: Dict[str, Any]) -> float:
    value = result.get("v", result.get("violation_penalty"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 100.0 if result.get("result_failed") or not result.get("execution_success") else 0.0
    return float(value)


def authoritative_acceptance(result: Dict[str, Any]) -> bool:
    if not result.get("execution_success") or result.get("result_failed", False):
        return False
    values = result.get("metrics")
    if isinstance(values, dict):
        condition = values.get("median_condition_number")
        kkt = values.get("max_kkt_residual")
    elif isinstance(values, (list, tuple)) and len(values) == len(ALL_METRICS):
        raw = dict(zip(ALL_METRICS, values))
        condition = raw.get("median_condition_number")
        kkt = raw.get("max_kkt_residual")
    else:
        return False
    # These are exactly the two checks in config_lightweight.cstr_acceptance.
    if isinstance(condition, bool) or not isinstance(condition, (int, float)):
        return False
    if isinstance(kkt, bool) or not isinstance(kkt, (int, float)):
        return False
    accepted = True
    if condition > 1e6:
        accepted = False
    if kkt > 1e8:
        accepted = False
    return accepted


def score_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt an existing evaluator result without changing failure semantics."""
    failed = bool(result.get("result_failed", False) or not result.get("execution_success"))
    raw = float("-inf") if failed else minimal_raw_score(result.get("metrics"))
    penalty = violation_penalty_from_result({**result, "result_failed": failed})
    selection = raw - penalty
    accepted = authoritative_acceptance({**result, "result_failed": failed})
    return {
        "metrics": metrics_dict(result.get("metrics")),
        "execution_success": not failed,
        "result_failed": failed,
        "accepted": accepted,
        "raw_score_nonruntime": raw,
        "raw_score_nonruntime_status": "negative_infinity_failure" if failed or not math.isfinite(raw) else "finite",
        "violation_penalty": penalty,
        "selection_score_nonruntime": selection,
        "selection_score_nonruntime_status": "negative_infinity_failure" if not math.isfinite(selection) else "finite",
        "simulation_duration_s": result.get("simulation_duration_s", result.get("duration_s")),
        "candidate": copy.deepcopy(result.get("candidate") or {}),
        "candidate_hash": result.get("candidate_id", ""),
    }


def incumbent_updated(before: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return bool(
        candidate.get("execution_success")
        and candidate.get("accepted")
        and candidate.get("selection_score_nonruntime") > before.get("selection_score_nonruntime")
    )


def tpe_objective(evaluation: Dict[str, Any]) -> Tuple[Optional[float], float]:
    score = evaluation.get("selection_score_nonruntime")
    feasible = bool(evaluation.get("execution_success") and evaluation.get("accepted"))
    if not feasible or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        return None, INFEASIBLE_OBJECTIVE
    normalized = 0.5 + math.atan(float(score)) / math.pi
    return normalized, FEASIBLE_OBJECTIVE_OFFSET + normalized


def candidate_hash(parameters: Dict[str, Any]) -> str:
    payload = "|".join(f"{name}={float(parameters[name]):.17g}" for name in score_only.NAMES)
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def secure_json_write(path: Any, value: Any, secret: str = "") -> None:
    text = json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if secret and secret in text:
        raise IntermediateIntegrityError("credential_redaction_failure")
    path.write_text(text, encoding="utf-8")
    try:
        import os
        os.chmod(path, 0o600)
    except OSError as exc:
        raise IntermediateIntegrityError("artifact_permission_failure") from exc


def build_score_only_prompt(space: Dict[str, Any], incumbent: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> str:
    lines = [
        "Task: Propose values for exactly seven CSTR configuration parameters.",
        "", "Allowed parameters:",
    ]
    for spec in space["variables"]:
        kind = "integer" if spec["type"] == "integer" else "number"
        lines.append(f"- {spec['name']}: {kind}, range [{spec['lo']}, {spec['hi']}]")
    lines.extend([
        "", "All other CSTR configuration settings are fixed to their baseline values.",
        "Do not modify structural settings.",
        "Objective: improve the accepted incumbent selection_score_nonruntime.",
        "", "Current incumbent parameters:",
        json.dumps({name: incumbent["parameters"][name] for name in score_only.NAMES}, separators=(",", ":")),
        "", f"Current incumbent selection_score_nonruntime: {incumbent['selection_score_nonruntime']:.17g}",
        "", "Previous candidate outcome:",
    ])
    if previous is None:
        lines.append("- previous_candidate_available: false")
    else:
        lines.extend([
            "- previous_candidate_available: true",
            f"- previous_candidate_selection_score_nonruntime: {previous['selection_score_nonruntime']:.17g}",
            f"- previous_candidate_accepted: {str(bool(previous['accepted'])).lower()}",
            f"- previous_candidate_execution_success: {str(bool(previous['execution_success'])).lower()}",
            f"- previous_candidate_incumbent_updated: {str(bool(previous['incumbent_updated'])).lower()}",
        ])
    lines.extend(["", "Output exactly one JSON object with exactly the seven parameter fields.",
                   "Use JSON numbers only. No explanation, no markdown, no extra fields."])
    return "\n".join(lines)


def build_full_feedback_object(incumbent: Dict[str, Any], previous: Optional[Dict[str, Any]], stall_count: int) -> Dict[str, Any]:
    def public_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if state is None:
            return {"available": False, "candidate_available": False, "parameters": None,
                    "raw_score_nonruntime": None, "violation_penalty": None,
                    "selection_score_nonruntime": None, "accepted": False,
                    "execution_success": False, "incumbent_updated": False,
                    "validation_category": "not_available", "simulation_failure": False,
                    "metrics": None}
        return {"available": True, "candidate_available": bool(state.get("candidate_available", True)),
                "parameters": copy.deepcopy(state.get("parameters")),
                "raw_score_nonruntime": state.get("raw_score_nonruntime"),
                "violation_penalty": state.get("violation_penalty"),
                "selection_score_nonruntime": state.get("selection_score_nonruntime"),
                "accepted": bool(state.get("accepted")),
                "execution_success": bool(state.get("execution_success")),
                "incumbent_updated": bool(state.get("incumbent_updated")),
                "validation_category": state.get("validation_category", "valid"),
                "simulation_failure": bool(state.get("simulation_failure")),
                "metrics": ({name: state.get("metrics", {}).get(name) for name in SCORE_METRICS}
                            if state.get("metrics") is not None else None)}
    previous_public = public_state(previous)
    reference = previous.get("reference_incumbent") if previous else None
    reference_metrics = reference.get("metrics") if reference else None
    prev_metrics = previous.get("metrics") if previous else None
    deltas = None
    improved: List[str] = []
    degraded: List[str] = []
    unchanged: List[str] = []
    if prev_metrics is not None and reference_metrics is not None:
        deltas = {}
        for name in SCORE_METRICS:
            delta = prev_metrics.get(name) - reference_metrics.get(name)
            deltas[name] = delta
            if abs(delta) <= 1e-12 * max(1.0, abs(prev_metrics[name]), abs(reference_metrics[name])):
                unchanged.append(name)
            elif delta < 0:
                improved.append(name)
            else:
                degraded.append(name)
    category = "not_available" if previous is None else previous.get("validation_category", "valid")
    summary = {
        "not_available": "No previous generation is available.",
        "valid": "Previous candidate was evaluated using non-runtime metrics.",
        "incumbent_updated": "Candidate updated the incumbent.",
        "insufficient_score": "Candidate was accepted but did not exceed the incumbent score.",
        "validation_failure": "Candidate output failed strict schema validation.",
        "execution_failure": "Candidate simulation did not execute successfully.",
    }.get(category, "Previous candidate outcome is recorded factually.")
    return {
        "schema_version": FULL_FEEDBACK_SCHEMA_VERSION,
        "current_incumbent": {
            "parameters": copy.deepcopy(incumbent["parameters"]),
            "raw_score_nonruntime": incumbent["raw_score_nonruntime"],
            "violation_penalty": incumbent["violation_penalty"],
            "selection_score_nonruntime": incumbent["selection_score_nonruntime"],
            "metrics": {name: incumbent["metrics"].get(name) for name in SCORE_METRICS},
            "accepted": incumbent["accepted"], "execution_success": incumbent["execution_success"],
        },
        "previous_generation": previous_public,
        "comparison": {
            "reference": "incumbent_before_previous_evaluation",
            "reference_parameters": reference.get("parameters") if reference else None,
            "reference_raw_score_nonruntime": reference.get("raw_score_nonruntime") if reference else None,
            "reference_violation_penalty": reference.get("violation_penalty") if reference else None,
            "reference_selection_score_nonruntime": reference.get("selection_score_nonruntime") if reference else None,
            "reference_metrics": {name: reference_metrics.get(name) for name in SCORE_METRICS} if reference_metrics else None,
            "metric_deltas": deltas, "improved_metrics": improved, "degraded_metrics": degraded,
            "unchanged_metrics": unchanged, "failure_category": category,
            "stall_count": stall_count, "diagnostic_summary": summary,
        },
    }


def build_full_feedback_prompt(space: Dict[str, Any], feedback: Dict[str, Any]) -> str:
    lines = ["Task: Propose values for exactly seven CSTR configuration parameters.", "", "Allowed parameters:"]
    for spec in space["variables"]:
        kind = "integer" if spec["type"] == "integer" else "number"
        lines.append(f"- {spec['name']}: {kind}, range [{spec['lo']}, {spec['hi']}]")
    lines.extend([
        "", "All other CSTR configuration settings are fixed to their baseline values.",
        "Do not modify structural settings.",
        "Objective: improve the accepted incumbent selection_score_nonruntime.",
        "", "Full-feedback evidence:",
        json.dumps(feedback, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "", "Output exactly one JSON object with exactly the seven parameter fields.",
        "Use JSON numbers only. No explanation, no markdown, no extra fields.",
    ])
    return "\n".join(lines)


def record_template(slot: Dict[str, Any], provenance_hash: str = "") -> Dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION, "validation_name": VALIDATION_NAME,
        "validation_only": VALIDATION_ONLY, "pilot_only": PILOT_ONLY,
        "publication_eligible": PUBLICATION_ELIGIBLE, "formal_experiment": FORMAL_EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION, "score_protocol_version": SCORE_PROTOCOL_VERSION,
        "acceptance_protocol_version": ACCEPTANCE_PROTOCOL_VERSION,
        "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
        "tpe_objective_version": TPE_OBJECTIVE_VERSION,
        "evaluator_hash": None, "score_hash": None, "acceptance_hash": None,
        "search_space_hash": None, "provenance_snapshot_hash": provenance_hash,
        **{key: slot[key] for key in ("block_id", "block_seed", "round_index", "order_cycle", "order_position", "global_slot_index", "method", "method_slot_index")},
        "proposal_attempted": True, "proposal_valid": None, "proposal_failure_category": None,
        "candidate_hash": None, "candidate_parameters": None,
        "complete_candidate_configuration": None, "logical_generation_index": None,
        "physical_request_count": 0, "retry_count": 0,
        "duplicate_detected": False, "duplicate_of_method": None, "duplicate_of_method_slot": None,
        "cache_hit": False, "cached_evaluation_hash": None,
        "candidate_simulation_performed": False, "runtime_remeasured": False,
        "runtime_measurement_origin": "not_applicable",
        "execution_success": False, "result_failed": False, "simulation_failure": False,
        "metrics": None, "raw_score_nonruntime": None, "violation_penalty": None,
        "selection_score_nonruntime": None, "raw_score_nonruntime_status": None,
        "selection_score_nonruntime_status": None, "accepted": False, "optimizer_objective": None,
        "normalized_optimizer_score": None,
        "incumbent_parameters_before": None, "incumbent_metrics_before": None,
        "incumbent_raw_score_nonruntime_before": None,
        "incumbent_violation_penalty_before": None,
        "incumbent_selection_score_nonruntime_before": None,
        "incumbent_updated": False, "no_update_reason": None,
        "incumbent_parameters_after": None, "incumbent_metrics_after": None,
        "incumbent_raw_score_nonruntime_after": None,
        "incumbent_violation_penalty_after": None,
        "incumbent_selection_score_nonruntime_after": None,
        "proposal_duration": None, "api_duration": None, "simulation_duration": None,
        "total_slot_duration": None, "prompt_tokens": None, "completion_tokens": None,
        "cached_tokens": None, "provider_request_id": None, "prompt_hash": None,
        "response_hash": None, "feedback_object_hash": None, "fallback_used": False,
        "credential_scan_passed": True,
        "record_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


CANONICAL_RECORD_FIELDS = frozenset(record_template(build_schedule(1)[0]))


def validate_record(record: Dict[str, Any]) -> None:
    if set(record) != CANONICAL_RECORD_FIELDS:
        raise IntermediateIntegrityError("record schema mismatch")
    if record["method"] not in METHODS or record["fallback_used"] is not False:
        raise IntermediateIntegrityError("record method/fallback mismatch")
    if record["validation_only"] is not True or record["pilot_only"] is not False or record["publication_eligible"] is not False:
        raise IntermediateIntegrityError("record protocol flags mismatch")
