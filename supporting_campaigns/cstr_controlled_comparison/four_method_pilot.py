#!/usr/bin/env python3
"""Implementation-only framework for the non-publication four-method pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna

import full_feedback_vgmpc as full_feedback
import score_only_vgmpc as score_only
from deepseek_requests_transport import request_chat_completion


PILOT_NAME = "cstr_four_method_pilot"
BLOCK_ID = 1
BLOCK_SEED = 20260815
SLOTS_PER_METHOD = 20
TOTAL_SLOTS = 80
METHODS = ("random_search", "tpe", "score_only_vgmpc", "full_feedback_vgmpc")
ORDER_CYCLES = (
    METHODS,
    ("tpe", "score_only_vgmpc", "full_feedback_vgmpc", "random_search"),
    ("score_only_vgmpc", "full_feedback_vgmpc", "random_search", "tpe"),
    ("full_feedback_vgmpc", "random_search", "tpe", "score_only_vgmpc"),
)
METHOD_SEED_OFFSETS = {
    "random_search": 11,
    "tpe": 22,
    "score_only_vgmpc": 33,
    "full_feedback_vgmpc": 44,
}
METRIC_NAMES = full_feedback.METRIC_NAMES
SCHEMA_VERSION = "CSTR_FOUR_METHOD_PILOT_v1"
PILOT_ONLY = True
PUBLICATION_ELIGIBLE = False
FORMAL_EXPERIMENT = False
FALLBACK_ENABLED = False
WARM_START_ENABLED = False
REPLACEMENT_SLOTS_ENABLED = False
AUTHENTICATION_REQUEST_LIMIT = 1
BASELINE_SIMULATION_LIMIT = 1
LOGICAL_LLM_GENERATION_LIMIT = 40
PHYSICAL_GENERATION_REQUEST_LIMIT = 80
CANDIDATE_SIMULATION_LIMIT = 80


class PilotIntegrityError(RuntimeError):
    """A block-level protocol or artifact integrity failure."""


def method_seeds(block_seed: int = BLOCK_SEED) -> Dict[str, int]:
    return {method: block_seed + offset for method, offset in METHOD_SEED_OFFSETS.items()}


def build_schedule() -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    counters = {method: 0 for method in METHODS}
    for round_index in range(1, SLOTS_PER_METHOD + 1):
        cycle = ORDER_CYCLES[(round_index - 1) % len(ORDER_CYCLES)]
        for position, method in enumerate(cycle, start=1):
            counters[method] += 1
            schedule.append({
                "block_id": BLOCK_ID,
                "block_seed": BLOCK_SEED,
                "round_index": round_index,
                "order_cycle": "ABCD"[(round_index - 1) % 4],
                "order_position": position,
                "global_slot_index": len(schedule) + 1,
                "method": method,
                "method_slot_index": counters[method],
            })
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: List[Dict[str, Any]]) -> None:
    if len(schedule) != TOTAL_SLOTS:
        raise AssertionError("schedule must contain exactly 80 slots")
    counts = {method: 0 for method in METHODS}
    positions = {method: [0, 0, 0, 0] for method in METHODS}
    for expected_index, slot in enumerate(schedule, start=1):
        if slot["global_slot_index"] != expected_index:
            raise AssertionError("global slot indexes must be contiguous")
        counts[slot["method"]] += 1
        positions[slot["method"]][slot["order_position"] - 1] += 1
    if counts != {method: SLOTS_PER_METHOD for method in METHODS}:
        raise AssertionError(f"method slot count mismatch: {counts}")
    if any(values != [5, 5, 5, 5] for values in positions.values()):
        raise AssertionError(f"balanced order mismatch: {positions}")


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_dict(values: Any) -> Optional[Dict[str, Optional[float]]]:
    return full_feedback.named_metrics(values)


def _parameters(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {name: candidate[name] for name in score_only.NAMES}


def _state_from_result(result: Dict[str, Any], fallback_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    candidate = result.get("candidate") or {}
    parameters = _parameters(candidate) if all(name in candidate for name in score_only.NAMES) else fallback_parameters
    return {
        "parameters": copy.deepcopy(parameters),
        "complete_configuration": copy.deepcopy(candidate) if candidate else None,
        "candidate_hash": result.get("candidate_id"),
        "metrics": _metric_dict(result.get("metrics")),
        "s": _finite(result.get("s")),
        "v": _finite(result.get("v")),
        "q": _finite(result.get("q")),
        "accepted": bool(result.get("accepted")),
        "execution_success": bool(result.get("execution_success")),
    }


def _baseline_state(baseline: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_result(baseline)
    if state["parameters"] is None:
        raise RuntimeError("canonical baseline did not contain the seven parameters")
    return state


def optimizer_objective(result: Dict[str, Any]) -> Optional[float]:
    q = _finite(result.get("q"))
    if q is None:
        return None
    return q if result.get("execution_success") and result.get("accepted") else q - 1.0


def _update_incumbent(before: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    before_q = before.get("q")
    candidate_q = candidate.get("q")
    updated = bool(
        candidate.get("execution_success")
        and candidate.get("accepted")
        and candidate_q is not None
        and before_q is not None
        and candidate_q > before_q
    )
    if updated:
        return copy.deepcopy(candidate), True, ""
    if not candidate.get("execution_success"):
        return copy.deepcopy(before), False, "execution_failure"
    if not candidate.get("accepted"):
        return copy.deepcopy(before), False, "rejected"
    return copy.deepcopy(before), False, "not_higher_than_incumbent"


def _candidate_hash(parameters: Dict[str, Any]) -> str:
    payload = "|".join(f"{name}={float(parameters[name]):.17g}" for name in score_only.NAMES)
    return hashlib.sha256(payload.encode()).hexdigest()


def random_parameters(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for spec in space["variables"]:
        if spec["type"] == "integer":
            values[spec["name"]] = rng.randint(int(spec["lo"]), int(spec["hi"]))
        elif spec["sampling"] == "log_uniform":
            values[spec["name"]] = math.exp(rng.uniform(math.log(spec["lo"]), math.log(spec["hi"])))
        else:
            values[spec["name"]] = rng.uniform(spec["lo"], spec["hi"])
    return values


def build_tpe_study(seed: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=2, multivariate=True
    )
    return optuna.create_study(direction="maximize", sampler=sampler)


def tpe_parameters(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for spec in space["variables"]:
        name = spec["name"]
        if spec["type"] == "integer":
            values[name] = trial.suggest_int(name, int(spec["lo"]), int(spec["hi"]))
        elif spec["sampling"] == "log_uniform":
            values[name] = trial.suggest_float(name, float(spec["lo"]), float(spec["hi"]), log=True)
        else:
            values[name] = trial.suggest_float(name, float(spec["lo"]), float(spec["hi"]))
    return values


def _json_write(path: Path, data: Any, secret: str) -> None:
    text = json.dumps(score_only.json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        score_only.secure_write(path, text, secret)
    except Exception as exc:
        raise PilotIntegrityError(f"artifact_write_failed:{path.name}") from exc


def _secure_write(path: Path, text: str, secret: str) -> None:
    try:
        score_only.secure_write(path, text, secret)
    except Exception as exc:
        raise PilotIntegrityError(f"artifact_write_failed:{path.name}") from exc


def _append_jsonl(path: Path, record: Dict[str, Any], secret: str) -> None:
    line = json.dumps(score_only.json_safe(record), ensure_ascii=False, allow_nan=False) + "\n"
    if secret and secret in line:
        raise PilotIntegrityError("credential_redaction_failure")
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        payload = line.encode("utf-8")
        while payload:
            written = os.write(descriptor, payload)
            if written <= 0:
                raise OSError("short append")
            payload = payload[written:]
        os.fsync(descriptor)
        os.chmod(path, 0o600)
    except Exception as exc:
        raise PilotIntegrityError("evaluation_record_write_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _record_template(slot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "block_id": slot["block_id"], "block_seed": slot["block_seed"],
        "round_index": slot["round_index"], "order_cycle": slot["order_cycle"],
        "order_position": slot["order_position"], "global_slot_index": slot["global_slot_index"],
        "method": slot["method"], "method_slot_index": slot["method_slot_index"],
        "proposal_attempted": True, "proposal_valid": None, "proposal_failure_category": None,
        "candidate_hash": None, "candidate_parameters": None,
        "complete_candidate_configuration": None, "logical_generation_index": None,
        "physical_request_count": 0, "retry_count": 0,
        "candidate_simulation_performed": False, "execution_success": False,
        "simulation_failure": False, "metrics": None, "s": None, "v": None, "q": None,
        "accepted": False, "optimizer_objective": None,
        "incumbent_parameters_before": None, "incumbent_metrics_before": None,
        "incumbent_q_before": None, "incumbent_updated": False, "no_update_reason": None,
        "incumbent_parameters_after": None, "incumbent_metrics_after": None,
        "incumbent_q_after": None, "proposal_duration": None, "api_duration": None,
        "simulation_duration": None, "total_slot_duration": None,
        "prompt_tokens": None, "completion_tokens": None, "cached_tokens": None,
        "provider_request_id": None, "prompt_hash": None, "response_hash": None,
        "feedback_object_hash": None, "fallback_used": False,
        "credential_scan_passed": True,
        "record_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


CANONICAL_RECORD_FIELDS = frozenset(_record_template(build_schedule()[0]))


def validate_record(record: Dict[str, Any]) -> None:
    if set(record) != CANONICAL_RECORD_FIELDS:
        raise ValueError("pilot record does not match the canonical schema")
    if record["method"] not in METHODS or record["fallback_used"] is not False:
        raise ValueError("pilot record contains an invalid method or fallback state")


def _usage_fields(record: Dict[str, Any], usage: Dict[str, Any]) -> None:
    record["prompt_tokens"] = usage.get("prompt_tokens")
    record["completion_tokens"] = usage.get("completion_tokens")
    record["cached_tokens"] = usage.get("prompt_cache_hit_tokens")
    record["provider_request_id"] = usage.get("provider_request_id")


def _score_prompt(space: Dict[str, Any], state: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> str:
    incumbent = copy.deepcopy(state["complete_configuration"] or state["parameters"])
    if previous is None:
        previous_score = None
    else:
        previous_score = {
            "q": previous.get("q"), "accepted": previous.get("accepted", False),
            "execution_success": previous.get("execution_success", False),
            "incumbent_updated": previous.get("incumbent_updated", False),
        }
    return score_only.build_prompt(space, incumbent, float(state["q"]), previous_score)


def _full_prompt(space: Dict[str, Any], state: Dict[str, Any], previous: Optional[Dict[str, Any]], stall: int) -> str:
    return full_feedback.build_prompt(space, full_feedback.build_feedback_object(state, previous, stall))


def _run_pilot_impl(
    api_key: str,
    output_dir: Path,
    *,
    baseline_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    transport_fn: Callable[..., Dict[str, Any]] = request_chat_completion,
) -> Dict[str, Any]:
    """Run the framework with injected dependencies; real execution is opt-in via the launcher."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot output: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    methods_dir = output_dir / "methods"
    prompts_dir = output_dir / "prompts"
    responses_dir = output_dir / "responses"
    feedback_dir = output_dir / "feedback_objects"
    for directory in (methods_dir, prompts_dir, responses_dir, feedback_dir):
        directory.mkdir(mode=0o700)
    for method in METHODS:
        (methods_dir / method).mkdir(mode=0o700)

    _json_write(output_dir / "block_status.json", {
        "schema_version": SCHEMA_VERSION, "block_id": BLOCK_ID,
        "status": "running", "completed_slots": 0, "expected_slots": TOTAL_SLOTS,
        "aborted": False, "incomplete": True,
    }, api_key)

    space = score_only.load_space()
    schedule = build_schedule()
    seeds = method_seeds(BLOCK_SEED)
    baseline = baseline_fn({"mode": "baseline"})
    common_baseline = _baseline_state(baseline)
    common_baseline_artifact = {"schema_version": SCHEMA_VERSION, "method": "common_baseline", **common_baseline}
    _json_write(output_dir / "common_baseline.json", common_baseline_artifact, api_key)
    _json_write(output_dir / "schedule.json", schedule, api_key)
    manifest = {
        "schema_version": SCHEMA_VERSION, "pilot_name": PILOT_NAME, "block_id": BLOCK_ID,
        "block_seed": BLOCK_SEED, "pilot_only": PILOT_ONLY,
        "publication_eligible": PUBLICATION_ELIGIBLE,
        "formal_experiment": FORMAL_EXPERIMENT, "slots_per_method": SLOTS_PER_METHOD,
        "total_slots": TOTAL_SLOTS, "methods": list(METHODS),
        "runtime_sensitive_metric": True, "runtime_metric_name": "avg_controller_time_ms",
        "balanced_schedule_used": True, "formal_comparison_treatment_not_yet_frozen": True,
        "pilot_protocol": True, "formal_objective_mapping_not_yet_frozen": True,
        "method_seeds": seeds, "tpe": {"direction": "maximize", "n_startup_trials": 2, "multivariate": True, "pruning": False, "early_stopping": False},
        "baseline_simulations": BASELINE_SIMULATION_LIMIT,
        "authentication_requests": AUTHENTICATION_REQUEST_LIMIT,
        "scheduled_slots": TOTAL_SLOTS, "total_logical_llm_generations": 40,
        "physical_generation_request_limit": PHYSICAL_GENERATION_REQUEST_LIMIT,
        "candidate_simulation_limit": CANDIDATE_SIMULATION_LIMIT,
        "fallback_generations": 0, "fourth_generation": False,
        "fallback_enabled": FALLBACK_ENABLED, "warm_start_enabled": WARM_START_ENABLED,
        "replacement_slots_enabled": REPLACEMENT_SLOTS_ENABLED,
    }
    _json_write(output_dir / "block_manifest.json", manifest, api_key)

    states = {method: copy.deepcopy(common_baseline) for method in METHODS}
    previous_score: Optional[Dict[str, Any]] = None
    previous_full: Optional[Dict[str, Any]] = None
    full_stall = 0
    random_rng = random.Random(seeds["random_search"])
    tpe_study = build_tpe_study(seeds["tpe"])
    records: List[Dict[str, Any]] = []
    physical_generation_requests = candidate_simulations = logical_llm = 0
    log_path = output_dir / "evaluations.jsonl"
    _secure_write(log_path, "", api_key)

    for slot in schedule:
        method = slot["method"]
        method_state = states[method]
        before = copy.deepcopy(method_state)
        record = _record_template(slot)
        record["incumbent_parameters_before"] = copy.deepcopy(before["parameters"])
        record["incumbent_metrics_before"] = copy.deepcopy(before["metrics"])
        record["incumbent_q_before"] = before["q"]
        slot_started = time.perf_counter()
        proposal_started = time.perf_counter()
        parameters = None
        request_result: Dict[str, Any] = {}
        raw_response = ""
        feedback_hash = None
        trial = None

        try:
            if method == "random_search":
                parameters = random_parameters(space, random_rng)
            elif method == "tpe":
                trial = tpe_study.ask()
                parameters = tpe_parameters(trial, space)
            elif method == "score_only_vgmpc":
                logical_llm += 1
                prompt = _score_prompt(space, method_state, previous_score)
                prompt_path = prompts_dir / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _secure_write(prompt_path, prompt, api_key)
                record["prompt_hash"] = score_only.hash_text(prompt)
                request_result = transport_fn(
                    api_key=api_key,
                    messages=[{"role": "system", "content": "Return only the requested JSON object."}, {"role": "user", "content": prompt}],
                    model="deepseek-chat", temperature=0.3, max_tokens=512,
                    connect_timeout=15, read_timeout=60,
                    response_format={"type": "json_object"}, allow_transport_retry=True,
                )
                physical = int(request_result.get("physical_request_count", 0))
                physical_generation_requests += physical
                if physical_generation_requests > PHYSICAL_GENERATION_REQUEST_LIMIT or physical > 2:
                    raise PilotIntegrityError("pilot_generation_request_limit_exceeded")
                record["physical_request_count"] = physical
                record["retry_count"] = int(request_result.get("retry_count", max(0, physical - 1)))
                raw_response = request_result.get("assistant_content") or ""
                response_path = responses_dir / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _secure_write(response_path, raw_response or "", api_key)
                record["response_hash"] = score_only.hash_text(raw_response or "")
                parameters, parse_error = score_only.strict_parse(raw_response, space) if raw_response else (None, "malformed_json")
                record["proposal_failure_category"] = None if parameters is not None else parse_error
                record["logical_generation_index"] = slot["method_slot_index"]
                record["api_duration"] = request_result.get("request_duration_s")
                _usage_fields(record, {**request_result.get("token_usage", {}), "provider_request_id": request_result.get("provider_request_id")})
                if parameters is None:
                    previous_score = None
            elif method == "full_feedback_vgmpc":
                logical_llm += 1
                feedback = full_feedback.build_feedback_object(method_state, previous_full, full_stall)
                feedback_text = json.dumps(feedback, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                feedback_path = feedback_dir / f"{method}_slot_{slot['method_slot_index']:03d}.json"
                _secure_write(feedback_path, feedback_text, api_key)
                feedback_hash = score_only.hash_text(feedback_text)
                record["feedback_object_hash"] = feedback_hash
                prompt = full_feedback.build_prompt(space, feedback)
                prompt_path = prompts_dir / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _secure_write(prompt_path, prompt, api_key)
                record["prompt_hash"] = score_only.hash_text(prompt)
                request_result = transport_fn(
                    api_key=api_key,
                    messages=[{"role": "system", "content": "Return only the requested JSON object."}, {"role": "user", "content": prompt}],
                    model="deepseek-chat", temperature=0.3, max_tokens=512,
                    connect_timeout=15, read_timeout=60,
                    response_format={"type": "json_object"}, allow_transport_retry=True,
                )
                physical = int(request_result.get("physical_request_count", 0))
                physical_generation_requests += physical
                if physical_generation_requests > PHYSICAL_GENERATION_REQUEST_LIMIT or physical > 2:
                    raise PilotIntegrityError("pilot_generation_request_limit_exceeded")
                record["physical_request_count"] = physical
                record["retry_count"] = int(request_result.get("retry_count", max(0, physical - 1)))
                raw_response = request_result.get("assistant_content") or ""
                response_path = responses_dir / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _secure_write(response_path, raw_response or "", api_key)
                record["response_hash"] = score_only.hash_text(raw_response or "")
                parameters, parse_error = score_only.strict_parse(raw_response, space) if raw_response else (None, "malformed_json")
                record["proposal_failure_category"] = None if parameters is not None else parse_error
                record["logical_generation_index"] = slot["method_slot_index"]
                record["api_duration"] = request_result.get("request_duration_s")
                _usage_fields(record, {**request_result.get("token_usage", {}), "provider_request_id": request_result.get("provider_request_id")})
                if parameters is None:
                    previous_full = full_feedback._invalid_previous(parse_error, before)
                    full_stall += 1
            record["proposal_duration"] = time.perf_counter() - proposal_started
            record["proposal_valid"] = parameters is not None
            if parameters is not None:
                record["candidate_parameters"] = copy.deepcopy(parameters)
                record["candidate_simulation_performed"] = True
                candidate_simulations += 1
                result = evaluate_fn({"mode": "candidate", "parameters": parameters, "incumbent_before": before["q"], "incumbent_candidate_id": before["candidate_hash"]})
                candidate = _state_from_result(result, parameters)
                record["execution_success"] = candidate["execution_success"]
                record["simulation_failure"] = not candidate["execution_success"]
                record["complete_candidate_configuration"] = copy.deepcopy(candidate["complete_configuration"])
                record["candidate_hash"] = candidate["candidate_hash"] or _candidate_hash(parameters)
                record["metrics"] = copy.deepcopy(candidate["metrics"])
                record["s"], record["v"], record["q"] = candidate["s"], candidate["v"], candidate["q"]
                record["accepted"] = candidate["accepted"]
                record["optimizer_objective"] = optimizer_objective(result)
                updated_state, updated, reason = _update_incumbent(before, candidate)
                record["incumbent_updated"] = updated
                record["no_update_reason"] = reason
                states[method] = updated_state
                record["incumbent_parameters_after"] = copy.deepcopy(updated_state["parameters"])
                record["incumbent_metrics_after"] = copy.deepcopy(updated_state["metrics"])
                record["incumbent_q_after"] = updated_state["q"]
                record["simulation_duration"] = result.get("simulation_duration_s")
                if method == "tpe":
                    tpe_study.tell(trial, record["optimizer_objective"] if record["optimizer_objective"] is not None else float("-inf"))
                if method == "score_only_vgmpc":
                    previous_score = {"q": candidate["q"], "accepted": candidate["accepted"], "execution_success": candidate["execution_success"], "incumbent_updated": updated}
                if method == "full_feedback_vgmpc":
                    previous_full = {"candidate_available": True, "parameters": parameters, "s": candidate["s"], "v": candidate["v"], "q": candidate["q"], "accepted": candidate["accepted"], "execution_success": candidate["execution_success"], "incumbent_updated": updated, "validation_category": "valid", "simulation_failure": not candidate["execution_success"], "metrics": candidate["metrics"], "reference_incumbent": before}
                    full_stall = 0 if updated else full_stall + 1
            else:
                record["no_update_reason"] = record["proposal_failure_category"] or "invalid_proposal"
                record["incumbent_parameters_after"] = copy.deepcopy(before["parameters"])
                record["incumbent_metrics_after"] = copy.deepcopy(before["metrics"])
                record["incumbent_q_after"] = before["q"]
                record["candidate_simulation_performed"] = False
                record["simulation_failure"] = False
                if method == "tpe":
                    # Invalid TPE proposal is not expected, but still consumes its trial slot.
                    tpe_study.tell(trial, float("-inf"))
        except PilotIntegrityError:
            raise
        except Exception:
            if parameters is not None:
                record["proposal_valid"] = True
                record["execution_success"] = False
                record["simulation_failure"] = True
                record["no_update_reason"] = "execution_failure"
                record["candidate_hash"] = record["candidate_hash"] or _candidate_hash(parameters)
            else:
                record["proposal_valid"] = False
                record["proposal_failure_category"] = "pilot_runtime_failure"
                record["no_update_reason"] = "execution_failure"
            record["incumbent_parameters_after"] = copy.deepcopy(before["parameters"])
            record["incumbent_metrics_after"] = copy.deepcopy(before["metrics"])
            record["incumbent_q_after"] = before["q"]
            if method == "tpe" and trial is not None:
                tpe_study.tell(trial, float("-inf"))
            if method == "score_only_vgmpc":
                previous_score = None
            if method == "full_feedback_vgmpc":
                previous_full = full_feedback._invalid_previous("pilot_runtime_failure", before)
                full_stall += 1
        record["total_slot_duration"] = time.perf_counter() - slot_started
        validate_record(record)
        records.append(record)
        _append_jsonl(log_path, record, api_key)

    if (len(records) != TOTAL_SLOTS or logical_llm != LOGICAL_LLM_GENERATION_LIMIT
            or candidate_simulations > CANDIDATE_SIMULATION_LIMIT):
        raise PilotIntegrityError("pilot_budget_invariant_failed")
    summary = {
        "schema_version": SCHEMA_VERSION, "pilot_name": PILOT_NAME, "block_id": BLOCK_ID,
        "block_seed": BLOCK_SEED, "pilot_only": True, "publication_eligible": False,
        "formal_experiment": False, "scheduled_slots": len(records),
        "slots_per_method": {method: sum(r["method"] == method for r in records) for method in METHODS},
        "total_logical_llm_generations": logical_llm,
        "physical_generation_requests": physical_generation_requests,
        "candidate_simulations": candidate_simulations, "fallback_generations": 0,
        "fourth_generation": False, "common_baseline_simulations": 1,
        "common_baseline_hash": common_baseline["candidate_hash"],
        "tpe_trials": len(tpe_study.trials), "method_seeds": seeds,
    }
    _json_write(output_dir / "block_summary.json", summary, api_key)
    scan = score_only.credential_scan(output_dir, api_key)
    validation = {
        "schedule_valid": True, "common_baseline_simulations": 1,
        "common_baseline_deepcopy_per_method": True, "scheduled_slots": len(records),
        "method_slot_counts": summary["slots_per_method"], "total_logical_llm_generations": logical_llm,
        "physical_generation_requests_not_above_limit": physical_generation_requests <= 80,
        "candidate_simulations_not_above_limit": candidate_simulations <= 80,
        "fallback_used": False, "fourth_generation": False,
        "publication_eligible": False, "pilot_only": True,
        "balanced_schedule_used": True, "runtime_sensitive_metric": True,
        "runtime_metric_name": "avg_controller_time_ms",
        "formal_comparison_treatment_not_yet_frozen": True,
        "formal_objective_mapping_not_yet_frozen": True,
        "credential_scan_passed": scan["credential_scan_passed"],
        "credential_matches": scan["credential_matches"],
    }
    _json_write(output_dir / "validation_report.json", validation, api_key)
    final_scan = score_only.credential_scan(output_dir, api_key)
    if not final_scan["credential_scan_passed"]:
        raise PilotIntegrityError("pilot_credential_scan_failed")
    _json_write(output_dir / "block_status.json", {
        "schema_version": SCHEMA_VERSION, "block_id": BLOCK_ID,
        "status": "complete", "completed_slots": len(records),
        "expected_slots": TOTAL_SLOTS, "aborted": False, "incomplete": False,
    }, api_key)
    return summary


def run_pilot(
    api_key: str,
    output_dir: Path,
    *,
    baseline_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    transport_fn: Callable[..., Dict[str, Any]] = request_chat_completion,
) -> Dict[str, Any]:
    """Run one block and durably mark any interrupted block as aborted."""
    output_dir = Path(output_dir)
    try:
        return _run_pilot_impl(
            api_key, output_dir, baseline_fn=baseline_fn,
            evaluate_fn=evaluate_fn, transport_fn=transport_fn,
        )
    except BaseException as exc:
        if output_dir.exists() and (output_dir / "block_status.json").exists():
            log_path = output_dir / "evaluations.jsonl"
            completed = 0
            try:
                if log_path.exists():
                    completed = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
                _json_write(output_dir / "block_status.json", {
                    "schema_version": SCHEMA_VERSION, "block_id": BLOCK_ID,
                    "status": "incomplete/aborted", "completed_slots": completed,
                    "expected_slots": TOTAL_SLOTS, "aborted": True, "incomplete": True,
                    "error_class": type(exc).__name__,
                }, api_key)
            except Exception:
                pass
        raise
