#!/usr/bin/env python3
"""Three-generation Full-feedback VG-MPC non-publication smoke test."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import score_only_vgmpc as score_only
from deepseek_requests_transport import request_chat_completion


METHOD = "full_feedback_vgmpc"
SEED = 20260814
MODEL = "deepseek-chat"
TEMPERATURE = 0.3
MAX_TOKENS = 512
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
PROMPT_VERSION = "full_feedback_v1"
PARSER_VERSION = score_only.PARSER_VERSION
# Absolute delta tolerance scaled by the larger metric magnitude (or 1).
METRIC_TOLERANCE = 1e-12
METRIC_NAMES = (
    "integral_error",
    "rms_tracking_error",
    "max_tracking_error",
    "final_tracking_error",
    "control_energy",
    "avg_controller_time_ms",
    "median_condition_number",
    "max_kkt_residual",
)
EXCLUDED_PROMPT_PHRASES = (
    "random search",
    "tpe",
    "score-only",
    "score only",
    "historical vg-mpc",
    "dual-generator",
    "dual generator",
    "future outcome",
    "source-code editing",
    "source code editing",
    "api key",
    "authorization header",
    "raw provider error",
    "raw matlab exception",
    "complete trajectory",
)


def parameters_only(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {name: candidate[name] for name in score_only.NAMES}


def named_metrics(values: Any) -> Optional[Dict[str, Optional[float]]]:
    if not isinstance(values, (list, tuple)) or len(values) != len(METRIC_NAMES):
        return None
    result: Dict[str, Optional[float]] = {}
    for name, value in zip(METRIC_NAMES, values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            result[name] = None
        else:
            numeric = float(value)
            result[name] = numeric if math.isfinite(numeric) else None
    return result


def finite_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def metric_comparison(
    candidate_metrics: Optional[Dict[str, Optional[float]]],
    reference_metrics: Optional[Dict[str, Optional[float]]],
) -> Tuple[Optional[Dict[str, Optional[float]]], List[str], List[str], List[str]]:
    if candidate_metrics is None or reference_metrics is None:
        return None, [], [], []
    deltas: Dict[str, Optional[float]] = {}
    improved: List[str] = []
    degraded: List[str] = []
    unchanged: List[str] = []
    for name in METRIC_NAMES:
        candidate = candidate_metrics.get(name)
        reference = reference_metrics.get(name)
        if candidate is None or reference is None:
            deltas[name] = None
            continue
        delta = candidate - reference
        scale = max(1.0, abs(candidate), abs(reference))
        if abs(delta) <= METRIC_TOLERANCE * scale:
            deltas[name] = 0.0
            unchanged.append(name)
        elif delta < 0:
            deltas[name] = delta
            improved.append(name)
        else:
            deltas[name] = delta
            degraded.append(name)
    return deltas, improved, degraded, unchanged


def validation_category(raw: str, parser_error: str) -> str:
    if parser_error == "markdown_code_fence":
        return "markdown_or_surrounding_text"
    if parser_error == "malformed_json":
        stripped = raw.strip()
        if stripped and (not stripped.startswith("{") or not stripped.endswith("}")):
            return "markdown_or_surrounding_text"
        return "malformed_json"
    if parser_error == "field_set_mismatch":
        try:
            parsed = json.loads(raw)
        except Exception:
            return "malformed_json"
        if isinstance(parsed, dict):
            missing = set(score_only.NAMES) - set(parsed)
            extra = set(parsed) - set(score_only.NAMES)
            if missing:
                return "missing_field"
            if extra:
                return "extra_field"
        return "invalid_type"
    return {
        "invalid_numeric_type": "invalid_type",
        "not_object": "invalid_type",
        "non_canonical_json": "invalid_type",
        "non_finite": "nonfinite_value",
        "non_integer_horizon": "noninteger_horizon",
        "out_of_range": "out_of_range",
        "empty_response": "malformed_json",
        "credential_echo_detected": "transport_failure",
        "ReadTimeout": "transport_failure",
        "ConnectTimeout": "transport_failure",
        "ConnectionError": "transport_failure",
    }.get(parser_error, parser_error or "malformed_json")


def failure_category(previous: Optional[Dict[str, Any]]) -> str:
    if previous is None:
        return "not_available"
    if not previous["candidate_available"]:
        return "validation_failure"
    if not previous["execution_success"]:
        return "execution_failure"
    if not previous["accepted"]:
        return "acceptance_failure"
    if previous["incumbent_updated"]:
        return "incumbent_updated"
    return "insufficient_score"


def diagnostic_summary(category: str) -> str:
    return {
        "not_available": "No previous generation is available.",
        "validation_failure": "Candidate output failed strict schema validation.",
        "execution_failure": "Candidate simulation did not execute successfully.",
        "acceptance_failure": "Candidate failed the benchmark acceptance rule.",
        "insufficient_score": "Candidate was accepted but did not exceed the incumbent score.",
        "incumbent_updated": "Candidate updated the incumbent.",
    }[category]


def state_from_evaluation(
    evaluation: Dict[str, Any], fallback_parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    candidate = evaluation.get("candidate") or fallback_parameters or {}
    parameters = parameters_only(candidate) if all(name in candidate for name in score_only.NAMES) else None
    return {
        "parameters": parameters,
        "s": finite_or_none(evaluation.get("s")),
        "v": finite_or_none(evaluation.get("v")),
        "q": finite_or_none(evaluation.get("q")),
        "metrics": named_metrics(evaluation.get("metrics")),
        "accepted": bool(evaluation.get("accepted")),
        "execution_success": bool(evaluation.get("execution_success")),
        "candidate_hash": evaluation.get("candidate_id", ""),
    }


def build_feedback_object(
    incumbent: Dict[str, Any], previous: Optional[Dict[str, Any]], stall_count: int
) -> Dict[str, Any]:
    category = failure_category(previous)
    if previous is None:
        previous_object = {
            "available": False,
            "candidate_available": False,
            "parameters": None,
            "s": None,
            "v": None,
            "q": None,
            "accepted": False,
            "execution_success": False,
            "incumbent_updated": False,
            "validation_category": "not_available",
            "simulation_failure": False,
            "metrics": None,
        }
        reference_parameters = reference_q = reference_metrics = None
        deltas, improved, degraded, unchanged = None, [], [], []
    else:
        previous_object = {
            "available": True,
            "candidate_available": bool(previous["candidate_available"]),
            "parameters": previous.get("parameters"),
            "s": previous.get("s"),
            "v": previous.get("v"),
            "q": previous.get("q"),
            "accepted": bool(previous.get("accepted")),
            "execution_success": bool(previous.get("execution_success")),
            "incumbent_updated": bool(previous.get("incumbent_updated")),
            "validation_category": previous.get("validation_category", ""),
            "simulation_failure": bool(previous.get("simulation_failure")),
            "metrics": previous.get("metrics"),
        }
        reference = previous["reference_incumbent"]
        reference_parameters = reference["parameters"]
        reference_q = reference["q"]
        reference_metrics = reference["metrics"]
        deltas, improved, degraded, unchanged = metric_comparison(
            previous.get("metrics"), reference_metrics
        )
    return {
        "current_incumbent": {
            "parameters": incumbent["parameters"],
            "s": incumbent["s"],
            "v": incumbent["v"],
            "q": incumbent["q"],
            "metrics": incumbent["metrics"],
            "accepted": incumbent["accepted"],
            "execution_success": incumbent["execution_success"],
        },
        "previous_generation": previous_object,
        "comparison": {
            "reference": "incumbent_before_previous_evaluation",
            "reference_parameters": reference_parameters,
            "reference_q": reference_q,
            "reference_metrics": reference_metrics,
            "metric_deltas": deltas,
            "improved_metrics": improved,
            "degraded_metrics": degraded,
            "unchanged_metrics": unchanged,
            "failure_category": category,
            "stall_count": stall_count,
            "diagnostic_summary": diagnostic_summary(category),
        },
    }


def build_prompt(space: Dict[str, Any], feedback: Dict[str, Any]) -> str:
    lines = [
        "Task: Propose values for exactly seven CSTR configuration parameters.",
        "",
        "Allowed parameters:",
    ]
    for spec in space["variables"]:
        kind = "integer" if spec["type"] == "integer" else "number"
        lines.append(f"- {spec['name']}: {kind}, range [{spec['lo']}, {spec['hi']}]")
    lines.extend(
        [
            "",
            "All other CSTR configuration settings are fixed to their baseline values.",
            "Do not modify structural settings.",
            "Objective: improve the accepted incumbent selection score q.",
            "",
            "Full-feedback evidence:",
            json.dumps(feedback, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "",
            "Output exactly one JSON object with exactly the seven parameter fields.",
            "Use JSON numbers only. No explanation, no markdown, no extra fields.",
        ]
    )
    return "\n".join(lines)


def prompt_isolation_check(prompt: str) -> Dict[str, Any]:
    lowered = prompt.lower()
    hits = [phrase for phrase in EXCLUDED_PROMPT_PHRASES if phrase in lowered]
    return {
        "passed": not hits,
        "excluded_phrase_hits": hits,
        "has_all_parameters": all(name in prompt for name in score_only.NAMES),
        "has_all_metrics": all(name in prompt for name in METRIC_NAMES),
        "json_only_instruction": "No explanation, no markdown, no extra fields." in prompt,
    }


def _invalid_previous(category: str, reference: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_available": False,
        "parameters": None,
        "s": None,
        "v": None,
        "q": None,
        "accepted": False,
        "execution_success": False,
        "incumbent_updated": False,
        "validation_category": category,
        "simulation_failure": False,
        "metrics": None,
        "reference_incumbent": reference,
    }


def run_secure(
    api_key: str,
    output_dir: Path,
    transport_fn: Callable[..., Dict[str, Any]] = request_chat_completion,
    matlab_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key must be a non-empty string")
    if matlab_fn is None:
        matlab_fn = score_only.matlab_call
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    space = score_only.load_space()
    output_dir.mkdir(parents=True, mode=0o700)
    prompts_dir = output_dir / "prompts"
    responses_dir = output_dir / "responses"
    feedback_dir = output_dir / "feedback_objects"
    for directory in (prompts_dir, responses_dir, feedback_dir):
        directory.mkdir(mode=0o700)

    def write_json(path: Path, data: Dict[str, Any]) -> None:
        text = json.dumps(score_only.json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        score_only.secure_write(path, text, api_key)

    manifest = {
        "schema_version": "NON_PUBLICATION_SMOKE_TEST_v1",
        "method": METHOD,
        "smoke_test": True,
        "publication_eligible": False,
        "provider": "deepseek",
        "endpoint": score_only.ENDPOINT,
        "backend": "requests.post",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "connect_timeout_s": CONNECT_TIMEOUT,
        "read_timeout_s": READ_TIMEOUT,
        "response_format": {"type": "json_object"},
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "run_seed": seed,
        "logical_generations": 3,
        "fallback_generations": 0,
        "fourth_generation_prohibited": True,
        "runtime_sensitive_metric": True,
        "formal_comparison_treatment_not_yet_frozen": True,
        "metric_names": list(METRIC_NAMES),
        "metric_delta_reference": "incumbent_before_previous_evaluation",
        "metric_tolerance": METRIC_TOLERANCE,
        "api_key_entered_securely": True,
        "api_key_saved": False,
        "authorization_header_saved": False,
        "local_credential_read": False,
        "credential_loaded_from_environment": False,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    baseline = matlab_fn({"mode": "baseline"})
    write_json(output_dir / "baseline.json", baseline)
    incumbent = state_from_evaluation(baseline)
    incumbent_id = baseline.get("candidate_id", "baseline")
    previous: Optional[Dict[str, Any]] = None
    stall_count = 0
    physical_total = retry_total = valid_count = invalid_count = simulations = 0
    accepted_count = rejected_count = failed_count = updates = 0
    retry_reasons: List[str] = []
    rows: List[Dict[str, Any]] = []
    isolation_checks: List[Dict[str, Any]] = []
    valid_details: List[Dict[str, Any]] = []
    log_path = output_dir / "evaluations.jsonl"
    score_only.secure_write(log_path, "", api_key)
    started = time.perf_counter()

    for generation in range(1, 4):
        stall_before = stall_count
        feedback = build_feedback_object(incumbent, previous, stall_count)
        feedback_text = json.dumps(feedback, sort_keys=True, separators=(",", ":"), allow_nan=False)
        feedback_path = feedback_dir / f"generation_{generation:03d}.json"
        score_only.secure_write(feedback_path, feedback_text + "\n", api_key)
        feedback_hash = score_only.hash_text(feedback_text + "\n")
        prompt = build_prompt(space, feedback)
        prompt_path = prompts_dir / f"generation_{generation:03d}.txt"
        score_only.secure_write(prompt_path, prompt, api_key)
        saved_prompt = prompt_path.read_text(encoding="utf-8")
        isolation = prompt_isolation_check(saved_prompt)
        isolation_checks.append({"generation": generation, "immediately_before_request": isolation})
        if not all(
            isolation[key]
            for key in ("passed", "has_all_parameters", "has_all_metrics", "json_only_instruction")
        ):
            raise RuntimeError("full_feedback_prompt_isolation_failed")

        request_result = transport_fn(
            api_key=api_key,
            messages=[
                {"role": "system", "content": "Return only the requested JSON object."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            connect_timeout=CONNECT_TIMEOUT,
            read_timeout=READ_TIMEOUT,
            response_format={"type": "json_object"},
            allow_transport_retry=True,
        )
        physical = int(request_result.get("physical_request_count", 0))
        retries = int(request_result.get("retry_count", max(0, physical - 1)))
        reasons = list(request_result.get("retry_reasons", []))
        physical_total += physical
        retry_total += retries
        retry_reasons.extend(reasons)
        if physical < 1 or physical > 2 or physical_total > 6:
            raise RuntimeError("generation_request_count_limit_exceeded")

        raw = request_result.get("assistant_content") or ""
        if api_key in raw:
            raw = ""
            parser_error = "credential_echo_detected"
        else:
            parser_error = ""
        response_body = raw or json.dumps(
            {
                "transport_error_category": request_result.get("transport_exception_class")
                or request_result.get("provider_error_type")
                or "empty_response"
            },
            separators=(",", ":"),
        )
        response_path = responses_dir / f"generation_{generation:03d}.txt"
        score_only.secure_write(response_path, response_body, api_key)
        response_hash = score_only.hash_text(response_body)
        parse_success = False
        if raw:
            try:
                json.loads(raw)
                parse_success = True
            except Exception:
                pass
        parameters, strict_error = score_only.strict_parse(raw, space) if raw else (None, parser_error or "empty_response")
        category = validation_category(raw, strict_error)
        reference = dict(incumbent)
        before_parameters = dict(incumbent["parameters"])
        before_metrics = dict(incumbent["metrics"]) if incumbent["metrics"] is not None else None
        before_q = incumbent["q"]
        candidate_state = None
        evaluation: Optional[Dict[str, Any]] = None
        candidate_hash = ""
        simulation_duration = 0.0

        if parameters is None:
            invalid_count += 1
            previous = _invalid_previous(category, reference)
            failure = "validation_failure"
            no_update_reason = "invalid_api_output"
            stall_count += 1
        else:
            valid_count += 1
            simulations += 1
            evaluation = matlab_fn(
                {
                    "mode": "candidate",
                    "parameters": parameters,
                    "incumbent_before": before_q,
                    "incumbent_candidate_id": incumbent_id,
                }
            )
            candidate_state = state_from_evaluation(evaluation, parameters)
            candidate_hash = candidate_state["candidate_hash"]
            simulation_duration = float(evaluation.get("simulation_duration_s", 0.0))
            updated = bool(evaluation.get("incumbent_updated"))
            previous = {
                "candidate_available": True,
                "parameters": parameters,
                "s": candidate_state["s"],
                "v": candidate_state["v"],
                "q": candidate_state["q"],
                "accepted": candidate_state["accepted"],
                "execution_success": candidate_state["execution_success"],
                "incumbent_updated": updated,
                "validation_category": "valid",
                "simulation_failure": not candidate_state["execution_success"],
                "metrics": candidate_state["metrics"],
                "reference_incumbent": reference,
            }
            failure = failure_category(previous)
            no_update_reason = evaluation.get("no_update_reason", "")
            accepted_count += int(candidate_state["accepted"])
            rejected_count += int(not candidate_state["accepted"])
            failed_count += int(not candidate_state["execution_success"])
            if updated:
                incumbent = candidate_state
                incumbent_id = candidate_hash
                updates += 1
                stall_count = 0
            else:
                stall_count += 1
            valid_details.append(
                {
                    "hash": candidate_hash,
                    "parameters": parameters,
                    "q": candidate_state["q"],
                    "accepted": candidate_state["accepted"],
                    "incumbent_updated": updated,
                }
            )

        after_parameters = dict(incumbent["parameters"])
        after_metrics = dict(incumbent["metrics"]) if incumbent["metrics"] is not None else None
        record = {
            "schema_version": "NON_PUBLICATION_SMOKE_TEST_v1",
            "method": METHOD,
            "smoke_test": True,
            "publication_eligible": False,
            "logical_generation": generation,
            "physical_request_count": physical,
            "retry_count": retries,
            "retry_reason": reasons,
            "prompt_hash": score_only.hash_text(saved_prompt),
            "response_hash": response_hash,
            "feedback_object_hash": feedback_hash,
            "parse_success": parse_success,
            "schema_validation_success": parameters is not None,
            "validation_error_category": "" if parameters is not None else category,
            "candidate_parameters": parameters,
            "candidate_hash": candidate_hash,
            "incumbent_before_parameters": before_parameters,
            "incumbent_before_metrics": before_metrics,
            "incumbent_before_q": before_q,
            "candidate_metrics": candidate_state["metrics"] if candidate_state else None,
            "candidate_s": candidate_state["s"] if candidate_state else None,
            "candidate_v": candidate_state["v"] if candidate_state else None,
            "candidate_q": candidate_state["q"] if candidate_state else None,
            "accepted": candidate_state["accepted"] if candidate_state else False,
            "execution_success": candidate_state["execution_success"] if candidate_state else False,
            "incumbent_after_parameters": after_parameters,
            "incumbent_after_metrics": after_metrics,
            "incumbent_after_q": incumbent["q"],
            "incumbent_updated": bool(previous["incumbent_updated"]),
            "no_update_reason": no_update_reason,
            "failure_category": failure,
            "stall_count_before": stall_before,
            "stall_count_after": stall_count,
            "api_duration_s": request_result.get("request_duration_s", 0.0),
            "simulation_duration_s": simulation_duration,
            "token_usage": request_result.get("token_usage", {}),
            "fallback_used": False,
            "provider_http_status": request_result.get("http_status"),
        }
        line = json.dumps(score_only.json_safe(record), ensure_ascii=False, allow_nan=False) + "\n"
        if api_key in line:
            raise RuntimeError("credential_redaction_failure")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line)
        rows.append(record)

    summary = {
        "schema_version": "NON_PUBLICATION_SMOKE_TEST_v1",
        "method": METHOD,
        "smoke_test": True,
        "publication_eligible": False,
        "baseline_q": finite_or_none(baseline.get("q")),
        "baseline_accepted": bool(baseline.get("accepted")),
        "logical_generations": len(rows),
        "physical_generation_requests": physical_total,
        "generation_retry_count": retry_total,
        "generation_retry_reasons": retry_reasons,
        "valid_candidates": valid_count,
        "invalid_responses": invalid_count,
        "candidate_simulations": simulations,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "failed_simulations": failed_count,
        "incumbent_updates": updates,
        "final_incumbent_score": incumbent["q"],
        "final_incumbent_candidate_id": incumbent_id,
        "final_stall_count": stall_count,
        "baseline_simulations": 1,
        "fallback_generations": 0,
        "fourth_generation_started": False,
        "valid_candidate_details": valid_details,
        "total_duration_s": time.perf_counter() - started,
    }
    write_json(output_dir / "summary.json", summary)
    scan = score_only.credential_scan(output_dir, api_key)
    validation = {
        "prompt_isolation_checks": isolation_checks,
        "credential_scan_passed": scan["credential_scan_passed"],
        "credential_matches": scan["credential_matches"],
        "api_key_entered_securely": True,
        "api_key_saved": False,
        "authorization_header_saved": False,
        "local_credential_read": False,
        "physical_request_count": physical_total,
        "logical_generation_count": len(rows),
        "baseline_simulation_count": 1,
        "candidate_simulation_count": simulations,
        "fallback_used": False,
        "fourth_generation_started": False,
        "full_feedback_only": True,
        "candidate_simulations_not_above_valid_candidates": simulations <= valid_count,
        "runtime_sensitive_metric": True,
        "formal_comparison_treatment_not_yet_frozen": True,
    }
    write_json(output_dir / "validation_report.json", validation)
    if not score_only.credential_scan(output_dir, api_key)["credential_scan_passed"]:
        raise RuntimeError("credential_scan_failed")
    return summary


# Descriptive aliases keep the public entry points easy to discover in tests and tools.
construct_feedback_object = build_feedback_object
build_full_feedback_prompt = build_prompt
