#!/usr/bin/env python3
"""Additive formal CSTR experiment runner with injected I/O."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna

import deepseek_requests_transport as transport
import intermediate_evaluation_cache as cache_module
import formal_protocol as protocol
import formal_provenance as provenance
import score_only_vgmpc as score_only


ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-chat"
TEMPERATURE = 0.3
MAX_TOKENS = 512
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
AUTHENTICATION_REQUEST_LIMIT = 1
PHYSICAL_GENERATION_REQUEST_LIMIT = 320
LOGICAL_GENERATION_LIMIT = 160
CANDIDATE_SIMULATION_LIMIT = 320
FALLBACK_GENERATIONS = 0


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _config_hash(configuration: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(protocol._safe(configuration), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: Any, secret: str) -> None:
    try:
        protocol.secure_json_write(path, value, secret)
    except Exception as exc:
        raise protocol.FormalIntegrityError(f"artifact_write_failed:{path.name}") from exc


def _write_atomic_json(path: Path, value: Any, secret: str) -> None:
    text = json.dumps(protocol._safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if secret and secret in text:
        raise protocol.FormalIntegrityError("credential_redaction_failure")
    descriptor = None
    temporary: Optional[Path] = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        payload = text.encode("utf-8")
        while payload:
            written = os.write(descriptor, payload)
            if written <= 0:
                raise OSError("short atomic state write")
            payload = payload[written:]
        os.fsync(descriptor); os.close(descriptor); descriptor = None
        os.chmod(temporary, 0o600); os.replace(temporary, path); temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_descriptor)
        finally: os.close(directory_descriptor)
    except Exception as exc:
        raise protocol.FormalIntegrityError(f"atomic_artifact_write_failed:{path.name}") from exc
    finally:
        if descriptor is not None: os.close(descriptor)
        if temporary is not None:
            try: temporary.unlink()
            except FileNotFoundError: pass


def _heartbeat(path: Path, block_id: int, stage: str, *, slot: Optional[Dict[str, Any]] = None,
               parameters: Optional[Dict[str, Any]] = None, candidate_hash_value: Optional[str] = None,
               last_completed_global_slot: int = 0, secret: str) -> None:
    _write_atomic_json(path, {
        "formal_protocol_version": protocol.FORMAL_PROTOCOL_VERSION,
        "execution_safety_version": protocol.EXECUTION_SAFETY_VERSION,
        "block_id": block_id, "attempt_id": path.parent.name, "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_completed_global_slot": last_completed_global_slot,
        "current_global_slot": slot.get("global_slot_index") if slot else None,
        "current_round": slot.get("round_index") if slot else None,
        "current_method": slot.get("method") if slot else None,
        "current_method_slot": slot.get("method_slot_index") if slot else None,
        "candidate_parameters": copy.deepcopy(parameters),
        "candidate_hash": candidate_hash_value,
    }, secret)


def _write_text(path: Path, text: str, secret: str) -> None:
    if secret and secret in text:
        raise protocol.FormalIntegrityError("credential_redaction_failure")
    try:
        path.write_text(text, encoding="utf-8")
        os.chmod(path, 0o600)
    except Exception as exc:
        raise protocol.FormalIntegrityError(f"artifact_write_failed:{path.name}") from exc


def _append_jsonl(path: Path, value: Dict[str, Any], secret: str) -> None:
    text = json.dumps(protocol._safe(value), ensure_ascii=False, allow_nan=False) + "\n"
    if secret and secret in text:
        raise protocol.FormalIntegrityError("credential_redaction_failure")
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        payload = text.encode()
        while payload:
            count = os.write(descriptor, payload)
            if count <= 0:
                raise OSError("short write")
            payload = payload[count:]
        os.fsync(descriptor)
        os.chmod(path, 0o600)
    except Exception as exc:
        raise protocol.FormalIntegrityError(f"jsonl_append_failed:{path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


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
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=2, multivariate=True)
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


def _parameters(configuration: Dict[str, Any]) -> Dict[str, Any]:
    return {name: configuration[name] for name in score_only.NAMES}


def _state(evaluation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_hash": evaluation["candidate_hash"],
        "complete_configuration": copy.deepcopy(configuration),
        "parameters": _parameters(configuration),
        "metrics": copy.deepcopy(evaluation["metrics"]),
        "raw_score_nonruntime": evaluation["raw_score_nonruntime"],
        "violation_penalty": evaluation["violation_penalty"],
        "selection_score_nonruntime": evaluation["selection_score_nonruntime"],
        "accepted": evaluation["accepted"], "execution_success": evaluation["execution_success"],
        "result_failed": evaluation["result_failed"],
    }


def _block_config(block_id: int) -> Dict[str, Any]:
    return {
        "experiment_name": protocol.EXPERIMENT_NAME, "block_id": block_id,
        "block_seed": protocol.block_seed(block_id), "method_seeds": protocol.method_seeds(block_id),
        "slots_per_method": protocol.SLOTS_PER_METHOD, "total_slots": protocol.TOTAL_SLOTS_PER_BLOCK,
        "schedule_offset": (block_id - 1) % 4, "methods": list(protocol.METHODS),
        "formal_protocol_version": protocol.FORMAL_PROTOCOL_VERSION,
        "execution_safety_version": protocol.EXECUTION_SAFETY_VERSION,
        "planned_block_count": protocol.PLANNED_BLOCKS,
        "score_protocol_version": protocol.SCORE_PROTOCOL_VERSION,
        "acceptance_protocol_version": protocol.ACCEPTANCE_PROTOCOL_VERSION,
        "duplicate_policy_version": protocol.DUPLICATE_POLICY_VERSION,
        "tpe_objective_version": protocol.TPE_OBJECTIVE_VERSION,
    }


def _source_hash(snapshot: Dict[str, Any], suffix: str) -> Optional[str]:
    for item in snapshot["files"]:
        if item["relative_to_runner"] == suffix:
            return item["sha256"]
    return None


def _set_protocol_hashes(record: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    record["evaluator_hash"] = _source_hash(snapshot, "evaluate_cstr_nonruntime_candidate.m")
    record["score_hash"] = snapshot["component_hashes"]["score_source_hash"]
    record["acceptance_hash"] = snapshot["component_hashes"]["acceptance_source_hash"]
    record["search_space_hash"] = _source_hash(snapshot, provenance.SEARCH_SPACE)


def _cache_hit_evaluation(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {name: entry["metrics"].get(name) for name in protocol.SCORE_METRICS}
    metrics[protocol.RUNTIME_METRIC] = None
    return {
        "candidate_hash": entry["candidate_hash"], "execution_success": entry["execution_success"],
        "result_failed": entry["result_failed"], "metrics": metrics,
        "raw_score_nonruntime": entry["raw_score_nonruntime"],
        "raw_score_nonruntime_status": "finite" if math.isfinite(entry["raw_score_nonruntime"]) else "negative_infinity_failure",
        "violation_penalty": entry["violation_penalty"],
        "selection_score_nonruntime": entry["selection_score_nonruntime"],
        "selection_score_nonruntime_status": "finite" if math.isfinite(entry["selection_score_nonruntime"]) else "negative_infinity_failure",
        "accepted": entry["accepted"], "simulation_duration_s": None,
    }


def _invalid_previous(before: Dict[str, Any], category: str) -> Dict[str, Any]:
    return {
        "candidate_available": False, "parameters": None, "raw_score_nonruntime": None,
        "violation_penalty": None, "selection_score_nonruntime": None,
        "accepted": False, "execution_success": False, "incumbent_updated": False,
        "validation_category": category, "simulation_failure": False, "metrics": None,
        "reference_incumbent": copy.deepcopy(before),
    }


def _initialize_method_artifacts(methods_dir: Path, seeds: Dict[str, int], block_id: int, secret: str) -> None:
    for method in protocol.METHODS:
        directory = methods_dir / method
        _write_json(directory / "method_config.json", {
            "formal_protocol_version": protocol.FORMAL_PROTOCOL_VERSION, "method": method,
            "block_id": block_id, "seed": seeds[method], "scheduled_slots": protocol.SLOTS_PER_METHOD,
            "early_stopping": False, "fallback": False, "replacement_slots": False,
        }, secret)
        _write_text(directory / "incumbent_history.jsonl", "", secret)
    _write_text(methods_dir / "random_search" / "proposal_sequence.jsonl", "", secret)
    for name in ("complete_trials.jsonl",):
        _write_text(methods_dir / "tpe" / name, "", secret)
    for method in ("score_only_vgmpc", "full_feedback_vgmpc"):
        _write_text(methods_dir / method / "prompt_response_index.jsonl", "", secret)
    for name in ("stall_history.jsonl", "feedback_index.jsonl", "metric_delta_history.jsonl"):
        _write_text(methods_dir / "full_feedback_vgmpc" / name, "", secret)


def _write_method_final_artifacts(methods_dir: Path, records: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]],
                                  tpe_study: optuna.Study, seeds: Dict[str, int], secret: str,
                                  previous_score: Optional[Dict[str, Any]], previous_full: Optional[Dict[str, Any]], stall: int) -> None:
    random_rows = [r for r in records if r["method"] == "random_search"]
    sequence_hash = hashlib.sha256("".join(r["candidate_hash"] or "" for r in random_rows).encode()).hexdigest()
    _write_json(methods_dir / "random_search" / "reproducibility_metadata.json", {"seed": seeds["random_search"], "rng": "python_random_mt19937"}, secret)
    _write_json(methods_dir / "random_search" / "proposal_sequence_hash.json", {"sha256": sequence_hash}, secret)
    _write_json(methods_dir / "random_search" / "final_state.json", states["random_search"], secret)

    trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params} for t in tpe_study.trials]
    for row in trials:
        _append_jsonl(methods_dir / "tpe" / "complete_trials.jsonl", row, secret)
    _write_json(methods_dir / "tpe" / "study_config.json", {"direction": "maximize", "trials": protocol.SLOTS_PER_METHOD, "warm_start": False}, secret)
    _write_json(methods_dir / "tpe" / "sampler_config.json", {"class": "TPESampler", "seed": seeds["tpe"], "n_startup_trials": 2, "multivariate": True, "pruning": False}, secret)
    best = tpe_study.best_trial if tpe_study.trials else None
    _write_json(methods_dir / "tpe" / "optimizer_best_trial.json", {"number": best.number, "objective": best.value, "params": best.params} if best else None, secret)
    tpe_rows = [r for r in records if r["method"] == "tpe"]
    accepted = [r for r in tpe_rows if r["execution_success"] and r["accepted"]]
    best_accepted = max(accepted, key=lambda r: r["selection_score_nonruntime"]) if accepted else None
    _write_json(methods_dir / "tpe" / "benchmark_best_accepted_candidate.json", best_accepted, secret)
    _write_json(methods_dir / "tpe" / "benchmark_final_incumbent.json", states["tpe"], secret)
    _write_json(methods_dir / "tpe" / "study_summary.json", {"trial_count": len(trials), "optimizer_best_trial": best.number if best else None, "benchmark_updates": sum(r["incumbent_updated"] for r in tpe_rows)}, secret)
    _write_json(methods_dir / "tpe" / "deterministic_reconstruction.json", {"seed": seeds["tpe"], "objectives": [r["optimizer_objective"] for r in tpe_rows]}, secret)

    for method in ("score_only_vgmpc", "full_feedback_vgmpc"):
        method_rows = [r for r in records if r["method"] == method]
        _write_json(methods_dir / method / "request_counters.json", {"logical": len(method_rows), "physical": sum(r["physical_request_count"] for r in method_rows), "retries": sum(r["retry_count"] for r in method_rows)}, secret)
        _write_json(methods_dir / method / "token_summary.json", {"prompt_tokens": sum(r["prompt_tokens"] or 0 for r in method_rows), "completion_tokens": sum(r["completion_tokens"] or 0 for r in method_rows), "cached_tokens": sum(r["cached_tokens"] or 0 for r in method_rows)}, secret)
        previous = previous_score if method == "score_only_vgmpc" else previous_full
        _write_json(methods_dir / method / "prior_generation_state.json", {
            "last_method_slot": method_rows[-1]["method_slot_index"] if method_rows else None,
            "state": previous,
        }, secret)
        _write_json(methods_dir / method / "final_state.json", states[method], secret)
    _write_json(methods_dir / "full_feedback_vgmpc" / "feedback_schema.json", {"schema_version": protocol.FULL_FEEDBACK_SCHEMA_VERSION, "metrics": list(protocol.SCORE_METRICS)}, secret)
    _write_json(methods_dir / "full_feedback_vgmpc" / "stall_final.json", {"stall_count": stall}, secret)


def _run_block_impl(api_key: str, output_dir: Path, block_id: int, *,
                    baseline_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                    transport_fn: Callable[..., Dict[str, Any]], root: Path = ROOT,
                    provenance_check: Callable[..., None] = provenance.assert_unchanged) -> Dict[str, Any]:
    if block_id not in range(1, protocol.PLANNED_BLOCKS + 1):
        raise ValueError("formal block_id must be in 1..10")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing formal output: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    directories = {name: output_dir / name for name in ("provenance", "methods", "prompts", "responses", "feedback_objects", "cache")}
    for directory in directories.values():
        directory.mkdir(mode=0o700)
    for method in protocol.METHODS:
        (directories["methods"] / method).mkdir(mode=0o700)
    _write_atomic_json(output_dir / "block_status.json", {
        "status": "running", "block_id": block_id, "completed_slots": 0,
        "expected_slots": protocol.TOTAL_SLOTS_PER_BLOCK, "incomplete": True, "aborted": False,
        "started_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }, api_key)
    heartbeat_path = output_dir / "current_slot.json"
    _heartbeat(heartbeat_path, block_id, "block_initialization", secret=api_key)

    config = _block_config(block_id)
    snapshot = provenance.collect_provenance(root, config)
    provenance_check(root, snapshot, config)
    _write_json(directories["provenance"] / "snapshot.json", snapshot, api_key)
    hashes = {item["relative_to_runner"]: item["sha256"] for item in snapshot["files"]}
    schedule = protocol.build_schedule(block_id)
    seeds = protocol.method_seeds(block_id)
    space = score_only.load_space()
    _write_json(output_dir / "schedule.json", schedule, api_key)
    _write_json(output_dir / "block_manifest.json", {
        **config, "formal_experiment": True, "formal_dry_run": False,
        "validation_only": False, "pilot_only": False, "publication_eligible": False,
        "score_metrics": list(protocol.SCORE_METRICS),
        "runtime_metric": protocol.RUNTIME_METRIC, "runtime_score_contributing": False,
        "parser_version": protocol.PARSER_VERSION, "score_only_prompt_template_version": protocol.SCORE_PROMPT_VERSION,
        "full_feedback_prompt_template_version": protocol.FULL_PROMPT_VERSION,
        "full_feedback_schema_version": protocol.FULL_FEEDBACK_SCHEMA_VERSION,
        "authentication_requests": 1, "logical_llm_generation_limit": LOGICAL_GENERATION_LIMIT,
        "physical_generation_request_limit": PHYSICAL_GENERATION_REQUEST_LIMIT,
        "candidate_simulation_limit": CANDIDATE_SIMULATION_LIMIT,
        "fallback_generations": 0, "no_resume": True, "provenance_snapshot_hash": snapshot["snapshot_hash"],
    }, api_key)
    _initialize_method_artifacts(directories["methods"], seeds, block_id, api_key)
    _write_text(output_dir / "evaluations.jsonl", "", api_key)

    _heartbeat(heartbeat_path, block_id, "baseline_evaluation", secret=api_key)
    baseline_raw = baseline_fn({"mode": "baseline"})
    baseline_eval = protocol.score_result(baseline_raw)
    baseline_configuration = copy.deepcopy(baseline_raw.get("candidate") or baseline_eval["candidate"])
    if not baseline_configuration or not baseline_eval["candidate_hash"]:
        raise protocol.FormalIntegrityError("malformed_common_baseline")
    baseline_state = _state(baseline_eval, baseline_configuration)
    baseline_artifact = {
        "schema_version": protocol.PROTOCOL_VERSION, "method": "common_baseline",
        "candidate_hash": baseline_state["candidate_hash"], "full_configuration": baseline_configuration,
        "configuration_hash": _config_hash(baseline_configuration), "parameters": baseline_state["parameters"],
        "raw_metrics": baseline_eval["metrics"],
        "score_contributing_metrics": {name: baseline_eval["metrics"].get(name) for name in protocol.SCORE_METRICS},
        "avg_controller_time_ms": baseline_eval["metrics"].get(protocol.RUNTIME_METRIC),
        "execution_success": baseline_eval["execution_success"], "result_failed": baseline_eval["result_failed"],
        "accepted": baseline_eval["accepted"], "raw_score_nonruntime": baseline_eval["raw_score_nonruntime"],
        "raw_score_nonruntime_status": baseline_eval["raw_score_nonruntime_status"],
        "violation_penalty": baseline_eval["violation_penalty"],
        "selection_score_nonruntime": baseline_eval["selection_score_nonruntime"],
        "selection_score_nonruntime_status": baseline_eval["selection_score_nonruntime_status"],
        "simulation_duration": baseline_eval["simulation_duration_s"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluator_source_hash": hashes.get("evaluate_cstr_nonruntime_candidate.m"),
        "evaluator_protocol_version": "cstr_nonruntime_matlab_adapter_v1",
        "score_source_hash": snapshot["component_hashes"]["score_source_hash"],
        "acceptance_source_hash": snapshot["component_hashes"]["acceptance_source_hash"],
        "search_space_hash": hashes.get("search_space_cstr_reduced.json"),
        "constructor_hash": hashes.get("construct_cstr_candidate.m"),
        "matlab_version": baseline_raw.get("matlab_version", snapshot["matlab_version"]), "python_version": snapshot["python_version"],
        "workspace_fingerprint": snapshot["workspace_merkle"], "provenance_snapshot_hash": snapshot["snapshot_hash"],
    }
    _write_json(output_dir / "common_baseline.json", baseline_artifact, api_key)
    _write_json(directories["provenance"] / "runtime_environment.json", {
        "matlab_version": baseline_artifact["matlab_version"],
        "python_version": snapshot["python_version"], "optuna_version": snapshot["optuna_version"],
        "requests_version": snapshot["requests_version"], "operating_system": snapshot["operating_system"],
        "architecture": snapshot["architecture"],
    }, api_key)
    states = {method: copy.deepcopy(baseline_state) for method in protocol.METHODS}
    if any(state != baseline_state for state in states.values()):
        raise protocol.FormalIntegrityError("baseline_deepcopy_equality_failed")
    evaluation_cache = cache_module.BlockEvaluationCache(snapshot["snapshot_hash"])
    baseline_payload = cache_module.cache_payload(baseline_eval, baseline_configuration, snapshot["snapshot_hash"])
    evaluation_cache.store(baseline_state["candidate_hash"], baseline_payload, origin_method="common_baseline", origin_slot=0)

    rng = random.Random(seeds["random_search"])
    tpe_study = build_tpe_study(seeds["tpe"])
    previous_score: Optional[Dict[str, Any]] = None
    previous_full: Optional[Dict[str, Any]] = None
    stall = 0
    records: List[Dict[str, Any]] = []
    logical = physical_total = candidate_simulations = 0

    for slot in schedule:
        _heartbeat(heartbeat_path, block_id, "provenance_check", slot=slot,
                   last_completed_global_slot=len(records), secret=api_key)
        provenance_check(root, snapshot, config)
        method = slot["method"]
        before = copy.deepcopy(states[method])
        record = protocol.record_template(slot, snapshot["snapshot_hash"])
        _set_protocol_hashes(record, snapshot)
        record["incumbent_parameters_before"] = copy.deepcopy(before["parameters"])
        record["incumbent_metrics_before"] = copy.deepcopy(before["metrics"])
        record["incumbent_raw_score_nonruntime_before"] = before["raw_score_nonruntime"]
        record["incumbent_violation_penalty_before"] = before["violation_penalty"]
        record["incumbent_selection_score_nonruntime_before"] = before["selection_score_nonruntime"]
        started = proposal_started = time.perf_counter()
        parameters: Optional[Dict[str, Any]] = None
        trial = None
        request_result: Dict[str, Any] = {}
        feedback_object: Optional[Dict[str, Any]] = None
        try:
            _heartbeat(heartbeat_path, block_id, "candidate_proposal", slot=slot,
                       last_completed_global_slot=len(records), secret=api_key)
            if method == "random_search":
                parameters = random_parameters(space, rng)
            elif method == "tpe":
                trial = tpe_study.ask(); parameters = tpe_parameters(trial, space)
            else:
                logical += 1
                record["logical_generation_index"] = slot["method_slot_index"]
                feedback = None
                if method == "score_only_vgmpc":
                    prompt = protocol.build_score_only_prompt(space, before, previous_score)
                else:
                    feedback_object = protocol.build_full_feedback_object(before, previous_full, stall)
                    feedback_text = json.dumps(feedback_object, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    feedback_path = directories["feedback_objects"] / f"{method}_slot_{slot['method_slot_index']:03d}.json"
                    _write_text(feedback_path, feedback_text, api_key)
                    record["feedback_object_hash"] = _hash_text(feedback_text)
                    prompt = protocol.build_full_feedback_prompt(space, feedback_object)
                prompt_path = directories["prompts"] / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _write_text(prompt_path, prompt, api_key); record["prompt_hash"] = _hash_text(prompt)
                _heartbeat(heartbeat_path, block_id, "llm_request", slot=slot,
                           last_completed_global_slot=len(records), secret=api_key)
                request_result = transport_fn(api_key=api_key,
                    messages=[{"role": "system", "content": "Return only the requested JSON object."}, {"role": "user", "content": prompt}],
                    model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT,
                    response_format={"type": "json_object"}, allow_transport_retry=True)
                physical = int(request_result.get("physical_request_count", 0)); physical_total += physical
                if physical > 2 or physical_total > PHYSICAL_GENERATION_REQUEST_LIMIT:
                    raise protocol.FormalIntegrityError("physical_generation_request_limit_exceeded")
                record["physical_request_count"] = physical
                record["retry_count"] = int(request_result.get("retry_count", max(0, physical - 1)))
                record["api_duration"] = request_result.get("request_duration_s")
                usage = request_result.get("token_usage", {}) or {}
                record["prompt_tokens"] = usage.get("prompt_tokens"); record["completion_tokens"] = usage.get("completion_tokens")
                record["cached_tokens"] = usage.get("prompt_cache_hit_tokens"); record["provider_request_id"] = request_result.get("provider_request_id")
                raw = request_result.get("assistant_content") or ""
                response_path = directories["responses"] / f"{method}_slot_{slot['method_slot_index']:03d}.txt"
                _write_text(response_path, raw, api_key); record["response_hash"] = _hash_text(raw)
                parameters, parse_error = score_only.strict_parse(raw, space) if raw else (None, "malformed_json")
                record["proposal_failure_category"] = None if parameters is not None else parse_error
                index_record = {"method_slot_index": slot["method_slot_index"], "prompt_hash": record["prompt_hash"], "response_hash": record["response_hash"], "feedback_object_hash": record["feedback_object_hash"]}
                _append_jsonl(directories["methods"] / method / "prompt_response_index.jsonl", index_record, api_key)
                if parameters is None:
                    if method == "score_only_vgmpc": previous_score = None
                    else: previous_full = _invalid_previous(before, parse_error); stall += 1
            record["proposal_duration"] = time.perf_counter() - proposal_started
            record["proposal_valid"] = parameters is not None
            if parameters is not None:
                candidate_id = protocol.candidate_hash(parameters)
                record["candidate_parameters"] = copy.deepcopy(parameters); record["candidate_hash"] = candidate_id
                _heartbeat(heartbeat_path, block_id, "cache_lookup", slot=slot, parameters=parameters,
                           candidate_hash_value=candidate_id, last_completed_global_slot=len(records), secret=api_key)
                hit = evaluation_cache.lookup(candidate_id, snapshot["snapshot_hash"])
                if hit is not None:
                    origin = evaluation_cache.origin(candidate_id)
                    record["duplicate_detected"] = True; record["cache_hit"] = True
                    record["duplicate_of_method"], record["duplicate_of_method_slot"] = origin
                    record["cached_evaluation_hash"] = hit["deterministic_evaluation_hash"]
                    record["candidate_simulation_performed"] = False; record["runtime_remeasured"] = False
                    record["runtime_measurement_origin"] = "not_remeasured_cache_hit"
                    evaluated = _cache_hit_evaluation(hit); configuration = copy.deepcopy(hit["complete_candidate_configuration"])
                else:
                    record["candidate_simulation_performed"] = True; record["runtime_remeasured"] = True
                    record["runtime_measurement_origin"] = "candidate_simulation"
                    candidate_simulations += 1
                    _heartbeat(heartbeat_path, block_id, "candidate_evaluation", slot=slot, parameters=parameters,
                               candidate_hash_value=candidate_id, last_completed_global_slot=len(records), secret=api_key)
                    raw_result = evaluate_fn({"mode": "candidate", "parameters": parameters,
                                              "incumbent_selection_score_before": before["selection_score_nonruntime"],
                                              "incumbent_candidate_id": before["candidate_hash"]})
                    _heartbeat(heartbeat_path, block_id, "candidate_evaluation_returned", slot=slot, parameters=parameters,
                               candidate_hash_value=candidate_id, last_completed_global_slot=len(records), secret=api_key)
                    evaluated = protocol.score_result(raw_result)
                    configuration = copy.deepcopy(raw_result.get("candidate") or evaluated.get("candidate") or parameters)
                    evaluated["candidate_hash"] = raw_result.get("candidate_id") or candidate_id
                    if evaluated["candidate_hash"] != candidate_id:
                        raise protocol.FormalIntegrityError("candidate_hash_mismatch")
                    payload = cache_module.cache_payload(evaluated, configuration, snapshot["snapshot_hash"])
                    record["cached_evaluation_hash"] = evaluation_cache.store(candidate_id, payload, origin_method=method, origin_slot=slot["method_slot_index"])
                record["complete_candidate_configuration"] = configuration
                record["execution_success"] = evaluated["execution_success"]; record["result_failed"] = evaluated["result_failed"]
                record["simulation_failure"] = not evaluated["execution_success"]; record["metrics"] = copy.deepcopy(evaluated["metrics"])
                record["raw_score_nonruntime"] = evaluated["raw_score_nonruntime"]; record["violation_penalty"] = evaluated["violation_penalty"]
                record["selection_score_nonruntime"] = evaluated["selection_score_nonruntime"]
                record["raw_score_nonruntime_status"] = evaluated["raw_score_nonruntime_status"]
                record["selection_score_nonruntime_status"] = evaluated["selection_score_nonruntime_status"]
                record["accepted"] = evaluated["accepted"]; record["simulation_duration"] = evaluated.get("simulation_duration_s")
                normalized, objective = protocol.tpe_objective(evaluated)
                record["normalized_optimizer_score"] = normalized; record["optimizer_objective"] = objective
                updated = protocol.incumbent_updated(before, evaluated)
                record["incumbent_updated"] = updated
                record["no_update_reason"] = "" if updated else ("execution_failure" if not evaluated["execution_success"] else "rejected" if not evaluated["accepted"] else "not_higher_than_incumbent")
                after = _state(evaluated, configuration) if updated else copy.deepcopy(before)
                states[method] = after
                if method == "tpe": tpe_study.tell(trial, objective)
                previous = {"candidate_available": True, "parameters": copy.deepcopy(parameters),
                            "raw_score_nonruntime": evaluated["raw_score_nonruntime"], "violation_penalty": evaluated["violation_penalty"],
                            "selection_score_nonruntime": evaluated["selection_score_nonruntime"], "accepted": evaluated["accepted"],
                            "execution_success": evaluated["execution_success"], "incumbent_updated": updated,
                            "validation_category": "incumbent_updated" if updated else "insufficient_score" if evaluated["accepted"] else "execution_failure" if not evaluated["execution_success"] else "acceptance_failure",
                            "simulation_failure": not evaluated["execution_success"],
                            "metrics": {name: evaluated["metrics"].get(name) for name in protocol.SCORE_METRICS},
                            "reference_incumbent": before}
                if method == "score_only_vgmpc": previous_score = previous
                elif method == "full_feedback_vgmpc": previous_full = previous; stall = 0 if updated else stall + 1
            else:
                normalized, objective = protocol.tpe_objective({})
                record["normalized_optimizer_score"] = normalized; record["optimizer_objective"] = objective
                record["no_update_reason"] = record["proposal_failure_category"] or "invalid_proposal"
                if method == "tpe" and trial is not None: tpe_study.tell(trial, objective)
        except protocol.FormalIntegrityError:
            raise
        except Exception:
            record["proposal_valid"] = parameters is not None
            record["proposal_failure_category"] = record["proposal_failure_category"] or "runtime_or_transport_failure"
            record["execution_success"] = False; record["result_failed"] = True; record["simulation_failure"] = parameters is not None
            record["raw_score_nonruntime"] = float("-inf"); record["violation_penalty"] = 100.0; record["selection_score_nonruntime"] = float("-inf")
            record["raw_score_nonruntime_status"] = "negative_infinity_failure"; record["selection_score_nonruntime_status"] = "negative_infinity_failure"
            record["accepted"] = False; record["optimizer_objective"] = 0.0; record["no_update_reason"] = "execution_failure"
            if method == "tpe" and trial is not None: tpe_study.tell(trial, 0.0)
            if method == "score_only_vgmpc": previous_score = None
            elif method == "full_feedback_vgmpc": previous_full = _invalid_previous(before, "execution_failure"); stall += 1

        after = states[method]
        record["incumbent_parameters_after"] = copy.deepcopy(after["parameters"])
        record["incumbent_metrics_after"] = copy.deepcopy(after["metrics"])
        record["incumbent_raw_score_nonruntime_after"] = after["raw_score_nonruntime"]
        record["incumbent_violation_penalty_after"] = after["violation_penalty"]
        record["incumbent_selection_score_nonruntime_after"] = after["selection_score_nonruntime"]
        record["total_slot_duration"] = time.perf_counter() - started
        protocol.validate_record(record)
        _heartbeat(heartbeat_path, block_id, "record_persistence", slot=slot, parameters=parameters,
                   candidate_hash_value=record["candidate_hash"], last_completed_global_slot=len(records), secret=api_key)
        records.append(record); _append_jsonl(output_dir / "evaluations.jsonl", record, api_key)
        _append_jsonl(directories["methods"] / method / "incumbent_history.jsonl", {
            "method_slot_index": slot["method_slot_index"], "candidate_hash": record["candidate_hash"],
            "incumbent_updated": record["incumbent_updated"], "incumbent_selection_score_after": record["incumbent_selection_score_nonruntime_after"]}, api_key)
        if method == "random_search":
            _append_jsonl(directories["methods"] / method / "proposal_sequence.jsonl", {"method_slot_index": slot["method_slot_index"], "parameters": parameters, "candidate_hash": record["candidate_hash"]}, api_key)
        if method == "full_feedback_vgmpc":
            _append_jsonl(directories["methods"] / method / "stall_history.jsonl", {"method_slot_index": slot["method_slot_index"], "stall_count_after": stall}, api_key)
            _append_jsonl(directories["methods"] / method / "feedback_index.jsonl", {"method_slot_index": slot["method_slot_index"], "feedback_object_hash": record["feedback_object_hash"]}, api_key)
            _append_jsonl(directories["methods"] / method / "metric_delta_history.jsonl", {
                "method_slot_index": slot["method_slot_index"],
                "metric_deltas": feedback_object["comparison"]["metric_deltas"] if feedback_object else None,
                "improved_metrics": feedback_object["comparison"]["improved_metrics"] if feedback_object else [],
                "degraded_metrics": feedback_object["comparison"]["degraded_metrics"] if feedback_object else [],
                "unchanged_metrics": feedback_object["comparison"]["unchanged_metrics"] if feedback_object else [],
            }, api_key)
        _heartbeat(heartbeat_path, block_id, "slot_complete", slot=slot, parameters=parameters,
                   candidate_hash_value=record["candidate_hash"], last_completed_global_slot=len(records), secret=api_key)

    provenance_check(root, snapshot, config)
    if (len(records) != protocol.TOTAL_SLOTS_PER_BLOCK
            or logical != LOGICAL_GENERATION_LIMIT
            or candidate_simulations > CANDIDATE_SIMULATION_LIMIT):
        raise protocol.FormalIntegrityError("block_budget_invariant_failed")
    _write_method_final_artifacts(directories["methods"], records, states, tpe_study, seeds, api_key,
                                  previous_score, previous_full, stall)
    _write_json(directories["cache"] / "cache_index.json", evaluation_cache.audit_index(), api_key)
    summary = {
        "experiment_name": protocol.EXPERIMENT_NAME, "block_id": block_id,
        "formal_experiment": True, "formal_dry_run": False,
        "validation_only": False, "pilot_only": False, "publication_eligible": False,
        "scheduled_slots": len(records), "slots_per_method": {m: sum(r["method"] == m for r in records) for m in protocol.METHODS},
        "logical_llm_generations": logical, "physical_generation_requests": physical_total,
        "candidate_simulations": candidate_simulations, "cache_hits": sum(r["cache_hit"] for r in records),
        "fallback_generations": 0, "baseline_simulations": 1,
        "method_seeds": seeds, "provenance_snapshot_hash": snapshot["snapshot_hash"],
    }
    _write_json(output_dir / "block_summary.json", summary, api_key)
    scan = score_only.credential_scan(output_dir, api_key)
    validation = {
        "status": "complete", "schedule_valid": True, "records_valid": True,
        "method_slot_counts": summary["slots_per_method"], "baseline_simulations": 1,
        "baseline_copies_equal": True, "runtime_score_contributing": False,
        "runtime_only_incumbent_updates": 0, "fallback_used": False,
        "slot_overflow": False, "method_directories_nonempty": all(any((directories["methods"] / m).iterdir()) for m in protocol.METHODS),
        "credential_scan_passed": scan["credential_scan_passed"], "credential_matches": scan["credential_matches"],
    }
    _write_json(output_dir / "validation_report.json", validation, api_key)
    if not score_only.credential_scan(output_dir, api_key)["credential_scan_passed"]:
        raise protocol.FormalIntegrityError("credential_scan_failed")
    _heartbeat(heartbeat_path, block_id, "block_complete", last_completed_global_slot=len(records), secret=api_key)
    _write_atomic_json(output_dir / "block_status.json", {"status": "complete", "block_id": block_id, "completed_slots": protocol.TOTAL_SLOTS_PER_BLOCK, "expected_slots": protocol.TOTAL_SLOTS_PER_BLOCK, "incomplete": False, "aborted": False}, api_key)
    return summary


def run_formal_block(api_key: str, output_dir: Path, block_id: int, *,
                         baseline_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                         evaluate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                         transport_fn: Callable[..., Dict[str, Any]] = transport.request_chat_completion,
                         root: Path = ROOT, provenance_check: Callable[..., None] = provenance.assert_unchanged) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    try:
        return _run_block_impl(api_key, output_dir, block_id, baseline_fn=baseline_fn,
                               evaluate_fn=evaluate_fn, transport_fn=transport_fn,
                               root=root, provenance_check=provenance_check)
    except BaseException as exc:
        status = output_dir / "block_status.json"
        if status.exists():
            completed = 0
            evaluations = output_dir / "evaluations.jsonl"
            try:
                if evaluations.exists(): completed = sum(bool(line.strip()) for line in evaluations.read_text().splitlines())
                heartbeat_path = output_dir / "current_slot.json"
                heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8")) if heartbeat_path.exists() else {}
                diagnostic = copy.deepcopy(getattr(exc, "diagnostics", {}))
                report = {
                    "status": "incomplete/aborted", "attempt_id": output_dir.name,
                    "block_id": block_id, "completed_slots": completed,
                    "last_completed_global_slot": completed,
                    "current_global_slot": heartbeat.get("current_global_slot"),
                    "current_round": heartbeat.get("current_round"),
                    "current_method": heartbeat.get("current_method"),
                    "current_method_slot": heartbeat.get("current_method_slot"),
                    "active_stage": heartbeat.get("stage"),
                    "candidate_parameters": heartbeat.get("candidate_parameters"),
                    "candidate_hash": heartbeat.get("candidate_hash"),
                    "abort_reason": getattr(exc, "reason", type(exc).__name__),
                    "exception_type": type(exc).__name__, "exception_message": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "block_started_timestamp": json.loads(status.read_text(encoding="utf-8")).get("started_timestamp"),
                    "timeout_stage": diagnostic.get("timeout_stage"),
                    "matlab_exit_code": diagnostic.get("child_returncode"),
                    "matlab_child_pid": diagnostic.get("child_pid"),
                    "matlab_process_group": diagnostic.get("child_process_group"),
                    "termination_actions": diagnostic.get("termination_actions", []),
                    "stdout_tail": diagnostic.get("stdout_tail", ""),
                    "stderr_tail": diagnostic.get("stderr_tail", ""),
                    "evaluations_last_modified": datetime.fromtimestamp(evaluations.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z") if evaluations.exists() else None,
                    "python_traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    "resume_allowed": False,
                }
                _write_atomic_json(output_dir / "abort_report.json", report, api_key)
                _write_atomic_json(status, {"status": "incomplete/aborted", "block_id": block_id, "completed_slots": completed, "expected_slots": protocol.TOTAL_SLOTS_PER_BLOCK, "incomplete": True, "aborted": True}, api_key)
            except Exception:
                pass
        raise
