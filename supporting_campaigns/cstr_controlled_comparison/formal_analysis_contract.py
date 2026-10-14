#!/usr/bin/env python3
"""Read-only audit, endpoint reconstruction, and frozen formal statistics."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import formal_protocol as protocol


ANALYSIS_CONTRACT_VERSION = "cstr_formal_analysis_contract_v1"
ANALYSIS_SEED = 20261001
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05
PAIR_ORDER = tuple(itertools.combinations(protocol.METHODS, 2))


class FormalAnalysisError(RuntimeError):
    """A formal block set is not eligible for comparative analysis."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FormalAnalysisError(f"unreadable_json:{path}") from exc
    if not isinstance(value, dict):
        raise FormalAnalysisError(f"json_object_required:{path}")
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise FormalAnalysisError(f"unreadable_jsonl:{path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise FormalAnalysisError(f"jsonl_object_required:{path}")
    return rows


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _require(condition: bool, reason: str, decisions: List[Dict[str, Any]]) -> None:
    decisions.append({"check": reason, "passed": bool(condition)})
    if not condition:
        raise FormalAnalysisError(reason)


def _required_artifacts(block: Path) -> Tuple[str, ...]:
    return (
        "block_manifest.json", "common_baseline.json", "schedule.json", "evaluations.jsonl", "current_slot.json",
        "block_summary.json", "validation_report.json", "block_status.json", "provenance/snapshot.json",
        "methods/random_search/final_state.json", "methods/random_search/proposal_sequence.jsonl",
        "methods/tpe/complete_trials.jsonl", "methods/tpe/optimizer_best_trial.json",
        "methods/tpe/benchmark_best_accepted_candidate.json", "methods/tpe/benchmark_final_incumbent.json",
        "methods/score_only_vgmpc/final_state.json", "methods/score_only_vgmpc/request_counters.json",
        "methods/full_feedback_vgmpc/final_state.json", "methods/full_feedback_vgmpc/request_counters.json",
        "methods/full_feedback_vgmpc/feedback_schema.json", "cache/cache_index.json",
    )


def _audit_records(block_id: int, baseline_score: float, rows: List[Dict[str, Any]],
                   schedule: List[Dict[str, Any]], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    _require(len(rows) == protocol.TOTAL_SLOTS_PER_BLOCK, "record_count_320", decisions)
    counts = Counter(row.get("method") for row in rows)
    _require(counts == Counter({method: protocol.SLOTS_PER_METHOD for method in protocol.METHODS}), "records_80_per_method", decisions)
    expected = protocol.build_schedule(block_id)
    schedule_keys = ("block_id", "round_index", "order_cycle", "order_position", "global_slot_index", "method", "method_slot_index")
    _require(len(schedule) == len(expected) and all(all(a.get(k) == b.get(k) for k in schedule_keys) for a, b in zip(schedule, expected)), "schedule_matches_frozen_rotation", decisions)
    _require(all(all(row.get(k) == expected[index].get(k) for k in schedule_keys) for index, row in enumerate(rows)), "records_follow_schedule", decisions)

    hashes = ("evaluator_hash", "score_hash", "acceptance_hash", "search_space_hash", "provenance_snapshot_hash")
    _require(all(row.get(name) for row in rows for name in hashes), "record_hashes_complete", decisions)
    _require(all(row.get("formal_experiment") is True and row.get("formal_dry_run") is False
                 and row.get("validation_only") is False and row.get("pilot_only") is False
                 and row.get("publication_eligible") is False for row in rows), "formal_collection_flags", decisions)
    _require(all(row.get("formal_protocol_version") == protocol.FORMAL_PROTOCOL_VERSION
                 and row.get("score_protocol_version") == protocol.SCORE_PROTOCOL_VERSION
                 and row.get("acceptance_protocol_version") == protocol.ACCEPTANCE_PROTOCOL_VERSION
                 and row.get("duplicate_policy_version") == protocol.DUPLICATE_POLICY_VERSION
                 and row.get("tpe_objective_version") == protocol.TPE_OBJECTIVE_VERSION
                 and row.get("planned_block_count") == protocol.PLANNED_BLOCKS
                 and row.get("scheduled_slots_per_method") == protocol.SLOTS_PER_METHOD for row in rows), "record_protocol_and_budget", decisions)

    for row in rows:
        metrics = row.get("metrics")
        if row.get("proposal_valid") and metrics is not None:
            if row.get("execution_success") and not row.get("result_failed"):
                raw = protocol.minimal_raw_score(metrics)
                _require(_same_number(row.get("raw_score_nonruntime"), raw), "score_recomputation", decisions)
                _require(_same_number(row.get("selection_score_nonruntime"), raw - float(row.get("violation_penalty"))), "selection_score_recomputation", decisions)
                expected_accepted = protocol.authoritative_acceptance({"execution_success": True, "result_failed": False, "metrics": metrics})
                _require(row.get("accepted") is expected_accepted, "acceptance_recomputation", decisions)
            else:
                _require(row.get("raw_score_nonruntime") is None and row.get("violation_penalty") == 100.0
                         and row.get("selection_score_nonruntime") is None and row.get("accepted") is False, "failure_semantics", decisions)
        before = row.get("incumbent_selection_score_nonruntime_before")
        score = row.get("selection_score_nonruntime")
        expected_update = bool(row.get("execution_success") and row.get("accepted") and score is not None and score > before)
        _require(row.get("incumbent_updated") is expected_update, "strict_incumbent_rule", decisions)
        expected_after = score if expected_update else before
        _require(_same_number(row.get("incumbent_selection_score_nonruntime_after"), expected_after), "incumbent_state_transition", decisions)
        if row.get("cache_hit"):
            _require(row.get("duplicate_detected") is True and row.get("candidate_simulation_performed") is False
                     and row.get("runtime_remeasured") is False and row.get("runtime_measurement_origin") == "not_remeasured_cache_hit"
                     and isinstance(metrics, dict) and metrics.get(protocol.RUNTIME_METRIC) is None, "cache_runtime_contract", decisions)
        if row.get("method") == "full_feedback_vgmpc":
            _require(not isinstance(metrics, dict) or protocol.RUNTIME_METRIC in metrics, "runtime_recorded_or_null", decisions)

    endpoints: Dict[str, Any] = {}
    for method in protocol.METHODS:
        method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["method_slot_index"])
        final_score = float(method_rows[-1]["incumbent_selection_score_nonruntime_after"])
        improvements = [float(row["incumbent_selection_score_nonruntime_after"]) - baseline_score for row in method_rows]
        absolute = final_score - baseline_score
        relative = absolute / abs(baseline_score) if baseline_score != 0 else None
        first = next((index for index, value in enumerate(improvements, 1) if value > 0), None)
        endpoints[method] = {
            "baseline_selection_score_nonruntime": baseline_score,
            "final_incumbent_selection_score_nonruntime": final_score,
            "absolute_improvement": absolute,
            "relative_improvement": relative,
            "normalized_best_so_far_auc": sum(improvements) / protocol.SLOTS_PER_METHOD,
            "slot_to_first_baseline_improvement": first,
            "accepted_rate": sum(bool(row["accepted"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "incumbent_update_count": sum(bool(row["incumbent_updated"]) for row in method_rows),
            "unique_proposal_rate": sum(bool(row["proposal_valid"]) and not bool(row["duplicate_detected"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "duplicate_rate": sum(bool(row["duplicate_detected"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "cache_hit_rate": sum(bool(row["cache_hit"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "invalid_output_rate": sum(not bool(row["proposal_valid"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "execution_failure_rate": sum(bool(row["proposal_valid"]) and not bool(row["execution_success"]) for row in method_rows) / protocol.SLOTS_PER_METHOD,
            "candidate_simulation_count": sum(bool(row["candidate_simulation_performed"]) for row in method_rows),
            "physical_api_request_count": sum(int(row.get("physical_request_count") or 0) for row in method_rows),
            "retry_count": sum(int(row.get("retry_count") or 0) for row in method_rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in method_rows),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in method_rows),
            "cached_tokens": sum(int(row.get("cached_tokens") or 0) for row in method_rows),
        }
    return endpoints


def audit_formal_block(block_dir: Path) -> Dict[str, Any]:
    """Audit one completed block without writing into it."""
    block = Path(block_dir).resolve()
    decisions: List[Dict[str, Any]] = []
    _require(block.is_dir(), "block_directory_exists", decisions)
    _require("cstr_four_method_formal_block" in block.name, "formal_directory_name", decisions)
    _require(all((block / relative).is_file() for relative in _required_artifacts(block)), "required_artifacts_complete", decisions)
    _require(not (block / "abort_report.json").exists(), "abort_report_absent", decisions)
    manifest = _read_json(block / "block_manifest.json")
    status = _read_json(block / "block_status.json")
    summary = _read_json(block / "block_summary.json")
    validation = _read_json(block / "validation_report.json")
    baseline = _read_json(block / "common_baseline.json")
    snapshot = _read_json(block / "provenance/snapshot.json")
    block_id = manifest.get("block_id")
    _require(block_id in range(1, protocol.PLANNED_BLOCKS + 1), "planned_block_id_1_to_10", decisions)
    _require(status.get("status") == "complete" and status.get("completed_slots") == 320
             and status.get("incomplete") is False and status.get("aborted") is False, "block_complete", decisions)
    _require(manifest.get("formal_experiment") is True and manifest.get("formal_dry_run") is False
             and manifest.get("validation_only") is False and manifest.get("pilot_only") is False
             and manifest.get("publication_eligible") is False, "manifest_formal_flags", decisions)
    _require(manifest.get("formal_protocol_version") == protocol.FORMAL_PROTOCOL_VERSION
             and manifest.get("execution_safety_version") == protocol.EXECUTION_SAFETY_VERSION
             and manifest.get("slots_per_method") == 80 and manifest.get("total_slots") == 320
             and manifest.get("planned_block_count") == 10, "manifest_frozen_protocol", decisions)
    _require(summary.get("scheduled_slots") == 320 and summary.get("logical_llm_generations") == 160
             and summary.get("baseline_simulations") == 1 and summary.get("physical_generation_requests", 321) <= 320, "summary_budgets", decisions)
    _require(validation.get("status") == "complete" and validation.get("credential_scan_passed") is True
             and validation.get("baseline_simulations") == 1 and validation.get("runtime_score_contributing") is False
             and validation.get("fallback_used") is False and validation.get("prior_outputs_unchanged") is True,
             "validation_credential_and_history_pass", decisions)
    _require(snapshot.get("snapshot_hash") == manifest.get("provenance_snapshot_hash") == summary.get("provenance_snapshot_hash"), "provenance_snapshot_consistent", decisions)
    heartbeat = _read_json(block / "current_slot.json")
    _require(heartbeat.get("stage") == "block_complete" and heartbeat.get("last_completed_global_slot") == 320,
             "completed_heartbeat", decisions)
    _require(all(item.get("sha256") and item.get("size") is not None for item in snapshot.get("files", [])), "provenance_files_complete", decisions)
    baseline_score = baseline.get("selection_score_nonruntime")
    _require(isinstance(baseline_score, (int, float)) and not isinstance(baseline_score, bool) and math.isfinite(float(baseline_score)), "finite_common_baseline", decisions)
    rows = _read_jsonl(block / "evaluations.jsonl")
    schedule = json.loads((block / "schedule.json").read_text(encoding="utf-8"))
    _require(isinstance(schedule, list), "schedule_is_list", decisions)
    endpoints = _audit_records(int(block_id), float(baseline_score), rows, schedule, decisions)
    _require(len(_read_jsonl(block / "methods/tpe/complete_trials.jsonl")) == 80, "tpe_trials_80", decisions)
    _require(len(_read_jsonl(block / "methods/random_search/proposal_sequence.jsonl")) == 80, "random_proposals_80", decisions)
    for method in ("score_only_vgmpc", "full_feedback_vgmpc"):
        counters = _read_json(block / f"methods/{method}/request_counters.json")
        _require(counters.get("logical") == 80 and counters.get("physical", 161) <= 160, f"{method}_generation_budget", decisions)
        prompts = sorted((block / "prompts").glob(f"{method}_slot_*.txt"))
        responses = sorted((block / "responses").glob(f"{method}_slot_*.txt"))
        _require(len(prompts) == 80 and len(responses) == 80, f"{method}_prompt_response_artifacts_80", decisions)
        forbidden_methods = [name for name in protocol.METHODS if name != method]
        _require(all(protocol.RUNTIME_METRIC not in path.read_text(encoding="utf-8")
                     and not any(name in path.read_text(encoding="utf-8") for name in forbidden_methods)
                     for path in prompts), f"{method}_prompt_runtime_and_method_isolation", decisions)
    feedback = sorted((block / "feedback_objects").glob("full_feedback_vgmpc_slot_*.json"))
    _require(len(feedback) == 80 and all(protocol.RUNTIME_METRIC not in path.read_text(encoding="utf-8") for path in feedback),
             "full_feedback_objects_80_runtime_excluded", decisions)
    return {
        "audit_contract_version": ANALYSIS_CONTRACT_VERSION,
        "block_id": block_id, "block_directory": str(block), "passed": True,
        "decisions": decisions, "endpoints": endpoints,
        "protocol_fingerprint": {
            "formal_protocol_version": manifest["formal_protocol_version"],
            "execution_safety_version": manifest["execution_safety_version"],
            "score_protocol_version": manifest["score_protocol_version"],
            "acceptance_protocol_version": manifest["acceptance_protocol_version"],
            "duplicate_policy_version": manifest["duplicate_policy_version"],
            "tpe_objective_version": manifest["tpe_objective_version"],
            "slots_per_method": manifest["slots_per_method"],
            "source_hashes": {item["relative_to_runner"]: item["sha256"] for item in snapshot["files"]},
            "component_hashes": snapshot["component_hashes"],
        },
    }


def validate_formal_blocks(block_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Require exactly the ten audited formal blocks and reject all drift."""
    if len(block_dirs) != protocol.PLANNED_BLOCKS:
        raise FormalAnalysisError("exactly_ten_block_directories_required")
    audits = [audit_formal_block(path) for path in block_dirs]
    ids = [audit["block_id"] for audit in audits]
    if sorted(ids) != list(range(1, protocol.PLANNED_BLOCKS + 1)) or len(set(ids)) != 10:
        raise FormalAnalysisError("block_ids_must_be_1_to_10_exactly_once")
    reference = audits[0]["protocol_fingerprint"]
    if any(audit["protocol_fingerprint"] != reference for audit in audits[1:]):
        raise FormalAnalysisError("formal_protocol_or_hash_drift")
    ordered = sorted(audits, key=lambda item: item["block_id"])
    table = [{"block_id": audit["block_id"], **{method: audit["endpoints"][method] for method in protocol.METHODS}} for audit in ordered]
    return {"analysis_contract_version": ANALYSIS_CONTRACT_VERSION, "validated": True,
            "block_ids": list(range(1, 11)), "audits": ordered, "endpoint_table": table,
            "decision_log": [{"decision": "include", "block_id": audit["block_id"], "reason": "completed_audited_formal_block"} for audit in ordered],
            "imputation": "none", "failed_or_zero_blocks_dropped": False}


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = ((index + 1) + end) / 2.0
        for target in order[index:end]: ranks[target] = average
        index = end
    return ranks


def friedman_test(matrix: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if len(matrix) != 10 or any(len(row) != 4 for row in matrix):
        raise FormalAnalysisError("friedman_input_must_be_10_by_4")
    ranks = [_rank(row) for row in matrix]
    sums = [sum(row[column] for row in ranks) for column in range(4)]
    n, k = 10, 4
    statistic = 12.0 / (n * k * (k + 1)) * sum(value * value for value in sums) - 3 * n * (k + 1)
    tie_term = sum(sum(count ** 3 - count for count in Counter(row).values() if count > 1) for row in matrix)
    correction = 1.0 - tie_term / (n * (k ** 3 - k))
    statistic = statistic / correction if correction > 0 else 0.0
    # Chi-square(3) survival function, matching the standard asymptotic Friedman test.
    root = math.sqrt(max(0.0, statistic) / 2.0)
    p_value = math.erfc(root) + 2.0 * root * math.exp(-root * root) / math.sqrt(math.pi)
    return {"statistic": statistic, "p_value": min(1.0, p_value), "degrees_of_freedom": 3,
            "alpha": ALPHA, "rejected": p_value < ALPHA, "calculation_mode": "chi_square_asymptotic"}


def wilcoxon_pratt(left: Sequence[float], right: Sequence[float]) -> Dict[str, Any]:
    if len(left) != len(right) or not left:
        raise FormalAnalysisError("paired_samples_required")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    zero_count = sum(value == 0 for value in differences)
    if zero_count == len(differences):
        return {"raw_p_value": 1.0, "effect_size": 0.0, "median_difference": 0.0,
                "zero_difference_count": zero_count, "calculation_mode": "all_zero_defined"}
    ranks = _rank([abs(value) for value in differences])
    positive = sum(rank for rank, value in zip(ranks, differences) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, differences) if value < 0)
    nonzero_rank_sum = positive + negative
    effect = (positive - negative) / nonzero_rank_sum
    absolute_nonzero = [abs(value) for value in differences if value != 0]
    no_ties = len(set(absolute_nonzero)) == len(absolute_nonzero)
    if zero_count == 0 and no_ties and len(differences) <= 20:
        observed = min(positive, negative)
        integer_ranks = list(range(1, len(differences) + 1))
        extreme = 0
        for signs in itertools.product((0, 1), repeat=len(integer_ranks)):
            plus = sum(rank for rank, sign in zip(integer_ranks, signs) if sign)
            if min(plus, sum(integer_ranks) - plus) <= observed + 1e-12: extreme += 1
        p_value = extreme / (2 ** len(integer_ranks)); mode = "exact_signed_rank_permutation"
    else:
        nonzero_ranks = [rank for rank, value in zip(ranks, differences) if value != 0]
        mean = sum(nonzero_ranks) / 2.0
        variance = sum(rank * rank for rank in nonzero_ranks) / 4.0
        statistic = positive
        correction = 0.5 if statistic > mean else -0.5 if statistic < mean else 0.0
        z = (statistic - mean - correction) / math.sqrt(variance) if variance else 0.0
        p_value = math.erfc(abs(z) / math.sqrt(2.0)); mode = "deterministic_normal_approximation_pratt"
    return {"raw_p_value": min(1.0, p_value), "effect_size": effect,
            "median_difference": statistics.median(differences), "zero_difference_count": zero_count,
            "calculation_mode": mode}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values); running = 0.0; size = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (size - rank) * float(value))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_bootstrap_median_difference(left: Sequence[float], right: Sequence[float], *,
                                       seed: int = ANALYSIS_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Any]:
    if len(left) != len(right) or not left or resamples != BOOTSTRAP_RESAMPLES:
        raise FormalAnalysisError("paired_bootstrap_requires_equal_samples_and_10000_resamples")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = sorted(statistics.median(differences[rng.randrange(len(differences))] for _ in differences) for _ in range(resamples))
    low = estimates[int(0.025 * resamples)]
    high = estimates[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return {"estimate": statistics.median(differences), "confidence_level": 0.95,
            "lower": low, "upper": high, "seed": seed, "resamples": resamples,
            "method": "paired_percentile_bootstrap_of_median_difference", "limitation": "n=10 paired blocks"}


def run_frozen_analysis(validated: Mapping[str, Any]) -> Dict[str, Any]:
    if validated.get("validated") is not True or validated.get("block_ids") != list(range(1, 11)):
        raise FormalAnalysisError("ten_validated_blocks_required_before_inference")
    table = validated["endpoint_table"]
    values = {method: [float(row[method]["absolute_improvement"]) for row in table] for method in protocol.METHODS}
    matrix = [[values[method][index] for method in protocol.METHODS] for index in range(10)]
    omnibus = friedman_test(matrix)
    posthoc: List[Dict[str, Any]] = []
    if omnibus["rejected"]:
        for index, (left, right) in enumerate(PAIR_ORDER):
            result = wilcoxon_pratt(values[left], values[right])
            result.update({"left": left, "right": right,
                           "bootstrap_95_percentile_ci": paired_bootstrap_median_difference(values[left], values[right], seed=ANALYSIS_SEED + index)})
            posthoc.append(result)
        adjusted = holm_adjust([row["raw_p_value"] for row in posthoc])
        for row, value in zip(posthoc, adjusted): row["holm_adjusted_p_value"] = value
    descriptive = {method: {"median": statistics.median(series),
                            "iqr": _percentile(series, 0.75) - _percentile(series, 0.25), "values": series}
                   for method, series in values.items()}
    return {"analysis_contract_version": ANALYSIS_CONTRACT_VERSION, "methods": list(protocol.METHODS),
            "primary_endpoint": "final_absolute_nonruntime_score_improvement_after_80_slots",
            "friedman": omnibus, "posthoc_condition": "only_if_friedman_rejects", "posthoc": posthoc,
            "method_descriptives": descriptive, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "analysis_seed": ANALYSIS_SEED, "zero_blocks_dropped": False, "imputation": "none"}


def _percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper: return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
