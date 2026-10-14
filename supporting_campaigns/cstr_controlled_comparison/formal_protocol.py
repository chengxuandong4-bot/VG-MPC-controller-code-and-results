#!/usr/bin/env python3
"""Frozen protocol primitives for the formal CSTR experiment."""
from __future__ import annotations

from typing import Any, Dict, List

import intermediate_protocol as shared


FORMAL_PROTOCOL_VERSION = "cstr_four_method_formal_experiment_v1"
EXECUTION_SAFETY_VERSION = "cstr_formal_process_lifecycle_v1"
PROTOCOL_VERSION = FORMAL_PROTOCOL_VERSION
EXPERIMENT_NAME = "cstr_four_method_formal_experiment"
SCORE_PROTOCOL_VERSION = shared.SCORE_PROTOCOL_VERSION
ACCEPTANCE_PROTOCOL_VERSION = shared.ACCEPTANCE_PROTOCOL_VERSION
DUPLICATE_POLICY_VERSION = shared.DUPLICATE_POLICY_VERSION
TPE_OBJECTIVE_VERSION = shared.TPE_OBJECTIVE_VERSION
SCORE_PROMPT_VERSION = shared.SCORE_PROMPT_VERSION
FULL_PROMPT_VERSION = shared.FULL_PROMPT_VERSION
FULL_FEEDBACK_SCHEMA_VERSION = shared.FULL_FEEDBACK_SCHEMA_VERSION
PARSER_VERSION = shared.PARSER_VERSION

FORMAL_EXPERIMENT = True
FORMAL_DRY_RUN = False
VALIDATION_ONLY = False
PILOT_ONLY = False
PUBLICATION_ELIGIBLE = False
PLANNED_BLOCKS = 10
METHODS = shared.METHODS
SLOTS_PER_METHOD = 80
ROUNDS_PER_BLOCK = 80
TOTAL_SLOTS_PER_BLOCK = 320
TOTAL_PLANNED_SLOTS = 3200
FORMAL_MASTER_SEED = 20261001
METHOD_SEED_OFFSETS = shared.METHOD_SEED_OFFSETS
SCORE_METRICS = shared.SCORE_METRICS
RUNTIME_METRIC = shared.RUNTIME_METRIC
ALL_METRICS = shared.ALL_METRICS

FormalIntegrityError = shared.IntermediateIntegrityError
IntermediateIntegrityError = FormalIntegrityError
minimal_raw_score = shared.minimal_raw_score
authoritative_acceptance = shared.authoritative_acceptance
incumbent_updated = shared.incumbent_updated
tpe_objective = shared.tpe_objective
candidate_hash = shared.candidate_hash
build_score_only_prompt = shared.build_score_only_prompt
build_full_feedback_object = shared.build_full_feedback_object
build_full_feedback_prompt = shared.build_full_feedback_prompt
secure_json_write = shared.secure_json_write
_safe = shared._safe


class FormalExecutionError(FormalIntegrityError):
    """An infrastructure failure that must abort, rather than consume, a slot."""

    def __init__(self, reason: str, diagnostics: Dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics or {}


def score_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the formal failure contract even if an evaluator supplied a stale penalty."""
    evaluated = shared.score_result(result)
    if result.get("result_failed", False) or not result.get("execution_success"):
        evaluated.update({
            "execution_success": False,
            "result_failed": True,
            "accepted": False,
            "raw_score_nonruntime": float("-inf"),
            "raw_score_nonruntime_status": "negative_infinity_failure",
            "violation_penalty": 100.0,
            "selection_score_nonruntime": float("-inf"),
            "selection_score_nonruntime_status": "negative_infinity_failure",
        })
    return evaluated


def block_seed(block_id: int) -> int:
    if block_id not in range(1, PLANNED_BLOCKS + 1):
        raise ValueError("formal block_id must be in 1..10")
    return FORMAL_MASTER_SEED + 1000 * (block_id - 1)


def method_seeds(block_id: int) -> Dict[str, int]:
    seed = block_seed(block_id)
    return {method: seed + offset for method, offset in METHOD_SEED_OFFSETS.items()}


def seed_table() -> List[Dict[str, int]]:
    return [
        {
            "block_id": block_id,
            "block_seed": block_seed(block_id),
            **method_seeds(block_id),
            "schedule_offset": (block_id - 1) % 4,
        }
        for block_id in range(1, PLANNED_BLOCKS + 1)
    ]


def build_schedule(block_id: int) -> List[Dict[str, Any]]:
    cycles = (
        (METHODS[0], METHODS[1], METHODS[2], METHODS[3]),
        (METHODS[1], METHODS[2], METHODS[3], METHODS[0]),
        (METHODS[2], METHODS[3], METHODS[0], METHODS[1]),
        (METHODS[3], METHODS[0], METHODS[1], METHODS[2]),
    )
    offset = (block_id - 1) % 4
    block_value = block_seed(block_id)
    counters = {method: 0 for method in METHODS}
    schedule: List[Dict[str, Any]] = []
    for round_index in range(1, ROUNDS_PER_BLOCK + 1):
        cycle_index = (round_index - 1 + offset) % 4
        for position, method in enumerate(cycles[cycle_index], 1):
            counters[method] += 1
            schedule.append({
                "experiment_name": EXPERIMENT_NAME,
                "block_id": block_id,
                "block_seed": block_value,
                "round_index": round_index,
                "order_cycle": "ABCD"[cycle_index],
                "order_position": position,
                "global_slot_index": len(schedule) + 1,
                "method": method,
                "method_slot_index": counters[method],
            })
    validate_schedule(schedule, block_id)
    return schedule


def validate_schedule(schedule: List[Dict[str, Any]], block_id: int) -> None:
    if len(schedule) != TOTAL_SLOTS_PER_BLOCK:
        raise FormalIntegrityError("formal_schedule_length_mismatch")
    for expected, slot in enumerate(schedule, 1):
        if slot["global_slot_index"] != expected or slot["block_id"] != block_id:
            raise FormalIntegrityError("formal_schedule_index_mismatch")
    for method in METHODS:
        method_rows = [slot for slot in schedule if slot["method"] == method]
        if len(method_rows) != SLOTS_PER_METHOD:
            raise FormalIntegrityError("formal_method_slot_count_mismatch")
        positions = [sum(row["order_position"] == p for row in method_rows) for p in range(1, 5)]
        if positions != [20, 20, 20, 20]:
            raise FormalIntegrityError("formal_position_balance_mismatch")


def record_template(slot: Dict[str, Any], provenance_hash: str = "") -> Dict[str, Any]:
    intermediate_slot = {**slot, "validation_name": EXPERIMENT_NAME}
    record = shared.record_template(intermediate_slot, provenance_hash)
    record.update({
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "validation_name": None,
        "formal_experiment": True,
        "formal_dry_run": False,
        "validation_only": False,
        "pilot_only": False,
        "publication_eligible": False,
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "protocol_version": None,
        "planned_block_count": PLANNED_BLOCKS,
        "scheduled_slots_per_method": SLOTS_PER_METHOD,
    })
    return record


CANONICAL_RECORD_FIELDS = frozenset(record_template(build_schedule(1)[0]))


def validate_record(record: Dict[str, Any]) -> None:
    if set(record) != CANONICAL_RECORD_FIELDS:
        raise FormalIntegrityError("formal_record_schema_mismatch")
    if not record["formal_experiment"] or record["formal_dry_run"]:
        raise FormalIntegrityError("formal_record_flag_mismatch")
    if record["validation_only"] or record["pilot_only"] or record["publication_eligible"]:
        raise FormalIntegrityError("formal_record_eligibility_mismatch")
    if record["formal_protocol_version"] != FORMAL_PROTOCOL_VERSION:
        raise FormalIntegrityError("formal_record_protocol_mismatch")
    if record["planned_block_count"] != PLANNED_BLOCKS or record["scheduled_slots_per_method"] != SLOTS_PER_METHOD:
        raise FormalIntegrityError("formal_record_budget_mismatch")
    if record["method"] not in METHODS or record["fallback_used"] is not False:
        raise FormalIntegrityError("formal_record_method_mismatch")
