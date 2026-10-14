#!/usr/bin/env python3
"""Block-global deterministic evaluation cache without optimization state."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import intermediate_protocol as protocol


ALLOWED_PAYLOAD_FIELDS = frozenset({
    "complete_candidate_configuration", "candidate_hash", "execution_success",
    "result_failed", "metrics", "violation_penalty", "raw_score_nonruntime",
    "selection_score_nonruntime", "accepted", "deterministic_evaluation_hash",
    "evaluator_provenance_hash",
})


def _canonical_hash(value: Dict[str, Any]) -> str:
    text = json.dumps(protocol._safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class BlockEvaluationCache:
    def __init__(self, provenance_hash: str):
        self.provenance_hash = provenance_hash
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._origins: Dict[str, Tuple[str, int]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def store(self, candidate_hash: str, payload: Dict[str, Any], *, origin_method: str, origin_slot: int) -> str:
        if payload.get("candidate_hash") != candidate_hash:
            raise protocol.IntermediateIntegrityError("cache_candidate_hash_mismatch")
        clean = {name: copy.deepcopy(payload.get(name)) for name in ALLOWED_PAYLOAD_FIELDS if name != "deterministic_evaluation_hash"}
        if clean.get("evaluator_provenance_hash") != self.provenance_hash:
            raise protocol.IntermediateIntegrityError("cache_provenance_mismatch")
        clean["metrics"] = {name: clean["metrics"].get(name) for name in protocol.SCORE_METRICS}
        clean["deterministic_evaluation_hash"] = _canonical_hash(clean)
        if set(clean) != ALLOWED_PAYLOAD_FIELDS:
            raise protocol.IntermediateIntegrityError("cache_payload_schema_mismatch")
        if candidate_hash in self._entries and self._entries[candidate_hash] != clean:
            raise protocol.IntermediateIntegrityError("cache_deterministic_result_mismatch")
        self._entries[candidate_hash] = clean
        self._origins.setdefault(candidate_hash, (origin_method, origin_slot))
        return clean["deterministic_evaluation_hash"]

    def lookup(self, candidate_hash: str, provenance_hash: str) -> Optional[Dict[str, Any]]:
        if provenance_hash != self.provenance_hash:
            raise protocol.IntermediateIntegrityError("cache_provenance_mismatch")
        entry = self._entries.get(candidate_hash)
        return copy.deepcopy(entry) if entry is not None else None

    def origin(self, candidate_hash: str) -> Optional[Tuple[str, int]]:
        return self._origins.get(candidate_hash)

    def audit_index(self) -> Dict[str, Any]:
        return {
            "duplicate_policy_version": protocol.DUPLICATE_POLICY_VERSION,
            "entry_count": len(self._entries),
            "entries": [
                {"candidate_hash": key, "deterministic_evaluation_hash": value["deterministic_evaluation_hash"],
                 "origin_method": self._origins[key][0], "origin_method_slot": self._origins[key][1]}
                for key, value in sorted(self._entries.items())
            ],
        }


def cache_payload(evaluation: Dict[str, Any], complete_configuration: Dict[str, Any], provenance_hash: str) -> Dict[str, Any]:
    return {
        "complete_candidate_configuration": copy.deepcopy(complete_configuration),
        "candidate_hash": evaluation["candidate_hash"],
        "execution_success": evaluation["execution_success"],
        "result_failed": evaluation["result_failed"],
        "metrics": {name: evaluation["metrics"].get(name) for name in protocol.SCORE_METRICS},
        "violation_penalty": evaluation["violation_penalty"],
        "raw_score_nonruntime": evaluation["raw_score_nonruntime"],
        "selection_score_nonruntime": evaluation["selection_score_nonruntime"],
        "accepted": evaluation["accepted"],
        "evaluator_provenance_hash": provenance_hash,
    }
