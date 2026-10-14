#!/usr/bin/env python3
"""Separate, non-mutating publication eligibility gate for formal results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import formal_analysis_contract as analysis
import formal_protocol as protocol


PUBLICATION_GATE_VERSION = "cstr_formal_publication_gate_v1"
FROZEN_ENDPOINTS = {
    "primary": "final_absolute_nonruntime_score_improvement_after_80_slots",
    "secondary": ["relative_improvement", "normalized_best_so_far_auc", "slot_to_first_baseline_improvement",
                  "accepted_rate", "incumbent_update_count", "engineering_efficiency_endpoints"],
}
FROZEN_INFERENCE = {
    "unit": "paired_block", "omnibus": "friedman", "alpha": 0.05,
    "posthoc": "six_paired_wilcoxon_two_sided_pratt_conditional_on_omnibus",
    "multiplicity": "holm_six", "effect": "matched_rank_biserial",
    "bootstrap": "paired_median_difference_percentile_10000_seeded",
}


class PublicationGateError(RuntimeError):
    """Publication eligibility requirements are incomplete or inconsistent."""


def immutable_table_hash(endpoint_table: Any) -> str:
    payload = json.dumps(endpoint_table, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_publication_eligibility(validated: Mapping[str, Any], analysis_result: Mapping[str, Any], *,
                                     analysis_source_hash: str, exclusion_rules_frozen: bool,
                                     immutable_tables_regenerated: bool, manuscript_traceability: Mapping[str, Any]) -> Dict[str, Any]:
    checks = {
        "ten_blocks_completed": validated.get("validated") is True and validated.get("block_ids") == list(range(1, 11)),
        "all_post_block_audits_passed": len(validated.get("audits", [])) == 10 and all(item.get("passed") is True for item in validated.get("audits", [])),
        "no_protocol_or_hash_drift": len({json.dumps(item.get("protocol_fingerprint"), sort_keys=True) for item in validated.get("audits", [])}) == 1,
        "no_aborted_attempt_included": all(any(d.get("check") == "abort_report_absent" and d.get("passed") for d in item.get("decisions", [])) for item in validated.get("audits", [])),
        "exclusion_rules_frozen": exclusion_rules_frozen is True,
        "analysis_source_hash_frozen": isinstance(analysis_source_hash, str) and len(analysis_source_hash) == 64 and analysis_source_hash == analysis.source_hash(),
        "endpoints_frozen": analysis_result.get("primary_endpoint") == FROZEN_ENDPOINTS["primary"],
        "inferential_plan_frozen": analysis_result.get("posthoc_condition") == "only_if_friedman_rejects" and analysis_result.get("bootstrap_resamples") == 10_000,
        "provenance_complete": all(any(d.get("check") == "provenance_files_complete" and d.get("passed") for d in item.get("decisions", [])) for item in validated.get("audits", [])),
        "credential_scans_passed": all(any(d.get("check") == "validation_credential_and_history_pass" and d.get("passed") for d in item.get("decisions", [])) for item in validated.get("audits", [])),
        "immutable_tables_regenerated": immutable_tables_regenerated is True,
        "manuscript_tables_traceable": manuscript_traceability.get("complete") is True
            and manuscript_traceability.get("endpoint_table_sha256") == immutable_table_hash(validated.get("endpoint_table")),
    }
    return {
        "publication_gate_version": PUBLICATION_GATE_VERSION,
        "formal_protocol_version": protocol.FORMAL_PROTOCOL_VERSION,
        "eligible": all(checks.values()), "checks": checks,
        "collection_records_modified": False, "collection_publication_eligible_flags_remain_false": True,
        "analysis_source_hash": analysis_source_hash, "frozen_endpoints": FROZEN_ENDPOINTS,
        "frozen_inferential_plan": FROZEN_INFERENCE,
        "immutable_endpoint_table_sha256": immutable_table_hash(validated.get("endpoint_table")),
    }


def write_publication_report(report: Mapping[str, Any], output_path: Path, block_dirs: Sequence[Path]) -> None:
    """Write only when explicitly called, outside all immutable block directories."""
    output = Path(output_path).resolve()
    blocks = [Path(path).resolve() for path in block_dirs]
    if output.name != "publication_eligibility_report.json":
        raise PublicationGateError("publication_report_filename_required")
    if any(output == block or block in output.parents for block in blocks):
        raise PublicationGateError("publication_report_must_be_outside_block_directories")
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
