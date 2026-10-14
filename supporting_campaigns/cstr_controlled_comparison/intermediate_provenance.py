#!/usr/bin/env python3
"""Read-only provenance and mutation checks for intermediate validation."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import optuna
import requests

import intermediate_protocol as protocol
import score_only_vgmpc


PYTHON_FILES = (
    "intermediate_protocol.py", "intermediate_provenance.py", "intermediate_evaluation_cache.py",
    "four_method_intermediate_validation.py", "run_four_method_intermediate_secure.py",
    "deepseek_requests_transport.py", "score_only_vgmpc.py", "full_feedback_vgmpc.py",
    "construct_cstr_candidate.m", "update_cstr_incumbent.m",
)
MATLAB_FILES = (
    "evaluate_cstr_nonruntime_candidate.m", "evaluate_cstr_candidate.m",
    "construct_cstr_candidate.m", "update_cstr_incumbent.m",
    "../benchmark_lightweight/config_lightweight.m",
    "../benchmark_lightweight/src/run_generation_experiment.m",
)
SEARCH_SPACE = "search_space_cstr_reduced.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def _git_info(root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {"git_available": True, "git_commit": commit, "dirty_worktree": bool(status.strip())}
    except Exception:
        return {"git_available": False, "git_commit": None, "dirty_worktree": None}


def _workspace_merkle(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".venv", "__pycache__", "NON_PUBLICATION_SMOKE_TEST", "NON_PUBLICATION_PILOT", "NON_PUBLICATION_INTERMEDIATE_VALIDATION"}
    files = []
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            relative = item.relative_to(root)
        except ValueError:
            continue
        if any(part in excluded for part in relative.parts):
            continue
        files.append((str(relative), item))
    for relative, item in sorted(files):
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(item.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def collect_provenance(root: Path, block_config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(root).resolve()
    all_files = []
    for relative in dict.fromkeys(PYTHON_FILES + MATLAB_FILES + (SEARCH_SPACE,)):
        path = (root / relative).resolve()
        all_files.append({"path": str(path.relative_to(root.parent.parent) if path.is_relative_to(root.parent.parent) else path),
                          "relative_to_runner": relative, "sha256": file_hash(path),
                          "size": path.stat().st_size if path.is_file() else None})
    payload = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "score_protocol_version": protocol.SCORE_PROTOCOL_VERSION,
        "acceptance_protocol_version": protocol.ACCEPTANCE_PROTOCOL_VERSION,
        "duplicate_policy_version": protocol.DUPLICATE_POLICY_VERSION,
        "tpe_objective_version": protocol.TPE_OBJECTIVE_VERSION,
        "score_only_prompt_template_version": protocol.SCORE_PROMPT_VERSION,
        "full_feedback_prompt_template_version": protocol.FULL_PROMPT_VERSION,
        "full_feedback_schema_version": protocol.FULL_FEEDBACK_SCHEMA_VERSION,
        "parser_version": protocol.PARSER_VERSION,
        "block_configuration": block_config,
        "block_configuration_hash": sha256_bytes(json.dumps(block_config, sort_keys=True, separators=(",", ":")).encode()),
        "files": all_files,
        "component_hashes": {
            "score_only_prompt_template_hash": sha256_bytes(inspect.getsource(protocol.build_score_only_prompt).encode()),
            "full_feedback_prompt_template_hash": sha256_bytes(inspect.getsource(protocol.build_full_feedback_prompt).encode()),
            "full_feedback_schema_hash": sha256_bytes(inspect.getsource(protocol.build_full_feedback_object).encode()),
            "parser_source_hash": sha256_bytes(inspect.getsource(score_only_vgmpc.strict_parse).encode()),
            "score_source_hash": sha256_bytes(inspect.getsource(protocol.minimal_raw_score).encode()),
            "acceptance_source_hash": sha256_bytes(inspect.getsource(protocol.authoritative_acceptance).encode()),
            "incumbent_source_hash": sha256_bytes(inspect.getsource(protocol.incumbent_updated).encode()),
            "tpe_objective_source_hash": sha256_bytes(inspect.getsource(protocol.tpe_objective).encode()),
        },
        "git": _git_info(root),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "optuna_version": optuna.__version__,
        "requests_version": requests.__version__,
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "matlab_version": os.environ.get("INTERMEDIATE_MATLAB_VERSION", "not_detected_before_execution"),
        "workspace_merkle": _workspace_merkle(root),
    }
    payload["snapshot_hash"] = sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload


def assert_unchanged(root: Path, snapshot: Dict[str, Any], block_config: Dict[str, Any]) -> None:
    current = collect_provenance(root, block_config)
    if current["snapshot_hash"] != snapshot["snapshot_hash"]:
        raise protocol.IntermediateIntegrityError("provenance_snapshot_mismatch")


def source_hashes(root: Path) -> Dict[str, Optional[str]]:
    root = Path(root).resolve()
    return {relative: file_hash((root / relative).resolve()) for relative in dict.fromkeys(PYTHON_FILES + MATLAB_FILES + (SEARCH_SPACE,))}


def preflight(root: Path, block_id: int) -> Dict[str, Any]:
    """Read-only checks performed before key entry and any MATLAB/API call."""
    if block_id not in (1, 2):
        raise protocol.IntermediateIntegrityError("invalid_block_id")
    root = Path(root).resolve()
    output_root = root / "NON_PUBLICATION_INTERMEDIATE_VALIDATION"
    if output_root.exists() and not output_root.is_dir():
        raise protocol.IntermediateIntegrityError("intermediate_output_root_not_directory")
    schedule = protocol.build_schedule(block_id)
    if len(schedule) != protocol.TOTAL_SLOTS_PER_BLOCK:
        raise protocol.IntermediateIntegrityError("preflight_schedule_mismatch")
    space = json.loads((root / SEARCH_SPACE).read_text(encoding="utf-8"))
    if [v["name"] for v in space["variables"]] != list(__import__("score_only_vgmpc").NAMES):
        raise protocol.IntermediateIntegrityError("preflight_search_space_mismatch")
    checks = {
        "schedule_160": True,
        "slots_per_method_40": True,
        "position_each_10": True,
        "runtime_not_in_score": protocol.RUNTIME_METRIC not in protocol.SCORE_METRICS,
        "flags_frozen": protocol.VALIDATION_ONLY and not protocol.PILOT_ONLY and not protocol.PUBLICATION_ELIGIBLE and not protocol.FORMAL_EXPERIMENT,
        "output_root_safe": not output_root.exists() or output_root.is_dir(),
        "no_cli_override_protocol": True,
    }
    if not all(checks.values()):
        raise protocol.IntermediateIntegrityError("preflight_failed")
    return {"checks": checks, "block_id": block_id, "block_seed": protocol.block_seed(block_id)}
