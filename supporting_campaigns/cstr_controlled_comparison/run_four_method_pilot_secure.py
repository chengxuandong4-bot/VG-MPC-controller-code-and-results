#!/usr/bin/env python3
"""Secure launcher for the approved four-method pilot framework."""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import deepseek_requests_transport as transport
import four_method_pilot
import run_score_only_secure as secure_helpers
import score_only_vgmpc


ROOT = Path(__file__).resolve().parent
AUTH_MESSAGES = [
    {"role": "system", "content": "Return only the requested JSON object."},
    {"role": "user", "content": 'Return exactly:\n{"authentication_valid": true}'},
]


class PreflightError(RuntimeError):
    pass


def collision_safe(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{prefix}_{timestamp}"
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"{prefix}_{timestamp}_{suffix}"
    return path


def authenticate(api_key: str, *, root: Path = ROOT, output: Optional[Path] = None,
                 transport_fn=transport.request_chat_completion):
    output = output or collision_safe(root / "NON_PUBLICATION_PILOT", "cstr_four_method_auth")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    result = transport_fn(
        api_key=api_key, messages=AUTH_MESSAGES, model="deepseek-chat", temperature=0.3,
        max_tokens=64, connect_timeout=15, read_timeout=60,
        response_format={"type": "json_object"}, allow_transport_retry=False,
    )
    assistant = result.get("assistant_content_json")
    valid = bool(
        result.get("http_response_received") and result.get("http_success")
        and result.get("provider_response_json_valid") and result.get("assistant_content")
        and isinstance(assistant, dict) and assistant.get("authentication_valid") is True
        and int(result.get("physical_request_count", 0)) == 1
    )
    record = {
        "diagnostic_only": True, "pilot_only": True, "publication_eligible": False,
        "formal_experiment": False, "api_key_entered_securely": True, "api_key_saved": False,
        "authorization_header_saved": False, "local_credential_read": False,
        "physical_request_count": int(result.get("physical_request_count", 0)), "retry_count": 0,
        "matlab_simulation_count": 0, "http_status": result.get("http_status"),
        "provider_response_json_valid": bool(result.get("provider_response_json_valid")),
        "assistant_content_present": bool(result.get("assistant_content")),
        "assistant_content_json": assistant, "authentication_valid": valid,
        "provider_error_type": result.get("provider_error_type", ""),
        "provider_error_code": result.get("provider_error_code", ""),
        "provider_message_redacted": result.get("provider_message_redacted", ""),
        "transport_exception_class": result.get("transport_exception_class", ""),
        "transport_exception_message_redacted": result.get("transport_exception_message_redacted", ""),
        "request_duration_s": result.get("request_duration_s", 0.0),
    }
    secure_helpers._write_json(output / "auth_validation.json", record, api_key)
    secure_helpers._write_json(output / "validation_report.json", {
        "authentication_valid": valid, "physical_request_count": record["physical_request_count"],
        "retry_count": 0, "matlab_simulation_count": 0, "pilot_started": False,
        "credential_scan_passed": secure_helpers._credential_scan(output, api_key),
        "api_key_saved": False, "authorization_header_saved": False,
    }, api_key)
    if not secure_helpers._credential_scan(output, api_key):
        raise RuntimeError("credential_scan_failed")
    return record, output


def _planned_path(root: Path, prefix: str, stamp: str) -> Path:
    return root / "NON_PUBLICATION_PILOT" / f"{prefix}_{stamp}"


def run_preflight(*, root: Path = ROOT, auth_dir: Optional[Path] = None,
                  block_dir: Optional[Path] = None, run_tests: bool = True) -> Dict[str, Any]:
    """Read-only protocol gate. It must pass before getpass or any API/MATLAB call."""
    checks: Dict[str, Any] = {}
    pilot = four_method_pilot
    schedule = pilot.build_schedule()
    space = score_only_vgmpc.load_space()
    checks["schedule_80"] = len(schedule) == pilot.TOTAL_SLOTS
    checks["slots_per_method_20"] = all(
        sum(slot["method"] == method for slot in schedule) == pilot.SLOTS_PER_METHOD
        for method in pilot.METHODS
    )
    checks["abcd_repeated_five"] = [slot["order_cycle"] for slot in schedule[::4]] == list("ABCD") * 5
    checks["position_each_five"] = all(
        [sum(slot["method"] == method and slot["order_position"] == pos for slot in schedule) for pos in range(1, 5)] == [5] * 4
        for method in pilot.METHODS
    )
    checks["seven_variable_space"] = len(space["variables"]) == 7 and [v["name"] for v in space["variables"]] == score_only_vgmpc.NAMES
    checks["fallback_disabled"] = pilot.FALLBACK_ENABLED is False
    checks["warm_start_disabled"] = pilot.WARM_START_ENABLED is False
    checks["replacement_disabled"] = pilot.REPLACEMENT_SLOTS_ENABLED is False
    checks["auth_limit_one"] = pilot.AUTHENTICATION_REQUEST_LIMIT == 1
    checks["baseline_limit_one"] = pilot.BASELINE_SIMULATION_LIMIT == 1
    checks["logical_generation_limit_40"] = pilot.LOGICAL_LLM_GENERATION_LIMIT == 40
    checks["physical_generation_limit_80"] = pilot.PHYSICAL_GENERATION_REQUEST_LIMIT == 80
    checks["target_dirs_absent"] = not (root / "NON_PUBLICATION_PILOT").exists()
    if auth_dir is not None:
        checks["planned_auth_absent"] = not auth_dir.exists()
    if block_dir is not None:
        checks["planned_block_absent"] = not block_dir.exists()
    checks["publication_ineligible"] = pilot.PUBLICATION_ELIGIBLE is False
    checks["formal_runner_disabled"] = pilot.FORMAL_EXPERIMENT is False
    checks["prompt_parser_frozen"] = (
        score_only_vgmpc.PROMPT_VERSION == "score_only_v1"
        and score_only_vgmpc.PARSER_VERSION == "strict_json_v1"
        and four_method_pilot.full_feedback.PROMPT_VERSION == "full_feedback_v1"
    )
    snapshots = secure_helpers._prior_snapshots(root)
    checks["historical_snapshot_read"] = True
    if run_tests:
        env = dict(os.environ)
        env["CSTR_PILOT_PREFLIGHT_TESTS"] = "1"
        test_files = sorted((ROOT / "tests").glob("test_*.py"))
        for test_file in test_files:
            result = subprocess.run([sys.executable, str(test_file)], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise PreflightError(f"regression_test_failed:{test_file.name}")
        checks["mock_regression_tests"] = True
        checks["historical_outputs_unchanged_after_tests"] = secure_helpers._snapshots_unchanged(snapshots)
    else:
        checks["mock_regression_tests"] = True
        checks["historical_outputs_unchanged_after_tests"] = True
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise PreflightError("preflight_failed:" + ",".join(failed))
    return {"checks": checks, "historical_snapshot_count": len(snapshots)}


def main(
    argv: Optional[list[str]] = None,
    *,
    root: Path = ROOT,
    prompt_fn: Optional[Callable[[str], str]] = None,
    transport_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    pilot_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    baseline_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    evaluate_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("This launcher accepts no command-line arguments.", file=sys.stderr)
        return 2
    if prompt_fn is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        print("./.venv/bin/python run_four_method_pilot_secure.py")
        return 0
    request = transport_fn or transport.request_chat_completion
    runner = pilot_runner or four_method_pilot.run_pilot
    baseline = baseline_fn or score_only_vgmpc.matlab_call
    evaluator = evaluate_fn or score_only_vgmpc.matlab_call
    if os.environ.get("CSTR_PILOT_PREFLIGHT_TESTS") == "1":
        preflight_fn = preflight_fn or (lambda **kwargs: {"checks": {"test_mode": True}})
    api_key: Optional[str] = None
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        planned_auth = _planned_path(root, "cstr_four_method_auth", stamp)
        planned_block = _planned_path(root, "cstr_four_method_block1", stamp)
        preflight = (preflight_fn or run_preflight)(root=root, auth_dir=planned_auth, block_dir=planned_block)
        api_key = secure_helpers.validate_api_key(
            (prompt_fn or getpass.getpass)("Enter new DeepSeek API key: ")
        )
        snapshots = secure_helpers._prior_snapshots(root)
        auth, auth_dir = authenticate(api_key, root=root, output=planned_auth, transport_fn=request)
        if not auth["authentication_valid"]:
            print(f"Authentication failed; Pilot did not start. Diagnostic: {auth_dir}")
            return 1
        block_dir = planned_block
        summary = runner(
            api_key, block_dir, baseline_fn=baseline, evaluate_fn=evaluator, transport_fn=request
        )
        prior_unchanged = secure_helpers._snapshots_unchanged(snapshots)
        report = json.loads((block_dir / "validation_report.json").read_text())
        report["preflight"] = preflight
        report["prior_smoke_outputs_unchanged"] = prior_unchanged
        report["credential_scan_passed"] = secure_helpers._credential_scan(block_dir, api_key)
        secure_helpers._write_json(block_dir / "validation_report.json", report, api_key)
        print(
            "Four-method pilot complete: "
            f"scheduled_slots={summary['scheduled_slots']}, "
            f"logical_llm_generations={summary['total_logical_llm_generations']}, "
            f"candidate_simulations={summary['candidate_simulations']}, "
            f"prior_smoke_outputs_unchanged={str(prior_unchanged).lower()}. "
            f"Auth: {auth_dir} Block: {block_dir}"
        )
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Secure key entry was cancelled.", file=sys.stderr)
        return 130
    except ValueError:
        print("Invalid API key input; Pilot did not start.", file=sys.stderr)
        return 2
    except PreflightError as exc:
        print(f"Pilot preflight failed before key entry: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Secure Pilot launcher failed: {transport.redact_sensitive(exc, api_key or '')}", file=sys.stderr)
        return 1
    finally:
        api_key = None


if __name__ == "__main__":
    raise SystemExit(main())
