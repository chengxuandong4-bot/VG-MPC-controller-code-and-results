#!/usr/bin/env python3
"""Secure hidden-key launcher for the Full-feedback VG-MPC smoke test."""
from __future__ import annotations

import getpass
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import deepseek_requests_transport as transport
import full_feedback_vgmpc as full_feedback
import run_score_only_secure as secure_helpers


ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-chat"
AUTH_MESSAGES = [
    {"role": "system", "content": "Return only the requested JSON object."},
    {"role": "user", "content": 'Return exactly:\n{"authentication_valid": true}'},
]


def authenticate(
    api_key: str,
    *,
    root: Path = ROOT,
    transport_fn: Callable[..., Dict[str, Any]] = transport.request_chat_completion,
) -> tuple[Dict[str, Any], Path]:
    output_dir = secure_helpers._collision_safe_dir(
        root / "NON_PUBLICATION_SMOKE_TEST", "full_feedback_secure_auth"
    )
    output_dir.mkdir(mode=0o700)
    result = transport_fn(
        api_key=api_key,
        messages=AUTH_MESSAGES,
        model=MODEL,
        temperature=0.3,
        max_tokens=64,
        connect_timeout=15,
        read_timeout=60,
        response_format={"type": "json_object"},
        allow_transport_retry=False,
    )
    assistant_json = result.get("assistant_content_json")
    authentication_valid = bool(
        result.get("http_response_received")
        and result.get("http_success")
        and result.get("provider_response_json_valid")
        and result.get("assistant_content")
        and isinstance(assistant_json, dict)
        and assistant_json.get("authentication_valid") is True
        and int(result.get("physical_request_count", 0)) == 1
    )
    common = {
        "diagnostic_only": True,
        "publication_eligible": False,
        "controller_generation": False,
        "scientific_candidate_evaluation": False,
        "api_key_entered_securely": True,
        "api_key_saved": False,
        "authorization_header_saved": False,
        "local_credential_read": False,
        "retry_count": 0,
        "matlab_simulation_count": 0,
        "physical_request_count": int(result.get("physical_request_count", 0)),
    }
    record = {
        **common,
        "request_completed": bool(result.get("request_completed")),
        "http_response_received": bool(result.get("http_response_received")),
        "http_status": result.get("http_status"),
        "reason": result.get("reason", ""),
        "response_content_type": result.get("response_content_type", ""),
        "provider_response_json_valid": bool(result.get("provider_response_json_valid")),
        "assistant_content_present": bool(result.get("assistant_content")),
        "assistant_content_json": assistant_json,
        "provider_error_type": result.get("provider_error_type", ""),
        "provider_error_code": result.get("provider_error_code", ""),
        "provider_message_redacted": result.get("provider_message_redacted", ""),
        "transport_exception_class": result.get("transport_exception_class", ""),
        "transport_exception_message_redacted": result.get("transport_exception_message_redacted", ""),
        "request_duration_s": result.get("request_duration_s", 0.0),
        "token_usage": result.get("token_usage", {}),
        "provider_request_id": result.get("provider_request_id", ""),
        "authentication_valid": authentication_valid,
    }
    environment = {
        **common,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "requests_version": transport.requests.__version__,
        "proxy_summary": secure_helpers._proxy_summary(),
        "tls_verification_enabled": True,
    }
    secure_helpers._write_json(output_dir / "auth_validation.json", record, api_key)
    secure_helpers._write_json(output_dir / "environment_summary.json", environment, api_key)
    scan = secure_helpers._credential_scan(output_dir, api_key)
    validation = {
        **common,
        "authentication_valid": authentication_valid,
        "credential_scan_passed": scan,
        "baseline_simulation_count": 0,
        "logical_generation_count": 0,
        "full_feedback_started": False,
    }
    secure_helpers._write_json(output_dir / "validation_report.json", validation, api_key)
    if not secure_helpers._credential_scan(output_dir, api_key):
        raise RuntimeError("credential_scan_failed")
    return record, output_dir


def main(
    argv: Optional[list[str]] = None,
    *,
    root: Path = ROOT,
    prompt_fn: Optional[Callable[[str], str]] = None,
    transport_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("This launcher accepts no command-line arguments.", file=sys.stderr)
        return 2
    if prompt_fn is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        print("./.venv/bin/python run_full_feedback_secure.py")
        return 0
    request = transport_fn or transport.request_chat_completion
    run = runner or full_feedback.run_secure
    api_key: Optional[str] = None
    try:
        prompt = prompt_fn or getpass.getpass
        api_key = secure_helpers.validate_api_key(prompt("Enter new DeepSeek API key: "))
        snapshots = secure_helpers._prior_snapshots(root)
        auth_record, auth_dir = authenticate(api_key, root=root, transport_fn=request)
        if not auth_record["authentication_valid"]:
            print(
                "Authentication failed; no baseline or Full-feedback generation was started. "
                f"Diagnostic: {auth_dir}"
            )
            return 1
        run_dir = secure_helpers._collision_safe_dir(
            root / "NON_PUBLICATION_SMOKE_TEST", "full_feedback_vgmpc_seed_20260814"
        )
        summary = run(api_key, run_dir, transport_fn=request)
        prior_unchanged = secure_helpers._snapshots_unchanged(snapshots)
        validation_path = run_dir / "validation_report.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["prior_outputs_unchanged"] = prior_unchanged
            validation["full_feedback_started"] = True
            secure_helpers._write_json(validation_path, validation, api_key)
            if not secure_helpers._credential_scan(run_dir, api_key):
                raise RuntimeError("credential_scan_failed")
        print(
            "Full-feedback secure smoke test complete: "
            f"auth_requests=1, logical_generations={summary['logical_generations']}, "
            f"physical_generation_requests={summary['physical_generation_requests']}, "
            f"candidate_simulations={summary['candidate_simulations']}, "
            f"fallback_generations=0, fourth_generation_started=false, "
            f"prior_outputs_unchanged={str(prior_unchanged).lower()}. "
            f"Auth: {auth_dir} Run: {run_dir}"
        )
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Secure key entry was cancelled.", file=sys.stderr)
        return 130
    except ValueError:
        print("Invalid API key input; no request or simulation was started.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Secure Full-feedback launcher failed: {transport.redact_sensitive(exc, api_key or '')}", file=sys.stderr)
        return 1
    finally:
        api_key = None


if __name__ == "__main__":
    raise SystemExit(main())
