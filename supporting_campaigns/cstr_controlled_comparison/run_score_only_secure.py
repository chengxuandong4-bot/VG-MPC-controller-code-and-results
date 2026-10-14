#!/usr/bin/env python3
"""One-prompt secure launcher for the three-generation Score-only smoke test."""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

import deepseek_requests_transport as deepseek_transport
import score_only_vgmpc


ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = ROOT / "NON_PUBLICATION_SMOKE_TEST"
MODEL = "deepseek-chat"
AUTH_MESSAGES = [
    {"role": "system", "content": "Return only the requested JSON object."},
    {"role": "user", "content": 'Return exactly:\n{"authentication_valid": true}'},
]
COMMON_AUTH = {
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
}


def validate_api_key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("API key input is required")
    if not value.strip():
        raise ValueError("API key input must be nonempty")
    if value != value.strip():
        raise ValueError("API key input has surrounding whitespace")
    if "\n" in value or "\r" in value:
        raise ValueError("API key input contains a line break")
    if value[0] in "\"'" or value[-1] in "\"'":
        raise ValueError("API key input must not be quoted")
    return value


def _collision_safe_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = root / f"{prefix}_{timestamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{prefix}_{timestamp}_{suffix}"
    return candidate


def _write_json(path: Path, data: Dict[str, Any], secret: str) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if secret in text:
        raise RuntimeError("credential_redaction_failure")
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def _credential_scan(path: Path, secret: str) -> bool:
    secret_bytes = secret.encode("utf-8")
    return all(secret_bytes not in item.read_bytes() for item in path.rglob("*") if item.is_file())


def _proxy_summary() -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    names = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    )
    for name in names:
        value = os.environ.get(name)
        if not value:
            result[name] = {"set": False}
            continue
        try:
            parsed = urlsplit(value if "://" in value else "http://" + value)
            result[name] = {
                "set": True,
                "scheme": parsed.scheme,
                "host": parsed.hostname or "",
                "port": parsed.port,
            }
        except Exception:
            result[name] = {"set": True, "scheme": "", "host": "", "port": None}
    return result


def authenticate(
    api_key: str,
    *,
    root: Path = ROOT,
    transport_fn: Callable[..., Dict[str, Any]] = deepseek_transport.request_chat_completion,
) -> tuple[Dict[str, Any], Path]:
    """Perform and record exactly one non-retrying authentication request."""
    output_dir = _collision_safe_dir(root / "NON_PUBLICATION_SMOKE_TEST", "score_only_secure_auth")
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
    )
    physical_count = int(result.get("physical_request_count", 0))
    if physical_count != 1:
        authentication_valid = False
    auth_record = {
        **COMMON_AUTH,
        "physical_request_count": physical_count,
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
        **COMMON_AUTH,
        "physical_request_count": physical_count,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "requests_version": deepseek_transport.requests.__version__,
        "proxy_summary": _proxy_summary(),
        "tls_verification_enabled": True,
    }
    _write_json(output_dir / "auth_validation.json", auth_record, api_key)
    _write_json(output_dir / "environment_summary.json", environment, api_key)
    initial_scan = _credential_scan(output_dir, api_key)
    validation = {
        **COMMON_AUTH,
        "physical_request_count": physical_count,
        "authentication_valid": authentication_valid,
        "credential_scan_passed": initial_scan,
        "baseline_simulation_count": 0,
        "logical_generation_count": 0,
        "full_feedback_started": False,
    }
    _write_json(output_dir / "validation_report.json", validation, api_key)
    if not _credential_scan(output_dir, api_key):
        raise RuntimeError("credential_scan_failed")
    return auth_record, output_dir


def _tree_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prior_snapshots(root: Path) -> Dict[Path, str]:
    smoke_root = root / "NON_PUBLICATION_SMOKE_TEST"
    if not smoke_root.exists():
        return {}
    return {item: _tree_fingerprint(item) for item in smoke_root.iterdir() if item.is_dir()}


def _snapshots_unchanged(snapshots: Dict[Path, str]) -> bool:
    return all(path.exists() and _tree_fingerprint(path) == fingerprint for path, fingerprint in snapshots.items())


def main(
    argv: Optional[list[str]] = None,
    *,
    root: Path = ROOT,
    prompt_fn: Optional[Callable[[str], str]] = None,
    transport_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    score_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("This launcher accepts no command-line arguments.", file=sys.stderr)
        return 2
    if prompt_fn is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        print("./.venv/bin/python run_score_only_secure.py")
        return 0
    api_key: Optional[str] = None
    prompt = prompt_fn or getpass.getpass
    request = transport_fn or deepseek_transport.request_chat_completion
    runner = score_runner or score_only_vgmpc.run_secure
    try:
        api_key = validate_api_key(prompt("Enter new DeepSeek API key: "))
        snapshots = _prior_snapshots(root)
        auth_record, auth_dir = authenticate(api_key, root=root, transport_fn=request)
        if not auth_record["authentication_valid"]:
            print(
                "Authentication failed; no baseline or controller generation was started. "
                f"Diagnostic: {auth_dir}"
            )
            return 1
        run_dir = _collision_safe_dir(
            root / "NON_PUBLICATION_SMOKE_TEST", "score_only_vgmpc_seed_20260813"
        )
        summary = runner(api_key, run_dir, transport_fn=request)
        prior_unchanged = _snapshots_unchanged(snapshots)
        validation_path = run_dir / "validation_report.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["prior_outputs_unchanged"] = prior_unchanged
            validation["full_feedback_started"] = False
            _write_json(validation_path, validation, api_key)
            if not _credential_scan(run_dir, api_key):
                raise RuntimeError("credential_scan_failed")
        print(
            "Score-only secure smoke test complete: "
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
        message = deepseek_transport.redact_sensitive(exc, api_key or "")
        print(f"Secure Score-only launcher failed: {message}", file=sys.stderr)
        return 1
    finally:
        api_key = None


if __name__ == "__main__":
    raise SystemExit(main())
