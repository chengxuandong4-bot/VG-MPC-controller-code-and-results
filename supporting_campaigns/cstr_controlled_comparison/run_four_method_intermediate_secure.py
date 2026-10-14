#!/usr/bin/env python3
"""Secure launcher for one approved intermediate-validation block."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import deepseek_requests_transport as transport
import intermediate_protocol as protocol
import intermediate_provenance as provenance
import four_method_intermediate_validation as runner
import score_only_vgmpc


ROOT = Path(__file__).resolve().parent
AUTH_MESSAGES = [
    {"role": "system", "content": "Return only the requested JSON object."},
    {"role": "user", "content": 'Return exactly:\n{"authentication_valid": true}'},
]
MATLAB = Path("/Applications/MATLAB_R2026a.app/bin/matlab")


def validate_api_key(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError("invalid API key input")
    if value[0] in "\"'" or value[-1] in "\"'":
        raise ValueError("quoted API key input")
    return value


def collision_safe(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True); os.chmod(root, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{prefix}_{stamp}"; suffix = 0
    while path.exists():
        suffix += 1; path = root / f"{prefix}_{stamp}_{suffix}"
    return path


def _write_json(path: Path, value: Any, secret: str) -> None:
    protocol.secure_json_write(path, value, secret)


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists(): return digest.hexdigest()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode()); digest.update(b"\0"); digest.update(item.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _historical_snapshots(root: Path) -> Dict[str, str]:
    snapshots: Dict[str, str] = {}
    for name in ("NON_PUBLICATION_SMOKE_TEST", "NON_PUBLICATION_PILOT"):
        base = root / name
        if base.exists():
            for item in base.iterdir():
                if item.is_dir(): snapshots[str(item)] = _tree_hash(item)
    return snapshots


def _snapshots_unchanged(snapshots: Dict[str, str]) -> bool:
    return all(Path(path).exists() and _tree_hash(Path(path)) == digest for path, digest in snapshots.items())


def authenticate(api_key: str, output: Path, transport_fn: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
    output.mkdir(parents=True, mode=0o700); os.chmod(output.parent, 0o700)
    result = transport_fn(api_key=api_key, messages=AUTH_MESSAGES, model="deepseek-chat", temperature=0.3,
                          max_tokens=64, connect_timeout=15, read_timeout=60,
                          response_format={"type": "json_object"}, allow_transport_retry=False)
    content = result.get("assistant_content_json")
    valid = bool(result.get("http_response_received") and result.get("http_success")
                 and result.get("provider_response_json_valid") and isinstance(content, dict)
                 and content.get("authentication_valid") is True
                 and int(result.get("physical_request_count", 0)) == 1)
    record = {
        "diagnostic_only": True, "validation_only": True, "publication_eligible": False,
        "formal_experiment": False, "api_key_entered_securely": True, "api_key_saved": False,
        "authorization_header_saved": False, "local_credential_read": False,
        "physical_request_count": int(result.get("physical_request_count", 0)), "retry_count": 0,
        "matlab_simulation_count": 0, "http_status": result.get("http_status"),
        "provider_response_json_valid": bool(result.get("provider_response_json_valid")),
        "assistant_content_present": bool(result.get("assistant_content")),
        "assistant_content_json": content if isinstance(content, dict) else None,
        "authentication_valid": valid,
        "provider_error_type": result.get("provider_error_type", ""),
        "provider_error_code": result.get("provider_error_code", ""),
        "provider_message_redacted": result.get("provider_message_redacted", ""),
        "request_duration_s": result.get("request_duration_s", 0.0),
    }
    _write_json(output / "auth_validation.json", record, api_key)
    _write_json(output / "validation_report.json", {
        "authentication_valid": valid, "physical_request_count": record["physical_request_count"],
        "retry_count": 0, "matlab_simulation_count": 0, "validation_started": False,
        "credential_scan_passed": all(api_key.encode() not in p.read_bytes() for p in output.rglob("*") if p.is_file()),
        "api_key_saved": False, "authorization_header_saved": False,
    }, api_key)
    return record


def matlab_call(request: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cstr_intermediate_bridge_") as temp:
        input_path = Path(temp) / "input.json"; output_path = Path(temp) / "output.json"
        input_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        esc = lambda value: str(value).replace("'", "''")
        expression = f"addpath('{esc(ROOT)}'); evaluate_cstr_nonruntime_candidate('{esc(input_path)}','{esc(output_path)}')"
        process = subprocess.run([str(MATLAB), "-batch", expression], text=True, capture_output=True)
        if process.returncode != 0 or not output_path.exists():
            raise RuntimeError("nonruntime_matlab_bridge_failure")
        return json.loads(output_path.read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None, *, root: Path = ROOT,
         prompt_fn: Optional[Callable[[str], str]] = None,
         transport_fn: Optional[Callable[..., Dict[str, Any]]] = None,
         baseline_fn: Optional[Callable[..., Dict[str, Any]]] = None,
         evaluate_fn: Optional[Callable[..., Dict[str, Any]]] = None,
         pilot_runner: Optional[Callable[..., Dict[str, Any]]] = None,
         preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--block", type=int)
    args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if unknown or args.block not in (1, 2):
        print("Usage: run_four_method_intermediate_secure.py --block {1,2}", file=sys.stderr)
        return 2
    if prompt_fn is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        print(f"./.venv/bin/python run_four_method_intermediate_secure.py --block {args.block}")
        return 0
    request = transport_fn or transport.request_chat_completion
    baseline = baseline_fn or matlab_call
    evaluator = evaluate_fn or matlab_call
    block_runner = pilot_runner or runner.run_validation_block
    key: Optional[str] = None
    try:
        preflight = (preflight_fn or provenance.preflight)(root=root, block_id=args.block)
        output_root = root / "NON_PUBLICATION_INTERMEDIATE_VALIDATION"
        auth_dir = collision_safe(output_root, f"cstr_four_method_validation_auth_block{args.block}")
        block_dir = collision_safe(output_root, f"cstr_four_method_validation_block{args.block}")
        if auth_dir.exists() or block_dir.exists():
            raise protocol.IntermediateIntegrityError("planned_output_collision")
        historical = _historical_snapshots(root)
        key = validate_api_key((prompt_fn or getpass.getpass)("Enter new DeepSeek API key: "))
        auth = authenticate(key, auth_dir, request)
        if not auth["authentication_valid"]:
            print(f"Authentication failed; validation block did not start. Auth: {auth_dir}")
            return 1
        summary = block_runner(key, block_dir, args.block, baseline_fn=baseline, evaluate_fn=evaluator,
                               transport_fn=request, root=root)
        prior_unchanged = _snapshots_unchanged(historical)
        report_path = block_dir / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["preflight"] = preflight; report["prior_outputs_unchanged"] = prior_unchanged
        report["credential_scan_passed"] = all(key.encode() not in p.read_bytes() for p in block_dir.rglob("*") if p.is_file())
        _write_json(report_path, report, key)
        print("Intermediate validation block complete: "
              f"block={args.block}, scheduled_slots={summary['scheduled_slots']}, "
              f"logical_generations={summary['logical_llm_generations']}, "
              f"physical_requests={summary['physical_generation_requests']}, "
              f"candidate_simulations={summary['candidate_simulations']}, "
              f"prior_outputs_unchanged={str(prior_unchanged).lower()}. Block: {block_dir}")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Secure key entry was cancelled.", file=sys.stderr); return 130
    except ValueError:
        print("Invalid API key input; validation did not start.", file=sys.stderr); return 2
    except Exception as exc:
        print(f"Secure intermediate launcher failed: {transport.redact_sensitive(exc, key or '')}", file=sys.stderr); return 1
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
