#!/usr/bin/env python3
"""Generate and strictly validate an AUV parent-relative parameter patch."""

import json
import math
import os
import sys

from llm_generate import call_llm, load_config


def build_prompts(request):
    allowed_lines = []
    for name, spec in request["allowed"].items():
        shape = spec.get("type", "float")
        bounds = f'[{spec["lo"]}, {spec["hi"]}]'
        allowed_lines.append(f"- {name}: {shape}, absolute bounds {bounds}")

    system = """You are tuning an executable AUV fault-tolerant controller.
Return ONLY one JSON object with this exact shape:
{"parent_id":"...", "changes":{"parameter": value}}

Rules:
1. Return a parent-relative PATCH, not a full controller.
2. Use only parameters explicitly listed as allowed.
3. Respect every absolute bound and required vector dimension.
4. Modify the required number of parameters.
5. Do not include explanations, markdown, or additional keys.
6. Do not change PI gains, ks_tau, health_decay, or Boolean mechanisms unless listed.
7. Optimize the normalized five-metric selection score while respecting all guardrails."""

    ratios = request["parent_seed_ratios"]
    weights = request["weights"]
    user = f"""Generation {request["generation"]}, role {request["role"]}.
Parent ID: {request["parent_id"]}

Assigned module and direction:
{request["instruction"]}

Normalized score weights:
IAE={weights[0]}, RMS={weights[1]}, Energy={weights[2]},
P1={weights[3]}, P2={weights[4]}.

Parent metric ratios versus seed:
IAE={ratios[0]:.9f}, RMS={ratios[1]:.9f}, Energy={ratios[2]:.9f},
P1={ratios[3]:.9f}, P2={ratios[4]:.9f}.

Guardrails:
IAE<=1.05, RMS<=1.03, Energy<=1.05, P1<=1.03, P2<=1.03.

Allowed patch fields:
{os.linesep.join(allowed_lines)}

Patch field count: minimum {request["min_changes"]}, maximum {request["max_changes"]}.

Current allowed-field values:
{json.dumps(request["parent_values"], ensure_ascii=False)}

Return only:
{{"parent_id":"{request["parent_id"]}","changes":{{...}}}}"""
    return system, user


def parse_patch(text, request):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"INVALID_SCHEMA JSON: {exc}"

    if not isinstance(data, dict) or set(data) != {"parent_id", "changes"}:
        return None, "INVALID_SCHEMA expected only parent_id and changes"
    if data["parent_id"] != request["parent_id"]:
        return None, "INVALID_SCHEMA parent_id mismatch"
    changes = data["changes"]
    if not isinstance(changes, dict):
        return None, "INVALID_SCHEMA changes must be an object"
    if not request["min_changes"] <= len(changes) <= request["max_changes"]:
        return None, (
            "INVALID_SCHEMA change count "
            f"{len(changes)} not in [{request['min_changes']},{request['max_changes']}]"
        )

    unknown = set(changes) - set(request["allowed"])
    if unknown:
        return None, f"INVALID_SCHEMA forbidden fields: {sorted(unknown)}"

    validated = {}
    for name, value in changes.items():
        spec = request["allowed"][name]
        expected = spec.get("type", "float")
        if expected.startswith("float["):
            length = int(expected[6:-1])
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
                or any(not math.isfinite(float(v)) for v in value)
            ):
                return None, f"INVALID_SCHEMA expected {name}: {expected}"
            parsed = [float(v) for v in value]
            if any(v < spec["lo"] or v > spec["hi"] for v in parsed):
                return None, f"INVALID_SCHEMA out of bounds: {name}"
            validated[name] = parsed
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"INVALID_SCHEMA expected {name}: float"
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < spec["lo"] or parsed > spec["hi"]:
                return None, f"INVALID_SCHEMA out of bounds: {name}"
            validated[name] = parsed

    return {"parent_id": request["parent_id"], "changes": validated}, ""


def merge_metadata(total, current, parse_attempt):
    if not total:
        total = dict(current)
    else:
        for field in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "api_calls", "codex_exec_calls", "retry_count",
        ):
            total[field] = total.get(field, 0) + current.get(field, 0)
        for field in (
            "provider", "transport", "requested_model", "response_model",
            "request_id", "http_status", "codex_cli_version",
        ):
            total[field] = current.get(field, total.get(field, ""))
    total["parse_retry_count"] = parse_attempt
    return total


def main():
    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path, encoding="utf-8") as handle:
        request = json.load(handle)

    system_prompt, user_prompt = build_prompts(request)
    config = load_config()
    max_parse_retries = int(config.get("max_parse_retries", 2))
    metadata = {}
    patch = None
    error = ""
    response = ""

    try:
        for attempt in range(max_parse_retries + 1):
            response, current = call_llm(config, system_prompt, user_prompt, "auv")
            metadata = merge_metadata(metadata, current, attempt)
            patch, error = parse_patch(response, request)
            if patch is not None:
                break
    except Exception as exc:
        error = f"Model invocation failed: {exc}"

    output = {
        "success": patch is not None,
        "error": error,
        "patch": patch or {},
        "api_metadata": metadata,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    with open(output_path.replace(".json", "_prompt.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": response,
            "success": patch is not None,
            "error": error,
            "api_metadata": metadata,
        }, handle, ensure_ascii=False, indent=2)

    print("PATCH OK" if patch is not None else f"PATCH FAILED: {error}")
    if patch is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
