#!/usr/bin/env python3
"""Frozen legacy CSTR generator with the current DeepSeek model endpoint.

Prompt construction, parsing, and fallback behavior are imported from the
June 2026 source snapshot. Only the API model selection and explicit
non-thinking toggle are changed. Credentials are read from the environment
or the legacy local configuration and are never written to output artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import requests


HERE = Path(__file__).resolve().parent
DEEPSEEK_ROOT = HERE.parent.parent
LEGACY_DIR = (
    DEEPSEEK_ROOT
    / "LLM"
    / "cstr_single_llm_v2"
    / "generation_snapshots"
    / "gen_001_S"
    / "src"
    / "unified_loop"
)
LEGACY_GENERATOR = LEGACY_DIR / "llm_generate.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_cstr_generator", LEGACY_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy generator: {LEGACY_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_request_config(legacy):
    config = legacy.load_config()
    active_config_path = DEEPSEEK_ROOT / "unified_loop" / "llm_config.json"
    if active_config_path.exists():
        with active_config_path.open("r", encoding="utf-8") as handle:
            active_config = json.load(handle)
        if active_config.get("api_key"):
            config["api_key"] = active_config["api_key"]
    elif os.environ.get("DEEPSEEK_API_KEY", ""):
        config["api_key"] = os.environ["DEEPSEEK_API_KEY"]
    config.update(
        {
            "model": "deepseek-v4-flash",
            "temperature": 0.3,
            "max_tokens": 512,
            "base_url": "https://api.deepseek.com/anthropic/v1/messages",
        }
    )
    if not config.get("api_key"):
        raise RuntimeError("DeepSeek API credential is not configured")
    return config


def call_llm(config, system_prompt, user_prompt):
    headers = {
        "x-api-key": config["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "thinking": {"type": "disabled"},
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }
    response = requests.post(
        config["base_url"], headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    data = response.json()
    text_blocks = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    if not text_blocks:
        raise RuntimeError("Anthropic response did not contain a text block")
    return "".join(text_blocks)


def main():
    legacy = load_legacy_module()
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/llm_input.json")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/llm_output.json")
    benchmark = sys.argv[3] if len(sys.argv) > 3 else "cstr"
    role = sys.argv[4] if len(sys.argv) > 4 else "S"

    with input_file.open("r", encoding="utf-8") as handle:
        input_data = json.load(handle)

    incumbent = input_data.get("incumbent", {})
    feedback = input_data.get("feedback", {})
    if isinstance(feedback, str):
        feedback = {"text": feedback}
    metrics = input_data.get("metrics", {})
    q_score = input_data.get("q_score", 0)
    best_q = input_data.get("best_q", -float("inf"))
    history = input_data.get("history", [])
    error_diag = input_data.get("error_diag", "")
    gen = input_data.get("gen", 1)
    stall_count = input_data.get("stall_count", 0)
    baseline_metrics = input_data.get("baseline_metrics", None)

    system_prompt = legacy.build_system_prompt(benchmark, role)
    user_prompt = legacy.build_user_prompt(
        benchmark,
        incumbent,
        feedback,
        metrics,
        q_score,
        best_q,
        history,
        error_diag,
        gen,
        role,
        baseline_metrics,
    )

    if "--dry-run" in sys.argv:
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "dry_run": True,
                    "request": {
                        "model": "deepseek-v4-flash",
                        "transport": "deepseek_anthropic_api",
                        "thinking": "disabled",
                        "temperature": 0.3,
                        "max_tokens": 512,
                    },
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"DRY-RUN: {len(system_prompt)} + {len(user_prompt)} chars")
        return

    success = False
    llm_response = ""
    error_msg = ""
    parsed_config = {}
    try:
        config = load_request_config(legacy)
        llm_response = call_llm(config, system_prompt, user_prompt)
        parsed_config, parse_ok, parse_msg = legacy.parse_llm_response(
            llm_response, benchmark, incumbent
        )
        if parse_ok:
            success = True
            error_msg = parse_msg
        else:
            error_msg = parse_msg
            parsed_config = legacy.generate_fallback(incumbent, benchmark, stall_count)
    except Exception as exc:
        error_msg = f"API call failed: {exc}"
        parsed_config = legacy.generate_fallback(incumbent, benchmark, stall_count)

    source_tag = "api" if success else "fallback"
    parsed_config["name"] = f"gen{gen:03d}_llm_{source_tag}_{role}"
    parsed_config["label"] = parsed_config["name"]
    output = {
        "success": success,
        "error": error_msg,
        "name": parsed_config["name"],
        "label": parsed_config["label"],
        "config": parsed_config,
        "llm_source": source_tag,
        "n_params": len(parsed_config),
    }
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    prompt_file = Path(str(output_file).replace(".json", "_prompt.json"))
    with prompt_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": benchmark,
                "role": role,
                "gen": gen,
                "request": {
                    "model": "deepseek-v4-flash",
                    "transport": "deepseek_anthropic_api",
                    "thinking": "disabled",
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "llm_response": llm_response[:2000] if llm_response else error_msg,
                "success": success,
                "error": error_msg,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    status = "OK" if success else f"FALLBACK ({error_msg[:80]})"
    print(f"LLM {status}: {len(parsed_config)} params in config")


if __name__ == "__main__":
    main()
