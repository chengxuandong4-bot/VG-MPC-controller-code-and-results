#!/usr/bin/env python3
"""Explicit CLI for analysis after all ten independently audited blocks exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import formal_analysis_contract as contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", action="append", type=Path, required=True,
                        help="Completed formal block directory; provide exactly ten times")
    parser.add_argument("--output", type=Path, required=True,
                        help="New analysis JSON path outside every block directory")
    args = parser.parse_args(argv)
    resolved_output = args.output.resolve()
    blocks = [path.resolve() for path in args.block]
    if any(resolved_output == block or block in resolved_output.parents for block in blocks):
        raise contract.FormalAnalysisError("analysis_output_must_be_outside_block_directories")
    if resolved_output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{resolved_output}")
    validated = contract.validate_formal_blocks(blocks)
    result = contract.run_frozen_analysis(validated)
    payload = {"validated_blocks": validated, "analysis": result,
               "analysis_source_hash": contract.source_hash()}
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
