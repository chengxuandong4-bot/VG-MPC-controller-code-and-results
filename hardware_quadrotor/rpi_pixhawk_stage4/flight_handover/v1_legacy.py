"""V1 historical characterization entry point with no real ROS transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


LEGACY_ACTUAL_PUBLISH_ERROR = "legacy_v1_actual_publish_forbidden"


def run_legacy_audit(results_dir: Path) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "mode": "flight_handover_v1_legacy_audit",
        "legacy_characterization_only": True,
        "hardware_allowed": False,
        "publisher_created": False,
        "actual_publish_count": 0,
        "exit_reason": "legacy_v1_audit_complete",
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0
