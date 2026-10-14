"""Flush-on-write CSV and summary logging for every Stage 4 run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable


FILES = {
    "telemetry": "telemetry_log.csv",
    "reference": "reference_log.csv",
    "raw_controller": "raw_controller_output_log.csv",
    "limited_setpoint": "limited_setpoint_log.csv",
    "transmitted": "transmitted_message_log.csv",
    "rc_mode": "rc_and_mode_log.csv",
    "safety": "safety_event_log.csv",
    "timing": "timing_log.csv",
}
FIELDS = {
    "telemetry": ("timestamp", "time_boot_ms", "mode", "armed", "gps_fix", "ekf_ok", "north_m", "east_m", "down_m", "roll_rad", "pitch_rad", "yaw_rad", "telemetry_age_s"),
    "reference": ("timestamp", "north_m", "east_m", "down_m", "yaw_rad"),
    "raw_controller": ("timestamp", "vx_north_mps", "vy_east_mps", "vz_down_mps", "yaw_rate_radps"),
    "limited_setpoint": ("timestamp", "vx_north_mps", "vy_east_mps", "vz_down_mps", "yaw_rate_radps", "saturated"),
    "transmitted": ("timestamp", "message_type", "reason", "vx_north_mps", "vy_east_mps", "vz_down_mps", "yaw_rate_radps"),
    "rc_mode": ("timestamp", "mode", "armed", "rc_channels", "pilot_override"),
    "safety": ("timestamp", "state", "event", "detail"),
    "timing": ("timestamp", "loop_elapsed_s", "loop_period_s"),
}


class LogBundle:
    def __init__(self, results_dir: str) -> None:
        self.path = Path(results_dir).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._handles = {}
        self._writers = {}
        for key, filename in FILES.items():
            handle = (self.path / filename).open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=FIELDS[key], extrasaction="ignore")
            writer.writeheader()
            handle.flush()
            self._handles[key] = handle
            self._writers[key] = writer

    def write(self, key: str, row: Dict[str, Any]) -> None:
        self._writers[key].writerow(row)
        self._handles[key].flush()

    def finish(self, summary: Dict[str, Any]) -> None:
        (self.path / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        lines = ["# Stage 4 Acceptance Report", ""]
        for key in sorted(summary):
            lines.append("- {0}: {1}".format(key, summary[key]))
        (self.path / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.flush()
                handle.close()
