"""Deterministic Stage 4 run metrics used by the acceptance report."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _get(value: object, name: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        return float(value.get(name, default))
    return float(getattr(value, name, default))


class RunMetrics:
    def __init__(self) -> None:
        self.position_errors: List[float] = []
        self.tilts_deg: List[float] = []
        self.loop_periods: List[float] = []
        self.gps_fix_min = 99
        self.ekf_unhealthy_samples = 0
        self.origin_north = None
        self.origin_east = None
        self.origin_down = None
        self.horizontal_radii: List[float] = []
        self.relative_heights: List[float] = []

    def observe(self, state: object, reference: object, loop_period_s: float) -> None:
        dn = _get(reference, "north_m") - _get(state, "north_m")
        de = _get(reference, "east_m") - _get(state, "east_m")
        dd = _get(reference, "down_m") - _get(state, "down_m")
        self.position_errors.append(math.sqrt(dn * dn + de * de + dd * dd))
        north = _get(state, "north_m")
        east = _get(state, "east_m")
        down = _get(state, "down_m")
        if self.origin_north is None:
            self.origin_north, self.origin_east, self.origin_down = north, east, down
        self.horizontal_radii.append(math.hypot(north - self.origin_north, east - self.origin_east))
        self.relative_heights.append(abs(down - self.origin_down))
        roll = _get(state, "roll_rad")
        pitch = _get(state, "pitch_rad")
        self.tilts_deg.append(math.degrees(math.hypot(roll, pitch)))
        if loop_period_s > 0.0 and math.isfinite(loop_period_s):
            self.loop_periods.append(loop_period_s)
        self.gps_fix_min = min(self.gps_fix_min, int(_get(state, "gps_fix_type")))
        if not bool(state.get("ekf_ok", False) if isinstance(state, dict) else getattr(state, "ekf_ok", False)):
            self.ekf_unhealthy_samples += 1

    def summary(self) -> Dict[str, Any]:
        count = len(self.position_errors)
        rms = math.sqrt(sum(value * value for value in self.position_errors) / count) if count else None
        mean_period = sum(self.loop_periods) / len(self.loop_periods) if self.loop_periods else None
        return {
            "rms_position_error_m": rms,
            "final_position_error_m": self.position_errors[-1] if count else None,
            "maximum_position_error_m": max(self.position_errors) if count else None,
            "maximum_tilt_deg": max(self.tilts_deg) if self.tilts_deg else None,
            "mean_control_loop_frequency_hz": (1.0 / mean_period) if mean_period else None,
            "minimum_gps_fix_type": None if self.gps_fix_min == 99 else self.gps_fix_min,
            "maximum_horizontal_radius_m": max(self.horizontal_radii) if self.horizontal_radii else None,
            "maximum_relative_height_m": max(self.relative_heights) if self.relative_heights else None,
            "ekf_unhealthy_sample_count": self.ekf_unhealthy_samples,
        }
