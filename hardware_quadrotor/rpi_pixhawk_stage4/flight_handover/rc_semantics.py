"""Measured-range-only RC semantic decoding with debounced one-shot edges."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence


CHANNELS = {
    "ch5": (("OPTICAL_FLOW", "optical_flow_range"), ("ATTITUDE", "attitude_range"), ("ONBOARD", "onboard_range")),
    "ch6": (("HOVER", "hover_range"), ("TRACKING", "tracking_range")),
    "ch7": (("ARM", "arm_range"), ("DISARM", "disarm_range")),
    "ch8": (("INIT", "init_range"), ("TAKEOFF_OR_RETURN", "takeoff_return_range"), ("EXECUTE", "execute_range"), ("LAND", "land_range")),
}


def valid_range(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in value) and float(value[0]) <= float(value[1])


class RcSemanticTracker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.minimum_messages = int(config["debounce_min_messages"])
        self.debounce_seconds = float(config["debounce_seconds"])
        self.stable = {name: None for name in CHANNELS}
        self.candidate = {name: None for name in CHANNELS}
        self.since = {name: None for name in CHANNELS}
        self.count = {name: 0 for name in CHANNELS}

    def _classify(self, channel: str, pwm: float) -> Optional[str]:
        matches = []
        for semantic, key in CHANNELS[channel]:
            limits = self.config[channel].get(key)
            if valid_range(limits) and float(limits[0]) <= pwm <= float(limits[1]):
                matches.append(semantic)
        if channel == "ch8" and matches == ["INIT", "LAND"]:
            return "INIT"
        return matches[0] if len(matches) == 1 else None

    def update(self, channels: Sequence[Any], now: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for channel in CHANNELS:
            index = int(self.config[channel]["index"])
            raw = channels[index] if len(channels) > index else None
            pwm = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)) else float("nan")
            semantic = self._classify(channel, pwm) if math.isfinite(pwm) else None
            if semantic != self.candidate[channel]:
                self.candidate[channel] = semantic
                self.since[channel] = float(now)
                self.count[channel] = 1
            else:
                self.count[channel] += 1
            previous = self.stable[channel]
            changed = False
            if semantic is not None and self.count[channel] >= self.minimum_messages and float(now) - float(self.since[channel]) >= self.debounce_seconds and semantic != previous:
                self.stable[channel] = semantic
                changed = previous is not None
            result[channel + "_pwm"] = pwm
            result[channel + "_semantic"] = self.stable[channel]
            result[channel + "_previous_semantic"] = previous
            result[channel + "_valid"] = semantic is not None
            result[channel + "_changed"] = changed
        result["ch5_onboard_edge"] = result["ch5_changed"] and result["ch5_previous_semantic"] == "OPTICAL_FLOW" and result["ch5_semantic"] == "ONBOARD"
        result["ch5_optical_flow_edge"] = result["ch5_changed"] and result["ch5_previous_semantic"] == "ONBOARD" and result["ch5_semantic"] == "OPTICAL_FLOW"
        result["ch6_tracking_edge"] = result["ch6_changed"] and result["ch6_previous_semantic"] == "HOVER" and result["ch6_semantic"] == "TRACKING"
        result["ch6_hover_edge"] = result["ch6_changed"] and result["ch6_previous_semantic"] == "TRACKING" and result["ch6_semantic"] == "HOVER"
        return result
