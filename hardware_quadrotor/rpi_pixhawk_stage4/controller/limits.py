"""Final independent flight limits applied after candidate_097."""

from __future__ import annotations

import math
from dataclasses import dataclass


MAX_HORIZONTAL_SPEED_MPS = 0.10
MAX_VERTICAL_SPEED_MPS = 0.10
MAX_YAW_RATE_RADPS = 0.08


@dataclass(frozen=True)
class LimitedTarget:
    vx_north_mps: float
    vy_east_mps: float
    vz_down_mps: float
    yaw_rate_radps: float
    saturated: bool
    invalid_input: bool

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def keys(self) -> tuple:
        return ("vx_north_mps", "vy_east_mps", "vz_down_mps", "yaw_rate_radps", "saturated", "invalid_input")

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.keys()}


def limit_target(vx_north_mps: float, vy_east_mps: float,
                 vz_down_mps: float, yaw_rate_radps: float) -> LimitedTarget:
    values = (vx_north_mps, vy_east_mps, vz_down_mps, yaw_rate_radps)
    if not all(math.isfinite(float(value)) for value in values):
        return LimitedTarget(0.0, 0.0, 0.0, 0.0, False, True)

    vx = float(vx_north_mps)
    vy = float(vy_east_mps)
    magnitude = math.hypot(vx, vy)
    saturated = False
    if magnitude > MAX_HORIZONTAL_SPEED_MPS:
        scale = MAX_HORIZONTAL_SPEED_MPS / magnitude
        vx *= scale
        vy *= scale
        saturated = True
    vz = max(-MAX_VERTICAL_SPEED_MPS, min(MAX_VERTICAL_SPEED_MPS, float(vz_down_mps)))
    yaw = max(-MAX_YAW_RATE_RADPS, min(MAX_YAW_RATE_RADPS, float(yaw_rate_radps)))
    saturated = saturated or vz != float(vz_down_mps) or yaw != float(yaw_rate_radps)
    return LimitedTarget(vx, vy, vz, yaw, saturated, False)
