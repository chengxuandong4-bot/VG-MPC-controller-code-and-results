"""Conservative first-flight reference: hold, then one smooth 0.10 m move."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencePoint:
    north_m: float
    east_m: float
    down_m: float
    yaw_rad: float

    def __getitem__(self, key: str) -> float:
        return float(getattr(self, key))


class FirstFlightReference:
    def __init__(self, origin: object, yaw_rad: float = 0.0) -> None:
        if isinstance(origin, (tuple, list)):
            self.north, self.east, self.down = map(float, origin[:3])
        else:
            self.north = float(getattr(origin, "north_m", getattr(origin, "north", 0.0)))
            self.east = float(getattr(origin, "east_m", getattr(origin, "east", 0.0)))
            self.down = float(getattr(origin, "down_m", getattr(origin, "down", 0.0)))
        self.yaw = float(yaw_rad)

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = max(0.0, min(1.0, value))
        return x * x * (3.0 - 2.0 * x)

    def at(self, elapsed_seconds: float) -> ReferencePoint:
        # Hold for 2 s, smoothly move +0.10 m north over the remaining 3 s.
        progress = self._smoothstep((float(elapsed_seconds) - 2.0) / 2.0)
        return ReferencePoint(self.north + 0.10 * progress, self.east, self.down, self.yaw)
