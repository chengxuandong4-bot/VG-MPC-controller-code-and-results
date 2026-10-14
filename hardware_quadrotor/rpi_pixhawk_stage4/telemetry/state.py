"""Normalized receive-only hardware state."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HardwareState:
    system_id: int = 0
    component_id: int = 0
    autopilot: int = -1
    vehicle_type: int = -1
    flight_mode: str = "UNKNOWN"
    armed: bool = False
    gps_fix_type: int = 0
    ekf_ok: bool = False
    local_position_valid: bool = False
    rc_valid: bool = False
    telemetry_age_s: float = 99.0
    rc_age_s: float = 99.0
    last_heartbeat_monotonic_s: float = 0.0
    heartbeat_age_s: float = 99.0
    heartbeat_fresh: bool = False
    received_monotonic_s: float = 0.0
    time_boot_ms: int = 0
    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = 0.0
    vn_mps: float = 0.0
    ve_mps: float = 0.0
    vd_mps: float = 0.0
    relative_alt_valid: bool = False
    relative_alt_m: float = 0.0
    horizontal_speed_m_s: float = 0.0
    vertical_speed_m_s: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    battery_remaining: int = -1
    rc_channels: Dict[int, int] = field(default_factory=dict)
