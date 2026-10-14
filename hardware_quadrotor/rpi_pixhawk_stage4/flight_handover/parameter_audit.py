"""Read-only PX4 RC mapping audit; no parameter write boundary exists here."""

from __future__ import annotations

import math
from typing import Any, Dict


REQUIRED_PARAMETERS = (
    "RC_MAP_FLTMODE", "RC_MAP_OFFB_SW", "RC_MAP_KILL_SW", "RC_MAP_RETURN_SW",
    "RC_MAP_ARM_SW", "COM_FLTMODE1", "COM_FLTMODE2", "COM_FLTMODE3",
    "COM_FLTMODE4", "COM_FLTMODE5", "COM_FLTMODE6", "COM_RC_IN_MODE",
)
MODE_VALUES = {"POSITION": 2, "STABILIZED": 8}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class ParameterAudit:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def evaluate(self, values: Dict[str, Any]) -> Dict[str, Any]:
        conflicts = []
        for name in REQUIRED_PARAMETERS:
            if name not in values or not _finite(values[name]):
                conflicts.append({"parameter": name, "reason": "parameter_unreadable", "actual": values.get(name), "expected": "finite_read_only_value"})
        if conflicts:
            return {"parameter_audit_passed": False, "parameter_conflicts": conflicts, "parameter_values": dict(values)}
        slots = self.config["px4_mode_slots"]
        expected = self.config["expected_modes"]
        for slot_name, expected_name in (("ch5_optical_flow_slot", "ch5_optical_flow"), ("ch5_attitude_slot", "ch5_attitude"), ("ch5_onboard_slot", "ch5_onboard_fallback")):
            slot = slots.get(slot_name)
            if slot is None:
                conflicts.append({"parameter": slot_name, "reason": "mode_slot_unconfirmed", "actual": None, "expected": "QGC_confirmed_1_to_6"})
                continue
            parameter = "COM_FLTMODE{0}".format(slot)
            wanted = MODE_VALUES[expected[expected_name]]
            if int(values[parameter]) != wanted:
                conflicts.append({"parameter": parameter, "reason": "mode_slot_conflict", "actual": values[parameter], "expected": wanted})
        if int(values["RC_MAP_OFFB_SW"]) != 0:
            conflicts.append({"parameter": "RC_MAP_OFFB_SW", "reason": "offboard_switch_must_be_disabled", "actual": values["RC_MAP_OFFB_SW"], "expected": 0})
        ch7_policy = self.config.get("ch7_policy")
        if ch7_policy != "px4_native_kill":
            conflicts.append({"parameter": "ch7_policy", "reason": "ch7_policy_missing_or_unsupported", "actual": ch7_policy, "expected": "px4_native_kill"})
        else:
            if int(values["RC_MAP_KILL_SW"]) != 7:
                conflicts.append({"parameter": "RC_MAP_KILL_SW", "reason": "ch7_native_kill_mapping_required", "actual": values["RC_MAP_KILL_SW"], "expected": 7})
            if int(values["RC_MAP_ARM_SW"]) == 7:
                conflicts.append({"parameter": "RC_MAP_ARM_SW", "reason": "ch7_arm_mapping_conflict", "actual": values["RC_MAP_ARM_SW"], "expected": "not_channel_7"})
        for channel, names, reason in (
            (6, ("RC_MAP_FLTMODE", "RC_MAP_OFFB_SW", "RC_MAP_KILL_SW", "RC_MAP_RETURN_SW", "RC_MAP_ARM_SW"), "ch6_mapping_conflict"),
            (8, ("RC_MAP_FLTMODE", "RC_MAP_OFFB_SW"), "ch8_mapping_conflict"),
        ):
            for name in names:
                if int(values[name]) == channel:
                    conflicts.append({"parameter": name, "reason": reason, "actual": values[name], "expected": "not_channel_{0}".format(channel)})
        return {"parameter_audit_passed": not conflicts, "parameter_conflicts": conflicts, "parameter_values": dict(values)}
