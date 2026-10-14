import json
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]
V2_CONFIG = STAGE_DIR / "config" / "alpha_stage4_flight_handover_v2.yaml"


def config_fixture():
    value = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    value["resolved_path"] = str(V2_CONFIG)
    value["rc"]["ch5"].update(optical_flow_range=[900, 1100], attitude_range=[1400, 1600], onboard_range=[1900, 2100])
    value["rc"]["ch6"].update(hover_range=[900, 1100], tracking_range=[1900, 2100])
    value["rc"]["ch7"].update(arm_range=[979, 1019], disarm_range=[1979, 2019])
    value["rc"]["ch8"].update(init_range=[900, 1100], takeoff_return_range=[1400, 1600], execute_range=[1900, 2100], land_range=[900, 1100])
    value["px4_mode_slots"].update(ch5_optical_flow_slot=1, ch5_attitude_slot=2, ch5_onboard_slot=3)
    return value


def parameters():
    return {
        "RC_MAP_FLTMODE": 5, "RC_MAP_OFFB_SW": 0, "RC_MAP_KILL_SW": 7,
        "RC_MAP_RETURN_SW": 0, "RC_MAP_ARM_SW": 0, "COM_FLTMODE1": 2,
        "COM_FLTMODE2": 8, "COM_FLTMODE3": 2, "COM_FLTMODE4": 2,
        "COM_FLTMODE5": 2, "COM_FLTMODE6": 2, "COM_RC_IN_MODE": 0,
    }


def snapshot(**changes):
    value = {
        "mavros_connected": True, "armed": True, "px4_mode": "POSCTL",
        "px4ctrl_state": "MANUAL_CTRL", "offboard_enabled": False,
        "ch5_semantic": "OPTICAL_FLOW", "ch6_semantic": "HOVER",
        "ch7_semantic": "ARM", "ch8_semantic": "INIT",
        "ch5_onboard_edge": False, "ch5_optical_flow_edge": False,
        "ch6_tracking_edge": False, "ch6_hover_edge": False,
        "cmd_fresh": False, "odom_fresh": True, "rc_fresh": True,
        "battery_fresh": True, "odom_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "roll": 0.0, "pitch": 0.0, "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "yaw": 0.1, "odom_reset_counter": 0, "subscriber_count": 1,
        "subscriber_nodes": ["/px4ctrl"],
    }
    value.update(changes)
    return value
