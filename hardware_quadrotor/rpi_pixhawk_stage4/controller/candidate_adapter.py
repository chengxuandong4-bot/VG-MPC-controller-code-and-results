"""Portable, schema-validated adapter for the frozen candidate_097 configuration."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_RELATIVE_PATH = Path(
    "rpi_pixhawk_stage4/config/best_candidate_config.json"
)
LEGACY_CANDIDATE_MARKER = "results_f450_auto_tuning/best_candidate_config.json"
INTERFACE_DEFAULTS_RELATIVE_PATH = Path(
    "sitl_outer_loop_stage/python_controller/controller_interface_defaults.json"
)
INTERFACE_DEFAULTS_SOURCE = (
    "sitl_outer_loop_stage/python_controller/controller_interface_defaults.json"
)
INTERFACE_VERSION = "candidate097_outer_loop_behavioral_equivalence_v1"
REQUIRED_CANDIDATE_FIELDS = (
    "k_i_z",
    "hover_bias_abs_limit",
    "k_i_xy",
    "xy_bias_abs_limit",
)
REQUIRED_INTERFACE_NUMERIC_FIELDS = (
    "horizontal_position_gain_per_s",
    "vertical_position_gain_per_s",
    "yaw_gain_per_s",
    "vertical_integral_limit_mps",
)


class CandidateConfigSchemaError(ValueError):
    """Raised when a candidate deployment configuration violates its schema."""


def portable_path_for_host(value: str, host_os: Optional[str] = None) -> PurePath:
    text = str(value)
    host = host_os or os.name
    windows_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    wsl_match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", text.replace("\\", "/"))
    if host == "posix" and windows_match:
        drive, rest = windows_match.groups()
        return PurePosixPath("/mnt/{0}/{1}".format(drive.lower(), rest.replace("\\", "/")))
    if host == "nt" and wsl_match:
        drive, rest = wsl_match.groups()
        suffix = (rest or "").replace("/", "\\")
        return PureWindowsPath("{0}:\\{1}".format(drive.upper(), suffix))
    if host == "posix":
        return PurePosixPath(text.replace("\\", "/"))
    return PureWindowsPath(text.replace("/", "\\"))


def _is_legacy_candidate_reference(value: str) -> bool:
    normalized = re.sub(r"/+", "/", str(value).replace("\\", "/")).lower()
    return LEGACY_CANDIDATE_MARKER in normalized


def resolve_candidate_path(value: Optional[str] = None) -> Path:
    deployment_path = (PROJECT_ROOT / DEFAULT_CANDIDATE_RELATIVE_PATH).resolve()
    if value is None:
        return deployment_path
    raw = str(value)
    if _is_legacy_candidate_reference(raw) and deployment_path.is_file():
        return deployment_path
    path = Path(portable_path_for_host(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("{0}_file_missing:{1}".format(label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateConfigSchemaError(
            "candidate_schema_error:invalid_json:{0}:{1}".format(label, exc)
        )
    if not isinstance(value, dict):
        raise CandidateConfigSchemaError(
            "candidate_schema_error:{0}_must_be_object".format(label)
        )
    return value


def _validate_finite_numeric(mapping: Dict[str, Any], fields: tuple, label: str) -> None:
    missing = [name for name in fields if name not in mapping]
    if missing:
        raise CandidateConfigSchemaError(
            "candidate_schema_error:missing_{0}_fields:{1}".format(
                label, ",".join(missing)
            )
        )
    for name in fields:
        value = mapping[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CandidateConfigSchemaError(
                "candidate_schema_error:nonfinite_or_nonnumeric_{0}_field:{1}".format(
                    label, name
                )
            )


def _validate_interface_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateConfigSchemaError(
            "candidate_schema_error:interface_defaults_must_be_object"
        )
    required = ("interface_version",) + REQUIRED_INTERFACE_NUMERIC_FIELDS
    missing = [name for name in required if name not in value]
    if missing:
        raise CandidateConfigSchemaError(
            "candidate_schema_error:missing_interface_defaults_fields:{0}".format(
                ",".join(missing)
            )
        )
    if value["interface_version"] != INTERFACE_VERSION:
        raise CandidateConfigSchemaError(
            "candidate_schema_error:unsupported_interface_version:{0}".format(
                value["interface_version"]
            )
        )
    _validate_finite_numeric(
        value,
        REQUIRED_INTERFACE_NUMERIC_FIELDS,
        "interface_defaults",
    )
    return dict(value)


def _load_authoritative_interface_defaults() -> Dict[str, Any]:
    path = PROJECT_ROOT / INTERFACE_DEFAULTS_RELATIVE_PATH
    payload = _read_json_object(path, "interface_defaults_source")
    return _validate_interface_defaults(payload)


def _select_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate = payload["candidate"] if "candidate" in payload else payload
    if not isinstance(candidate, dict):
        raise CandidateConfigSchemaError(
            "candidate_schema_error:candidate_must_be_object"
        )
    return candidate


def load_candidate_config(value: Optional[str] = None) -> Dict[str, Any]:
    path = resolve_candidate_path(value)
    payload = _read_json_object(path, "candidate_config")
    candidate = _select_candidate(payload)
    candidate_id = candidate.get(
        "candidate_id", candidate.get("id", payload.get("candidate_id"))
    )
    if candidate_id != "candidate_097":
        raise CandidateConfigSchemaError(
            "candidate_schema_error:candidate_id_must_be_candidate_097"
        )
    _validate_finite_numeric(candidate, REQUIRED_CANDIDATE_FIELDS, "candidate")

    if "interface_defaults" in candidate:
        interface_defaults = _validate_interface_defaults(
            candidate["interface_defaults"]
        )
        interface_source = candidate.get(
            "interface_defaults_source", INTERFACE_DEFAULTS_SOURCE
        )
        authoritative_path = PROJECT_ROOT / INTERFACE_DEFAULTS_RELATIVE_PATH
        if authoritative_path.is_file():
            authoritative = _load_authoritative_interface_defaults()
            if interface_defaults != authoritative:
                raise CandidateConfigSchemaError(
                    "candidate_schema_error:interface_defaults_mismatch_authoritative_source"
                )
    elif candidate is not payload and "interface_defaults" in payload:
        interface_defaults = _validate_interface_defaults(
            payload["interface_defaults"]
        )
        interface_source = payload.get(
            "interface_defaults_source", INTERFACE_DEFAULTS_SOURCE
        )
        authoritative_path = PROJECT_ROOT / INTERFACE_DEFAULTS_RELATIVE_PATH
        if authoritative_path.is_file():
            authoritative = _load_authoritative_interface_defaults()
            if interface_defaults != authoritative:
                raise CandidateConfigSchemaError(
                    "candidate_schema_error:interface_defaults_mismatch_authoritative_source"
                )
    else:
        interface_defaults = _load_authoritative_interface_defaults()
        interface_source = INTERFACE_DEFAULTS_SOURCE

    if not isinstance(interface_source, str) or not interface_source.strip():
        raise CandidateConfigSchemaError(
            "candidate_schema_error:invalid_interface_defaults_source"
        )

    result = dict(candidate)
    result["candidate_id"] = candidate_id
    result["interface_defaults"] = interface_defaults
    result["interface_defaults_source"] = interface_source
    result["resolved_path"] = str(path)
    return result

class Candidate097Adapter:
    """Python 3.8-compatible behavioral equivalent of frozen Stage 2 candidate_097."""

    def __init__(self, candidate_config: Optional[str] = None) -> None:
        self.config = load_candidate_config(candidate_config)
        defaults = self.config["interface_defaults"]
        self.kp_horizontal = float(defaults["horizontal_position_gain_per_s"])
        self.kp_vertical = float(defaults["vertical_position_gain_per_s"])
        self.kp_yaw = float(defaults["yaw_gain_per_s"])
        self.vertical_integral_limit = min(0.15, float(defaults["vertical_integral_limit_mps"]))
        self.ix_north = 0.0
        self.ix_east = 0.0
        self.iz_down = 0.0

    @staticmethod
    def _value(source: object, name: str) -> float:
        return float(source[name]) if isinstance(source, dict) else float(getattr(source, name))

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def step(self, state: object, reference: object, dt_seconds: float) -> Dict[str, Any]:
        names = ("north_m", "east_m", "down_m", "yaw_rad")
        try:
            state_values = {name: self._value(state, name) for name in names}
            reference_values = {name: self._value(reference, name) for name in names}
            values = list(state_values.values()) + list(reference_values.values()) + [float(dt_seconds)]
        except (KeyError, AttributeError, TypeError, ValueError):
            return {"vx_north_mps": 0.0, "vy_east_mps": 0.0, "vz_down_mps": 0.0, "yaw_rate_radps": 0.0, "invalid_input": True}
        if float(dt_seconds) <= 0.0 or not all(math.isfinite(value) for value in values):
            return {"vx_north_mps": 0.0, "vy_east_mps": 0.0, "vz_down_mps": 0.0, "yaw_rate_radps": 0.0, "invalid_input": True}
        north_error = reference_values["north_m"] - state_values["north_m"]
        east_error = reference_values["east_m"] - state_values["east_m"]
        down_error = reference_values["down_m"] - state_values["down_m"]
        xy_limit = float(self.config["xy_bias_abs_limit"])
        self.ix_north = max(-xy_limit, min(xy_limit, self.ix_north + float(self.config["k_i_xy"]) * north_error * dt_seconds))
        self.ix_east = max(-xy_limit, min(xy_limit, self.ix_east + float(self.config["k_i_xy"]) * east_error * dt_seconds))
        self.iz_down = max(-self.vertical_integral_limit, min(self.vertical_integral_limit, self.iz_down + float(self.config["k_i_z"]) * down_error * dt_seconds))
        raw_vx = self.kp_horizontal * north_error + self.ix_north
        raw_vy = self.kp_horizontal * east_error + self.ix_east
        raw_vz = self.kp_vertical * down_error + self.iz_down
        raw_yaw = self.kp_yaw * self._wrap_angle(reference_values["yaw_rad"] - state_values["yaw_rad"])
        horizontal = math.hypot(raw_vx, raw_vy)
        scale = min(1.0, 0.20 / horizontal) if horizontal else 1.0
        return {
            "vx_north_mps": raw_vx * scale,
            "vy_east_mps": raw_vy * scale,
            "vz_down_mps": max(-0.15, min(0.15, raw_vz)),
            "yaw_rate_radps": max(-0.10, min(0.10, raw_yaw)),
            "raw_vx_north_mps": raw_vx, "raw_vy_east_mps": raw_vy,
            "raw_vz_down_mps": raw_vz, "raw_yaw_rate_radps": raw_yaw,
            "invalid_input": False,
        }
