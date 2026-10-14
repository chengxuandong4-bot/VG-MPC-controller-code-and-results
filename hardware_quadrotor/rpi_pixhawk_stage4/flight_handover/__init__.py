"""Physically separated legacy characterization and safe V2 handover paths."""

from .state_v2 import FlightHandoverV2, StaticHoldSource
from .v2_safe import load_v2_config

__all__ = ["FlightHandoverV2", "StaticHoldSource", "load_v2_config"]
