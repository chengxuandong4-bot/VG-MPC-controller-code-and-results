"""File-backed emergency stop checked synchronously by the control loop."""

from pathlib import Path
from typing import Optional


class EmergencyStopFile:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path).expanduser() if path else None

    def triggered(self) -> bool:
        return bool(self.path is not None and self.path.exists())
