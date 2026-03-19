from datetime import datetime
from dataclasses import dataclass
from typing import Any
from enum import StrEnum

class ClearanceType(StrEnum):
    CLEARED_FLIGHT_LEVEL = "cleared-flight-level"
    HEADING = "heading"
    DIRECT_TO = "direct-to"

@dataclass(frozen=True, slots=True)
class Clearance:
    timestamp: datetime
    callsign: str

    type: ClearanceType
    clearance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "callsign": self.callsign,
            "clearance_type": self.type.value,
            "clearance": self.clearance,
        }