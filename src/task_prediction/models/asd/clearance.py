from datetime import datetime
from dataclasses import dataclass
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