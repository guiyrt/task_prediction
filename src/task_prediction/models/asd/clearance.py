from datetime import datetime
from dataclasses import dataclass
from enum import StrEnum

class ClearanceType(StrEnum):
    CLEARED_FLIGHT_LEVEL = "cleared-flight-level"
    CLEARED_SPEED = "cleared-speed"
    DIRECT_TO = "direct-to"
    HEADING = "heading"
    ROUTE_CLEARANCE = "route-clearance"

@dataclass(frozen=True, slots=True)
class Clearance:
    timestamp: datetime
    callsign: str

    type: ClearanceType
    clearance: str