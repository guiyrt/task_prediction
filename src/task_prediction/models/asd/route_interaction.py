from datetime import datetime
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

class RouteInteractionType(IntEnum):
    ROUTE_INTERACTION_TYPE_NOT_SET = 0
    POINT_SELECTED = 1
    POINT_DESELECTED = 2
    POINT_ADDED = 3
    POINT_REMOVED = 4
    DIRECT_TO = 5
    DRAG_STARTED = 6
    DRAG_ENDED = 7
    CHANGES_APPLIED = 8
    CHANGES_CANCELLED = 9

@dataclass(frozen=True, slots=True)
class RouteInteraction:
    timestamp: datetime
    callsign: str

    type: RouteInteractionType
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "callsign": self.callsign,
            "action_type": self.type.value,
            "value": self.value,
        }