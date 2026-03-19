from datetime import datetime
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

class AwareActionStatus(IntEnum):
    AWARE_ACTION_STATUS_NOT_SET = 0
    ACCEPTED = 1
    DISMISSED = 2
    EXPIRED = 3
    REMOVED = 4

@dataclass(frozen=True, slots=True)
class AwareActionInteraction:
    timestamp: datetime
    callsign: str
    
    uuid: str
    status: AwareActionStatus
    details: str
    suggestion_mode_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "callsign": self.callsign,
            "action_uuid": self.uuid,
            "action_status": self.status.value,
            "action_details": self.details,
            "suggestion_mode_enabled": self.suggestion_mode_enabled
        }