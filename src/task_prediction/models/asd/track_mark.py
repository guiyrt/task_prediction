from datetime import datetime
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

class TrackMarkType(IntEnum):
    MARK_TYPE_NOT_SET = 0
    FIRST_TYPE = 1
    SECOND_TYPE = 2
    THIRD_TYPE = 3
    FOURTH_TYPE = 4

class TrackMarkVariant(IntEnum):
    MARK_VARIANT_NOT_SET = 0
    FIRST_VARIANT = 1
    SECOND_VARIANT = 2
    THIRD_VARIANT = 3

@dataclass(frozen=True, slots=True)
class TrackMark:
    timestamp: datetime
    callsign: str

    type: TrackMarkType
    variant: TrackMarkVariant
    scope: str
    set: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "callsign": self.callsign,
            "mark_type": self.type.value,
            "mark_variant": self.variant.value,
            "mark_scope": self.scope,
            "mark_set": self.set
        }