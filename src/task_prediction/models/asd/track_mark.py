from datetime import datetime
from dataclasses import dataclass
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
    is_set: bool