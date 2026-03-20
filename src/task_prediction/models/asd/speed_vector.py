from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum

class SpeedVectorMode(IntEnum):
    SPEED_VECTOR_MODE_UNKNOWN = 0
    SPEED_VECTOR_MODE_ALL_OFF = 1
    SPEED_VECTOR_MODE_ALL_ON = 2
    SPEED_VECTOR_MODE_SELECTED = 3

@dataclass(frozen=True, slots=True)
class SpeedVectorBase:
    timestamp: datetime

@dataclass(frozen=True, slots=True)
class SpeedVectorModeUpdated(SpeedVectorBase):
    mode: SpeedVectorMode

@dataclass(frozen=True, slots=True)
class SpeedVectorVisibility(SpeedVectorBase):
    visible: bool
    callsign: str

@dataclass(frozen=True, slots=True)
class SpeedVectorLength(SpeedVectorBase):
    length_seconds: int