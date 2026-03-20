from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum

class SepToolType(IntEnum):
    SEP_TOOL_TYPE_NOT_SET = 0
    A = 1
    B = 2
    C = 3
    D = 4

@dataclass(frozen=True, slots=True)
class SepToolBase:
    timestamp: datetime
    type: SepToolType
    measurement_id: int

@dataclass(frozen=True, slots=True)
class SepToolOpened(SepToolBase):
    callsign: str

@dataclass(frozen=True, slots=True)
class SepToolConnected(SepToolBase):
    callsign_1: str
    callsign_2: str

@dataclass(frozen=True, slots=True)
class SepToolClosed(SepToolBase):
    closed: bool