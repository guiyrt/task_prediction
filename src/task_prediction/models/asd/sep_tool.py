from datetime import datetime
from dataclasses import dataclass
from typing import Any
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.type.value,
            "measurement_id": self.measurement_id,
            "opened__callsign": None,
            "connected__callsign_1": None,
            "connected__callsign_2": None,
            "closed": None    
        }

@dataclass(frozen=True, slots=True)
class SepToolOpened(SepToolBase):
    callsign: str

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["opened__callsign"] = self.callsign
        
        return d

@dataclass(frozen=True, slots=True)
class SepToolConnected(SepToolBase):
    callsign_1: str
    callsign_2: str

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["connected__callsign_1"] = self.callsign_1
        d["connected__callsign_2"] = self.callsign_2
        
        return d

@dataclass(frozen=True, slots=True)
class SepToolClosed(SepToolBase):
    closed: bool

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["closed"] = self.closed
        
        return d