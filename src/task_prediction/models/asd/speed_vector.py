from datetime import datetime
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

class SpeedVectorMode(IntEnum):
    SPEED_VECTOR_MODE_UNKNOWN = 0
    SPEED_VECTOR_MODE_ALL_OFF = 1
    SPEED_VECTOR_MODE_ALL_ON = 2
    SPEED_VECTOR_MODE_SELECTED = 3

@dataclass(frozen=True, slots=True)
class SpeedVectorBase:
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode_updated__mode": None,
            "visibility__visible": None,
            "visibility__callsign": None,
            "length__length_seconds": None    
        }

@dataclass(frozen=True, slots=True)
class SpeedVectorModeUpdated(SpeedVectorBase):
    mode: SpeedVectorMode

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["mode_updated__mode"] = self.mode.value
        
        return d

@dataclass(frozen=True, slots=True)
class SpeedVectorVisibility(SpeedVectorBase):
    visible: bool
    callsign: str

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["visibility__visible"] = self.visible
        d["visibility__callsign"] = self.callsign
        
        return d

@dataclass(frozen=True, slots=True)
class SpeedVectorLength(SpeedVectorBase):
    length_seconds: int

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        d["length__length_seconds"] = self.length_seconds
        
        return d