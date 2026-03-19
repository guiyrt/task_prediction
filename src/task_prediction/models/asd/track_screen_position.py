from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class TrackScreenPosition:
    timestamp: datetime
    callsign: str

    pos: ScreenPosition
    is_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "callsign": self.callsign,
            "x": self.pos.x,
            "y": self.pos.y,
            "visible": self.is_visible,
        }