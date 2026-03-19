from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class TrackLabelPosition:
    timestamp: datetime
    callsign: str

    top_left: ScreenPosition
    width: int
    height: int
    is_visible: bool
    is_hovered: bool
    is_selected: bool
    on_pip: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.top_left.x,
            "y": self.top_left.y,
            "width": self.width,
            "height": self.height,
            "visible": self.is_visible,
            "hovered": self.is_hovered,
            "selected": self.is_selected,
            "on_pip": self.on_pip,
            "callsign": self.callsign,
        }