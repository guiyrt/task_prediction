from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class MousePosition:
    timestamp: datetime
    pos: ScreenPosition

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.pos.x,
            "y": self.pos.y,
        }