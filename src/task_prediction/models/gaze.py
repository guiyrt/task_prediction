from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class GazePosition:
    timestamp: datetime
    pos: ScreenPosition | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.pos.x if self.pos is not None else None,
            "y": self.pos.y if self.pos is not None else None,
        }