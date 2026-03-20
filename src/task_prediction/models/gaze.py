from datetime import datetime
from dataclasses import dataclass

from .screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class GazePosition:
    timestamp: datetime
    pos: ScreenPosition | None