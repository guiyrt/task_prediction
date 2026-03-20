from datetime import datetime
from dataclasses import dataclass

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class MousePosition:
    timestamp: datetime
    pos: ScreenPosition