from datetime import datetime
from dataclasses import dataclass

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class TrackScreenPosition:
    timestamp: datetime
    callsign: str

    pos: ScreenPosition
    is_visible: bool