from datetime import datetime
from dataclasses import dataclass

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