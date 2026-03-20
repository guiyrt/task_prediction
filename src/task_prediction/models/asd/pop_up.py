from datetime import datetime
from dataclasses import dataclass
from enum import StrEnum

class PopupMenu(StrEnum):
    CFLMenu = "CFLMenu"
    HeadingMenu = "HeadingMenu"
    WaypointMenu = "WaypointMenu"
    ASPMenu = "ASPMenu"

@dataclass(frozen=True, slots=True)
class Popup:
    timestamp: datetime
    callsign: str

    name: PopupMenu
    opened: bool