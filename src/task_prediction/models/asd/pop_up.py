from datetime import datetime
from dataclasses import dataclass
from typing import Any
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "name": self.name.value,
            "opened": self.opened,
            "callsign": self.callsign,
        }