from datetime import datetime
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class KeyboardShortcut:
    timestamp: datetime
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action_name": self.name,
        }