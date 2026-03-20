from datetime import datetime
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class KeyboardShortcut:
    timestamp: datetime
    name: str