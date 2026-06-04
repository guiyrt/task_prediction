from dataclasses import dataclass
from datetime import datetime

from ..models import ActiveTaskType

@dataclass(frozen=True, slots=True)
class ActiveTaskListEntry:
    timestamp: datetime
    participant_id: int
    scenario_id: int
    task_type: ActiveTaskType
    callsign: str
    rank: int