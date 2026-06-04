from dataclasses import dataclass
from datetime import datetime

from .prediction import TaskType

@dataclass(frozen=True, slots=True)
class TaskGroundTruth:
    participant_id: int
    scenario_id: int
    timestamp: datetime
    true_task: TaskType | None
    true_callsigns: list[str]