from dataclasses import dataclass
from datetime import datetime

from .prediction import TaskType
from .asa_support_mode import AsaSupportMode

@dataclass(frozen=True, slots=True)
class TaskLabel:
    participant_id: int
    scenario_id: int
    asa_support_mode: AsaSupportMode
    start_time: datetime
    end_time: datetime
    task_type: TaskType
    callsigns: list[str]

    def __str__(self):
        return f"{self.participant_id:03d}_scenario_{self.scenario_id} (ASA SUPPORT MODE: {self.asa_support_mode.name}): " \
               f"TASK: {self.task_type.name}, START: {self.start_time.isoformat()}, END:{self.end_time.isoformat()}, " \
               f"CALLSIGNS: {','.join(self.callsigns)}"