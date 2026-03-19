from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum

class TaskType(IntEnum):
    AIRCRAFT_REQUEST = 0
    CONFLICT_RESOLUTION = 1
    ENTRY_CONDITIONS = 2
    ENTRY_CONFLICT_RESOLUTION = 3
    ENTRY_COORDINATION = 4
    EXIT_CONDITIONS = 5
    EXIT_CONFLICT_RESOLUTION = 6
    EXIT_COORDINATION = 7
    NON_CONFORMANCE_RESOLUTION = 8
    QUALITY_OF_SERVICE = 9
    RETURN_TO_ROUTE = 10
    TRANSFER = 11
    ZONE_CONFLICT = 12

@dataclass(frozen=True, slots=True)
class TaskPrediction:
    timestamp: datetime

    is_active: bool
    active_proba: float

    pred_task: TaskType | None = None
    task_probas: dict[TaskType, float]