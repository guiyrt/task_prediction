from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum

class TaskType(IntEnum):
    AIRCRAFT_REQUEST = 0
    ASSUME = 1
    CONFLICT_RESOLUTION = 2
    ENTRY_CONDITIONS = 3
    ENTRY_CONFLICT_RESOLUTION = 4
    ENTRY_COORDINATION = 5
    EXIT_CONDITIONS = 6
    EXIT_CONFLICT_RESOLUTION = 7
    EXIT_COORDINATION = 8
    NON_CONFORMANCE_RESOLUTION = 9
    QUALITY_OF_SERVICE = 10
    RETURN_TO_ROUTE = 11
    TRANSFER = 12
    ZONE_CONFLICT = 13

class TaskPredStatus(IntEnum):
    WARMING_UP = 0
    NO_ASD_EVENTS = 1
    NO_GAZE = 2
    INVALID_GAZE = 3
    ERROR = 4
    OK = 5

@dataclass(frozen=True, slots=True)
class TaskPredTelemetry:
    """Diagnostic data accompanying every prediction."""
    # Events
    gaze_availability_pct: float
    gaze_validity_pct: float
    asd_events_count: int
    
    # Processing time
    feature_extraction_time_ms: float = 0.0
    inference_time_ms: float = 0.0

@dataclass(frozen=True, slots=True)
class InferenceResult:
    is_active: bool
    active_proba: float

    pred_task: TaskType | None = None
    task_probas: dict[TaskType, float] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TaskPrediction:
    timestamp: datetime
    telemetry: TaskPredTelemetry
    status: TaskPredStatus
    pred: InferenceResult | None