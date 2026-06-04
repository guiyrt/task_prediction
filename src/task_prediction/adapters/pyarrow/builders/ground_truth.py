import pyarrow as pa
from typing import Any

from ....models import TaskGroundTruth
from .base import TableDefinition, TIMESTAMP_COL, CATEGORY_TYPE

def build_task_gt_cols(batch: list[TaskGroundTruth]) -> dict[str, list[Any]]:
    size = len(batch)
    
    participant_id, scenario_id = [None] * size, [None] * size
    timestamp, true_task, true_callsigns = [None] * size, [None] * size, [[] for _ in range(size)]
    
    for i, row in enumerate(batch):
        participant_id[i] = row.participant_id
        scenario_id[i] = row.scenario_id
        timestamp[i] = row.timestamp
        true_task[i] = row.true_task.name if row.true_task is not None else None
        
        if row.true_callsigns:
            true_callsigns[i] = row.true_callsigns

    return {
        "participant_id": participant_id,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "true_task": true_task,
        "true_callsigns": true_callsigns
    }

TASK_GT_DEFINITION: TableDefinition[TaskGroundTruth] = TableDefinition(
    name="ground_truth",
    schema=pa.schema([
        pa.field("participant_id", pa.int8(), nullable=False),
        pa.field("scenario_id", pa.int8(), nullable=False),
        TIMESTAMP_COL,
        pa.field("true_task", CATEGORY_TYPE, nullable=True),
        pa.field("true_callsigns", pa.list_(pa.string()), nullable=False)
    ]),
    extractor=build_task_gt_cols,
)