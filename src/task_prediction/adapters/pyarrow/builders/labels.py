import pyarrow as pa
from typing import Any

from ....models import TaskLabel
from .base import TableDefinition, CATEGORY_TYPE

def build_task_label_cols(batch: list[TaskLabel]) -> dict[str, list[Any]]:
    size = len(batch)
    
    # Pre-allocate lists for performance
    participant_id, scenario_id, asa_support_mode = [None] * size, [None] * size, [None] * size
    start_time, end_time = [None] * size, [None] * size
    task_type, callsigns = [None] * size, [None] * size
    
    for i, row in enumerate(batch):
        participant_id[i] = row.participant_id
        scenario_id[i] = row.scenario_id
        asa_support_mode[i] = row.asa_support_mode.name
        start_time[i] = row.start_time
        end_time[i] = row.end_time
        task_type[i] = row.task_type.name
        callsigns[i] = row.callsigns

    return {
        "participant_id": participant_id,
        "scenario_id": scenario_id,
        "asa_support_mode": asa_support_mode,
        "start_time": start_time,
        "end_time": end_time,
        "task_type": task_type,
        "callsigns": callsigns,
    }

TASK_LABEL_DEFINITION: TableDefinition[TaskLabel] = TableDefinition(
    name="task_labels",
    schema=pa.schema([
        pa.field("participant_id", pa.int8(), nullable=False),
        pa.field("scenario_id", pa.int8(), nullable=False),
        pa.field("asa_support_mode", CATEGORY_TYPE, nullable=False),
        pa.field("start_time", pa.timestamp('ms', tz='UTC'), nullable=False),
        pa.field("end_time", pa.timestamp('ms', tz='UTC'), nullable=False),
        pa.field("task_type", CATEGORY_TYPE, nullable=False),
        pa.field("callsigns", pa.list_(pa.string()), nullable=False),
    ]),
    extractor=build_task_label_cols,
)