import pyarrow as pa
from typing import Any

from ....models import ActiveTaskListEntry
from .base import TableDefinition, TIMESTAMP_COL, CATEGORY_TYPE

def build_atl_cols(batch: list[ActiveTaskListEntry]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, participant_id, scenario_id = [None] * size, [None] * size, [None] * size
    task_type, callsign, rank = [None] * size, [None] * size, [None] * size
    
    for i, row in enumerate(batch):
        timestamp[i] = row.timestamp
        participant_id[i] = row.participant_id
        scenario_id[i] = row.scenario_id
        task_type[i] = row.task_type.name
        callsign[i] = row.callsign
        rank[i] = row.rank

    return {
        "participant_id": participant_id,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "task_type": task_type,
        "callsign": callsign,
        "rank": rank
    }

ATL_DEFINITION: TableDefinition[ActiveTaskListEntry] = TableDefinition(
    name="active_task_list",
    schema=pa.schema([
        pa.field("participant_id", pa.int8(), nullable=False),
        pa.field("scenario_id", pa.int8(), nullable=False),
        TIMESTAMP_COL,
        pa.field("task_type", CATEGORY_TYPE, nullable=False),
        pa.field("callsign", CATEGORY_TYPE, nullable=False),
        pa.field("rank", pa.uint16(), nullable=True),
    ]),
    extractor=build_atl_cols,
)