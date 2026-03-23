import pyarrow as pa
from typing import Any

from .base import TableDefinition, TIMESTAMP_COL, CATEGORY_TYPE
from ....models import AsdEvent

def build_asd_activity_timeline_cols(batch: list[AsdEvent]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, event_type = [None] * size, [None] * size

    for i, ev in enumerate(batch):
        timestamp[i] = ev.timestamp
        event_type[i] = ev.__class__.__name__

    return {
        "timestamp": timestamp,
        "event_type": event_type
    }

ASD_ACTIVITY_TIMELINE_DEFINITION: TableDefinition[AsdEvent] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("event_type", CATEGORY_TYPE, nullable=False),
    ]),
    extractor=build_asd_activity_timeline_cols,
)