import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models import AwareActionInteraction

def build_aware_action_interaction_cols(
    batch: list[AwareActionInteraction]
) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, callsign = [None] * size, [None] * size,
    uuid, status, details =  [None] * size, [None] * size, [None] * size
    suggestion_mode_enabled = [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        uuid[i], status[i], details[i], suggestion_mode_enabled[i] = row.uuid, row.status.name, row.details, row.suggestion_mode_enabled
        suggestion_mode_enabled[i] =  row.suggestion_mode_enabled

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "action_uuid": uuid,
        "action_status": status,
        "action_details": details,
        "suggestion_mode_enabled": suggestion_mode_enabled
    }

AWARE_ACTION_INTERACTION_DEFINITION: TableDefinition[AwareActionInteraction] = TableDefinition(
    name="aware_action_interaction",
    schema = pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("action_uuid", pa.string(), nullable=False),
        pa.field("action_status", CATEGORY_TYPE, nullable=False),
        pa.field("action_details", pa.string(), nullable=False),
        pa.field("suggestion_mode_enabled", pa.bool_(), nullable=False),
    ]),
    extractor=build_aware_action_interaction_cols,
)