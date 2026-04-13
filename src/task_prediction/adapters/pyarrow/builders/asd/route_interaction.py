import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models.asd.route_interaction import RouteInteraction

def build_route_interaction_cols(batch: list[RouteInteraction]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, callsign = [None] * size, [None] * size
    a_type, value = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        a_type[i], value[i] = row.type.name, row.value

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "action_type": a_type,
        "value": value
    }

ROUTE_INTERACTION_DEFINITION: TableDefinition[RouteInteraction] = TableDefinition(
    name="route_interaction",
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("action_type", CATEGORY_TYPE, nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    extractor=build_route_interaction_cols,
)