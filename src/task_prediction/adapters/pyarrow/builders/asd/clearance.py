import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models import Clearance

def build_clearance_cols(
    batch: list[Clearance]
) -> dict[str, list[Any]]:
    size = len(batch)

    timestamps, callsigns = [None] * size, [None] * size
    types, clearances = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamps[i], callsigns[i] = row.timestamp, row.callsign
        types[i], clearances[i] = row.type.name, row.clearance

    return {
        "timestamp": timestamps,
        "callsign": callsigns,
        "clearance_type": types,
        "clearance": clearances
    }

CLEARANCE_DEFINITION: TableDefinition[Clearance] = TableDefinition(
    schema = pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("clearance_type", CATEGORY_TYPE, nullable=False),
        pa.field("clearance", pa.string(), nullable=False),
    ]),
    extractor=build_clearance_cols,
)