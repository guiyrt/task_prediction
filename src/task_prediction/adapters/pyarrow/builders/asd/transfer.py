import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models.asd.transfer import Transfer

def build_transfer_cols(batch: list[Transfer]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, callsign = [None] * size, [None] * size
    transfer_type = [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        transfer_type[i] = row.type.name

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "transfer_type": transfer_type
    }

TRANSFER_DEFINITION: TableDefinition[Transfer] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("transfer_type", CATEGORY_TYPE, nullable=False),
    ]),
    extractor=build_transfer_cols,
)