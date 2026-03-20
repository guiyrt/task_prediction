import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models.asd.pop_up import Popup

def build_popup_cols(batch: list[Popup]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, callsign = [None] * size, [None] * size
    name, opened = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        name[i], opened[i] = row.name.name, row.opened

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "name": name,
        "opened": opened
    }

POPUP_DEFINITION: TableDefinition[Popup] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("name", CATEGORY_TYPE, nullable=False),
        pa.field("opened", pa.bool_(), nullable=False),
    ]),
    extractor=build_popup_cols,
)