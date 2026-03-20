import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL
from .....models.asd.mouse_position import MousePosition

def build_mouse_position_cols(batch: list[MousePosition]) -> dict[str, list[Any]]:
    size = len(batch)
   
    timestamp = [None] * size
    x, y = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i] = row.timestamp
        x[i], y[i] = row.pos.x, row.pos.y

    return {
        "timestamp": timestamp,
        "x": x,
        "y": y
    }

MOUSE_POSITION_DEFINITION: TableDefinition[MousePosition] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("x", pa.uint16(), nullable=False),
        pa.field("y", pa.uint16(), nullable=False),
    ]),
    extractor=build_mouse_position_cols,
)