import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL
from .....models.asd.track_screen_position import TrackScreenPosition

def build_track_screen_cols(batch: list[TrackScreenPosition]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, callsign = [None] * size, [None] * size
    x, y, is_visible = [None] * size, [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        x[i], y[i], is_visible[i] = row.pos.x, row.pos.y, row.is_visible

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "x": x,
        "y": y,
        "is_visible": is_visible
    }

TRACK_SCREEN_POSITION_DEFINITION: TableDefinition[TrackScreenPosition] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("x", pa.int16(), nullable=False),
        pa.field("y", pa.int16(), nullable=False),
        pa.field("is_visible", pa.bool_(), nullable=False),
    ]),
    extractor=build_track_screen_cols,
)