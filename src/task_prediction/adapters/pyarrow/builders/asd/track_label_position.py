import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL
from .....models.asd.track_label_position import TrackLabelPosition

def build_track_label_cols(batch: list[TrackLabelPosition]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, callsign = [None] * size, [None] * size
    x, y, w, h = [None] * size, [None] * size, [None] * size, [None] * size
    vis, hov, sel, pip = [None] * size, [None] * size, [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        x[i], y[i] = row.top_left.x, row.top_left.y
        w[i], h[i] = row.width, row.height
        vis[i], hov[i], sel[i], pip[i] = row.is_visible, row.is_hovered, row.is_selected, row.on_pip

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "is_visible": vis,
        "is_hovered": hov,
        "is_selected": sel,
        "on_pip": pip
    }

TRACK_LABEL_POSITION_DEFINITION: TableDefinition[TrackLabelPosition] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("x", pa.int16(), nullable=False),
        pa.field("y", pa.int16(), nullable=False),
        pa.field("width", pa.uint16(), nullable=False),
        pa.field("height", pa.uint16(), nullable=False),
        pa.field("is_visible", pa.bool_(), nullable=False),
        pa.field("is_hovered", pa.bool_(), nullable=False),
        pa.field("is_selected", pa.bool_(), nullable=False),
        pa.field("on_pip", pa.bool_(), nullable=False),
    ]),
    extractor=build_track_label_cols,
)