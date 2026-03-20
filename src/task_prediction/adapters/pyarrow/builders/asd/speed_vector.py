import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CATEGORY_TYPE
from .....models.asd.speed_vector import (
    SpeedVectorBase, SpeedVectorModeUpdated, SpeedVectorVisibility, SpeedVectorLength
)

def build_speed_vector_cols(batch: list[SpeedVectorBase]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, callsign, = [None] * size, [None] * size
    mode, visible, length = [None] * size, [None] * size, [None] * size

    for i, ev in enumerate(batch):
        timestamp[i] = ev.timestamp

        if isinstance(ev, SpeedVectorModeUpdated):
            mode[i] = ev.mode.name
        elif isinstance(ev, SpeedVectorVisibility):
            visible[i], callsign[i] = ev.visible, ev.callsign
        elif isinstance(ev, SpeedVectorLength):
            length[i] = ev.length_seconds

    return {
        "timestamp": timestamp,
        "mode_updated__mode": mode,
        "visibility__visible": visible,
        "visibility__callsign": callsign,
        "length__length_seconds": length
    }

SPEED_VECTOR_DEFINITION: TableDefinition[SpeedVectorBase] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("mode_updated__mode", CATEGORY_TYPE),
        pa.field("visibility__visible", pa.bool_()),
        pa.field("visibility__callsign", pa.string()),
        pa.field("length__length_seconds", pa.int32()),
    ]),
    extractor=build_speed_vector_cols,
)