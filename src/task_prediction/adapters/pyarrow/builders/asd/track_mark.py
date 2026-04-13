import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CALLSIGN_COL, CATEGORY_TYPE
from .....models.asd.track_mark import TrackMark

def build_track_mark_cols(batch: list[TrackMark]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, callsign = [None] * size, [None] * size
    m_type, variant = [None] * size, [None] * size,
    scope, m_set = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], callsign[i] = row.timestamp, row.callsign
        m_type[i], variant[i] = row.type.name, row.variant.name
        scope[i], m_set[i] = row.scope, row.is_set

    return {
        "timestamp": timestamp,
        "callsign": callsign,
        "mark_type": m_type,
        "mark_variant": variant,
        "mark_scope": scope,
        "mark_set": m_set
    }

TRACK_MARK_DEFINITION: TableDefinition[TrackMark] = TableDefinition(
    name="track_mark",
    schema=pa.schema([
        TIMESTAMP_COL,
        CALLSIGN_COL,
        pa.field("mark_type", CATEGORY_TYPE, nullable=False),
        pa.field("mark_variant", CATEGORY_TYPE, nullable=False),
        pa.field("mark_scope", pa.string(), nullable=False),
        pa.field("mark_set", pa.bool_(), nullable=False),
    ]),
    extractor=build_track_mark_cols,
)