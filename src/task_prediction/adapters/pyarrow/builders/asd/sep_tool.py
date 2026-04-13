import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL, CATEGORY_TYPE
from .....models.asd.sep_tool import SepToolBase, SepToolOpened, SepToolConnected, SepToolClosed

def build_sep_tool_cols(batch: list[SepToolBase]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, s_type, m_id = [None] * size, [None] * size, [None] * size
    op_cs, con_cs1, con_cs2, closed = [None] * size, [None] * size, [None] * size, [None] * size

    for i, ev in enumerate(batch):
        timestamp[i], s_type[i], m_id[i] = ev.timestamp, ev.type.name, ev.measurement_id
        
        if isinstance(ev, SepToolOpened):
            op_cs[i] = ev.callsign
        elif isinstance(ev, SepToolConnected):
            con_cs1[i], con_cs2[i] = ev.callsign_1, ev.callsign_2
        elif isinstance(ev, SepToolClosed):
            closed[i] = ev.closed

    return {
        "timestamp": timestamp,
        "type": s_type,
        "measurement_id": m_id,
        "opened__callsign": op_cs,
        "connected__callsign_1": con_cs1,
        "connected__callsign_2": con_cs2,
        "closed": closed
    }

SEP_TOOL_DEFINITION: TableDefinition[SepToolBase] = TableDefinition(
    name="sep_tool",
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("type", CATEGORY_TYPE, nullable=False),
        pa.field("measurement_id", pa.uint8(), nullable=False),
        pa.field("opened__callsign", pa.string()),
        pa.field("connected__callsign_1", pa.string()),
        pa.field("connected__callsign_2", pa.string()),
        pa.field("closed", pa.bool_()),
    ]),
    extractor=build_sep_tool_cols,
)