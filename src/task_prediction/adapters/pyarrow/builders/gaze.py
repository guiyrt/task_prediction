import pyarrow as pa
from typing import Any

from .base import TableDefinition, TIMESTAMP_COL
from ....models.gaze import GazePosition

def build_gaze_cols(batch: list[GazePosition]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp = [None] * size
    x, y = [None] * size, [None] * size
    
    for i, row in enumerate(batch):
        timestamp[i] = row.timestamp
        
        if row.pos:
            x[i], y[i] = row.pos.x, row.pos.y
            
    return {
        "timestamp": timestamp,
        "x": x,
        "y": y
    }

GAZE_DEFINITION: TableDefinition[GazePosition] = TableDefinition(
    name="gaze",
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("x", pa.uint16()),
        pa.field("y", pa.uint16()),
    ]),
    extractor=build_gaze_cols,
)