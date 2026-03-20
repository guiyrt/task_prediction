import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL
from .....models.asd.keyboard_shortcut import KeyboardShortcut

def build_keyboard_shortcut_cols(batch: list[KeyboardShortcut]) -> dict[str, list[Any]]:
    size = len(batch)

    timestamp, name = [None] * size, [None] * size

    for i, row in enumerate(batch):
        timestamp[i], name[i] = row.timestamp, row.name

    return {
        "timestamp": timestamp,
        "action_name": name
    }

KEYBOARD_SHORTCUT_DEFINITION: TableDefinition[KeyboardShortcut] = TableDefinition(
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("action_name", pa.string(), nullable=False),
    ]),
    extractor=build_keyboard_shortcut_cols,
)