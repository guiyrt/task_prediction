import pyarrow as pa
from typing import Callable, Any, Final
from dataclasses import dataclass

TIMESTAMP_COL: Final[pa.field] = pa.field("timestamp", pa.timestamp('ms', tz='UTC'), nullable=False)
CALLSIGN_COL: Final[pa.field] = pa.field("callsign", pa.string(), nullable=False)
CATEGORY_TYPE: Final[pa.DataType] = pa.dictionary(pa.uint8(), pa.string())

@dataclass(frozen=True, slots=True)
class TableDefinition[T]:
    """Declarative configuration and execution binding for a specific event type."""
    schema: pa.Schema
    extractor: Callable[[list[T]], dict[str, list[Any]]]
    compression: str = "zstd"
    use_dictionary: list[str] | bool = False

    @property
    def parquet_kwargs(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "compression": self.compression,
            "use_dictionary": self.use_dictionary
        }

    def build_empty(self) -> pa.Table:
        return pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in self.schema], 
            schema=self.schema
        )

    def build_table(self, batch: list[T]) -> pa.Table:
        """Executes the pure function and builds the table safely by key."""
        if not batch:
            return self.build_empty()
        
        cols_dict = self.extractor(batch)
        
        # Build the table strictly relying on the schema
        return pa.Table.from_arrays(
            [
                pa.array(cols_dict[f.name], type=f.type)
                for f in self.schema
            ],
            schema=self.schema,
        )