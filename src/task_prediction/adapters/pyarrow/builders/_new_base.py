"""
This file represents a new way to define ParquetTables, one that doesn't require manual load and extract functions.
It is an implementation draft, that would replace the current base.py `TableDefinition`.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from enum import Enum
from typing import Any, Iterator, Type, get_type_hints, get_origin, get_args, Union, NamedTuple
from collections.abc import Mapping, Collection
from types import NoneType, UnionType
from pathlib import Path
from functools import cache
from dataclasses import is_dataclass, fields

class FieldInfo(NamedTuple):
    requires_transform: bool
    type_hint: type

class BaseParquetTable[T]:
    """Abstract base class for PyArrow Table definitions."""
    
    name: str
    schema: pa.Schema
    data_cls: Type[T]
    
    compression: str = "zstd"
    use_dictionary: list[str] | bool = False

    @classmethod
    def _requires_transform(cls, type_hint: type) -> bool:
        """
        Recursively checks if a type hint is an Enum or contains an Enum.
        Returns True for `TaskType`, `list[TaskType]`, `dict[str, TaskType]`, etc.
        Returns False for `int`, `list[str]`, `dict[str, float]`, etc.
        """
        if get_origin(type_hint) is not None:
            return any(cls._requires_transform(arg) for arg in get_args(type_hint))
        
        return issubclass(type_hint, Enum) \
            or is_dataclass(type_hint)

    @classmethod
    @cache
    def _get_fields_info(cls) -> dict[str, FieldInfo]:
        """
        Caches a boolean mask for each field indicating if it requires 
        expensive recursive serialization/deserialization.
        """
        return {
            field: FieldInfo(cls._requires_transform(typ), typ)
            for field, typ in get_type_hints(cls.data_cls).items()
        }

    # ==========================================
    # EXTRACTION (Serialization)
    # ==========================================
    @classmethod
    def _serialize_value(cls, val: Any) -> Any:
        """Recursively prepares Enum-containing objects for PyArrow."""
        match val:
            case Enum():
                return val.name
            case Mapping():
                return {cls._serialize_value(k): cls._serialize_value(v) for k, v in val.items()}
            case str() | bytes():
                return val
            case Collection():
                return [cls._serialize_value(item) for item in val]
            case _ if is_dataclass(val):
                return {f.name: cls._serialize_value(getattr(val, f.name)) for f in fields(val)}
            case _:
                return val

    @classmethod
    def extract(cls, batch: list[T]) -> dict[str, list[Any]]:
        size = len(batch)
        cols = {f.name: [None] * size for f in cls.schema}
        
        fields_info = cls._get_fields_info()
        
        for i, row_dto in enumerate(batch):
            for field_name in cols.keys():
                val = getattr(row_dto, field_name)

                cols[field_name][i] = (
                    cls._serialize_value(val)
                    if fields_info[field_name].requires_transform
                    else val
                )
                    
        return cols

    @classmethod
    def _deserialize_value(cls, val: Any, type_hint: type) -> Any:
        """Recursively re-hydrates PyArrow primitives back to Enums."""
        if val is None and isinstance(val, type_hint):
            return None
        
        origin = get_origin(type_hint)
        
        if origin is None:
            if issubclass(type_hint, Enum) and isinstance(val, str):
                return type_hint[val]
            elif is_dataclass(type_hint) and isinstance(val, dict[str, Any]):
                return origin(**{
                    f.name: cls._deserialize_value(val[f.name], f.type)
                    for f in fields(type_hint)
                })
            elif isinstance(val, type_hint):
                return val
            else:
                raise ValueError(f"Expected type {type_hint}, got {type(val)}.")
        
        elif origin in (Union, UnionType):
            args = [a for a in get_args(type_hint) if a is not NoneType]
            
            if len(args) > 1:
                raise ValueError(
                    f"Ambiguous deserialization for type hinting {args}."
                    "If type cannot be more precise, consider overloading 'load' method."
                )

            return cls._deserialize_value(val, args[0])
        
        elif issubclass(origin, Mapping) and isinstance(val, Mapping):
            k_type, v_type = get_args(type_hint)
            
            return {
                cls._deserialize_value(k, k_type): cls._deserialize_value(v, v_type)
                for k, v in val.items()
            }
            
        elif issubclass(origin, Collection) and isinstance(val, Collection):
            return origin(cls._deserialize_value(item, get_args(type_hint)[0]) for item in val)
        
        else:
            raise ValueError(f"Got unexpected origin type '{origin}' for value of '{type(val)}'")

    @classmethod
    def load(cls, row_dict: dict[str, Any]) -> T:
        fields_info = cls._get_fields_info()
        
        for k, v in row_dict.items():
            if fields_info[k].requires_transform:
                row_dict[k] = cls._deserialize_value(v, fields_info[k].type_hint)

        return cls.data_cls(**row_dict)

    @classmethod
    def build_table(cls, batch: list[T]) -> pa.Table:
        cols_dict = cls.extract(batch)
        return pa.Table.from_arrays(
            [pa.array(cols_dict[f.name], type=f.type) for f in cls.schema],
            schema=cls.schema,
        )
    
    @classmethod
    def build_df(cls, batch: list[T], index_col: str | None = None, sort: bool = False) -> pd.DataFrame:
        df = cls.build_table(batch).to_pandas(types_mapper=pd.ArrowDtype)
        
        if index_col is not None:
            df.set_index(index_col, drop=False, inplace=True)
        
        if sort:
            df.sort_index(kind="stable", inplace=True)
        
        return df

    @classmethod
    def iter_parquet(cls, path: Path, batch_size: int = 10_000) -> Iterator[T]:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            for row_dict in batch.to_pylist(maps_as_pydicts='strict'):
                yield cls.load(row_dict)




# TODO: update dataclasses to use pydantic dataclasses
#from pydantic.dataclasses import dataclass
#from pydantic import ConfigDict
#@dataclass(frozen=True, slots=True, config=ConfigDict(strict=True))
"""
import pyarrow as pa
from .base import BaseParquetTable, CATEGORY_TYPE, TIMESTAMP_COL
from ....models import TaskPrediction

class TaskPredictionTable(BaseParquetTable[TaskPrediction]):
    name = "task_predictions"
    data_cls = TaskPrediction
    
    schema = pa.schema([
        TIMESTAMP_COL,
        pa.field("status", CATEGORY_TYPE, nullable=False),
        
        # Define telemetry as a nested Struct mirroring TaskPredTelemetry
        pa.field("telemetry", pa.struct([
            pa.field("gaze_availability_pct", pa.float32()),
            pa.field("gaze_validity_pct", pa.float32()),
            pa.field("asd_events_count", pa.uint32()),
            pa.field("feature_extraction_time_ms", pa.float32()),
            pa.field("inference_time_ms", pa.float32()),
        ]), nullable=False),
        
        # Define pred as a nested Struct mirroring InferenceResult
        pa.field("pred", pa.struct([
            pa.field("is_active", pa.bool_()),
            pa.field("active_proba", pa.float32()),
            pa.field("pred_task", CATEGORY_TYPE, nullable=True),
            # task_probas is exactly what PyArrow's Map is designed for
            pa.field("task_probas", pa.map_(pa.string(), pa.float32())), 
        ]), nullable=True) # Nullable because it is None if status != OK
    ])
"""