import asyncio
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, UTC
from pathlib import Path
from typing import Final

from .base import PredictionSink
from ..models import TaskPrediction
from ..utils.logging import ThrottledLogger
from ..utils.end_token import EndToken, _END

logger = logging.getLogger(__name__)

class ParquetSink(PredictionSink):
    # Dictionary encoding for strings that repeat frequently (Enums)
    _ENUM_TYPE = pa.dictionary(pa.int32(), pa.string())

    _SCHEMA: Final[pa.Schema] = pa.schema([
        ("timestamp", pa.timestamp('ms', tz='UTC')),

        # Data Quality & Telemetry
        ("status", _ENUM_TYPE),
        ("gaze_availability_pct", pa.float32()),
        ("gaze_validity_pct", pa.float32()),
        ("asd_events_count", pa.uint32()),
        ("feature_extraction_time_ms", pa.float32()),
        ("inference_time_ms", pa.float32()),
        
        # Stage 1 (Active or Idle)
        ("is_active", pa.bool_()),
        ("active_proba", pa.float32()),
        
        # Stage 2 (Task prediction)
        ("pred_task", _ENUM_TYPE),
        ("task_probas", pa.map_(_ENUM_TYPE, pa.float32())),
    ])

    def __init__(
        self,
        output_dir: Path,
        drop_when_full: bool,
        max_buffer_size: int,
        queue_size: int,
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.drop_when_full = drop_when_full
        
        # Setup file
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"task_prediction__{datetime.now(UTC):%Y%m%d_%H%M%S}.parquet"
        
        # Internal state
        self._queue: asyncio.Queue[TaskPrediction | EndToken] = asyncio.Queue(maxsize=queue_size)
        self._worker_task: asyncio.Task | None = None
        self._writer: pq.ParquetWriter | None = None

        # Logging and stats
        self._total_rows = 0
        self._total_preds_dropped = 0
        self._drop_logger = ThrottledLogger(logger, interval_sec=1)

        logger.info(f"ParquetSink ready. Writing to: {self.output_path}")
    
    async def send(self, pred: TaskPrediction) -> None:
        # Fail fast, zero latency added (drop data)
        # For running in production environment
        if self.drop_when_full:
            try:
                self._queue.put_nowait(pred)
            except asyncio.QueueFull:
                self._total_preds_dropped += 1
                self._drop_logger.warning("Queue is full, dropping prediction.")

        else:
            await self._queue.put(pred)
    
    async def _worker(self) -> None:
        """
        High-throughput, constant-frequency worker.
        Relies on the strictly paced producer to guarantee flush intervals.
        """
        buffer: list[TaskPrediction] = []
        queue = self._queue
        max_buf = self.max_buffer_size

        while True:
            # 1. Wait for the next item
            item = await queue.get()
            if item is _END:
                break
            buffer.append(item)

            # 2. Greedy Drain (Instantly grab anything else sitting in the queue)
            while len(buffer) < max_buf:
                try:
                    next_item = queue.get_nowait()
                    if next_item is _END:
                        await self._flush(buffer)
                        return 
                    buffer.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # 3. Buffer Full Check -> Flush to disk
            if len(buffer) >= max_buf:
                await self._flush(buffer)
                buffer.clear()
        
        # 4. Final flush on graceful shutdown
        if buffer:
            await self._flush(buffer)

    async def _flush(self, batch: list[TaskPrediction]) -> None:
        """Offloads the expensive conversion and IO to a thread."""
        if not batch:
            return
        
        try:
            self._total_rows += await asyncio.to_thread(self._write_sync, batch)
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            self._total_preds_dropped += len(batch)

    def _write_sync(self, batch: list[TaskPrediction]) -> int:
        """Writes batch to Parquet using columnar pre-allocation."""
        size = len(batch)

        # Pre-allocate columns
        timestamps = [None] * size
        status = [None] * size
        gaze_avail = [None] * size
        gaze_valid = [None] * size
        asd_count = [None] * size
        feat_time = [None] * size
        inf_time = [None] * size
        is_active = [None] * size
        active_proba = [None] * size
        pred_task = [None] * size
        task_probas = [None] * size

        for i, p in enumerate(batch):
            timestamps[i] = p.timestamp
            
            # Extract Enum name (e.g. "OK", "NO_GAZE")
            status[i] = p.status.name
            
            # Telemetry
            t = p.telemetry
            gaze_avail[i] = t.gaze_availability_pct
            gaze_valid[i] = t.gaze_validity_pct
            asd_count[i] = t.asd_events_count
            feat_time[i] = t.feature_extraction_time_ms
            inf_time[i] = t.inference_time_ms

            # Inference Result
            inference = p.pred
            if inference is not None:
                is_active[i] = inference.is_active
                active_proba[i] = inference.active_proba
                
                if inference.pred_task is not None:
                    pred_task[i] = inference.pred_task.name
                
                if inference.task_probas:
                    # Convert map keys to string names
                    task_probas[i] = [(task.name, prob) for task, prob in inference.task_probas.items()]

        table = pa.Table.from_arrays(
            [
                pa.array(timestamps, type=self._SCHEMA[0].type),
                pa.array(status, type=self._SCHEMA[1].type),
                pa.array(gaze_avail, type=self._SCHEMA[2].type),
                pa.array(gaze_valid, type=self._SCHEMA[3].type),
                pa.array(asd_count, type=self._SCHEMA[4].type),
                pa.array(feat_time, type=self._SCHEMA[5].type),
                pa.array(inf_time, type=self._SCHEMA[6].type),
                pa.array(is_active, type=self._SCHEMA[7].type),
                pa.array(active_proba, type=self._SCHEMA[8].type),
                pa.array(pred_task, type=self._SCHEMA[9].type),
                pa.array(task_probas, type=self._SCHEMA[10].type),
            ],
            schema=self._SCHEMA
        )
        
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path, 
                schema=self._SCHEMA, 
                compression="zstd",
            )
        
        self._writer.write_table(table)
        return size

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        if self._worker_task:
            await self._queue.put(_END)
            await self._worker_task
            self._worker_task = None
        
        if self._writer:
            await asyncio.to_thread(self._writer.close)
            self._writer = None
            logger.info(f"Parquet closed. Written: {self._total_rows:,}, Dropped: {self._total_preds_dropped:,}")