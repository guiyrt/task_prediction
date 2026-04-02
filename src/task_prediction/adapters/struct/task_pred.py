import struct
import io
from datetime import datetime, timezone

from ...models import TaskType, InferenceResult, TaskPredTelemetry, TaskPrediction, TaskPredStatus

# Header: Timestamp(d), Status(i), HasPred(?), GazeAvail(f), GazeVal(f), ASD(i), Feat(f), Infer(f)
_HEADER_STRUCT = struct.Struct("<di?ffiff")
# Inference: IsActive(?), ActiveProba(f), PredTask(i) + 14 TaskProbas(14f)
_INF_STRUCT = struct.Struct(f"<?fi{len(TaskType)}f")

def pred_to_struct(pred: TaskPrediction) -> bytes:
    buffer = io.BytesIO()
    
    # Pack Fixed Header
    has_pred = pred.pred is not None
    buffer.write(_HEADER_STRUCT.pack(
        pred.timestamp.timestamp(),
        pred.status.value,
        has_pred,
        pred.telemetry.gaze_availability_pct,
        pred.telemetry.gaze_validity_pct,
        pred.telemetry.asd_events_count,
        pred.telemetry.feature_extraction_time_ms,
        pred.telemetry.inference_time_ms,
    ))

    # Pack InferenceResult (if exists)
    if has_pred and pred.pred:
        # Flatten task probas into a list in Enum order
        probas = [pred.pred.task_probas.get(t, 0.0) for t in TaskType]
        pred_task_val = pred.pred.pred_task.value if pred.pred.pred_task is not None else -1
        buffer.write(
            _INF_STRUCT.pack(
                pred.pred.is_active,
                pred.pred.active_proba,
                pred_task_val,
                *probas
            )
        )

    return buffer.getvalue()

def pred_from_struct(data: bytes) -> "TaskPrediction":
    buffer = io.BytesIO(data)
    
    # Unpack Header
    header_data = buffer.read(_HEADER_STRUCT.size)
    ts, status_val, has_pred, g_av, g_val, asd, f_ms, i_ms = _HEADER_STRUCT.unpack(header_data)
    
    # Unpack Inference
    inference = None
    if has_pred:
        inf_data = buffer.read(_INF_STRUCT.size)
        unpacked_inf = _INF_STRUCT.unpack(inf_data)
        
        inference = InferenceResult(
            *unpacked_inf[0:2],
            TaskType(unpacked_inf[2]) if unpacked_inf[2] != -1 else None,
            {TaskType(i): val for i, val in enumerate(unpacked_inf[3:])}
        )

    return TaskPrediction(
        timestamp=datetime.fromtimestamp(ts, timezone.utc),
        status=TaskPredStatus(status_val),
        telemetry=TaskPredTelemetry(g_av, g_val, asd, f_ms, i_ms),
        pred=inference,
    )