import logging
import nats
from pathlib import Path

from ..configs import AppSettings
from ..state.stream_buffer import StreamBuffer
from ..inference.predictor import TaskPredictor
from .system import TaskPredictionSystem
from ..sinks import ParquetSink, NATSSink, PredictionSink, TerminalSink

logger = logging.getLogger(__name__)

def create_system(settings: AppSettings) -> TaskPredictionSystem:
    predictor = TaskPredictor(
        model_dir=settings.model.model_dir,
        alpha_smooth=settings.model.alpha_smooth,
        always_validate_input=settings.model.always_validate_input
    )
    
    buffer = StreamBuffer(
        short_sec=settings.data.short_sec,
        mid_sec=settings.data.mid_sec,
        long_sec=settings.data.long_sec,
        max_history_sec=settings.data.max_history_sec,
        gaze_freq_hz=settings.data.gaze_freq_hz,
        min_gaze_availability_pct=settings.data.min_availability_pct,
        min_gaze_validity_pct=settings.data.min_validity_pct
    )
    
    return TaskPredictionSystem(
        predictor=predictor, 
        buffer=buffer,
        force_stage_b=settings.model.force_stage_b
    )

def create_sinks(
    settings: AppSettings,
    nc: nats.NATS | None = None,
    output_dir: Path | None = None,
) -> list[PredictionSink]:
    sinks = []
    
    if settings.nats.enabled:
        if nc is not None:
            sinks.append(
                NATSSink(
                    nc=nc,
                    subject=settings.nats.subject
                )
            )
        else:
            ValueError("NATS sink enabled, but no NATS instance passed to factory.")
    
    if settings.parquet.enabled:
        sinks.append(
            ParquetSink(
                output_dir=output_dir or settings.data_dir,
                drop_when_full=settings.parquet.drop_when_full,
                max_buffer_size=settings.parquet.max_buffer_size,
                queue_size=settings.parquet.queue_size
            )
        )
    
    if settings.terminal.enabled:
        sinks.append(
            TerminalSink(
                refresh_per_sec=settings.terminal.refresh_per_sec
            )
        )
        
    return sinks