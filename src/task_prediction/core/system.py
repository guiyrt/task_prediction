import logging
import time
from datetime import datetime
from dataclasses import replace

from ..models import GazePosition, AsdEvent, TaskPrediction, TaskPredStatus
from ..state.stream_buffer import StreamBuffer
from ..inference.predictor import TaskPredictor
from ..features.pipeline import extract_all_features

logger = logging.getLogger(__name__)

class TaskPredictionSystem:
    def __init__(self, predictor: TaskPredictor, buffer: StreamBuffer, force_stage_b: bool):
        self.predictor = predictor
        self.buffer = buffer

        self.force_stage_b = force_stage_b

    def ingest_gaze(self, event: GazePosition) -> None:
        """Buffers a domain-native gaze point."""
        self.buffer.ingest_gaze(event)

    def ingest_asd(self, event: AsdEvent) -> None:
        """Buffers a domain-native ASD event."""
        self.buffer.ingest_asd(event)

    def get_prediction(self, current_time: datetime) -> TaskPrediction:
        """
        Synchronously calculates the prediction for a given point in time.
        """
        # Start telemetry as None, add more data each step
        telemetry = None

        try:
            self.buffer.prune(current_time)
            buffer_out = self.buffer.get_windows(current_time)
            telemetry = buffer_out.telemetry

            # 3. Short-Circuit: If stream is unhealthy, return early
            if buffer_out.status is not TaskPredStatus.OK:
                self.predictor.reset_state()
                
                return TaskPrediction(
                    timestamp=current_time,
                    status=buffer_out.status,
                    telemetry=buffer_out.telemetry,
                    pred=None
                )
            

            # Compute features
            start_feat_time = time.perf_counter()
            features = extract_all_features(buffer_out.snapshots)
            feat_time = (time.perf_counter() - start_feat_time) * 1_000.0
            telemetry = replace(telemetry, feature_extraction_time_ms=feat_time)
            
            # ML Inference
            start_inf_time = time.perf_counter()
            pred = self.predictor.predict(
                features=features, 
                force_stage_b=self.force_stage_b
            )
            inf_time = (time.perf_counter() - start_inf_time) * 1_000.0
            telemetry = replace(telemetry, inference_time_ms=inf_time)
            
            return TaskPrediction(
                timestamp=current_time,
                telemetry=telemetry,
                status=buffer_out.status,
                pred=pred
            )

        except Exception as e:
            logger.error("Critical error during feature extraction or inference: %s", e, exc_info=True)
            self.predictor.reset_state()
            
            return TaskPrediction(
                timestamp=current_time,
                status=TaskPredStatus.ERROR,
                telemetry=telemetry,
                pred=None
            )