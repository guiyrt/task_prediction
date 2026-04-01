import logging
import nats

from .base import PredictionSink
from ..models import TaskPrediction

from aware_protos.zhaw.protobuf import task_prediction_pb2

logger = logging.getLogger(__name__)

class NATSSink(PredictionSink):
    def __init__(
        self,
        nc: nats.NATS,
        subject: str = "intent.task_prediction"
    ):
        self.nc = nc
        self.subject = subject
        
        # Pre-instantiate Protobuf object for memory reuse
        self._proto = task_prediction_pb2.TaskPrediction()

    async def send(self, pred: TaskPrediction) -> None:
        try:
            p = self._proto
            p.Clear()

            p.timestamp.FromDatetime(pred.timestamp)
            p.status = pred.status.value + 1 # 0 is UNSPECIFIED in proto, so we add one

            inference_result = pred.pred

            if inference_result is not None:
                p.is_active = inference_result.is_active
                p.active_proba = inference_result.active_proba

                if inference_result.pred_task is not None:
                    p.pred_task = inference_result.pred_task.value + 1 # 0 is UNSPECIFIED in proto, so we add one

                if inference_result.task_probas:
                    # Convert the dict applying the same +1 offset, then use .update() to inject it into the Protobuf map
                    mapped_probas = {
                        task_enum.value + 1: proba
                        for task_enum, proba in inference_result.task_probas.items()
                    }
                    p.task_probas.update(mapped_probas)
            
            # Serialize and publish
            await self.nc.publish(self.subject, p.SerializeToString())
            
        except Exception as e:
            logger.error(f"Failed to publish prediction to NATS: {e}")