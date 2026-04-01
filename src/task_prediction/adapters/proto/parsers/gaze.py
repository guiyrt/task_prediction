import logging
from datetime import timezone

from aware_protos.zhaw.protobuf import gaze_pb2

from ....models import GazePosition, ScreenPosition

logger = logging.getLogger(__name__)

def parse_gaze_proto(payload: bytes) -> GazePosition | None:
    try:
        event = gaze_pb2.GazeScreenPosition()
        event.ParseFromString(payload)

        return GazePosition(
            timestamp=event.timestamp.ToDatetime(timezone.utc),
            pos=(
                ScreenPosition(event.x, event.y)
                if event.is_valid
                else None
            )
        )
    
    except Exception as e:
        logger.critical("Failed to parse gaze protobuf event: %s.", e)
        return None