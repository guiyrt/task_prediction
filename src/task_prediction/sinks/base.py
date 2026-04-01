from abc import ABC, abstractmethod
from ..models import TaskPrediction

class PredictionSink(ABC):
    async def start(self) -> None:
        """Initialize sink resources."""
        pass

    @abstractmethod
    async def send(self, data: TaskPrediction) -> None:
        """
        Push data to the sink.
        Must be non-blocking to the caller.
        """
        pass

    async def close(self) -> None:
        """Clean up sink resources."""
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()