from datetime import datetime, timezone
import asyncio
import time
from typing import Sequence, Final

import nats

from .base import PredictionRunner
from ..adapters.proto.parsers import parse_asd_proto, parse_gaze_proto
from ..core.system import TaskPredictionSystem
from ..sinks import PredictionSink


class ServerRunner(PredictionRunner):
    """
    Orchestrates the Real-time prediction engine.
    """

    _LAG_WARNING_MS: Final[int] = 50
    _LAG_RESET_MS: Final[int] = 500
    
    def __init__(
        self,
        system: TaskPredictionSystem,
        sinks: Sequence[PredictionSink],
        nc: nats.NATS,
        sampling_interval_ms: int,
    ):
        super().__init__(system, sinks, sampling_interval_ms)
        self.nc = nc
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """The main entry point for the application."""
        self.logger.info("Starting Intent Engine...")
        self._running = True
        
        await self.start_sinks()

        # Create the tasks for our 3 concurrent loops
        self._tasks = [
            asyncio.create_task(self._gaze_loop(), name="Gaze_NATS"),
            asyncio.create_task(self._asd_loop(), name="ASD_NATS"),
            asyncio.create_task(self._predict_loop(), name="Predictor")
        ]
    
    async def stop(self) -> None:
            self._running = False
            
            for t in self._tasks:
                t.cancel()
            
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.close_sinks()

            self.logger.info("Shutdown complete.")

    async def _asd_loop(self) -> None:
        """Consumes JSON-encoded ASD Events."""
        try:
            # Subscribe using the shared client
            sub = await self.nc.subscribe("polaris.ASDEvent")

            async for msg in sub.messages:
                try:
                    asd_event = await asyncio.to_thread(
                        parse_asd_proto,
                        msg.data,
                        msg.header and msg.header.get("deflate") == "1"
                    )

                    if asd_event is not None:
                        self.system.ingest_asd(asd_event)
                
                except Exception as e:
                    self.logger.error("Failed to process single ASD message: %s", e)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"ASD NATS loop failed: {e}", exc_info=True)

    async def _gaze_loop(self):
        """Consumes High-Frequency Binary Gaze Events."""
        try:
            # Subscribe using the shared client
            sub = await self.nc.subscribe("intent.gaze")

            async for msg in sub.messages:
                try:
                    gaze_event = parse_gaze_proto(msg.data)

                    if gaze_event is not None:
                        self.system.ingest_gaze(gaze_event)

                except Exception as e:
                    self.logger.error("Failed to process single gaze message: %s", e)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Gaze NATS loop failed: {e}", exc_info=True)

    async def _predict_loop(self):
        """Predicts, broadcasts, and sleeps."""
        interval_sec = self.sampling_interval_ms / 1000.0        
        next_tick = time.monotonic()

        try:
            while self._running:
                try:
                    current_time = datetime.now(timezone.utc)
                    pred = await asyncio.to_thread(self.system.get_prediction, current_time)
                    await self.broadcast(pred)
                except Exception as e:
                    self.logger.error("Failed to create prediction: %s", e)
                
                next_tick += interval_sec
                sleep_duration = next_tick - time.monotonic()
                
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                else:
                    lag_ms = abs(sleep_duration) * 1000

                    if lag_ms > self._LAG_RESET_MS:
                        next_tick = time.monotonic()
                        self.logger.warning(f"System clock reset, was {lag_ms:.1f}ms behind schedule")
                    
                    elif lag_ms > self._LAG_WARNING_MS:
                        self.logger.warning(f"System lagging {lag_ms:.1f}ms behind schedule")

        except asyncio.CancelledError:
            pass