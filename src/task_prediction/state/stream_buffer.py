import numpy as np
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
from typing import Final
from dataclasses import dataclass
import logging
import time

from ..models import GazePosition, AsdEvent, TaskPredTelemetry, TaskPredStatus
from ..models.asd import *
from ..adapters.pyarrow.builders import (
    GAZE_DEFINITION,
    ASD_EVENT_DEFINITIONS,
)
from .snapshots import EventsSnapshot, MultiscaleSnapshots

logger = logging.getLogger(__name__)

_IGNORED_EVENTS: Final[tuple[type[AsdEvent], ...]] = (
    SepToolBase, SpeedVectorBase, RouteInteraction, TrackMark, KeyboardShortcut
)

@dataclass(frozen=True, slots=True)
class StreamBufferOut:
    status: TaskPredStatus
    telemetry: TaskPredTelemetry
    snapshots: MultiscaleSnapshots | None

class StreamBuffer:
    def __init__(
        self,
        short_sec: int = 5,
        mid_sec: int = 10,
        long_sec: int = 25,
        max_history_sec: int = 27,
        gaze_freq_hz: int = 120,
        min_gaze_availability_pct: float = 0.75,
        min_gaze_validity_pct: float = 0.5,
    ):
        # Window durations
        self.short_duration = timedelta(seconds=short_sec)
        self.mid_duration = timedelta(seconds=mid_sec)
        self.long_duration = timedelta(seconds=long_sec)
        self.long_sec = long_sec

        # Max duration for pruning
        self.max_history = timedelta(seconds=max_history_sec)

        # Health Configs
        self.expected_gaze_events = long_sec * gaze_freq_hz
        self.min_gaze_availability_pct = min_gaze_availability_pct
        self.min_gaze_validity_pct = min_gaze_validity_pct

        # State tracking
        self._start_time: float = time.monotonic()

        # _ts for bisecting, _events for the raw data
        self._gaze_dt: list[datetime] = []
        self._gaze_events: list[GazePosition] = []
        self._asd_dt: list[datetime] = []
        self._asd_events: list[AsdEvent] = []

    def ingest_gaze(self, event: GazePosition) -> None:
        # Maintain order, in case network scrambles packets
        if not self._gaze_dt or event.timestamp >= self._gaze_dt[-1]:
            self._gaze_dt.append(event.timestamp)
            self._gaze_events.append(event)
        else:
            logger.warning("Received gaze event out of order by %d ms", (self._gaze_dt[-1] - event.timestamp).total_seconds() * 1_000.0)
            idx = bisect_right(self._gaze_dt, event.timestamp)
            self._gaze_dt.insert(idx, event.timestamp)
            self._gaze_events.insert(idx, event)

    def ingest_asd(self, event: AsdEvent) -> None:
        # Maintain order, in case network scrambles packets
        if not self._asd_dt or event.timestamp >= self._asd_dt[-1]:
            self._asd_dt.append(event.timestamp)
            self._asd_events.append(event)
        else:
            logger.warning("Received ASD event out of order by %d ms", (self._asd_dt[-1] - event.timestamp).total_seconds() * 1_000.0)
            idx = bisect_right(self._asd_dt, event.timestamp)
            self._asd_dt.insert(idx, event.timestamp)
            self._asd_events.insert(idx, event)

    def _prune_events[T](self, cutoff: datetime, ts: list[datetime], events: list[T]) -> tuple[list[datetime], list[T]]:
        idx = bisect_left(ts, cutoff)
        return ts[idx:], events[idx:]

    def prune(self, current_time: datetime) -> None:
        """Prunes both buffers using binary search to find the cutoff index."""
        cutoff = current_time - self.max_history

        self._gaze_dt, self._gaze_events = self._prune_events(cutoff, self._gaze_dt, self._gaze_events)
        self._asd_dt, self._asd_events = self._prune_events(cutoff, self._asd_dt, self._asd_events)

    def get_windows(
        self, 
        anchor_time: datetime
    ) -> StreamBufferOut:
        
        # Anchor times for all windows
        t_long = anchor_time - self.long_duration
        t_mid = anchor_time - self.mid_duration
        t_short = anchor_time - self.short_duration

        # Extract LONG slices
        g_idx_long = bisect_left(self._gaze_dt, t_long)
        a_idx_long = bisect_left(self._asd_dt, t_long)

        gaze_batch = self._gaze_events[g_idx_long:]
        asd_batch = self._asd_events[a_idx_long:]

        # Calculate Telemetry
        asd_count = len(asd_batch)
        gaze_count = len(gaze_batch)
        
        availability_pct = gaze_count / self.expected_gaze_events
        
        valid_gaze_count = sum(1 for g in gaze_batch if g.pos is not None)
        validity_pct = (
            valid_gaze_count / gaze_count
            if gaze_count > 0
            else np.nan
        )

        telemetry = TaskPredTelemetry(
            gaze_availability_pct=availability_pct,
            gaze_validity_pct=validity_pct,
            asd_events_count=asd_count
        )

        # Strict Hierarchy Health Gates (Short-circuit on failure)
        status = TaskPredStatus.OK

        if time.monotonic() - self._start_time < self.long_sec:
            status = TaskPredStatus.WARMING_UP

        elif asd_count == 0:
            status = TaskPredStatus.NO_ASD_EVENTS

        elif availability_pct < self.min_gaze_availability_pct:
            status = TaskPredStatus.NO_GAZE

        elif validity_pct < self.min_gaze_validity_pct:
            status = TaskPredStatus.INVALID_GAZE

        # If status is not OK, return without extracting features
        if status is not TaskPredStatus.OK:
            return StreamBufferOut(status, telemetry, None)

        # Status is OK, get data windows
        gaze_df = GAZE_DEFINITION.build_df(gaze_batch)

        asd_batches: dict[type[AsdEvent], list[AsdEvent]] = defaultdict(list)
        for ev in asd_batch:
            if isinstance(ev, _IGNORED_EVENTS):
                continue
            asd_batches[get_base_asd_event_type(ev)].append(ev)

        def get_asd_df(cls_type: type[AsdEvent]) -> pd.DataFrame:
            batch = asd_batches.get(cls_type, [])
            return ASD_EVENT_DEFINITIONS[cls_type].build_df(batch)

        long_snapshot = EventsSnapshot(
            gaze=gaze_df,
            mouse_position=get_asd_df(MousePosition),
            track_screen_position=get_asd_df(TrackScreenPosition),
            track_label_position=get_asd_df(TrackLabelPosition),
            popup=get_asd_df(Popup),
            transfer=get_asd_df(Transfer),
            clearance=get_asd_df(Clearance),
            distance_measurement=get_asd_df(DistanceMeasurementBase)
        )

        # Cascading slicing for the smaller windows
        mid_snapshot = long_snapshot.slice_by_time(t_mid)
        short_snapshot = mid_snapshot.slice_by_time(t_short)

        windows = MultiscaleSnapshots(
            short=short_snapshot,
            mid=mid_snapshot,
            long=long_snapshot
        )

        return StreamBufferOut(status, telemetry, windows)