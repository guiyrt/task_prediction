from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

from ..models import GazePosition, AsdEvent
from ..models.asd import (
    MousePosition, TrackLabelPosition, TrackScreenPosition, 
    Popup, Clearance, Transfer
)
from ..adapters.pyarrow.builders import (
    GAZE_DEFINITION,
    ASD_EVENT_DEFINITIONS,
    ASD_ACTIVITY_TIMELINE_DEFINITION
)
from .snapshots import EventsSnapshot, MultiscaleSnapshots


class StreamBuffer:
    def __init__(
        self,
        short_sec: int = 5,
        mid_sec: int = 10,
        long_sec: int = 25,
        max_history_sec: int = 27,
    ):
        # Window durations
        self.short_duration = timedelta(seconds=short_sec)
        self.mid_duration = timedelta(seconds=mid_sec)
        self.long_duration = timedelta(seconds=long_sec)

        # Max duration for pruning
        self.max_history = timedelta(seconds=max_history_sec)

        # Parallel lists: _ts for bisecting, _events for the raw data
        self._gaze_ts: list[datetime] = []
        self._gaze_events: list[GazePosition] = []
        
        self._asd_ts: list[datetime] = []
        self._asd_events: list[AsdEvent] = []

    def ingest_gaze(self, event: GazePosition) -> None:
        self._gaze_ts.append(event.timestamp)
        self._gaze_events.append(event)

    def ingest_asd(self, event: AsdEvent) -> None:
        self._asd_ts.append(event.timestamp)
        self._asd_events.append(event)

    def prune(self, current_time: datetime) -> None:
        """Prunes both buffers using binary search to find the cutoff index."""
        cutoff = current_time - self.max_history
        
        # Gaze
        g_idx = bisect_left(self._gaze_ts, cutoff)
        if g_idx > 0:
            self._gaze_ts = self._gaze_ts[g_idx:]
            self._gaze_events = self._gaze_events[g_idx:]

        # ASD
        a_idx = bisect_left(self._asd_ts, cutoff)
        if a_idx > 0:
            self._asd_ts = self._asd_ts[a_idx:]
            self._asd_events = self._asd_events[a_idx:]

    def get_windows(self, anchor_time: datetime) -> MultiscaleSnapshots:
        # Anchor times for all windows
        t_long = anchor_time - timedelta(seconds=self.long_duration)
        t_mid = anchor_time - timedelta(seconds=self.mid_duration)
        t_short = anchor_time - timedelta(seconds=self.short_duration)

        # Process gaze
        g_idx_long = bisect_left(self._gaze_ts, t_long)
        g_batch_long = self._gaze_events[g_idx_long:]
        gaze_df = GAZE_DEFINITION.build_df(g_batch_long)

        # Process ASD (categorize batches by type)
        a_idx_long = bisect_left(self._asd_ts, t_long)
        asd_long_slice = self._asd_events[a_idx_long:]
        asd_batches: dict[type[AsdEvent], list[AsdEvent]] = defaultdict(list)
        for ev in asd_long_slice:
            asd_batches[type(ev)].append(ev)

        # Helper to build a safe DataFrame for a specific event type
        def get_asd_df(cls_type: type[AsdEvent]) -> pd.DataFrame:
            batch = asd_batches.get(cls_type, [])
            return ASD_EVENT_DEFINITIONS[cls_type].build_df(batch)

        # Build the Long Snapshot
        long_snapshot = EventsSnapshot(
            gaze=gaze_df,
            mouse=get_asd_df(MousePosition),
            track_label=get_asd_df(TrackLabelPosition),
            track_screen=get_asd_df(TrackScreenPosition),
            popup=get_asd_df(Popup),
            clearance=get_asd_df(Clearance),
            transfer=get_asd_df(Transfer),
            asd_timeline=ASD_ACTIVITY_TIMELINE_DEFINITION.build_df(asd_long_slice)
        )

        # Cascading slicing for the smaller windows
        mid_snapshot = long_snapshot.slice_by_time(t_mid)
        short_snapshot = mid_snapshot.slice_by_time(t_short)

        return MultiscaleSnapshots(
            short=short_snapshot,
            mid=mid_snapshot,
            long=long_snapshot
        )