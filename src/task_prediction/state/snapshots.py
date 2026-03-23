from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass(slots=True)
class EventsSnapshot:
    """The complete state of the world for a specific time duration."""
    gaze: pd.DataFrame
    mouse: pd.DataFrame
    track_label: pd.DataFrame
    track_screen: pd.DataFrame
    popup: pd.DataFrame
    clearance: pd.DataFrame
    transfer: pd.DataFrame
    asd_timeline: pd.DataFrame

    def slice_by_time(self, start_time: datetime) -> "EventsSnapshot":
        """Creates a zero-copy view of the snapshot from start_time onwards."""
        return EventsSnapshot(
            gaze=self.gaze.loc[start_time:],
            mouse=self.mouse.loc[start_time:],
            track_label=self.track_label.loc[start_time:],
            track_screen=self.track_screen.loc[start_time:],
            popup=self.popup.loc[start_time:],
            clearance=self.clearance.loc[start_time:],
            transfer=self.transfer.loc[start_time:],
            asd_timeline=self.asd_timeline.loc[start_time:],
        )

@dataclass(slots=True)
class MultiscaleSnapshots:
    """The 3 views of the world provided to the predictor."""
    short: EventsSnapshot  # 5s
    mid: EventsSnapshot # 10s
    long: EventsSnapshot # 25s