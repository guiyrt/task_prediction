from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass(slots=True)
class EventsSnapshot:
    """The complete state of the world for a specific time duration."""
    gaze: pd.DataFrame
    mouse_position: pd.DataFrame
    track_screen_position: pd.DataFrame
    track_label_position: pd.DataFrame
    popup: pd.DataFrame
    transfer: pd.DataFrame
    clearance: pd.DataFrame
    distance_measurement: pd.DataFrame

    def slice_by_time(self, start_time: datetime) -> "EventsSnapshot":
        """Creates a zero-copy view of the snapshot from start_time onwards."""
        return EventsSnapshot(
            gaze=self.gaze.loc[start_time:],
            mouse_position=self.mouse_position.loc[start_time:],
            track_screen_position=self.track_screen_position.loc[start_time:],
            track_label_position=self.track_label_position.loc[start_time:],
            popup=self.popup.loc[start_time:],
            transfer=self.transfer.loc[start_time:],
            clearance=self.clearance.loc[start_time:],
            distance_measurement=self.distance_measurement.loc[start_time:],
        )

@dataclass(slots=True)
class MultiscaleSnapshots:
    """The 3 views of the world provided to the predictor."""
    short: EventsSnapshot # 5s
    mid: EventsSnapshot # 10s
    long: EventsSnapshot # 25s