from typing import Final

from ...state.snapshots import EventsSnapshot
from ...utils.types import FeatureDict

DEFAULT_BLINK_FEATURES: Final[FeatureDict] = {
    "Blink_Rate_(blinks/s)": 0.0,
}

def extract_blink_features(
    snapshot: EventsSnapshot, 
    blink_threshold_ms: float = 100.0
) -> FeatureDict:
    """
    Identifies runs of missing gaze data. If a contiguous run exceeds the 
    blink_threshold_ms, the rows within that run are counted towards the blink rate.
    
    Note: To maintain parity with legacy XGBoost training, 'Blink Rate' is calculated 
    as (Number of Rows inside Blinks) / (Total Window Duration in Seconds).
    """
    df = snapshot.gaze

    if df.empty or len(df) < 2:
        return DEFAULT_BLINK_FEATURES.copy()

    # Total duration of the window in seconds
    # (Using the index difference between last and first row)
    total_time_s = (df.index[-1] - df.index[0]).total_seconds()
    if total_time_s <= 0:
        return DEFAULT_BLINK_FEATURES.copy()

    # Identify missing gaze points
    missing_mask = df["x"].isna() | df["y"].isna()

    if not missing_mask.any():
        return DEFAULT_BLINK_FEATURES.copy()

    # Group contiguous runs of missing/valid data
    # .ne().shift().cumsum() creates a unique ID for each continuous block of True or False
    run_ids = missing_mask.ne(missing_mask.shift(fill_value=False)).cumsum()

    # Calculate the duration of each run in milliseconds
    # We use the DatetimeIndex to find the time delta of each row, then sum per run
    dt_ms = df.index.to_series().diff().dt.total_seconds().fillna(0.0) * 1000.0
    
    # Sum durations only for the runs where gaze is actually missing
    durations_by_run = dt_ms[missing_mask].groupby(run_ids[missing_mask]).sum()

    # Find the IDs of the runs that exceed the blink threshold
    valid_blink_runs = durations_by_run[durations_by_run > blink_threshold_ms].index

    if valid_blink_runs.empty:
        return DEFAULT_BLINK_FEATURES.copy()

    # Count the total NUMBER OF ROWS that belong to these valid blink runs
    # (Replicating legacy logic: int(self.df["Blink"].sum()))
    blink_row_count = (missing_mask & run_ids.isin(valid_blink_runs)).sum()

    return {
        "Blink_Rate_(blinks/s)": float(blink_row_count / total_time_s)
    }