import numpy as np

from ...utils.types import FeatureDict
from ...state.snapshots import EventsSnapshot

def extract_mouse_features(
    snapshot: EventsSnapshot,
    idle_threshold_sec: float = 0.5,
    angle_threshold_deg: int = 30,
    stop_threshold_px_sec: int = 5,
    burst_threshold_px_sec: int = 50,
    stillness_threshold_px_sec: int = 5
) -> FeatureDict:
    features: FeatureDict = {
        "Avg_Mouse_Velocity_(px/s)": 0.0,
        "Avg_Mouse_Acceleration_(px/s²)": 0.0,
        "Movement_Frequency_(movements/s)": 0.0,
        "Total_Idle_Time_(s)": 0.0,
        "Path_Direction_Changes": 0,
        "Total_Distance_Traveled_(px)": 0.0,
        "Number_of_Stops": 0,
        "Movement_Bursts": 0,
        "Stillness_Periods": 0,
    }

    df = snapshot.mouse_position.groupby(level="timestamp").mean()

    # Short-circuit for empty or insufficient data
    if df.empty or len(df) < 2:
        return features

    # Delta x and y
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0.0)
    dx = df["x"].diff().fillna(0.0)
    dy = df["y"].diff().fillna(0.0)
    disp = np.sqrt(dx**2 + dy**2)
    features["Total_Distance_Traveled_(px)"] = disp.sum()
    
    safe_dt = np.where(dt == 0, np.nan, dt)

    # Velocity
    velocity = disp / safe_dt
    if not velocity.isna().all():
        features["Avg_Mouse_Velocity_(px/s)"] = velocity.mean()
        features["Number_of_Stops"] = (velocity < stop_threshold_px_sec).sum()
        features["Movement_Bursts"] = (velocity > burst_threshold_px_sec).sum()
        features["Stillness_Periods"] = (velocity < stillness_threshold_px_sec).sum()

    # Acceleration
    acceleration = velocity.diff().fillna(0.0) / safe_dt
    if not acceleration.isna().all():
        features["Avg_Mouse_Acceleration_(px/s²)"] = acceleration.mean()

    # Movement frequency
    total_time = dt.sum()
    moved = (dx.abs() > 0) | (dy.abs() > 0)
    features["Movement_Frequency_(movements/s)"] = (
        moved.sum() / total_time
        if total_time > 0
        else 0.0
    )
    
    # Idle time
    idle_periods = dt[~moved]
    features["Total_Idle_Time_(s)"] = idle_periods[idle_periods >= idle_threshold_sec].sum()

    # Direction changes
    angles = np.arctan2(dy, dx)
    angle_diff = np.abs(np.diff(angles))
    features["Path_Direction_Changes"] = np.sum(angle_diff > np.radians(angle_threshold_deg))

    return features