import pandas as pd

def load_and_prep_data(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df.sort_values(by=["scenario_id", "participant_id", "start_time"], inplace=True)
    df["duration_sec"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    return df

def get_duration_frequency_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
        
    stats = df.groupby(group_cols).agg(
        frequency=("task_type", "count"),
        duration_avg=("duration_sec", "mean"),
        duration_std=("duration_sec", "std"),
        duration_min=("duration_sec", "min"),
        duration_max=("duration_sec", "max")
    ).reset_index()
    
    # Round floats for cleaner display
    float_cols = ['duration_avg', 'duration_std', 'duration_min', 'duration_max']
    stats[float_cols] = stats[float_cols].round(2)
    return stats

def format_seconds(seconds: float) -> str:
    """Helper to format seconds into 'Mm Ss' or just 'Ss'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"