import numpy as np
import pandas as pd
import pymovements as pm
from typing import Final

from ...state.snapshots import EventsSnapshot
from ...utils.types import FeatureDict

DEFAULT_GAZE_METRICS: Final[dict[str, float | int]] = {
    "Fixation_Count": 0,
    "Total_Fixation_Duration_(s)": 0.0,
    "Avg_Fixation_Duration_(s)": 0.0,
    "Saccade_Count": 0,
    "Avg_Saccade_Amplitude_(px)": 0.0,
    "Avg_Saccade_Velocity_(px/s)": 0.0,
    "Avg_Gaze_Velocity_(px/s)": 0.0,
    "Avg_Gaze_Acceleration_(px/s²)": 0.0,
    "Gaze_Dispersion_(area_px²)": 0.0,
}

def _estimate_sampling_rate_hz(dt_s: pd.Series) -> float:
    """Estimates the hardware sampling rate based on the median time delta."""
    dt_valid = dt_s[dt_s > 0].dropna()
    if dt_valid.empty:
        return 120.0  # Safe default fallback
    return float(1.0 / dt_valid.median())

def _get_valid_segments(df: pd.DataFrame, min_samples: int):
    """Yields contiguous chunks of valid gaze data (no NaNs) longer than min_samples."""
    valid = df["x"].notna() & df["y"].notna()
    if not valid.any():
        return

    # Create a unique ID for each contiguous run of valid/invalid data
    run_ids = valid.ne(valid.shift(fill_value=False)).cumsum()
    
    # Group by run ID, yielding only the valid chunks
    for _, chunk in df[valid].groupby(run_ids[valid]):
        if len(chunk) >= min_samples:
            yield chunk

def extract_gaze_metrics(
    snapshot: EventsSnapshot,
    fixation_time_thresh_ms: int = 100,
    fixation_dispersion_thresh_px: int = 50,
    saccade_radius_thresh_px: float = 50.0,
    saccade_min_duration_ms: float = 20.0,
    saccade_threshold_factor: float = 6.0,
    saccade_min_vel_std: float = 1e-8,
) -> FeatureDict:
    
    df = snapshot.gaze
    features = DEFAULT_GAZE_METRICS.copy()

    if df.empty or len(df) < 2:
        return features
    
    # Ensures precision for sqrt/division and overflow
    df["x"] = df["x"].astype(np.float64)
    df["y"] = df["y"].astype(np.float64)

    # 1. Base Kinematics (Velocity, Acceleration, Dispersion)
    dt_s = df.index.to_series().diff().dt.total_seconds().fillna(0.0)
    dx = df["x"].diff().fillna(0.0)
    dy = df["y"].diff().fillna(0.0)
    disp = np.sqrt(dx**2 + dy**2)

    safe_dt = np.where(dt_s == 0, np.nan, dt_s)
    gaze_vel = disp / safe_dt
    gaze_acc = gaze_vel.diff().fillna(0.0) / safe_dt

    features["Avg_Gaze_Velocity_(px/s)"] = float(gaze_vel.mean() if not gaze_vel.isna().all() else 0.0)
    features["Avg_Gaze_Acceleration_(px/s²)"] = float(gaze_acc.mean() if not gaze_acc.isna().all() else 0.0)

    valid_x = df["x"].dropna()
    valid_y = df["y"].dropna()
    if len(valid_x) > 3 and len(valid_y) > 3:
        features["Gaze_Dispersion_(area_px²)"] = float((valid_x.max() - valid_x.min()) * (valid_y.max() - valid_y.min()))

    # 2. Fixations (pymovements I-DT)
    sr_hz = _estimate_sampling_rate_hz(dt_s)
    fix_dur_samples = max(1, int(np.ceil((fixation_time_thresh_ms / 1000.0) * sr_hz)))
    fix_durations_s = []

    for seg in _get_valid_segments(df, max(2, fix_dur_samples)):
        positions = seg[["x", "y"]].to_numpy(dtype=float)
        res = pm.events.idt(
            positions=positions,
            dispersion_threshold=float(fixation_dispersion_thresh_px),
            minimum_duration=int(fix_dur_samples),
        )
        for dur in res.fixations["duration"]:
            fix_durations_s.append(dur / sr_hz)

    if fix_durations_s:
        features["Fixation_Count"] = len(fix_durations_s)
        features["Total_Fixation_Duration_(s)"] = float(np.sum(fix_durations_s))
        features["Avg_Fixation_Duration_(s)"] = features["Total_Fixation_Duration_(s)"] / features["Fixation_Count"]

    # 3. Saccades (pymovements Engbert / Fallback)
    sac_min_samples = max(2, int(np.ceil((saccade_min_duration_ms / 1000.0) * sr_hz)))
    
    # Global variance check for pymovements viability
    if len(valid_x) < (sac_min_samples + 1):
        pos_all = np.empty((0, 2))
    else:
        pos_all = df.loc[valid_x.index, ["x", "y"]].to_numpy(dtype=float)

    v_all = np.diff(pos_all, axis=0) * sr_hz if len(pos_all) > 1 else np.empty((0, 2))
    
    # If velocity variance is too low, fallback to simple displacement threshold
    if len(v_all) == 0 or np.nanstd(v_all[:, 0]) < saccade_min_vel_std or np.nanstd(v_all[:, 1]) < saccade_min_vel_std:
        saccade_mask = disp >= saccade_radius_thresh_px
        features["Saccade_Count"] = int(saccade_mask.sum())
        if features["Saccade_Count"] > 0:
            features["Avg_Saccade_Amplitude_(px)"] = float(disp[saccade_mask].mean())
            features["Avg_Saccade_Velocity_(px/s)"] = float((disp[saccade_mask] / safe_dt[saccade_mask]).mean())
        return features

    # Run Pymovements Microsaccades
    sac_amps = []
    sac_vels = []

    for seg in _get_valid_segments(df, sac_min_samples + 1):
        pos = seg[["x", "y"]].to_numpy(dtype=float)
        v = np.diff(pos, axis=0) * sr_hz

        if np.nanstd(v[:, 0]) < saccade_min_vel_std or np.nanstd(v[:, 1]) < saccade_min_vel_std:
            # Segment degenerate -> Abort PM entirely and use Fallback for consistency
            saccade_mask = disp >= saccade_radius_thresh_px
            features["Saccade_Count"] = int(saccade_mask.sum())
            if features["Saccade_Count"] > 0:
                features["Avg_Saccade_Amplitude_(px)"] = float(disp[saccade_mask].mean())
                features["Avg_Saccade_Velocity_(px/s)"] = float((disp[saccade_mask] / safe_dt[saccade_mask]).mean())
            return features

        try:
            sacc = pm.events.microsaccades(
                velocities=v,
                threshold="engbert2015",
                threshold_factor=float(saccade_threshold_factor),
                minimum_duration=int(sac_min_samples),
            )
        except ValueError:
            # PM refused (variance/threshold issues) -> Fallback
            saccade_mask = disp >= saccade_radius_thresh_px
            features["Saccade_Count"] = int(saccade_mask.sum())
            if features["Saccade_Count"] > 0:
                features["Avg_Saccade_Amplitude_(px)"] = float(disp[saccade_mask].mean())
                features["Avg_Saccade_Velocity_(px/s)"] = float((disp[saccade_mask] / safe_dt[saccade_mask]).mean())
            return features

        # Extract Events
        # Pymovements returns an EventDataFrame or a list of events depending on version
        if hasattr(sacc, "saccades"):
            onsets = np.asarray(sacc.saccades["onset"], dtype=int)
            durs   = np.asarray(sacc.saccades["duration"], dtype=int)
            iterable_events = zip(onsets, durs)
        else:
            iterable_events = ((int(ev.onset), int(ev.offset) - int(ev.onset)) for ev in sacc)

        for onset_v, dur in iterable_events:
            if dur < sac_min_samples:
                continue
            
            offset_v = onset_v + dur
            onset_p = onset_v
            offset_p = min(offset_v, len(pos) - 1)

            amp = float(np.linalg.norm(pos[offset_p] - pos[onset_p]))
            speed = np.linalg.norm(v[onset_v:offset_v], axis=1)
            mean_speed = float(np.nanmean(speed)) if speed.size else 0.0

            sac_amps.append(amp)
            sac_vels.append(mean_speed)

    features["Saccade_Count"] = len(sac_amps)
    if sac_amps:
        features["Avg_Saccade_Amplitude_(px)"] = float(np.mean(sac_amps))
        features["Avg_Saccade_Velocity_(px/s)"] = float(np.mean(sac_vels))

    return features