import pandas as pd
import numpy as np

from ...utils.types import FeatureDict
from ...state.snapshots import EventsSnapshot

def _extract_lifecycle_features(df: pd.DataFrame, prefix: str) -> FeatureDict:
    """Helper to compute Appear/Disappear/Persist based on callsigns."""
    flights_by_epoch = df.groupby(level="timestamp")["callsign"].apply(set)
    first_flights = flights_by_epoch.iloc[0]
    last_flights = flights_by_epoch.iloc[-1]
    flights_ever = set().union(*flights_by_epoch.tolist())

    return {
        f"n_flights_{prefix}_ever": len(flights_ever),
        f"n_flights_{prefix}_appear": len(flights_ever - first_flights),
        f"n_flights_{prefix}_disappear": len(flights_ever - last_flights),
        f"n_flights_{prefix}_persist": len(first_flights & last_flights),
        f"n_flights_{prefix}_transient": len(flights_ever - (first_flights | last_flights)),
    }

def _extract_coord_features(prefix: str, col: pd.Series) -> FeatureDict:
    return {
        f"{prefix}_mean": col.mean(),
        f"{prefix}_std": col.std() if len(col) > 1 else 0.0,
        f"{prefix}_min": col.min(),
        f"{prefix}_max": col.max(),
    }

def extract_track_screen_position_features(snapshot: EventsSnapshot) -> FeatureDict:
    """Computes combined spatial and lifecycle features for Screen Tracks and Labels."""
    features: FeatureDict = {
        "n_flights_screen_ever": 0,
        "n_flights_screen_appear": 0,
        "n_flights_screen_disappear": 0,
        "n_flights_screen_persist": 0,
        "n_flights_screen_transient": 0,
        "track_screen_n_visible": 0,
        "track_screen_visible_ratio": 0.0,
        "track_screen_x_mean": np.nan,
        "track_screen_x_std": np.nan,
        "track_screen_x_min": np.nan,
        "track_screen_x_max": np.nan,
        "track_screen_y_mean": np.nan,
        "track_screen_y_std": np.nan,
        "track_screen_y_min": np.nan,
        "track_screen_y_max": np.nan,
    }

    tsp = snapshot.track_screen_position

    tsp["x"] = tsp["x"].astype(np.float64)
    tsp["y"] = tsp["y"].astype(np.float64)

    if tsp.empty:
        return features
    
    features.update(_extract_lifecycle_features(tsp, "screen"))
    features.update(_extract_coord_features("track_screen_x", tsp["x"].dropna()))
    features.update(_extract_coord_features("track_screen_y", tsp["y"].dropna()))

    vis_count = tsp["is_visible"].sum()
    features["track_screen_n_visible"] = vis_count
    features["track_screen_visible_ratio"] = vis_count / len(tsp)
    
    return features

def extract_track_label_position_features(snapshot: EventsSnapshot) -> FeatureDict:
    features: FeatureDict = {
        "n_flights_label_ever": 0,
        "n_flights_label_appear": 0,
        "n_flights_label_disappear": 0,
        "n_flights_label_persist": 0,
        "n_flights_label_transient": 0,
        "track_label_n_visible": 0,
        "track_label_n_hovered": 0,
        "track_label_n_selected": 0,
        "track_label_n_on_pip": 0,
        "track_label_hovered_ratio": 0.0,
        "track_label_selected_ratio": 0.0,
        "track_label_on_pip_ratio": 0.0,
        "track_label_x_mean": np.nan,
        "track_label_x_std": np.nan,
        "track_label_x_min": np.nan,
        "track_label_x_max": np.nan,
        "track_label_y_mean": np.nan,
        "track_label_y_std": np.nan,
        "track_label_y_min": np.nan,
        "track_label_y_max": np.nan,
        "track_label_width_mean": 0.0,
        "track_label_height_mean": 0.0,
        "track_label_area_mean": 0.0,
        "track_label_area_total": 0.0
    }

    tlp = snapshot.track_label_position

    if tlp.empty:
        return features
    
    tlp["x"] = tlp["x"].astype(np.float64)
    tlp["y"] = tlp["y"].astype(np.float64)
    tlp["height"] = tlp["height"].astype(np.float64)
    tlp["width"] = tlp["width"].astype(np.float64)

    features.update(_extract_lifecycle_features(tlp, "label"))
    features.update(_extract_coord_features("track_label_x", tlp["x"].dropna()))
    features.update(_extract_coord_features("track_label_y", tlp["y"].dropna()))

    features["track_label_n_visible"] = tlp["is_visible"].sum()
    features["track_label_n_hovered"] = tlp["is_hovered"].sum()
    features["track_label_n_selected"] = tlp["is_selected"].sum()
    features["track_label_n_on_pip"] = tlp["on_pip"].sum()

    vis_count = max(features["track_label_n_visible"], 1)
    features["track_label_hovered_ratio"] = features["track_label_n_hovered"] / vis_count
    features["track_label_selected_ratio"] = features["track_label_n_selected"] / vis_count
    features["track_label_on_pip_ratio"] = features["track_label_n_on_pip"] / vis_count

    area = tlp["width"] * tlp["height"]
    features["track_label_width_mean"] = tlp["width"].mean()
    features["track_label_height_mean"] = tlp["height"].mean()
    features["track_label_area_mean"] = area.mean()
    features["track_label_area_total"] = area.sum()

    return features