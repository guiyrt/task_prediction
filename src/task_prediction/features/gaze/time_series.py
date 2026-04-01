import numpy as np
import pandas as pd
from typing import Final
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from ...state.snapshots import EventsSnapshot
from ...utils.types import FeatureDict

# MinimalFCParameters generates exactly these 10 features per column
_TSFRESH_FEATURES: Final[list[str]] = [
    "sum_values", "median", "mean", "length", "standard_deviation",
    "variance", "root_mean_square", "maximum", "absolute_maximum", "minimum"
]

DEFAULT_TSFRESH_FEATURES: Final[FeatureDict] = {
    **{f"Gaze_point_X__DACS_px___{feat}": 0.0 for feat in _TSFRESH_FEATURES},
    **{f"Gaze_point_Y__DACS_px___{feat}": 0.0 for feat in _TSFRESH_FEATURES},
}

def extract_tsfresh_features(snapshot: EventsSnapshot) -> FeatureDict:
    """
    Extracts TSFresh MinimalFCParameters features from the gaze data.
    Runs exactly on one window (dummy id=1) and returns a flat dictionary.
    """
    df = snapshot.gaze

    if df.empty or len(df) < 2:
        return DEFAULT_TSFRESH_FEATURES.copy()

    # 1. Prepare data (collapse duplicate timestamps)
    # Grouping by the DatetimeIndex guarantees unique timestamps per row
    df_clean = df[["x", "y"]].groupby(level="timestamp").mean()

    if len(df_clean) < 2:
        return DEFAULT_TSFRESH_FEATURES.copy()

    # 2. Fill NaNs (Replicating legacy logic exactly)
    # The legacy code used: ffill -> bfill -> interpolate(linear)
    df_clean = (
        df_clean
        .ffill()
        .bfill()
        .interpolate(method="linear", limit_direction="both")
    )
    
    # Global fallback (if the entire column was NaN)
    medians = df_clean.median()
    df_clean = df_clean.fillna(medians).fillna(0.0)

    # 3. Add Dummy ID & Timestamp columns for TSFresh API
    # TSFresh requires explicit columns, it does not read the Pandas Index
    df_clean["id"] = 1
    df_clean["time_idx"] = np.arange(len(df_clean))  # Safe, monotonic integer sequence

    # 4. Run TSFresh
    # We force n_jobs=0 (single thread) because spawning processes for 
    # a single small DataFrame is significantly slower than executing it inline.
    extracted_features = extract_features(
        df_clean,
        column_id="id",
        column_sort="time_idx",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
        disable_progressbar=True
    )

    # 5. Impute missing features (TSFresh utility)
    impute(extracted_features)

    # Convert the 1-row DataFrame into dict
    raw_dict = extracted_features.iloc[0].to_dict()

    # LEGACY: changing names to match model expected feature names
    return {
        k.replace("x__", "Gaze_point_X__DACS_px___").replace("y__", "Gaze_point_Y__DACS_px___"): v 
        for k, v in raw_dict.items()
    }