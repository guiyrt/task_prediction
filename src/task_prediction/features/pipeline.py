from ..utils.types import FeatureDict
from ..state.snapshots import MultiscaleSnapshots
from .asd import *
from .gaze import *

def extract_all_features(windows: MultiscaleSnapshots) -> FeatureDict:
    final_features = {}

    for prefix, snapshot in [
        ("short", windows.short), 
        ("mid", windows.mid), 
        ("long", windows.long)
    ]:
        scale_feats = dict(
            **extract_gaze_metrics(snapshot),
            **extract_blink_features(snapshot),
            **extract_tsfresh_features(snapshot),
            **extract_mouse_features(snapshot),
            **extract_activity_features(snapshot),
            **extract_track_screen_position_features(snapshot),
            **extract_track_label_position_features(snapshot),
            **extract_transfer_features(snapshot),
            **extract_popup_features(snapshot),
            **extract_clearance_features(snapshot),
        )

        # Append the scale prefix
        for key, val in scale_feats.items():
            final_features[f"{prefix}_{key}"] = val

    return final_features