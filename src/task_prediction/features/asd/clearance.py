from ...utils.types import FeatureDict
from ...state.snapshots import EventsSnapshot
from ...models.asd import ClearanceType

_CLEARANCE_TYPES = [e.value for e in ClearanceType]

def extract_clearance_features(snapshot: EventsSnapshot) -> FeatureDict:
    features: FeatureDict = {
        "clearance_count": 0,
        "clearance_unique_flights": 0,
        "clearance_any": 0,
        "clearance_inter_event_mean_ms": 0.0,
        "clearance_inter_event_median_ms": 0.0,
        "clearance_inter_event_std_ms": 0.0,
        "clearance_max_per_flight": 0,
        "clearance_mean_per_flight": 0.0,
        **{f"clearance_type_{ctype}_count": 0 for ctype in _CLEARANCE_TYPES},
        **{f"clearance_type_{ctype}_present": 0 for ctype in _CLEARANCE_TYPES},
    }
    
    df = snapshot.clearance

    if df.empty:
        return features
    
    features["clearance_count"] = len(df)
    features["clearance_any"] = 1
    features["clearance_unique_flights"] = df["callsign"].nunique()

    # Per-type counts
    type_counts = df["clearance_type"].value_counts().to_dict()
    for ctype in _CLEARANCE_TYPES:
        c = type_counts.get(ctype, 0)
        features[f"clearance_type_{ctype}_count"] = c
        features[f"clearance_type_{ctype}_present"] = 1 if c > 0 else 0

    # Timing features
    deltas = df.index.to_series().diff().dt.total_seconds().dropna() * 1000.0
    if not deltas.empty:
        features["clearance_inter_event_mean_ms"] = deltas.mean()
        features["clearance_inter_event_median_ms"] = deltas.median()
        features["clearance_inter_event_std_ms"] = deltas.std() if len(deltas) > 1 else 0.0

    # Clearances per flight distribution
    per_flight_counts = df.groupby("callsign").size()
    features["clearance_max_per_flight"] = per_flight_counts.max()
    features["clearance_mean_per_flight"] = per_flight_counts.mean()

    return features