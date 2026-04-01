import pandas as pd
from dataclasses import fields
from scipy.stats import entropy

from ...utils.types import FeatureDict
from ...state.snapshots import EventsSnapshot

_ACTIVITY_FIELDS = [
    f.name for f in fields(EventsSnapshot) 
    if f.name != "gaze"
]

def extract_activity_features(snapshot: EventsSnapshot) -> FeatureDict:
    features: FeatureDict = {
        "n_events_total": 0,
        "n_events_unique": 0,
        "events_per_ms": 0.0,
        "events_per_timestamp": 0.0,
        "event_type_entropy": 0.0,
        **{f"event_{name}_count": 0 for name in _ACTIVITY_FIELDS},
    }

    # Count events
    counts = []
    all_timestamps = []

    for field_name in _ACTIVITY_FIELDS:
        df = getattr(snapshot, field_name)
        count = len(df)
        features[f"event_{field_name}_count"] = count
        counts.append(count)
        all_timestamps.append(df.index.to_series())

    n_total = sum(counts)
    features["n_events_total"] = n_total

    if n_total == 0:
        return features

    # Unique event types
    features["n_events_unique"] = sum(1 for c in counts if c > 0)

    combined_ts = pd.concat(all_timestamps)
    duration_ms = (combined_ts.max() - combined_ts.min()).total_seconds() * 1000.0
    features["events_per_ms"] = n_total / max(duration_ms, 1) # Duration could be 0, max with 1 to avoid zero division
    features["events_per_timestamp"] = n_total / combined_ts.nunique()

    # Entropy
    probs = [c / n_total for c in counts if c > 0]
    features["event_type_entropy"] = float(entropy(probs))

    return features