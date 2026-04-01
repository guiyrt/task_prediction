from ...state.snapshots import EventsSnapshot
from ...models.asd import TransferType
from ...utils.types import FeatureDict

_TRANSFER_TYPES = [e.name for e in TransferType]

def extract_transfer_features(snapshot: EventsSnapshot) -> FeatureDict:
    features: FeatureDict = {
        **{f"transfer_type_{t}_count": 0 for t in _TRANSFER_TYPES},
        **{f"transfer_type_{t}_present": 0 for t in _TRANSFER_TYPES},
    }

    df = snapshot.transfer

    if not df.empty:    
        counts = df["transfer_type"].value_counts().to_dict()

        for t in _TRANSFER_TYPES:
            c = counts.get(t, 0)
            features[f"transfer_type_{t}_count"] = c
            features[f"transfer_type_{t}_present"] = 1 if c > 0 else 0

    return features