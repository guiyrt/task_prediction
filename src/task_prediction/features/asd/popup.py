from ...utils.types import FeatureDict
from ...state.snapshots import EventsSnapshot

def extract_popup_features(snapshot: EventsSnapshot) -> FeatureDict:
    features: FeatureDict = {
        "n_popup_open": 0,
        "n_popup_close": 0,
        "popup_any": 0,
        "popup_overlap": 0,
        "popup_dwell_total_ms": 0.0,
        "popup_dwell_mean_ms": 0.0,
        "popup_dwell_max_ms": 0.0,
        "popup_per_flight_mean": 0.0,
        "popup_per_flight_max": 0,
        "popup_revisit_count": 0,
        "popup_inter_time_mean_ms": 0.0,
        "popup_inter_time_median_ms": 0.0,
        "popup_inter_time_std_ms": 0.0,
    }

    df = snapshot.popup

    if df.empty:
        return features

    features["popup_any"] = 1

    # Basic Counts
    features["n_popup_open"] = df["opened"].sum()
    features["n_popup_close"] = len(df) - features["n_popup_open"]

    # Overlap (More than 2 popups open simultaneously)
    open_close = df["opened"].map({False: -1, True: 1})
    features["popup_overlap"] = bool((open_close.cumsum() > 2).any())

    # Flight Features (Opens per flight)
    opens = df[df["opened"]]
    if not opens.empty:
        nb_popup_per_flight = opens.groupby("callsign").size()
        features["popup_per_flight_mean"] = nb_popup_per_flight.mean()
        features["popup_per_flight_max"] = nb_popup_per_flight.max()

    # Dwell Time & Revisit Logic
    durations_ms = []
    revisit_count = 0
    last_state = {}

    # Iterate over unique Popups (Menu + Callsign)
    for (name, callsign), group in df.groupby(["name", "callsign"]):
        open_time = None
        
        for timestamp, row in group.iterrows():
            is_open = row["opened"]
            key = (name, callsign)

            # Revisit: Previously closed, now open again
            if last_state.get(key) is False and is_open:
                revisit_count += 1
            last_state[key] = is_open

            # Dwell duration tracking
            if is_open:
                open_time = timestamp

            elif not is_open and open_time is not None:
                # Convert datetime difference back to milliseconds for model compatibility
                duration_ms = (timestamp - open_time).total_seconds() * 1000.0
                durations_ms.append(duration_ms)
                open_time = None

    features["popup_revisit_count"] = revisit_count

    if durations_ms:
        features["popup_dwell_total_ms"] = sum(durations_ms)
        features["popup_dwell_mean_ms"] = sum(durations_ms) / len(durations_ms)
        features["popup_dwell_max_ms"] = max(durations_ms)

    # Inter-popup time
    deltas_ms = df.index.to_series().diff().dt.total_seconds().dropna() * 1000.0
    if not deltas_ms.empty:
        features["popup_inter_time_mean_ms"] = deltas_ms.mean()
        features["popup_inter_time_median_ms"] = deltas_ms.median()
        features["popup_inter_time_std_ms"] = deltas_ms.std() if len(deltas_ms) > 1 else 0.0

    return features