import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

from ...models import TaskType, AsaSupportMode

@dataclass(frozen=True, slots=True)
class LifecycleTask:
    task: TaskType
    start_time: datetime
    end_time: datetime
    duration_sec: float
    other_callsigns: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class CallsignLifecycle:
    callsign: str
    lead_time_sec: float
    tail_time_sec: float
    tasks: list[LifecycleTask] = field(default_factory=list)

def build_base_callsign_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core transformation: Explodes tasks by callsign, drops IDLEs natively,
    sorts chronologically, and identifies multi-aircraft interactions.
    """
    if df.empty or "callsigns" not in df.columns:
        return pd.DataFrame()

    work_df = df.copy()
    work_df["all_callsigns"] = work_df["callsigns"]
    
    # Explode drops IDLE tasks natively because their callsign list is empty
    exploded = work_df.explode("callsigns").dropna(subset=["callsigns"])
    
    if exploded.empty:
        return pd.DataFrame()

    # PERFORMANCE OPTIMIZATION: List comprehension with zip is ~10x faster than .apply(axis=1)
    exploded["other_callsigns"] = [
        tuple(c for c in all_c if c != current_c) 
        for current_c, all_c in zip(exploded["callsigns"], exploded["all_callsigns"])
    ]
    
    exploded.sort_values(
        by=["scenario_id", "participant_id", "callsigns", "start_time"], 
        inplace=True
    )
    
    return exploded

def get_sequence_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Consumes the base timeline to calculate ATCO sequence agreement."""
    exploded_df = build_base_callsign_timeline(df)
    
    if exploded_df.empty:
        return pd.DataFrame()
        
    # 1. Group by participant to get their specific sequence as a TUPLE
    seq_df = exploded_df.groupby(["scenario_id", "participant_id", "callsigns", "asa_support_mode"]).agg(
        task_sequence=("task_type", tuple)
    ).reset_index()
    
    def get_participants(series, mode_filter):
        return tuple(sorted(series[seq_df.loc[series.index, "asa_support_mode"] == mode_filter]))

    consensus_df = seq_df.groupby(["scenario_id", "callsigns", "task_sequence"]).agg(
        total_count=("participant_id", "count"),
        participants=("participant_id", lambda x: tuple(sorted(x.unique()))),
        # Mode-specific participant lists
        participants_none=("participant_id", lambda x: get_participants(x, AsaSupportMode.NONE.name)),
        participants_advisory=("participant_id", lambda x: get_participants(x, AsaSupportMode.ADVISORY.name)),
        participants_execution=("participant_id", lambda x: get_participants(x, AsaSupportMode.EXECUTION.name))
    ).reset_index()

    # Calculate counts from the participant tuples
    consensus_df["count_none"] = consensus_df["participants_none"].apply(len)
    consensus_df["count_advisory"] = consensus_df["participants_advisory"].apply(len)
    consensus_df["count_execution"] = consensus_df["participants_execution"].apply(len)
    
    # Sort: Most agreed-upon sequences first
    consensus_df.sort_values(
        by=["scenario_id", "callsigns", "total_count"], 
        ascending=[True, True, False], 
        inplace=True
    )

    return consensus_df

# ==========================================
# 3. MODE 1: SINGLE RUN VIEW (Lifecycles)
# ==========================================
def get_callsign_lifecycles(df: pd.DataFrame, scen_start: pd.Timestamp, scen_end: pd.Timestamp) -> list[CallsignLifecycle]:
    """Consumes the base timeline to build strongly typed lifecycle objects."""
    exploded_df = build_base_callsign_timeline(df)
    
    if exploded_df.empty:
        return []
        
    lifecycles: list[CallsignLifecycle] = []
    
    for callsign, group in exploded_df.groupby("callsigns"):
        tasks = [
            LifecycleTask(
                task=TaskType[row["task_type"]],
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration_sec=row["duration_sec"],
                other_callsigns=row["other_callsigns"]
            )
            for _, row in group.iterrows()
        ]
            
        # 3. Assemble and append the lifecycle object
        lifecycles.append(
            CallsignLifecycle(
                callsign=str(callsign),
                lead_time_sec=(group.iloc[0]["start_time"] - scen_start).total_seconds(),
                tail_time_sec=(scen_end - group.iloc[-1]["end_time"]).total_seconds(),
                tasks=tasks
            )
        )

    # Sort by interaction order
    lifecycles.sort(key=lambda x: x.lead_time_sec)
        
    return lifecycles