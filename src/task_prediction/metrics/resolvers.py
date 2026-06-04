import pandas as pd
import numpy as np

from ..models import TaskType

def resolve_stage1(df: pd.DataFrame, idle_label: str = TaskType.IDLE.name) -> tuple[np.ndarray, np.ndarray]:
    """
    Mode: Stage 1 (Binary Task Detection)
    True: 1 if true_task != IDLE, else 0
    Pred: 1 if is_active == True, else 0
    """
    if df.empty:
        return np.array([]), np.array([])
    
    y_true = (df["true_task"] != idle_label).astype(int).values
    y_pred = df["is_active"].astype(int).values
    
    return y_true, y_pred

def resolve_stage2(df: pd.DataFrame, idle_label: str = TaskType.IDLE.name) -> tuple[np.ndarray, np.ndarray]:
    """
    Mode: Stage 2 Isolation
    Evaluates how well Stage 2 differentiates tasks, strictly ignoring periods 
    where the true state is IDLE. (Stage 2 is never trained to predict IDLE).
    """
    # Mask out IDLE time
    active_df = df[df["true_task"] != idle_label]
    
    if active_df.empty:
        return np.array([]), np.array([])

    y_true = active_df["true_task"].values
    y_pred = np.array([
        max(probas, key=lambda x: x[1])[0]
        for probas in active_df["task_probas"].values
    ])

    return y_true, y_pred

def resolve_joint(df: pd.DataFrame, idle_label: str = TaskType.IDLE.name) -> tuple[np.ndarray, np.ndarray]:
    """
    Mode: Full Production Pipeline
    Respects the Stage 1 gate. If Stage 1 is False, prediction is forced to IDLE.
    Otherwise, prediction is Stage 2's output. Evaluated across the whole timeline.
    """
    if df.empty:
        return np.array([]), np.array([])

    y_true = df["true_task"].values
    y_pred = np.where(
        df["is_active"],
        df["pred_task"],
        idle_label
    )
    
    return y_true, y_pred