import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Any
from collections import Counter

def evaluate_continuous(values: np.ndarray) -> dict[str, float | None]:
    """
    Calculates standard statistics for continuous telemetry/quality metrics.
    Automatically filters out NaN values before computing.
    """

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values))
    }

def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """
    Evaluates Stage 1 (Active vs Idle) performance.
    Expects binary arrays (0 and 1).
    """
    if len(y_true) == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    counter_y_true = Counter(y_true)
    counter_y_pred = Counter(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_distribution": {"IDLE": counter_y_true[0], "ACTIVE": counter_y_true[1]},
        "pred_distribution": {"IDLE": counter_y_pred[0], "ACTIVE": counter_y_pred[1]},
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classes": ["IDLE", "ACTIVE"],
    }

def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """
    Evaluates Multiclass performance (Stage 2 Only and Joint Pipeline).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: A definitive list of all possible classes (to guarantee consistent 
                confusion matrix dimensions, even if a class is missing in this specific run).
    """
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "cm": None,
            "classes": labels
        }
    
    counter_y_true = Counter(y_true)
    counter_y_pred = Counter(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "true_distribution": {task: counter_y_true.get(task, 0) for task in labels},
        "pred_distribution": {task: counter_y_pred.get(task, 0) for task in labels},
        "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": labels,
    }

def evaluate_accuracy(df: pd.DataFrame, labels: list[str], idle_label: str = "IDLE") -> dict[str, Any]:
    """
    Unified evaluation function. Returns global metrics AND per-task metrics.
    """
    def make_signature(task, calls):
        c_str = ",".join(sorted([str(c) for c in calls])) if isinstance(calls, (list, np.ndarray)) else ""
        return f"{task}_{c_str}"

    signatures = pd.Series([
        make_signature(t, c) for t, c in zip(df["true_task"], df["true_callsigns"])
    ], index=df.index)

    instance_ids = (signatures != signatures.shift(1)).cumsum()

    temp_df = pd.DataFrame({
        "instance_id": instance_ids,
        "true_task": df["true_task"],
        "hit_s2_task": (df["pred_task_stage2"] == df["true_task"]).fillna(False).astype(bool),
        "hit_joint_task": (df["pred_task"] == df["true_task"]).fillna(False).astype(bool),
    })

    temp_df["hit_callsign"] = [
        (str(p) in t) if (pd.notna(p) and isinstance(t, (list, np.ndarray))) else False
        for p, t in zip(df["pred_callsign"], df["true_callsigns"])
    ]

    temp_df["hit_s2_strict"] = temp_df["hit_s2_task"] & temp_df["hit_callsign"]
    temp_df["hit_joint_strict"] = temp_df["hit_joint_task"] & temp_df["hit_callsign"]

    active = temp_df[temp_df["true_task"] != idle_label]

    # --- HELPER: Calculates metrics for any given slice ---
    def _calc_slice_metrics(slice_df: pd.DataFrame) -> dict[str, float | None]:
        if slice_df.empty:
            return {f"{prefix}_{m}_acc": None for prefix in ["frame", "inst_any", "inst_cov", "inst_50p", "inst_75p"] for m in ["callsign", "s2_task", "s2_strict", "joint_task", "joint_strict"]}

        frame_means = slice_df[["hit_callsign", "hit_s2_task", "hit_s2_strict", "hit_joint_task", "hit_joint_strict"]].mean()
        
        instance_coverage = slice_df.groupby("instance_id")[
            ["hit_callsign", "hit_s2_task", "hit_s2_strict", "hit_joint_task", "hit_joint_strict"]
        ].mean()

        inst_any = (instance_coverage > 0).mean()
        inst_cov = instance_coverage.mean()
        inst_50p = (instance_coverage >= 0.50).mean()
        inst_75p = (instance_coverage >= 0.75).mean()

        res = {}
        for m in ["callsign", "s2_task", "s2_strict", "joint_task", "joint_strict"]:
            res[f"frame_{m}_acc"] = float(frame_means[f"hit_{m}"])
            res[f"inst_any_{m}_acc"] = float(inst_any[f"hit_{m}"])
            res[f"inst_cov_{m}_acc"] = float(inst_cov[f"hit_{m}"])
            res[f"inst_50p_{m}_acc"] = float(inst_50p[f"hit_{m}"])
            res[f"inst_75p_{m}_acc"] = float(inst_75p[f"hit_{m}"])
        return res

    # --- EXECUTE ---
    return {
        "global": _calc_slice_metrics(active),
        **{
            task_name: _calc_slice_metrics(active[active["true_task"] == task_name])
            for task_name in labels
        }
    }

def evaluate_atl(df: pd.DataFrame, idle_label: str = "IDLE") -> dict[str, Any]:
    """
    Evaluates the prediction validity against the Active Task List (ATL).
    Measures Contextual Precision, Recall, and the Validity of Correct/Incorrect
    predictions for both Stage 2 and the Joint pipeline.
    """
    total_predictions = len(df)
    if total_predictions == 0:
        return {
            "context_coverage": None,
            "gt_contextual_recall": None,
            "joint_contextual_precision": None,
            "joint_contextual_recall": None,
            "stage2_contextual_recall": None,
            "joint_correct_validity": None,
            "joint_incorrect_validity": None,
            "stage2_correct_validity": None,
            "stage2_incorrect_validity": None,
        }

    # ==========================================
    # 1. Context Coverage (Data Gaps)
    # ==========================================
    no_context_count = int(df["atl_timestamp"].isna().sum())
    context_coverage = (total_predictions - no_context_count) / total_predictions

    # ==========================================
    # 2. Joint Contextual Precision (Precision)
    #    "When the joint system predicted an action, was it possible?"
    # ==========================================
    precision_mask = (df["pred_task"] != idle_label) & df["atl_timestamp"].notna()
    total_predictions_made = int(precision_mask.sum())
    
    joint_contextual_precision = None
    if total_predictions_made > 0:
        valid_predictions = int(df.loc[precision_mask, "atl_rank"].notna().sum())
        joint_contextual_precision = valid_predictions / total_predictions_made

    # ==========================================
    # 3. Joint Contextual Recall (Recall)
    #    "When a task happened, did the joint system suggest something possible?"
    # ==========================================
    recall_mask = (df["true_task"] != idle_label) & df["atl_timestamp"].notna()
    total_true_task_time = int(recall_mask.sum())
    
    joint_contextual_recall = None
    if total_true_task_time > 0:
        covered_task_time = int(df.loc[recall_mask, "atl_rank"].notna().sum())
        joint_contextual_recall = covered_task_time / total_true_task_time

    # ==========================================
    #    Ground Truth Baseline Recall
    #    "Given a true task, and using the predicted callsign, how often was it valid?"
    # ==========================================
    gt_contextual_recall = None
    if total_true_task_time > 0:
        gt_covered_time = int(df.loc[recall_mask, "atl_rank_gt"].notna().sum())
        gt_contextual_recall = gt_covered_time / total_true_task_time

    # ==========================================
    # 4. Stage 2 Contextual Recall
    #    "Given an active task, does Stage 2's argmax select a possible option?"
    # ==========================================
    stage2_contextual_recall = None
    if total_true_task_time > 0:
        s2_covered_time = int(df.loc[recall_mask, "atl_rank_stage2"].notna().sum())
        stage2_contextual_recall = s2_covered_time / total_true_task_time

    # ==========================================
    # 5. Joint: Correct vs. Incorrect Validity
    # ==========================================
    # JOINT CORRECT: Predicted active task == True active task
    joint_correct_mask = (
        (df["true_task"] != idle_label) & 
        (df["pred_task"] == df["true_task"]) & 
        df["atl_timestamp"].notna()
    )
    total_joint_correct = int(joint_correct_mask.sum())
    joint_correct_validity = None
    if total_joint_correct > 0:
        valid_joint_correct = int(df.loc[joint_correct_mask, "atl_rank"].notna().sum())
        joint_correct_validity = valid_joint_correct / total_joint_correct

    # JOINT INCORRECT: Predicted active task != True active task
    joint_incorrect_mask = (
        (df["pred_task"] != idle_label) & 
        (df["pred_task"] != df["true_task"]) & 
        df["atl_timestamp"].notna()
    )
    total_joint_incorrect = int(joint_incorrect_mask.sum())
    joint_incorrect_validity = None
    if total_joint_incorrect > 0:
        valid_joint_incorrect = int(df.loc[joint_incorrect_mask, "atl_rank"].notna().sum())
        joint_incorrect_validity = valid_joint_incorrect / total_joint_incorrect

    # ==========================================
    # 6. Stage 2: Correct vs. Incorrect Validity (YOUR MAIN FOCUS)
    # ==========================================
    # STAGE 2 CORRECT: Stage 2's argmax == True active task
    s2_correct_mask = (
        (df["true_task"] != idle_label) & 
        (df["pred_task_stage2"] == df["true_task"]) & 
        df["atl_timestamp"].notna()
    )
    total_s2_correct = int(s2_correct_mask.sum())
    stage2_correct_validity = None
    if total_s2_correct > 0:
        valid_s2_correct = int(df.loc[s2_correct_mask, "atl_rank_stage2"].notna().sum())
        stage2_correct_validity = valid_s2_correct / total_s2_correct

    # STAGE 2 INCORRECT: Stage 2's argmax != True active task
    s2_incorrect_mask = (
        (df["true_task"] != idle_label) & 
        (df["pred_task_stage2"] != df["true_task"]) & 
        df["atl_timestamp"].notna()
    )
    total_s2_incorrect = int(s2_incorrect_mask.sum())
    stage2_incorrect_validity = None
    if total_s2_incorrect > 0:
        valid_s2_incorrect = int(df.loc[s2_incorrect_mask, "atl_rank_stage2"].notna().sum())
        stage2_incorrect_validity = valid_s2_incorrect / total_s2_incorrect

    return {
        "context_coverage": float(context_coverage),
        "gt_contextual_recall": float(gt_contextual_recall) if gt_contextual_recall is not None else None,
        "joint_contextual_precision": float(joint_contextual_precision) if joint_contextual_precision is not None else None,
        "joint_contextual_recall": float(joint_contextual_recall) if joint_contextual_recall is not None else None,
        "stage2_contextual_recall": float(stage2_contextual_recall) if stage2_contextual_recall is not None else None,
        "joint_correct_validity": float(joint_correct_validity) if joint_correct_validity is not None else None,
        "joint_incorrect_validity": float(joint_incorrect_validity) if joint_incorrect_validity is not None else None,
        "stage2_correct_validity": float(stage2_correct_validity) if stage2_correct_validity is not None else None,
        "stage2_incorrect_validity": float(stage2_incorrect_validity) if stage2_incorrect_validity is not None else None,
    }