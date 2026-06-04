import pandas as pd
import json
from pathlib import Path
from typing import Any
from collections import Counter

from .core import evaluate_binary, evaluate_multiclass, evaluate_continuous, evaluate_atl, evaluate_accuracy
from .resolvers import resolve_stage1, resolve_stage2, resolve_joint
from .exports import create_acc_excel_tables, create_global_cms
from ..models import ACTIVE_TASK_NAMES

class PipelineEvaluator:
    def __init__(
        self,
        output_dir: Path,
        active_labels: list[str],
        idle_label: str,
        continuous_metrics_to_eval: list[str]
    ):
        """
        Args:
            task_labels: List of all valid active task names (excluding IDLE).
            idle_label: The string/enum representing the idle state.
        """
        self.idle_label = idle_label
        self.active_labels = active_labels
        self.joint_labels = active_labels + [self.idle_label]
        self.continuous_metrics_to_eval = continuous_metrics_to_eval
        
        # Storage for persistence
        self.output_dir: Path = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def _save_metrics(self, metrics: dict[str, Any], filename: str) -> None:
        with open(self.output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _evaluate_group(self, df_group: pd.DataFrame):
        """Runs the 3 modes on a specific slice of data."""
        # Predictions that executed correctly
        ok_df_group = df_group[df_group["status"] == "OK"]

        # Time between predictions
        df_group["prediction_interval_sec"] = (
            df_group.groupby(["participant_id", "scenario_id"])["timestamp"]
            .diff()
            .dt.total_seconds()
        )

        return {
            "status_distribution": Counter(df_group["status"]),
            "prediction_interval_sec": evaluate_continuous(df_group["prediction_interval_sec"].dropna()),
            **{val: evaluate_continuous(ok_df_group[val]) for val in self.continuous_metrics_to_eval},
            "stage_1": evaluate_binary(*resolve_stage1(ok_df_group, self.idle_label)),
            "stage_2": evaluate_multiclass(*resolve_stage2(ok_df_group, self.idle_label), self.active_labels),
            "joint": evaluate_multiclass(*resolve_joint(ok_df_group, self.idle_label), self.joint_labels),
            "accuracy": evaluate_accuracy(ok_df_group, self.active_labels, self.idle_label),
            # "atl": evaluate_atl(ok_df_group)
        }

    def evaluate(self, aligned_df: pd.DataFrame):
        """Orchestrates the hierarchical evaluation of the aligned dataset."""
        print("Evaluating Global Metrics...")
        global_metrics = self._evaluate_group(aligned_df)
        self._save_metrics(global_metrics, "global.json")
        create_global_cms(global_metrics, self.output_dir)
        create_acc_excel_tables(global_metrics["accuracy"], ACTIVE_TASK_NAMES, self.output_dir)

        print("Evaluating per Participant...")
        self._save_metrics(
            metrics={
                f"participant_{p_id:03}": self._evaluate_group(group)
                for p_id, group in aligned_df.groupby("participant_id")
            },
            filename="per_participant.json"
        )

        print("Evaluating per Scenario...")
        self._save_metrics(
            metrics={
                f"scenario_{s_id}": self._evaluate_group(group)
                for s_id, group in aligned_df.groupby("scenario_id")
            },
            filename="per_scenario.json"
        )

        print("Evaluating per Run...")
        self._save_metrics(
            metrics={
                f"run_p{p_id:03}_s{s_id}": self._evaluate_group(group)
                for (p_id, s_id), group in aligned_df.groupby(["participant_id", "scenario_id"])
            },
            filename="per_run.json"
        )