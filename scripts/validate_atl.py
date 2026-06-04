import argparse
import io
import zipfile
import re
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Iterable
import pandas as pd
import logging
import json
import numpy as np

from task_prediction.models import TaskType, ActiveTaskType

logger = logging.getLogger(__name__)

class TaskCallsignsTuple(NamedTuple):
    task: ActiveTaskType
    callsigns: frozenset[str]

# Map exact log prefixes to Enum names
LOG_TO_ENUM_MAP: dict[str, ActiveTaskType] = {
    "aircraft_request": TaskType.AIRCRAFT_REQUEST,
    "assume_aircraft": TaskType.ASSUME,
    "conflict_resolution": TaskType.CONFLICT_RESOLUTION,
    "entry_conditions": TaskType.ENTRY_CONDITIONS,
    "entry_conflict_resolution": TaskType.ENTRY_CONFLICT_RESOLUTION,
    "entry_coordination": TaskType.ENTRY_COORDINATION,
    "exit_conditions": TaskType.EXIT_CONDITIONS,
    "exit_conflict_resolution": TaskType.EXIT_CONFLICT_RESOLUTION,
    "exit_coordination": TaskType.EXIT_COORDINATION,
    "non_conformance_resolution": TaskType.NON_CONFORMANCE_RESOLUTION,
    "qos_improvement": TaskType.QUALITY_OF_SERVICE,
    "return_to_route": TaskType.RETURN_TO_ROUTE,
    "transfer_aircraft": TaskType.TRANSFER,
    "zone_conflict": TaskType.ZONE_CONFLICT,
}

PREFIXES = sorted(LOG_TO_ENUM_MAP.keys(), key=len, reverse=True)
FILENAME_PATTERN = re.compile(r"(?P<participant_id>\d+)_[a-zA-Z0-9]+_scenario_(?P<scenario_id>\d+)_log_ATL\.txt$")


def parse_active_task_item(item_str: str) -> TaskCallsignsTuple:
    """Parses simulator strings into an Enum name and a frozenset of callsigns."""
    for prefix in PREFIXES:
        if item_str.startswith(prefix):
            remainder = item_str[len(prefix):].strip("_")
            callsigns = frozenset(remainder.split("_")) if remainder else frozenset()
            return TaskCallsignsTuple(LOG_TO_ENUM_MAP[prefix], callsigns)
            
    raise ValueError(f"Could not derive task from '{item_str}'.")


def parse_active_tasks_log(lines: Iterable[str]) -> dict[datetime, set[TaskCallsignsTuple]]:
    """Parses the text log and returns a dictionary mapping timestamps to sets of active tasks."""
    pattern = re.compile(r"Tasks at ([\d\-T:]+):\s*(\[.*\])")
    data: dict[datetime, set[TaskCallsignsTuple]] = {}
    
    for line in lines:
        match = pattern.search(line)
        if match:
            sim_ts_str, array_str = match.groups()
            timestamp = datetime.strptime(sim_ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            raw_items = ast.literal_eval(array_str)
            
            data[timestamp] = {
                parse_active_task_item(item_str)
                for item_str in raw_items
            }

    return data


def process_zip_dataset(zip_path: Path) -> dict[tuple[int, int], dict[datetime, set[TaskCallsignsTuple]]]:
    """Reads the zip file and returns the parsed ATL registry grouped by (participant_id, scenario_id)."""
    atl_registry = {}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            if file_info.is_dir():
                continue
                
            match = FILENAME_PATTERN.match(file_info.filename)
            if not match:
                continue
                
            p_id, s_id = int(match.group("participant_id")), int(match.group("scenario_id"))
            
            with z.open(file_info) as f:
                text_stream = io.TextIOWrapper(f, encoding="utf-8")
                atl_registry[(p_id, s_id)] = parse_active_tasks_log(text_stream)
                
    return atl_registry


def audit_gt_validity(preds_df: pd.DataFrame, atl_run_data: dict[datetime, set[TaskCallsignsTuple]]) -> dict:
    """Evaluates the predictions dataframe against the in-memory ATL registry."""
    # Filter out IDLE, failed inference, and missing contexts
    active_preds = preds_df[
        (preds_df["true_task"] != "IDLE") & 
        (preds_df["status"] == "OK") & 
        preds_df["atl_timestamp"].notna()
    ]

    if active_preds.empty or not atl_run_data:
        return {"total_checked": 0, "valid_gt_pct": None}

    valid_count = 0
    total_count = 0

    for row in active_preds.itertuples(index=False):
        # Fetch valid set of tasks for this exact timestamp
        valid_set = atl_run_data.get(row.atl_timestamp)
        
        if not valid_set:
            total_count += 1
            continue

        # Reconstruct the Ground Truth into our strict NamedTuple
        # Convert string to Enum safely, and callsigns list to frozenset
        true_task_enum = TaskType[str(row.true_task)]
        
        # Handle PyArrow lists / Numpy arrays safely
        try:
            callsigns_iterable = row.true_callsigns if not pd.isna(row.true_callsigns).all() else []
        except ValueError: # Catch truth value of array ambiguous if it's a list
            callsigns_iterable = row.true_callsigns if row.true_callsigns is not None else []
            
        t_callsigns = frozenset(callsigns_iterable)
        
        target_struct = TaskCallsignsTuple(true_task_enum, t_callsigns)

        # O(1) Lookup Check
        if target_struct in valid_set:
            valid_count += 1
            
        total_count += 1

    return {
        "total_checked": total_count,
        "valid_gt_pct": float((valid_count / total_count * 100)) if total_count > 0 else None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Ground-Truth Validity directly from ZIP logs."
    )
    parser.add_argument("zip_path", type=Path, help="Path to the input ZIP archive containing the ATL logs.")
    parser.add_argument("predictions_parquet", type=Path, help="Path to predictions.parquet.")
    parser.add_argument("--output", type=Path, default=Path("gt_sync_audit.json"), help="Output JSON report path.")
    args = parser.parse_args()
    
    if not args.zip_path.exists():
        parser.error(f"The input zip file does not exist: {args.zip_path}")
    if not args.predictions_parquet.exists():
        parser.error(f"The predictions file does not exist: {args.predictions_parquet}")
    
    logger.info("Parsing ATL logs from Zip...")
    atl_registry = process_zip_dataset(args.zip_path)
    
    if not atl_registry:
        logger.warning("No matching log files were found in the zip. Exiting.")
        return

    logger.info("Loading Predictions Parquet...")
    preds_df = pd.read_parquet(args.predictions_parquet, dtype_backend="pyarrow")

    global_results = {}
    
    logger.info("Starting Ground-Truth Validity Audit...")
    for (scen_id, part_id), group_preds in preds_df.groupby(["scenario_id", "participant_id"]):
        run_key = f"run_p{part_id}_s{scen_id}"
        
        # Get the parsed ATL dictionary for this specific run
        atl_run_data = atl_registry.get((part_id, scen_id), {})
        
        stats = audit_gt_validity(group_preds, atl_run_data)
        global_results[run_key] = stats
        
        if stats["valid_gt_pct"] is not None:
            logger.info(f"{run_key} GT Validity: {stats['valid_gt_pct']:.2f}%")

    # Calculate Global Summary
    valid_runs = [r["valid_gt_pct"] for r in global_results.values() if r["valid_gt_pct"] is not None]

    global_results["global_summary"] = {
        "average_gt_validity_pct": float(np.mean(valid_runs)) if valid_runs else None,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=2)
        
    logger.info("Audit Complete! Saved report to %s", args.output)
    logger.info("Global Ground-Truth Validity Rate: %.2f%%", global_results["global_summary"]["average_gt_validity_pct"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()