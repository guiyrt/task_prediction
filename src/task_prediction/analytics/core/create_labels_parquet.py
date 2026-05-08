import re
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from typing import NamedTuple
from datetime import datetime, timezone

from task_prediction.models import TaskLabel, TaskType, AsaSupportMode
from task_prediction.adapters.pyarrow.builders import TASK_LABEL_DEFINITION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RunId(NamedTuple):
    participant_id: int
    scenario_id: int

def _parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d  %H:%M:%S.%f").replace(tzinfo=timezone.utc)

def _read_asa_support_mode(dataset_folder: Path) -> dict[RunId, AsaSupportMode]:
    mode_df = pd.read_csv(dataset_folder / "asa_support_modes.csv")
    
    mode_mapping = {}
    
    for _, row in mode_df.iterrows():
        p_id = int(row["participant_id"])
        for s_id in range(1, 5):
            mode_mapping[RunId(p_id, s_id)] = AsaSupportMode[row[f"scenario_{s_id}"]]
    
    return mode_mapping

def process_scenario_labels(json_path: Path, participant_id: int, scenario_id: int, asa_support_mode: AsaSupportMode) -> list[TaskLabel]:
    with json_path.open("r", encoding="utf-8") as f:
        raw_labels: list[dict] = json.load(f)
    
    if len(raw_labels) == 0:
        logger.info("No labels present in %s.", json_path)
        return []
        
    labels = [
        TaskLabel(
            participant_id=participant_id,
            scenario_id=scenario_id,
            asa_support_mode=asa_support_mode,
            start_time=_parse_timestamp(item["start"]),
            end_time=_parse_timestamp(item["end"]),
            task_type=TaskType(item["task_type"] - 1), # 0 was considered "Invalid" in labelling tool
            callsigns=item.get("callsigns", [])
        )
        for item in raw_labels
    ]

    # Identify and skip labels with negative duration
    valid_labels: list[TaskLabel] = []
    for label in labels:
        if label.end_time > label.start_time:
            valid_labels.append(label)
        else:
            logger.warning("Ignored label with null or negative duration: %s", str(label))

    # Sort by start time
    valid_labels.sort(key=lambda x: (x.start_time, x.end_time))

    # Contains sorted tasks, including IDLE
    timeline: list[TaskLabel] = []
    
    # Track the furthest timestamp we have processed so far
    max_end_so_far = valid_labels[0].start_time 
    
    for label in valid_labels:
        # If the current task starts AFTER our furthest seen end time, we found a gap!
        if label.start_time > max_end_so_far:
            timeline.append(TaskLabel(
                participant_id=participant_id,
                scenario_id=scenario_id,
                asa_support_mode=asa_support_mode,
                start_time=max_end_so_far,
                end_time=label.start_time,
                task_type=TaskType.IDLE,
                callsigns=[]
            ))
        
        # Always append the actual task
        timeline.append(label)
        
        # Advance the high-water mark if this task extends further than any previous task
        if label.end_time > max_end_so_far:
            max_end_so_far = label.end_time

    return timeline

def process_dataset_labels(dataset_folder: Path):
    asa_mode_mapping = _read_asa_support_mode(dataset_folder) 
    session_pattern = re.compile(r"^(\d{3})_.+_scenario_(\d)$")
    
    all_labels: list[TaskLabel] = []
    
    for entry in dataset_folder.iterdir():
        if entry.is_dir():
            match = session_pattern.match(entry.name)
            if match:
                run_id = RunId(int(match.group(1)), int(match.group(2)))
                
                json_path = entry / "taskRecognition" / f"{entry.name}_task_labelled.json"
                if json_path.exists():
                    all_labels.extend(
                        process_scenario_labels(
                            json_path,
                            run_id.participant_id,
                            run_id.scenario_id,
                            asa_mode_mapping[run_id]
                        )
                    )
                else:
                    logger.warning("Labels missing for session: %s", entry.name)

    if not all_labels:
        logger.error("No labels found across any scenario folders.")
        return

    # Build the single PyArrow table
    pq.write_table(
        TASK_LABEL_DEFINITION.build_table(all_labels),
        dataset_folder / "labels.parquet",
        compression="zstd"
    )
    logger.info("Successfully wrote %d labels.", len(all_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=Path)
    parser.add_argument("-f", "--force", default=False, action="store_true")
    args = parser.parse_args()

    if not (args.dataset_folder / "labels.parquet").exists() or args.force:
        process_dataset_labels(args.dataset_folder)