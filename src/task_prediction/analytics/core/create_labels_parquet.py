import re
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime, timezone

from ...models import TaskLabel, TaskType, AsaSupportMode, RunId, TaskGroundTruth
from ...adapters.pyarrow.builders import TASK_LABEL_DEFINITION, TASK_GT_DEFINITION
from .ground_truth import build_ground_truth_boundaries
from .alignment import align_preds_with_gt, align_preds_with_aircraft_attention

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d  %H:%M:%S.%f").replace(tzinfo=timezone.utc)

def _read_asa_support_mode(dataset_folder: Path) -> dict[RunId, AsaSupportMode]:
    mode_df = pd.read_csv(dataset_folder / "asa_support_modes.csv")
    
    mode_mapping: dict[RunId, AsaSupportMode] = {}
    
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
    all_gt: list[TaskGroundTruth] = []
    all_preds: list[pd.DataFrame] = []
    all_ac_attentions: list[pd.DataFrame] = []
    
    for entry in dataset_folder.iterdir():
        if entry.is_dir():
            match = session_pattern.match(entry.name)
            if match:
                run_id = RunId(int(match.group(1)), int(match.group(2)))
                
                json_path = entry / "taskRecognition" / f"{entry.name}_task_labelled.json"
                if json_path.exists():
                    run_labels = process_scenario_labels(
                            json_path,
                            run_id.participant_id,
                            run_id.scenario_id,
                            asa_mode_mapping[run_id]
                        )
                    
                    all_gt.extend(build_ground_truth_boundaries(run_labels, run_id))
                    all_labels.extend(run_labels)
                    
                else:
                    logger.warning("Labels missing for session: %s", entry.name)

                pred_path = entry / "taskRecognition" / f"{entry.name}_task_prediction.parquet"
                if pred_path.exists():
                    run_preds_df = pd.read_parquet(pred_path, dtype_backend="pyarrow")
                    
                    # Inject the missing identifiers
                    run_preds_df["participant_id"] = run_id.participant_id
                    run_preds_df["scenario_id"] = run_id.scenario_id

                    run_preds_df = run_preds_df.astype({
                        "participant_id": "int8[pyarrow]", 
                        "scenario_id": "int8[pyarrow]"
                    })
                    
                    all_preds.append(run_preds_df)
                
                ac_attention_path = entry / "taskRecognition" / f"{entry.name}_aircraft_attention.parquet"
                if ac_attention_path.exists():
                    run_ac_attention_df = pd.read_parquet(ac_attention_path, dtype_backend="pyarrow")
                    
                    # Inject the missing identifiers
                    run_ac_attention_df["participant_id"] = run_id.participant_id
                    run_ac_attention_df["scenario_id"] = run_id.scenario_id

                    run_ac_attention_df = run_ac_attention_df.astype({
                        "participant_id": "int8[pyarrow]", 
                        "scenario_id": "int8[pyarrow]"
                    })
                    
                    all_ac_attentions.append(run_ac_attention_df)

    if not all_labels:
        logger.error("No labels found across any scenario folders. Exiting.")
        return
    
    pq.write_table(
        TASK_LABEL_DEFINITION.build_table(all_labels),
        dataset_folder / "labels.parquet",
        compression="zstd"
    )

    pq.write_table(
        gt_table := TASK_GT_DEFINITION.build_table(all_gt),
        dataset_folder / "ground_truth.parquet",
        compression="zstd"
    )

    logger.info("Successfully wrote %d labels.", len(all_labels))

    if not all_ac_attentions:
        logger.error("No aircraft attention files found.")
    else:
        ac_attention_df = pd.concat(all_ac_attentions, ignore_index=True)
        ac_attention_df.to_parquet(dataset_folder / "aircraft_attentions.parquet")
    
    if not all_preds:
        logger.error("No prediction files found.")
    else:
        # Combine into a single master predictions DataFrame
        preds_df = pd.concat(all_preds, ignore_index=True)
        aligned_df = align_preds_with_gt(preds_df, gt_table.to_pandas(types_mapper=pd.ArrowDtype))

        def get_argmax_task(probas: list[tuple[str, float]] | None) -> str | None:
            return pd.NA if probas is pd.NA else max(probas, key=lambda x: x[1])[0]

        # Apply to every row to get the hypothetical Stage 2 prediction
        aligned_df["pred_task_stage2"] = aligned_df["task_probas"].apply(get_argmax_task)

        if all_ac_attentions:
            aligned_df = align_preds_with_aircraft_attention(aligned_df, ac_attention_df)
        
        aligned_df.to_parquet(dataset_folder / "predictions.parquet")

        logger.info("Successfully wrote %d predictions with labels.", len(aligned_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=Path)
    parser.add_argument("-f", "--force", default=False, action="store_true")
    args = parser.parse_args()

    if (args.dataset_folder / "labels.parquet").exists() and not args.force:
        logger.info("File 'labels.parquet' already exists. Use '--force' to overwrite.")
        exit()

    if (args.dataset_folder / "ground_truth.parquet").exists() and not args.force:
        logger.info("File 'ground_truth.parquet' already exists. Use '--force' to overwrite.")
        exit()

    if (args.dataset_folder / "predictions.parquet").exists() and not args.force:
        logger.info("File 'predictions.parquet' already exists. Use '--force' to overwrite.")
        exit()
    
    process_dataset_labels(args.dataset_folder)