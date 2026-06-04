import argparse
import io
import zipfile
import re
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Iterable
import pyarrow.parquet as pq
import logging

from task_prediction.models import TaskType, ActiveTaskType, ActiveTaskListEntry
from task_prediction.adapters.pyarrow.builders import ATL_DEFINITION

logger = logging.getLogger(__name__)

class TaskCallsignPair(NamedTuple):
    task: ActiveTaskType
    callsigns: list[str]

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

# Sort descending by length to safely match prefixes ('entry_conflict_resolution' before 'entry_conflict')
PREFIXES = sorted(LOG_TO_ENUM_MAP.keys(), key=len, reverse=True)

# Matches filenames like "001_april_scenario_1_log_ATL.txt"
FILENAME_PATTERN = re.compile(r"(?P<participant_id>\d+)_[a-zA-Z0-9]+_scenario_(?P<scenario_id>\d+)_log_ATL\.txt$")

def parse_active_task_item(item_str: str) -> TaskCallsignPair:
    """Parses simulator strings into an Enum name and a list of callsigns."""
    for prefix in PREFIXES:
        if item_str.startswith(prefix):
            return TaskCallsignPair(LOG_TO_ENUM_MAP[prefix], item_str[len(prefix):].strip("_").split("_"))
            
    raise ValueError(f"Could not derive task from '{item_str}'.")

def parse_active_tasks_log(lines: Iterable[str], participant_id: int, scenario_id: int) -> list[ActiveTaskListEntry]:
    """Parses the text log and returns a pre-exploded list of rows."""
    pattern = re.compile(r"Tasks at ([\d\-T:]+):\s*(\[.*\])")
    rows = []
    
    for line in lines:
        match = pattern.search(line)
        if match:
            sim_ts_str, array_str = match.groups()
            timestamp = datetime.strptime(sim_ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            raw_items = ast.literal_eval(array_str)
            
            # Enumerate gives us the exact Priority Rank from the simulator (0-indexed)
            for i, item_str in enumerate(raw_items):
                task_type, callsigns = parse_active_task_item(item_str)
                    
                # Pre-explode the data! (Creates multiple rows for multi-aircraft conflicts)
                for callsign in callsigns:
                    rows.append(
                        ActiveTaskListEntry(
                            timestamp,
                            participant_id,
                            scenario_id,
                            task_type,
                            callsign,
                            i + 1 # 1-based rank
                        )
                    )

    return rows

def process_zip_dataset(zip_path: Path) -> list[ActiveTaskListEntry]:
    """Reads the zip file, finds matching logs, and processes them."""
    all_atls = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            # Skip directories inside the zip
            if file_info.is_dir():
                continue
                
            match = FILENAME_PATTERN.match(file_info.filename)
            if not match:
                continue
                
            p_id, s_id = int(match.group("participant_id")), int(match.group("scenario_id"))
            
            with z.open(file_info) as f:
                # Wrap binary stream in TextIOWrapper for line-by-line text reading
                text_stream = io.TextIOWrapper(f, encoding="utf-8")
                rows = parse_active_tasks_log(text_stream, p_id, s_id)
                all_atls.extend(rows)
                
    return all_atls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and parse active task logs from a zip archive into a Parquet dataset."
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to the input ZIP archive containing the log files."
    )
    parser.add_argument(
        "dataset_folder",
        type=Path,
        help="Directory where the parsed Parquet dataset will be saved."
    )
    
    args = parser.parse_args()
    
    # Simple validation
    if not args.zip_path.exists():
        parser.error(f"The input zip file does not exist: {args.zip_path}")
        
    if not args.dataset_folder.exists():
        parser.error(f"The input dataset folder does not exist: {args.dataset_folder}")
    
    all_atls = process_zip_dataset(args.zip_path)
    
    if not all_atls:
        logger.warning("No matching log files were found or processed.")
        return

    pq.write_table(
        ATL_DEFINITION.build_table(all_atls),
        output_path := args.dataset_folder / "atl.parquet",
        compression="zstd"
    )
    logger.info("Successfully wrote dataset to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()