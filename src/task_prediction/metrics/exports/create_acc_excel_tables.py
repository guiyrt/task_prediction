import argparse
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any

from task_prediction.models import ACTIVE_TASK_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_acc_excel_tables(
    accuracy_results: dict[str, Any], 
    task_order: list[str] = ACTIVE_TASK_NAMES, 
    output_dir: Path = Path(".")
):
    """
    Exports 3 formatted tables (Callsign, Task, Strict) to an Excel file.
    Preserves strict row order and uses '-' for tasks with no occurrences.
    """
    # Convert the nested dictionary into a Pandas DataFrame
    df = pd.DataFrame(accuracy_results).T
    
    # Reorder rows to match exact task_order, with 'global' at the bottom
    ordered_index = task_order + ['global']
    df = df.reindex(ordered_index)
    
    # Rename 'global' to 'Total', and clean up Enum names for readability
    # e.g. "ENTRY_CONFLICT_RESOLUTION" -> "Entry Conflict Resolution"
    clean_index = {idx: idx.replace("_", " ").title() for idx in task_order}
    clean_index['global'] = 'Total'
    df.rename(index=clean_index, inplace=True)
    df.index.name = "Task Type"
    
    # 4. Define the 3 tables we want to extract
    target_tables = {
        "Callsign": "callsign",
        "Task (Stage 2)": "s2_task",
        "Callsign & Task (Stage 2)": "s2_strict",
        "Task (Joint)": "joint_task",
        "Callsign & Task (joint)": "joint_strict",
    }

    # Define the columns we want and their "Pretty" names
    metric_prefixes = ["frame", "inst_any", "inst_cov", "inst_50p", "inst_75p"]
    pretty_names = ["Frame", "Inst (Any)", "Inst (Avg Cov)", "Inst (≥50%)", "Inst (≥75%)"]

    # Extract, format, and save to Excel
    # We use engine='openpyxl' to support modern .xlsx formatting
    output_path = output_dir / "accuracy_metrics_tables.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for table_title, metric_suffix in target_tables.items():
            
            # Reconstruct the exact column names used in the JSON dictionary
            raw_columns = [f"{prefix}_{metric_suffix}_acc" for prefix in metric_prefixes]
            
            # Ensure the columns exist in the JSON (handles missing data gracefully)
            available_cols = [c for c in raw_columns if c in df.columns]
            table_df = df[available_cols].copy()
            
            # Rename columns to the pretty names
            # (Zip ensures we only rename columns that were actually found)
            table_df.columns = [
                pretty_names[i] for i, c in enumerate(raw_columns) if c in df.columns
            ]
            
            # Format the floats to 3 decimals, then replace NaNs with '-'
            table_df = table_df.round(3).fillna("-")
            
            # Write to its own sheet in the Excel file
            table_df.to_excel(writer, sheet_name=table_title)

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready Word/Excel tables from a JSON metrics file."
    )
    parser.add_argument(
        "json_metrics_file",
        type=Path,
        help="Path to the JSON file containing the evaluate_accuracy output."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory path to save the resulting Excel file."
    )
    
    args = parser.parse_args()
    
    if not args.json_metrics_file.exists():
        parser.error(f"The input JSON file does not exist: {args.json_metrics_file}")
        
    # Load the JSON file
    with open(args.json_metrics_file, 'r', encoding='utf-8') as f:
        try:
            results_dict = json.load(f)
        except json.JSONDecodeError as e:
            parser.error(f"Failed to parse JSON file: {e}")
            
    # Generate the tables using the ACTIVE_TASK_NAMES imported from your models
    create_acc_excel_tables(results_dict["accuracy"], ACTIVE_TASK_NAMES, args.output)

if __name__ == "__main__":
    main()