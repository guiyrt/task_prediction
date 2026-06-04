import argparse
import logging
import pandas as pd
from pathlib import Path

from ..models import TaskType, ACTIVE_TASK_NAMES
from .evaluator import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Live Inference Predictions")
    parser.add_argument("dataset_folder", type=Path, help="Folder containing all runs and processed predictions.parquet.")
    
    args = parser.parse_args()
    # 1. Load Data
    logger.info("Loading Data...")

    # Just look at actual prediction, when the system was OK
    preds_df = pd.read_parquet(args.dataset_folder / "predictions.parquet", dtype_backend="pyarrow")

    # Setup Evaluator
    evaluator = PipelineEvaluator(
        args.dataset_folder / "metrics",
        ACTIVE_TASK_NAMES,
        TaskType.IDLE.name,
        ["gaze_availability_pct", "gaze_availability_pct", "asd_events_count", "feature_extraction_time_ms", "inference_time_ms"]
    )
    
    # Run Evaluation
    logger.info("Starting hierarchical evaluation...")
    evaluator.evaluate(preds_df)
    
    # Export
    logger.info("Evaluation Complete!")

if __name__ == "__main__":
    main()