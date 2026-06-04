import pandas as pd
import pyarrow as pa
import logging

logger = logging.getLogger(__name__)

def align_preds_with_gt(
    predictions_df: pd.DataFrame,
    gt_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Safely aligns irregular prediction timestamps with the LIFO ground truth.
    Uses the 'by' parameter to align across all runs simultaneously.
    """
    # Ensure strictly sorted by time (Required by merge_asof)
    predictions_df = predictions_df.sort_values(by="timestamp").reset_index(drop=True)
    gt_df = gt_df.sort_values(by="timestamp").reset_index(drop=True)

    # Vectorized As-Of Merge
    aligned_df = pd.merge_asof(
        predictions_df,
        gt_df,
        on="timestamp",
        by=["participant_id", "scenario_id"],
        direction="backward"
    )

    # Drop out-of-bounds predictions (Predictions made before the first label, or after the terminal NA)
    initial_count = len(aligned_df)
    aligned_df = aligned_df.dropna(subset=["true_task"])
    
    dropped = initial_count - len(aligned_df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} out-of-bounds predictions.")

    return aligned_df

def align_preds_with_aircraft_attention(
    predictions_df: pd.DataFrame, 
    attention_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts the most attended callsign and merges it into the task predictions.
    """
    if attention_df.empty:
        logger.warning("Attention DataFrame is empty. Filling pred_callsign with NA.")
        predictions_df["pred_callsign"] = pd.NA
        return predictions_df

    # Extract the Top-1 Callsign
    top_callsigns = [
        c[0] if c else pd.NA 
        for c in attention_df["callsigns"]
    ]
    
    # Create slim dataframe to avoid duplicating the heavy list arrays in the merge
    slim_attention = attention_df[["participant_id", "scenario_id", "timestamp"]].copy()
    slim_attention["pred_callsign"] = pd.Series(top_callsigns, dtype=pd.ArrowDtype(pa.dictionary(pa.int16(), pa.string())))

    # Strict sorting required by merge_asof
    preds_sorted = predictions_df.sort_values("timestamp").reset_index(drop=True)
    slim_attention = slim_attention.sort_values("timestamp").reset_index(drop=True)

    # Attach the latest attention state to the task prediction
    return pd.merge_asof(
        preds_sorted,
        slim_attention,
        on="timestamp",
        by=["participant_id", "scenario_id"],
        direction="backward"
    )