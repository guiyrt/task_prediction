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

def align_preds_with_atl_rank(
    preds_df: pd.DataFrame, 
    atl_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Appends three rank columns to the predictions:
      - `atl_rank_joint`: Rank of the Joint prediction (None if Stage 1 is False).
      - `atl_rank_stage2`: Rank of Stage 2's highest-probability prediction.
      - `atl_rank_gt`: Rank of the True Task paired with the Predicted Callsign (The theoretical ceiling).
    """
    if atl_df.empty:
        logger.warning("Active tasks DataFrame is empty. Assigning rank '<NA>' to all predictions.")
        preds_df["atl_rank_joint"] = pd.NA
        preds_df["atl_rank_stage2"] = pd.NA
        preds_df["atl_rank_gt"] = pd.NA
        return preds_df

    # 2. Cast join keys to PyArrow strings to prevent dictionary-index mismatches
    preds_df["pred_task_str"] = preds_df["pred_task"].astype("string[pyarrow]")
    preds_df["pred_task_stage2_str"] = preds_df["pred_task_stage2"].astype("string[pyarrow]")
    preds_df["true_task_str"] = preds_df["true_task"].astype("string[pyarrow]")
    preds_df["pred_callsign_str"] = preds_df["pred_callsign"].astype("string[pyarrow]")
    
    atl_df = atl_df.copy()
    atl_df["task_type_str"] = atl_df["task_type"].astype("string[pyarrow]")
    atl_df["callsign_str"] = atl_df["callsign"].astype("string[pyarrow]")

    # ==========================================
    # FIX: Deduplicate Task-Callsign Pairs
    # ==========================================
    # A single callsign can be involved in multiple tasks of the same type simultaneously 
    # (e.g. multi-aircraft conflicts `conflict_resolution_A_B` & `conflict_resolution_A_C`).
    # We sort by rank so we keep the highest priority rank (lowest int) for duplicated pairs,
    # avoiding extra rows being generated during the pd.merge(how="left") operations.
    atl_df = atl_df.sort_values(
        by=["participant_id", "scenario_id", "timestamp", "task_type_str", "callsign_str", "rank"]
    )
    atl_df = atl_df.drop_duplicates(
        subset=["participant_id", "scenario_id", "timestamp", "task_type_str", "callsign_str"],
        keep="first"
    )

    # 3. Extract unique timestamps for as-of alignment
    atl_times = atl_df[["participant_id", "scenario_id", "timestamp"]].drop_duplicates()
    atl_times = atl_times.rename(columns={"timestamp": "atl_timestamp"})
    
    preds_sorted = preds_df.sort_values(by="timestamp").reset_index(drop=True)
    atl_times = atl_times.sort_values(by="atl_timestamp").reset_index(drop=True)

    # 4. Time alignment (merge_asof)
    merged_df = pd.merge_asof(
        preds_sorted,
        atl_times,
        left_on="timestamp",
        right_on="atl_timestamp",
        by=["participant_id", "scenario_id"],
        direction="backward"
    )

    # ==========================================
    # 5. Join 1: The JOINT Pipeline Rank
    # ==========================================
    atl_flat_joint = atl_df[[
        "participant_id", "scenario_id", "timestamp", "task_type_str", "callsign_str", "rank"
    ]].rename(columns={
        "timestamp": "atl_timestamp",
        "task_type_str": "pred_task_str",
        "callsign_str": "pred_callsign_str",
        "rank": "atl_rank"
    })

    final_df = pd.merge(
        merged_df,
        atl_flat_joint,
        on=["participant_id", "scenario_id", "atl_timestamp", "pred_task_str", "pred_callsign_str"],
        how="left"
    )

    # ==========================================
    # 6. Join 2: The STAGE 2 Only Rank
    # ==========================================
    atl_flat_stage2 = atl_df[[
        "participant_id", "scenario_id", "timestamp", "task_type_str", "callsign_str", "rank"
    ]].rename(columns={
        "timestamp": "atl_timestamp",
        "task_type_str": "pred_task_stage2_str",
        "callsign_str": "pred_callsign_str",
        "rank": "atl_rank_stage2"
    })

    final_df = pd.merge(
        final_df,
        atl_flat_stage2,
        on=["participant_id", "scenario_id", "atl_timestamp", "pred_task_stage2_str", "pred_callsign_str"],
        how="left"
    )

    # ==========================================
    # 7. Join 3: The GROUND TRUTH Baseline Rank
    # ==========================================
    atl_flat_gt = atl_df[[
        "participant_id", "scenario_id", "timestamp", "task_type_str", "callsign_str", "rank"
    ]].rename(columns={
        "timestamp": "atl_timestamp",
        "task_type_str": "true_task_str",
        "callsign_str": "pred_callsign_str",
        "rank": "atl_rank_gt"
    })

    final_df = pd.merge(
        final_df,
        atl_flat_gt,
        on=["participant_id", "scenario_id", "atl_timestamp", "true_task_str", "pred_callsign_str"],
        how="left"
    )

    # ==========================================
    # 8. Final Cleanups & Type Casts
    # ==========================================
    final_df["atl_rank"] = final_df["atl_rank"].astype("uint8[pyarrow]")
    final_df["atl_rank_stage2"] = final_df["atl_rank_stage2"].astype("uint8[pyarrow]")
    final_df["atl_rank_gt"] = final_df["atl_rank_gt"].astype("uint8[pyarrow]")
    
    # Ensure IDLE predictions are strictly forced to <NA>
    invalid_joint_mask = final_df["pred_task"].isna() | (final_df["pred_task"] == "IDLE")
    final_df.loc[invalid_joint_mask, "atl_rank"] = pd.NA
    
    # Drop temporary join keys and intermediate columns
    columns_to_drop = [
        "pred_task_str", "pred_task_stage2_str", "true_task_str", "pred_callsign_str"
    ]
    final_df = final_df.drop(columns=[c for c in columns_to_drop if c in final_df.columns])

    # Clean up temporary columns in the original in-memory preds_df
    preds_df.drop(
        columns=["pred_task_stage2", "pred_task_str", "pred_task_stage2_str", "true_task_str", "pred_callsign_str"], 
        errors="ignore", 
        inplace=True
    )

    return final_df