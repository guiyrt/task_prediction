import logging
import xgboost as xgb
import numpy as np
from pathlib import Path
import json
from typing import Final

from ..models import InferenceResult, TaskType
from ..utils.types import FeatureDict

logger = logging.getLogger(__name__)

class TaskPredictor:
    """
    Handles hierarchical XGBoost inference with real-time EMA smoothing.
    
    Stage A: Idle vs Active
    Stage B: Task Classification (if Active)
    """
    
    def __init__(
        self, 
        model_dir: Path,
        alpha_smooth: float = 0.6,
        always_validate_input: bool = False
    ):
        self.alpha = alpha_smooth
        
        # Load models
        self.booster_a = xgb.Booster()
        self.booster_a.load_model(model_dir / "stageA_model.ubj")
        self.booster_a.set_param({"device": "cpu", "n_jobs": 1})

        self.booster_b = xgb.Booster()
        self.booster_b.load_model(model_dir / "stageB_model.ubj")
        self.booster_b.set_param({"device": "cpu", "n_jobs": 1})

        # Load Metadata
        with open(model_dir / "stageA_metadata.json", "r") as f:
            metadata_a = json.load(f)
        with open(model_dir / "stageB_metadata.json", "r") as f:
            metadata_b = json.load(f)

        if metadata_a["feature_names"] != metadata_b["feature_names"]:
            raise ValueError(
                "CRITICAL: Stage A and Stage B models expect different features "
                "or feature ordering. Please check the training pipeline."
            )
            
        self.threshold_a = metadata_a["threshold"]
        self._feature_order: Final[list[str]] = metadata_a["feature_names"]
        self._expected_keys: Final[set[str]] = set(self._feature_order)
        self._active_classes: Final[list[int]] = metadata_b["classes"]
        
        self.is_initial_validation = True
        self.always_validate_input = always_validate_input
        
        # State for Real-Time EMA Smoothing
        self._pB: np.ndarray | None = None

    def _prepare_input(self, features: FeatureDict) -> np.ndarray:
        return np.array([features.get(k, 0.0) for k in self._feature_order], dtype=np.float32).reshape(1, -1)

    def _validate_input(self, features: FeatureDict) -> None:
        # Convert sets for comparison
        actual_keys = set(features.keys())

        # Features that are expected but not present
        if (missing := self._expected_keys - actual_keys):
            logger.warning(f"MISSING FEATURES ({len(missing)}): {list(missing)}... (will be filled with 0.0)")
        
        # Features that are present but not used
        if (extra := actual_keys - self._expected_keys):
            logger.warning(f"EXTRA FEATURES ({len(extra)}): {list(extra)}... (will be filled with 0.0)")
        
        for k, val in features.items():
            if not isinstance(val, (float, np.number, int, bool)):
                logger.error(f"DATA QUALITY ERROR: Feature '{k}' is {type(val)}. Expected numeric.")

    def reset_state(self) -> None:
        self._pB = None

    def predict(
        self, 
        features: FeatureDict, 
        force_stage_b: bool = False
    ) -> InferenceResult:
        """Executes hierarchical inference for a given snapshot in time."""
        
        if self.is_initial_validation or self.always_validate_input:
            self._validate_input(features)
            self.is_initial_validation = False
        
        x = self._prepare_input(features)
        
        # --- STAGE A: Active vs Idle ---
        preds_a = self.booster_a.inplace_predict(x)
        
        if preds_a.ndim > 1 and preds_a.shape[1] > 1:
            p_idle = float(preds_a[0][0])
            p_active = float(preds_a[0][1])
        else:
            p_active = float(preds_a.ravel()[0])
            p_idle = 1.0 - p_active
        
        # Short-circuit logic: If clearly idle, skip Stage B to save compute.
        is_active_gate = (p_active >= self.threshold_a)

        if not is_active_gate and not force_stage_b:
            # We are Idle. Reset Stage B smoothing state because the user stopped a task.
            self.reset_state()
            
            return InferenceResult(
                is_active=False,
                active_proba=p_active,
                pred_task=None,
                task_probas={}
            )
            
        # --- STAGE B: Task Classification ---
        pB_raw = self.booster_b.inplace_predict(x)[0]
        
        # Apply EMA Smoothing manually for real-time
        self._pB = (
            (self.alpha * pB_raw) + ((1.0 - self.alpha) * self._pB)
            if self._pB is not None
            else pB_raw
        )
        
        # --- COMBINE STAGES ---
        # P(Task_i) = P(Active) * P(Task_i | Active)
        combined_probas = {
            TaskType(task_id): p_active * float(self._pB[i])
            for i, task_id in enumerate(self._active_classes)
        }
            
        # Find the highest combined probability
        # Note: We compare against p_idle just to mathematically verify if Idle is still 
        # the dominant class, even though we passed the threshold gate.
        max_task = None
        max_p = p_idle
        is_active_final = False
        
        for task_type, proba in combined_probas.items():
            if proba > max_p:
                max_p = proba
                max_task = task_type
                is_active_final = True

        return InferenceResult(
            is_active=is_active_final,
            active_proba=p_active,
            pred_task=max_task if is_active_final else None,
            task_probas=combined_probas
        )