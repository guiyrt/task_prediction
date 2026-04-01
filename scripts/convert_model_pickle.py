import joblib
import json
import xgboost as xgb
import argparse
from pathlib import Path

def convert_model(pkl_path: Path, prefix: str):
    print(f"Converting {pkl_path}...")
    bundle = joblib.load(pkl_path)
    model: xgb.XGBClassifier = bundle["model"]
    folder: Path = pkl_path.parent
    
    # Save the tree structures natively (Future-proof)
    model.save_model(folder / f"{prefix}_model.ubj")
    
    # Extract and save the Python metadata safely
    metadata = {
        "feature_names": list(model.feature_names_in_),
        "classes": [int(c) for c in model.classes_] if hasattr(model, 'classes_') else []
    }

    if "threshold" in bundle:
        metadata["threshold"] = bundle["threshold"]
    
    with open(folder / f"{prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {prefix}_model.ubj and {prefix}_metadata.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    args = p.parse_args()

    convert_model(args.folder / "model_stageA.pkl", "stageA")
    convert_model(args.folder / "model_stageB.pkl", "stageB")