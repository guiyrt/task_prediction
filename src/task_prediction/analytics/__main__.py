import argparse
import logging
from pathlib import Path

from .app.dashboard import create_app
from .core.create_labels_parquet import process_dataset_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ATC Task Analytics Dashboard")
    parser.add_argument(
        "dataset_folder", 
        type=Path, 
        help="Path to the folder containing labels.parquet"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8050, 
        help="Port to run the dashboard on (default: 8050)"
    )
    
    args = parser.parse_args()

    try:
        if not (args.dataset_folder / "labels.parquet").exists():
            process_dataset_labels(args.dataset_folder)

        # Generate the Dash app instance dynamically
        app = create_app(args.dataset_folder)
        
        # Run the server
        logging.info(f"Starting dashboard on http://127.0.0.1:{args.port}/")
        app.run(debug=True, host="127.0.0.1", port=args.port)
        
    except Exception as e:
        logging.error(f"Failed to start dashboard: {e}")