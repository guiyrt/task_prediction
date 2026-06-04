# task_prediction

## Commands

### uv run task-pred
Starts task prediction

### uv run analytics <DATASET_PATH>
Starts UI to analyze labels. NOTE: requires `process_labels_and_preds`, will auto-execute if not previously executed.

### uv run metrics <DATASET_PATH>
Generates prediction metrics into `<DATASET_PATH>/metrics`. NOTE: requires `process_labels_and_preds`
At the moment, ATL metrics do not consider order, as prioritization list was not redy in time.


## Data processing scripts

### uv run process_polaris_db <DATASET_PATH>
Finds runs in DATASET_PATH and extracts /asdEvents, assuming files `simulator/XXX_Y_scenario_Z_simdata.db` exist.

### uv run process_labels_and_preds <DATASET_PATH>
Creates the following files:
- `aircraft_attentions.parquet`: Table with all callsign predictions centralized.
- `labels.parquet`: Table with all of the annotations centralized.
- `ground_truth.parquet`: Table with non-overlapping labels, representing what task is the ground truth at any given time (by start time).
- `predictions.parquet`: Contains all the data required to compute metrics, as it includes ground truth, stage 1, stage 2 and joint predictions, as well as callsigns and active task list (ATL) rank.

### uv run process_atl <ZIPFILE_PATH> <DATASET_PATH>
Creates `atl.parquet`, that contains the "exploded" pairs of (task, callsign, rank), where rank is the position in the task list. At the moment, we don't care about the rank itself because there is no prioritization, but it's there so it works once it is available.