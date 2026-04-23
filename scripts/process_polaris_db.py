import re
import sqlite3
import logging
import argparse
from pathlib import Path
from contextlib import contextmanager
from contextlib import closing
from tqdm import tqdm
import tempfile
import zipfile
from collections import defaultdict
import pyarrow.parquet as pq
from typing import Iterator

from task_prediction.adapters.proto.parsers.asd import parse_asd_proto, get_base_asd_event_type
from task_prediction.adapters.pyarrow.builders import ASD_EVENT_DEFINITIONS
from task_prediction.models import AsdEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_db_filepath(session_path: Path):
	session_id = session_path.name
	return session_path / "simulator" / f"{session_id}_simdata.zip"

@contextmanager
def resolve_database(session_path: Path):
    """
    Handles the XXX_scenario_Y -> XXX_sept_scenario_Y.zip convention.
    Always extracts the zip to a temporary directory and dynamically finds the .db file.
    """
    # 1. Parse the folder name to reconstruct the zip name
    zip_path = get_db_filepath(session_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip file not found: {zip_path}")

    logger.info("Extracting %s to temporary storage...", zip_path.name)

    # 2. Extract securely to an auto-deleting temp folder
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:    
            
            # Look for any file ending in .db inside the zip archive
            db_files = [f for f in zip_ref.namelist() if f.endswith('.db')]
            
            if not db_files:
                raise FileNotFoundError(f"No .db file found inside {zip_path.name}")
            
            if len(db_files) > 1:
                logger.warning(
                    "Multiple .db files found inside %s: %s. Using the first one: '%s'", 
                    zip_path.name, db_files, db_files[0]
                )
            
            target_db = db_files[0]
            
            # Extract only that file
            extracted_path = Path(zip_ref.extract(target_db, tmpdir))
            
            logger.info("Extraction complete. Processing '%s'...", target_db)
            
            # 3. Yield control back to the main script
            yield extracted_path

def extract_data(db_path: Path, out_folder: Path, session_id: str) -> None:
    data: dict[type[AsdEvent], list[AsdEvent]] = defaultdict(list)

    # Open SQLite db
    with closing(sqlite3.connect(f"{db_path.as_uri()}?mode=ro&immutable=1", uri=True)) as con:
        con.text_factory = bytes
        cur = con.cursor()
        
        total_rows = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        cur.execute("SELECT data FROM events ORDER BY id ASC")

        # We use a standard for-loop so we can 'continue' to skip rows
        pbar = tqdm(total=total_rows, unit="evt")
        
        for raw_data, in cur:
            event = parse_asd_proto(raw_data, from_string=True)

            if event is not None:
                data[get_base_asd_event_type(event)].append(event)

            pbar.update(1)
        pbar.close()
    
    for event_type, event_data in data.items():
        event_def = ASD_EVENT_DEFINITIONS[event_type]
        
        pq.ParquetWriter(
            out_folder / f"{session_id}_{event_def.name}.parquet",
            **event_def.parquet_kwargs
        ).write_table(
            table=event_def.build_table(event_data)
        )

def process_session(session_path: Path, force: bool = False) -> None:
    if not session_path.is_dir():
        logger.error(f"Folder not found: {session_path}")

    out_folder: Path = session_path / "asdEvents"

    if out_folder.exists() and not force:
        logger.debug("Skipping %s, asdEvents already exists.")
        return
    
    out_folder.mkdir(exist_ok=True)
    
    try:
        with resolve_database(session_path) as db_path:
            extract_data(db_path, out_folder, session_path.name)
    except FileNotFoundError:
        logger.warning("Skipped %s, zip file not found.", str(session_path))

def discover_sessions(root_path: Path) -> Iterator[Path]:
    """
    Yields valid session directories within the root_path.
    Uses os.scandir internally via Path.iterdir for better performance.
    """
    session_pattern = re.compile(r"^\d{3}_.+_scenario_\d$")

    if not root_path.is_dir():
        logger.error(f"Root path is not a directory: {root_path}")
        return

    for entry in root_path.iterdir():
        if entry.is_dir() and bool(session_pattern.match(entry.name)):
            yield entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions_path", type=Path, help="Path to the session folder.")
    parser.add_argument("-f", "--force", default=False, action="store_true", help="Overwrite asd_events folder, if exists.")
    args = parser.parse_args()

    sessions = list(discover_sessions(args.sessions_path))

    if not sessions:
        logger.warning(f"No folders matching pattern 'DDD_X_scenario_D' found in: %s", args.sessions_path)
        exit()

    logger.info(f"Found {len(sessions)} matching sessions. Starting processing...")
    
    for session_path in sessions:
        process_session(session_path, force=args.force)