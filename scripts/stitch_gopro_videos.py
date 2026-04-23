import argparse
import asyncio
import logging
import tempfile
import sys
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gopro-batch")

async def _stream_ffmpeg_progress(stderr: asyncio.StreamReader):
    """Consumes FFmpeg stderr, prints progress, and filters out noise."""
    ignore_phrases = [
        "All samples in data stream",
        "Auto-inserting h264_mp4toannexb",
        "Could not find codec parameters",
        "Consider increasing the value"
    ]
    
    try:
        while True:
            try:
                line_bytes = await stderr.readuntil((b"\n", b"\r"))
                line = line_bytes.decode('utf-8', errors='replace').strip()
                
                if line.startswith("frame="):
                    print(f"\r{line}", end="", flush=True)
                elif line and not any(phrase in line for phrase in ignore_phrases):
                    print(f"[FFmpeg] {line}")
                    
            except asyncio.IncompleteReadError as e:
                line = e.partial.decode('utf-8', errors='replace').strip()
                if line and not any(phrase in line for phrase in ignore_phrases):
                    print(f"\r{line}", end="", flush=True)
                break
    except Exception as e:
        logger.debug(f"Progress stream error: {e}")

async def run_compression(cmd: list[str], input_files: list[Path], output_file: Path) -> bool:
    """Executes FFmpeg and returns True if successful."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )

    if process.stderr:
        await asyncio.gather(_stream_ffmpeg_progress(process.stderr), process.wait())
    else:
        await process.wait()

    print()  # Final newline after progress bar
    await asyncio.sleep(0.1) # Settle signal handlers
    
    if process.returncode == 0:
        size_in = sum(f.stat().st_size for f in input_files) / (1024**3)
        size_out = output_file.stat().st_size / (1024**3)
        logger.info(f"Compression Complete! Space Saved: {size_in - size_out:.2f} GB")
        return True
    else:
        logger.error(f"FFmpeg failed with exit code {process.returncode}")
        return False

def extract_start_time(session_json_path: Path) -> datetime | None:
    """Extracts the final GoPro sync timestamp from the session metadata."""
    try:
        with open(session_json_path, 'r') as f:
            data = json.load(f)
            
        sync_list = data.get("external_sync", {}).get("GoPro", [])
        if not sync_list:
            logger.warning(f"No GoPro external_sync data found in {session_json_path.name}")
            return None
            
        # Get the last timestamp in the list
        last_stamp_str = sync_list[-1]
        
        # Python 3.11+ handles ISO format with Z or +00:00 seamlessly
        return datetime.fromisoformat(last_stamp_str)
        
    except Exception as e:
        logger.error(f"Failed to parse metadata from {session_json_path.name}: {e}")
        return None

def process_session(session_path: Path, args: argparse.Namespace) -> None:
    session_id = session_path.name
    vid_dir = session_path / "videoRecordings"
    meta_dir = session_path / "metadata"
    
    out_mkv = vid_dir / f"{session_id}_goprorecording.mkv"
    out_meta = vid_dir / f"{session_id}_goprorecording_metadata.json"
    session_json = meta_dir / f"{session_id}_session.json"

    # 1. Guard clauses
    if out_mkv.exists() and not args.force:
        logger.debug(f"Skipping {session_id} - Output MKV already exists.")
        return

    if not vid_dir.exists() or not meta_dir.exists():
        logger.warning(f"Skipping {session_id} - Missing videoRecordings or metadata folder.")
        return

    if not session_json.exists():
        logger.warning(f"Skipping {session_id} - Missing {session_json.name}.")
        return

    # Find Raw GoPro files
    files = sorted([f for f in vid_dir.iterdir() if f.suffix.upper() == ".MP4" and f.name.startswith("GH")])
    if not files:
        logger.warning(f"Skipping {session_id} - No raw GoPro ({args.ext}) files found.")
        return

    # Extract Metadata Time
    start_time = extract_start_time(session_json)
    if not start_time:
        return

    logger.info(f"Processing Session: {session_id} ({len(files)} chapters found)")

    # Create Concat List
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        for f in files:
            tf.write(f"file '{f.resolve()}'\n")
        concat_path = Path(tf.name)

    # Build Command
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        # Explicitly map only Video and Audio to drop telemetry
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "hevc_nvenc",
        "-preset", "p7",
        "-rc", "vbr", "-cq", str(args.cq),
        "-spatial-aq", "1", "-temporal-aq", "1",
        "-c:a", "libopus", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-r", str(args.fps),
        str(out_mkv)
    ]

    # 5. Execute and Finalize
    try:
        success = asyncio.run(run_compression(cmd, files, out_mkv))
        
        if success:
            # Generate the new metadata JSON
            meta_payload = {
                "start_epoch_sec": start_time.timestamp(),
                "start_utc_iso": start_time.isoformat(),
                "fps": str(args.fps)
            }
            with open(out_meta, 'w') as f:
                json.dump(meta_payload, f, indent=2)
            logger.info(f"Generated metadata: {out_meta.name}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Aborting batch...")
        sys.exit(1)
    finally:
        if concat_path.exists():
            concat_path.unlink()

def discover_sessions(root_path: Path) -> Iterator[Path]:
    """Yields valid session directories within the root_path."""
    session_pattern = re.compile(r"^\d{3}_.+_scenario_\d+$")

    if not root_path.is_dir():
        logger.error(f"Root path is not a directory: {root_path}")
        return

    for entry in root_path.iterdir():
        if entry.is_dir() and bool(session_pattern.match(entry.name)):
            yield entry

def main():
    parser = argparse.ArgumentParser(description="Batch process and compress GoPro scenario recordings.")
    parser.add_argument("sessions_path", type=Path, help="Path to the root directory containing session folders.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing mkv outputs.")
    parser.add_argument("--cq", type=int, default=24, help="Quality target (lower is better, default 24)")
    parser.add_argument("--fps", type=int, choices=[30, 60], default=30, help="Force output frame rate (e.g., 30)")

    args = parser.parse_args()

    sessions = list(discover_sessions(args.sessions_path))

    if not sessions:
        logger.warning(f"No folders matching pattern 'DDD_X_scenario_Y' found in: {args.sessions_path}")
        sys.exit(0)

    logger.info(f"Found {len(sessions)} matching sessions. Starting processing pipeline...")
    
    for session_path in sessions:
        process_session(session_path, args)

if __name__ == "__main__":
    main()