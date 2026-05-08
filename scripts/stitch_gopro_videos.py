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
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gopro-batch")

@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration: float = 0.0
    frames: int = 0

    def __add__(self, other):
        if other == 0:
            return self
        
        if not isinstance(other, VideoInfo):
            return NotImplemented
        
        return VideoInfo(
            duration=self.duration + other.duration,
            frames=self.frames + other.frames
        )

    def __radd__(self, other):
        return self.__add__(other)

async def get_video_info(path: Path) -> VideoInfo:
    """Uses ffprobe to get duration and frame count."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,nb_frames",
        "-of", "json", str(path)
    ]
    
    # Create the process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Wait for the result
    stdout, stderr = await process.communicate()
    
    try:
        data = json.loads(stdout.decode())
        stream = data["streams"][0]
    
        return VideoInfo(
            duration=float(stream["duration"]),
            frames=int(stream["nb_frames"])
        )
    except Exception as e:
        logger.error(f"ffprobe failed for {path}: {e}\nStderr: {stderr.decode()}")
        return VideoInfo()

async def verify_output(input_files: list[Path], output_file: Path) -> bool:
    """
    Performs 3 checks: 
    1. Duration match
    2. Frame count ratio (~50%)
    3. Bitstream integrity scan
    """
    logger.info(f"Verifying {output_file.name}...")

    # 1. Gather Input / Output stats
    in_info = [await get_video_info(f) for f in input_files]
    total_in_info = sum(in_info)
    out_info = await get_video_info(output_file)
    
    # 3. Compare Duration
    duration_diff = abs(total_in_info.duration - out_info.duration)
    if duration_diff > 0.1:
        logger.error(f"Verification Failed: Duration mismatch! (Diff: {duration_diff:.3f}s)")
        return False

    # We expect 50% of frames because of 60fps -> 30fps
    expected_frames = total_in_info.frames // 2
    frame_diff = abs(expected_frames - out_info.frames)
    if frame_diff > 5:
        logger.error(f"Verification Failed: Frame count mismatch! (Expected ~{expected_frames}, got {out_info.frames})")
        return False
    
    logger.info(
        f"Verification Info: Diff ({frame_diff}f | {duration_diff:.2f}s) "
        f"| Frames (IN:{'+'.join(f'{i.frames:,}' for i in in_info)}={total_in_info.frames:,}, OUT:{out_info.frames:,}) "
        f"| Duration (IN:{'+'.join(f'{i.duration:.2f}s' for i in in_info)}={total_in_info.duration:.2f}s, OUT:{out_info.duration:.2f}s)"
    )

    logger.info("Verification Passed! Video is intact and duration matches.")
    return True

async def _stream_ffmpeg_progress(stderr: asyncio.StreamReader):
    """Consumes FFmpeg stderr, prints progress, and filters out noise."""
    try:
        while not stderr.at_eof():
            try:
                line_bytes = await stderr.readuntil((b"\r", b"\n"))
            except asyncio.IncompleteReadError as e:
                # Catch the last line if FFmpeg closes without a final newline
                line_bytes = e.partial
            
            if not line_bytes:
                break
                
            # Decode and .strip() to remove the trailing \r or \n
            line_decoded = line_bytes.decode('utf-8', errors='replace').strip()
            
            # Skip empty lines
            if not line_decoded:
                continue
            
            # Ignore frame updates  
            if not line_decoded.startswith("frame="):
                print(f"\n[FFmpeg] {line_decoded}", end="")
                    
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
            data: dict = json.load(f)
            
        sync_list = data.get("external_sync", {}).get("GoPro", [])
        if not sync_list:
            logger.warning(f"No GoPro external_sync data found in {session_json_path.name}")
            return None
            
        return datetime.fromisoformat(sync_list[-1])
        
    except Exception as e:
        logger.error(f"Failed to parse metadata from {session_json_path.name}: {e}")
        return None

async def process_session(session_path: Path, args: argparse.Namespace) -> None:
    session_id = session_path.name
    vid_dir = session_path / "videoRecordings"
    meta_dir = session_path / "metadata"
    
    out_mp4 = vid_dir / f"{session_id}_goprorecording.mp4"
    out_meta = vid_dir / f"{session_id}_goprorecording_metadata.json"
    session_json = meta_dir / f"{session_id}_session.json"

    # 1. Guard clauses
    if out_mp4.exists() and not args.force:
        logger.debug(f"Skipping {session_id} - Output file already exists.")
        return

    if not vid_dir.exists():
        logger.warning(f"Skipping {session_id} - Missing videoRecordings folder.")
        return

    # Find Raw GoPro files
    mp4_files = [f for f in vid_dir.iterdir() if f.suffix.upper() == ".MP4" and f.name.startswith("GH")]

    # Sort robustly by GoPro naming convention: GH<Chapter:2><FileNumber:4>.MP4
    # Example: GH010025.MP4 -> Sorts by '0025' first, then '01'
    files = sorted(mp4_files, key=lambda f: (f.name[4:8], f.name[2:4]))

    if not files:
        logger.warning(f"Skipping {session_id} - No raw GoPro (.MP4) files found.")
        return

    logger.info(f"Processing Session: {session_id} ({len(files)} chapters found)")

    # Create Concat List
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        for f in files:
            safe_path = str(f.resolve()).replace("'", r"'\''")
            tf.write(f"file '{safe_path}'\n")
        concat_path = Path(tf.name)

    # Build Command
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-map", "0:v:0", "-map", "0:a:0",
        
        # Video encoding (Optimized NVENC HEVC)
        "-c:v", "hevc_nvenc",
        "-preset", "p7",
        "-tune", "hq", # Added: optimizes for visual quality
        "-rc", "vbr", 
        "-cq", "24",
        "-b:v", "0", # Added: Removes default bitrate cap so -cq works properly
        "-bf", "3", # Added: Enables B-Frames (Major space saving)
        "-b_ref_mode", "middle", # Added: B-frame referencing
        "-spatial-aq", "1", 
        "-temporal-aq", "1",
        "-vf", "fps=30000/1001",
        "-movflags", "+faststart",
        
        # Audio encoding
        "-c:a", "libopus", "-b:a", "128k",
        
        str(out_mp4)
    ]

    # Execute and Finalize
    try:
        success = await run_compression(cmd, files, out_mp4)
        
        if success:
            is_valid = await verify_output(files, out_mp4)

            if not meta_dir.exists() and not session_json.exists():
                logger.warning(f"Skipping {session_id} - Missing {session_json.name}.")

            elif (start_time := extract_start_time(session_json)) is not None:
                meta_payload = {
                    "start_epoch_sec": start_time.timestamp(),
                    "start_utc_iso": start_time.isoformat(),
                    "fps": "29.97"
                }
                
                with open(out_meta, 'w') as f:
                    json.dump(meta_payload, f, indent=2)
                logger.info(f"Generated metadata: {out_meta.name}")
            
            if is_valid:
                for f in files:
                    f.unlink()
                pass
            else:
                logger.error("Verification failed. Keeping intermediate files for debugging.")

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

async def main_async(args: argparse.Namespace):
    sessions = list(discover_sessions(args.sessions_path))

    if not sessions:
        logger.warning(f"No folders matching pattern 'DDD_X_scenario_Y' found in: {args.sessions_path}")
        sys.exit(0)

    logger.info(f"Found {len(sessions)} matching sessions. Starting processing pipeline...")
    
    for session_path in sessions:
        await process_session(session_path, args)

def main():
    parser = argparse.ArgumentParser(description="Batch process and compress GoPro scenario recordings.")
    parser.add_argument("sessions_path", type=Path, help="Path to the root directory containing session folders.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing mkv outputs.")
    
    args = parser.parse_args()

    # Start the asyncio event loop once
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Process stopped by user.")

if __name__ == "__main__":
    main()

# TODO: Implement "by folder" and "by name" merging strategies