import sys
import time
import logging
import socket
import os
import threading
import asyncio
from collections import deque
from functools import lru_cache
from datetime import datetime
from typing import Optional, Final

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

from ..models import TaskPrediction, TaskPredStatus, TaskType, InferenceResult
from ..adapters.struct.task_pred import pred_to_struct, pred_from_struct
from .base import PredictionSink

logger = logging.getLogger(__name__)

# IPC Configuration
SOCKET_PATH: Final = "/tmp/task_prediction.sock"

class HeaderView:
    """Top bar tracking system state, rate, and uptime."""
    _RADAR_FRAMES: Final[str] = "⠁⠂⠄⡀⢀⠠⠐⠈"
    _RADAR_STATES: Final[int] = 8

    _STATUS_STYLES: Final[dict[TaskPredStatus, str]] = {
        TaskPredStatus.WARMING_UP: "bold yellow",
        TaskPredStatus.NO_ASD_EVENTS: "bold dark_orange",
        TaskPredStatus.NO_GAZE: "bold dark_orange",
        TaskPredStatus.INVALID_GAZE: "bold dark_orange",
        TaskPredStatus.ERROR: "bold red",
        TaskPredStatus.OK: "bold green",
    }

    def __init__(self):
        self.start_time: float = time.monotonic()
        self.timestamp: Optional[datetime] = None
        self.status: TaskPredStatus = TaskPredStatus.WARMING_UP
        self._times: deque[float] = deque(maxlen=10)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def rate_hz(self) -> float:
        if len(self._times) < 2: return 0.0
        duration = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / duration if duration > 0 else 0.0

    def __rich__(self) -> Table:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center")
        grid.add_column(justify="right")

        now = time.monotonic()
        ts_str = self.timestamp.strftime('%d %b %Y, %H:%M:%S') if self.timestamp else "Waiting..."
        time_info = Text.assemble(
            ("UTC: ", "bold dim"), (f"{ts_str} | ", "dim"),
            ("Uptime: ", "bold white"), (time.strftime("%H:%M:%S", time.gmtime(now - self.start_time)), "white")
        )

        status_style = self._STATUS_STYLES.get(self.status, "white")
        status_info = Text.assemble(
            ("Status: ", "bold dim"), (self.status.name, status_style),
            ("  |  Rate: ", "bold dim"), (f"{self.rate_hz:.2f} Hz", "bright_black")
        )

        radar_color = "bold green" if self.status == TaskPredStatus.OK else status_style
        radar_heartbeat = Text.assemble(
            ("│ ", "bright_black"),
            (self._RADAR_FRAMES[int(now * 10) % self._RADAR_STATES], radar_color)
        )

        grid.add_row(time_info, status_info, radar_heartbeat)
        return grid


class TaskRankingView:
    """Left panel: Idle status and strict-ordered tasks."""
    _BAR_WIDTH: Final[int] = 20
    _COLOR_DIM: Final[str] = "bright_black"

    def __init__(self):
        self.payload: Optional[InferenceResult] = None
        self._cached_table: Optional[Table] = None

    def update(self, payload: Optional[InferenceResult]):
        self.payload = payload
        self._cached_table = None 

    def _create_base_table(self) -> Table:
        table = Table(expand=True, box=box.SIMPLE_HEAVY, show_header=True, border_style="dim")
        table.add_column("Task", justify="left")
        table.add_column("Score", justify="right", width=7)
        table.add_column("Graph", justify="left")
        return table

    @lru_cache(maxsize=1024)
    def _render_row(self, name: str, proba: float, style: str) -> tuple[Text, Text, Text]:
        filled = int(proba * self._BAR_WIDTH)
        bar = ("█" * filled) + ("░" * (self._BAR_WIDTH - filled))
        
        return (
            Text(name, style=style),
            Text(f"{proba * 100:>5.1f}%", style=style),
            Text(bar, style=style)
        )

    def __rich__(self) -> Table:
        if self._cached_table is not None:
            return self._cached_table

        table = self._create_base_table()

        if not self.payload:
            table.add_row("Waiting for models...", "-", "░" * self._BAR_WIDTH, style=self._COLOR_DIM)
            self._cached_table = table
            return table

        is_act = self.payload.is_active
        idle_prob = 1.0 - self.payload.active_proba

        max_task = None
        if self.payload.task_probas:
            max_task, _ = max(self.payload.task_probas.items(), key=lambda x: x[1])

        # 1. Idle Row Logic (Rounded to 3 decimals to keep .1 precision)
        idle_style = self._COLOR_DIM if is_act else "bold green"
        table.add_row(*self._render_row("IDLE", round(idle_prob, 3), idle_style))
        table.add_section()

        # 2. Task Rows Logic
        for task in TaskType:
            proba = self.payload.task_probas.get(task, 0.0)
            
            if is_act:
                style = "bold green" if task == max_task else "white"
            else:
                style = self._COLOR_DIM

            # Rounded to 3 decimals to keep .1 precision
            table.add_row(*self._render_row(task.name, round(proba, 3), style))

        self._cached_table = table
        return table


class TelemetryView:
    """Right panel: Point-in-time metrics and simple, highly readable graphs."""
    _BAR_WIDTH: Final[int] = 30

    def __init__(self):
        self.last_pred: Optional[TaskPrediction] = None

    def update(self, pred: TaskPrediction):
        self.last_pred = pred

    def _render_gaze_bar(self, avail: float, valid: float) -> Text:
        """Renders: [████ Valid | ░░ Invalid | ░░ Missing]"""
        valid_blocks = int(valid * self._BAR_WIDTH)
        avail_blocks = int(avail * self._BAR_WIDTH)
        
        invalid_blocks = max(0, avail_blocks - valid_blocks)
        missing_blocks = max(0, self._BAR_WIDTH - avail_blocks)

        # Handle rounding artifacts to keep bar exactly width size
        total = valid_blocks + invalid_blocks + missing_blocks
        if total < self._BAR_WIDTH: 
            missing_blocks += (self._BAR_WIDTH - total)

        bar = Text()
        bar.append("█" * valid_blocks, style="bold white")           
        bar.append("░" * invalid_blocks, style="bold bright_black")  
        bar.append("░" * missing_blocks, style="dim")                
        return bar

    def __rich__(self) -> Group:
        if not self.last_pred or not self.last_pred.pred:
            return Group(Text("Waiting for telemetry...", style="dim"))

        t = self.last_pred.telemetry
        total_lat = t.feature_extraction_time_ms + t.inference_time_ms

        # Bounds check 0.0 -> 1.0
        avail = max(0.0, min(1.0, t.gaze_availability_pct))
        valid = max(0.0, min(avail, t.gaze_validity_pct))
        invalid_pct = avail - valid
        missing_pct = 1.0 - avail

        # --- 1. Instant Metrics (Top) ---
        metrics = Table.grid(expand=True, padding=(0, 2))
        metrics.add_column(justify="left", style="bold dim")
        metrics.add_column(justify="right", style="white")
        
        metrics.add_row("Total Latency", f"{total_lat:.1f} ms")
        metrics.add_row(" ├─ Feature Ext.", f"{t.feature_extraction_time_ms:.1f} ms")
        metrics.add_row(" └─ Inference", f"{t.inference_time_ms:.2f} ms")
        metrics.add_row("ASD Events", str(t.asd_events_count))
        
        # --- 2. Simple Graphs (Bottom) ---
        graphs = Table(box=None, expand=True, show_header=False, padding=(1, 0, 0, 0))
        graphs.add_column()
        
        graphs.add_row(Text("Gaze Quality (Current Packet)", style="bold dim"))
        graphs.add_row(self._render_gaze_bar(avail, valid))
        
        # Breakdown right below the bar
        legend = Text.assemble(
            (f"{valid:.1%} Valid", "bold white"),
            (" | ", "dim"),
            (f"{invalid_pct:.1%} Invalid", "bold bright_black"),
            (" | ", "dim"),
            (f"{missing_pct:.1%} Missing", "dim")
        )
        graphs.add_row(legend)

        return Group(metrics, graphs)


class TerminalSink(PredictionSink):
    """Rich-based live terminal visualization with IPC streaming."""
    def __init__(self, refresh_per_sec: int = 10):
        self._isatty = sys.stdout.isatty()
        self._live: Optional[Live] = None
        self.refresh_per_sec = refresh_per_sec

        # UI Components
        self.header = HeaderView()
        self.tasks = TaskRankingView()
        self.telemetry = TelemetryView()

        # Layout Setup
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        self.layout["body"].split_row(
            Layout(name="tasks", ratio=5),
            Layout(name="telemetry", ratio=5) 
        )

        self.layout["header"].update(Panel(self.header, title="Prediction Sink", border_style="blue"))
        self.layout["tasks"].update(Panel(self.tasks, title="Task Estimations", border_style="white"))
        self.layout["telemetry"].update(Panel(self.telemetry, title="Prediction Diagnostics", border_style="cyan"))

        # IPC State
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._server_running = False

    async def start(self) -> None:
        if self._isatty:
            if self._live is None:
                self._live = Live(self.layout, screen=True, refresh_per_second=self.refresh_per_sec)
                self._live.start()
        else:
            logger.info(f"TerminalSink: Background mode. Starting IPC on {SOCKET_PATH}")
            self._start_ipc_server()

    def _start_ipc_server(self) -> None:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        
        self._server_running = True
        self._server_thread = threading.Thread(target=self._ipc_listener_loop, daemon=True)
        self._server_thread.start()

    def _ipc_listener_loop(self) -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        server.settimeout(1.0)
        
        while self._server_running:
            try:
                conn, _ = server.accept()
                with self._lock:
                    self._clients.append(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self._server_running:
                    logger.error(f"IPC Listener Error: {e}")
                break
    
    def _update_state(self, pred: TaskPrediction) -> None:
        self.header.timestamp = pred.timestamp
        self.header.status = pred.status
        self.header.tick()
        self.telemetry.update(pred)
        self.tasks.update(pred.pred)

    async def send(self, pred: TaskPrediction) -> None:
        if self._isatty and self._live:
            self._update_state(pred)
        elif not self._isatty and self._clients:
            self._broadcast(pred)

    def _broadcast(self, pred: TaskPrediction) -> None:
        """Serializes and sends binary data to all attached control centers."""
        try:
            payload = pred_to_struct(pred)
            # Frame: 4-byte Big-Endian Length + Data
            header = len(payload).to_bytes(4, 'big')
            message = header + payload

            with self._lock:
                to_remove = []
                for client in self._clients:
                    try:
                        client.sendall(message)
                    except (BrokenPipeError, ConnectionResetError):
                        to_remove.append(client)
                
                for client in to_remove:
                    self._clients.remove(client)
        except Exception as e:
            logger.debug(f"Broadcast failed: {e}")

    async def close(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        
        if self._server_running:
            with self._lock:
                for client in self._clients:
                    try:
                        client.close()
                    except:
                        pass
                self._clients.clear()

            if os.path.exists(SOCKET_PATH):
                try:
                    os.remove(SOCKET_PATH)
                except:
                    pass

            self._server_running = False

async def listen_from_ipc() -> None:
    """Entry point for the ephemeral 'monitor' mode."""
    if not sys.stdout.isatty():
        print("Error: Monitor must be run in a TTY (interactive terminal).")
        return

    try:
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError):
        print(f"Error: Could not connect to service. Ensure application is running and socket exists at {SOCKET_PATH}")
        return

    sink = TerminalSink()
    await sink.start()

    try:
        while True:
            # Read length prefix
            header = await reader.readexactly(4)
            length = int.from_bytes(header, 'big')

            # Read full payload
            data = await reader.readexactly(length)
            
            # Feed data to sink
            await sink.send(pred_from_struct(data))

    except asyncio.exceptions.IncompleteReadError:
        # Happens cleanly if the main service shuts down while we are watching
        pass
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass # Clean exit on Ctrl+C
    except Exception as e:
        print(f"Monitor Stream Interrupted: {e}")
    finally:
        # Clean up UI and connection
        await sink.close()
        writer.close()
        await writer.wait_closed()