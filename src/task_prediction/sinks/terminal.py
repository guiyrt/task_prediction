import sys
import time
import logging
from collections import deque
from functools import lru_cache
from datetime import datetime
from typing import Optional, Final, Tuple

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

from ..models import TaskPrediction, TaskPredStatus, TaskType, TaskPredTelemetry, InferenceResult
from .base import PredictionSink

logger = logging.getLogger(__name__)


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
    def _render_row(self, name: str, proba: float, style: str) -> Tuple[Text, Text, Text]:
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
    """Rich-based live terminal visualization."""
    def __init__(self, refresh_per_sec: int = 10):
        self._isatty = sys.stdout.isatty()
        self._live: Optional[Live] = None
        self.refresh_per_sec = refresh_per_sec

        self.header = HeaderView()
        self.tasks = TaskRankingView()
        self.telemetry = TelemetryView()

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

    async def start(self) -> None:
        if self._isatty and self._live is None:
            self._live = Live(self.layout, screen=True, refresh_per_second=self.refresh_per_sec)
            self._live.start()
        elif not self._isatty:
            logger.info("TerminalSink disabled: stdout is not a TTY.")

    async def send(self, pred: TaskPrediction) -> None:
        if not self._isatty or not self._live: return
        self.header.timestamp = pred.timestamp
        self.header.status = pred.status
        self.header.tick()
        self.telemetry.update(pred)
        self.tasks.update(pred.pred)

    async def close(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None