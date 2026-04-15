import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from task_prediction.adapters.proto.parsers import parse_asd_proto

import nats
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Import your generated protos
from aware_protos.zhaw.protobuf import gaze_pb2, task_prediction_pb2, aircraft_attention_target_pb2

# (Assuming your parse_asd_proto is imported here)
# from your_parser_module import parse_asd_proto

# ---------------------------------------------------------------------------
# CONFIGURATION & THRESHOLDS
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "gaze": {"expected_hz": 120.0, "stale_s": 2 / 120.0},
    "task": {"expected_hz": 1/3, "stale_s": 6.0},
    "attention": {"expected_hz": 10.0, "stale_s": 0.2},
    "asd": {"expected_hz": 0.0, "stale_s": 10.0}, 
}

# Pre-compute a constant list of Tasks so the UI never flickers.
# Formats "TASK_TYPE_AIRCRAFT_REQUEST" -> "Aircraft Request"
CONSTANT_TASKS = [
    (v, k.replace("TASK_TYPE_", "").replace("_", " ").title())
    for k, v in task_prediction_pb2.TaskPrediction.TaskType.items()
    if k != "TASK_TYPE_UNSPECIFIED"
]
# Sort alphabetically to guarantee a fixed order
CONSTANT_TASKS.sort(key=lambda x: x[1])

class MonitorState:
    def __init__(self):
        self.data: Dict[str, Any] = {"gaze": None, "task": None, "attention": None, "asd": None}
        self.last_ts: Dict[str, float] = {k: 0.0 for k in THRESHOLDS}
        self.counts: Dict[str, int] = {k: 0 for k in THRESHOLDS}
        self.hz: Dict[str, float] = {k: 0.0 for k in THRESHOLDS}
        
        # New: Grouped history for ASD events
        # Structure: [{"event": "MousePosition", "count": 2}, ...]
        self.asd_history: List[Dict[str, Any]] = [] 
        
        self._last_calc = time.time()

    def update_hz(self):
        now = time.time()
        dt = now - self._last_calc
        if dt >= 1.0:
            for k in self.counts:
                self.hz[k] = self.counts[k] / dt
                self.counts[k] = 0
            self._last_calc = now

    def is_stale(self, key: str) -> bool:
        return (time.time() - self.last_ts[key]) > THRESHOLDS[key]["stale_s"]

state = MonitorState()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def get_style(key: str) -> tuple[str, str]:
    if state.last_ts[key] == 0:
        return "white", "dim"
    if state.is_stale(key):
        return "red", "dim"
    return "green", "bold"

def make_bar(proba: float, width: int = 15) -> Text:
    """Generates a text-based progress bar for UI stability."""
    proba = max(0.0, min(1.0, proba))
    filled = int(round(proba * width))
    empty = width - filled
    # Use standard unicode block characters
    return Text.from_markup(f"[bright_cyan]{'█' * filled}[/][dim]{'░' * empty}[/]")

# ---------------------------------------------------------------------------
# NATS LISTENERS
# ---------------------------------------------------------------------------
async def gaze_sub(nc):
    sub = await nc.subscribe("intent.gaze")
    msg_proto = gaze_pb2.GazeScreenPosition()
    async for msg in sub.messages:
        msg_proto.ParseFromString(msg.data)
        state.data["gaze"] = msg_proto
        state.last_ts["gaze"] = time.time()
        state.counts["gaze"] += 1

async def task_sub(nc):
    sub = await nc.subscribe("intent.task_prediction")
    msg_proto = task_prediction_pb2.TaskPrediction()
    async for msg in sub.messages:
        msg_proto.ParseFromString(msg.data)
        state.data["task"] = msg_proto
        state.last_ts["task"] = time.time()
        state.counts["task"] += 1

async def attention_sub(nc):
    sub = await nc.subscribe("intent.aircraft_attention_target")
    msg_proto = aircraft_attention_target_pb2.AircraftAttentionTarget()
    async for msg in sub.messages:
        msg_proto.ParseFromString(msg.data)
        state.data["attention"] = msg_proto
        state.last_ts["attention"] = time.time()
        state.counts["attention"] += 1

async def asd_sub(nc):
    sub = await nc.subscribe("polaris.ASDEvent")
    async for msg in sub.messages:
        try:
            # We will use the pure JSON extraction as requested, but if you want 
            # to use your `parse_asd_proto` wrapper, you can do:
            # event_obj = parse_asd_proto(msg.data)
            # event_type = type(event_obj).__name__ if event_obj else "Unknown"
            
            asd_event = await asyncio.to_thread(
                parse_asd_proto,
                msg.data,
                msg.header and msg.header.get("deflate") == "1"
            )

            if asd_event is not None:
                # 2. Get the class name (e.g., "MousePosition")
                event_type = asd_event.__class__.__name__
                
                # 3. Update the grouping history logic
                if not state.asd_history or state.asd_history[0]["event"] != event_type:
                    # New event type: Insert at the top
                    state.asd_history.insert(0, {"event": event_type, "count": 1})
                    state.asd_history = state.asd_history[:15] # Keep last 15
                else:
                    # Same as the top: just bump the count
                    state.asd_history[0]["count"] += 1

                # 4. Update standard monitoring metrics
                state.data["asd"] = event_type
                state.last_ts["asd"] = time.time()
                state.counts["asd"] += 1
            
        except Exception:
            pass

# ---------------------------------------------------------------------------
# UI BUILDER
# ---------------------------------------------------------------------------
def generate_ui():
    state.update_hz()
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(
        Layout(name="gaze", ratio=1), Layout(name="task", ratio=2), # Give task more width for bars
        Layout(name="attention", ratio=1), Layout(name="asd", ratio=1)
    )
    
    # 1. HEADER
    layout["header"].update(Panel(
        Text(f"PREDICTION ENGINE CONTROL CENTER | {datetime.now().strftime('%H:%M:%S')}", justify="center", style="bold cyan"),
        box=box.DOUBLE
    ))

    # 2. GAZE
    b_col, t_sty = get_style("gaze")
    g_text = Text(style=t_sty)
    if state.data["gaze"]:
        g = state.data["gaze"]
        g_text.append(f"\nX: {g.x}\nY: {g.y}\n\nValid: {g.is_valid}")
    layout["gaze"].update(Panel(g_text, title=f"Gaze {state.hz['gaze']:.1f}Hz", border_style=b_col))

    # 3. TASK (Constant view with Bars)
    b_col, t_sty = get_style("task")
    t_table = Table.grid(expand=True, padding=(0, 1))
    t_table.add_column("Task", justify="right", style="bold")
    t_table.add_column("Bar", justify="left")
    t_table.add_column("Prob", justify="right", width=5)

    if state.data["task"]:
        d = state.data["task"]
        status = task_prediction_pb2.TaskPrediction.TaskPredStatus.Name(d.status).replace("TASK_PRED_STATUS_", "")
        
        # Calculate IDLE probability and add as the first row
        idle_prob = 1.0 - d.active_proba if d.is_active else 1.0
        t_table.add_row(
            Text("IDLE", style="bold green" if idle_prob > 0.5 else "dim"),
            make_bar(idle_prob),
            Text(f"{idle_prob*100:3.0f}%", style="bold green" if idle_prob > 0.5 else "dim")
        )
        
        # Add all predefined tasks
        for enum_val, t_name in CONSTANT_TASKS:
            prob = d.task_probas.get(enum_val, 0.0)
            row_style = "bold magenta" if prob > 0.5 else ("dim" if prob < 0.05 else t_sty)
            t_table.add_row(
                Text(t_name, style=row_style),
                make_bar(prob),
                Text(f"{prob*100:3.0f}%", style=row_style)
            )
            
        layout["task"].update(Panel(t_table, title=f"Task {state.hz['task']:.2f}Hz | {status}", border_style=b_col))
    else:
        layout["task"].update(Panel("Awaiting initial data...", title="Task", border_style=b_col))

    # 4. ATTENTION
    b_col, t_sty = get_style("attention")
    a_table = Table(box=box.SIMPLE, expand=True, show_header=False)
    if state.data["attention"]:
        for t in state.data["attention"].targets[:15]:
            a_table.add_row(Text(t.callsign, style=t_sty), Text(f"{t.score:.2f}", style=t_sty))
    layout["attention"].update(Panel(a_table, title=f"Attention {state.hz['attention']:.1f}Hz", border_style=b_col))

    # 5. ASD EVENTS (Grouped History View)
    b_col, t_sty = get_style("asd")
    asd_table = Table.grid(expand=True, padding=(0, 1))
    asd_table.add_column("Count", justify="right", style="bold cyan", width=5)
    asd_table.add_column("Event", justify="left")

    if state.asd_history:
        for i, item in enumerate(state.asd_history):
            # The top item stays bright unless stale. Older items fade.
            style = "bold yellow" if (i == 0 and not state.is_stale("asd")) else "dim"
            asd_table.add_row(
                f"[{item['count']}]",
                Text(item["event"], style=style)
            )
    else:
        asd_table.add_row("", Text("Waiting for events...", style="dim"))

    layout["asd"].update(Panel(asd_table, title=f"ASD {state.hz['asd']:.1f}Hz", border_style=b_col))

    return layout

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
async def main():
    nc = await nats.connect("nats://192.168.68.64:4222")
    
    for coro in [gaze_sub(nc), task_sub(nc), attention_sub(nc), asd_sub(nc)]:
        asyncio.create_task(coro)

    with Live(generate_ui(), refresh_per_second=10, screen=True) as live:
        while True:
            live.update(generate_ui())
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass