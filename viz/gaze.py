import tkinter as tk
from tkinter import font as tkfont
import time
import collections
import math
import threading
import queue
import asyncio
import nats

from aware_protos.zhaw.protobuf.gaze_pb2 import GazeScreenPosition

# --- Configuration ---
NATS_HOST = "nats://192.168.68.64:4222"
SUBJECT = "intent.gaze"

# Coordinate Mapping
SOURCE_WIDTH, SOURCE_HEIGHT = 3840, 2160  # The 4K space from the container

# Visual Tuning
TRAIL_MAX = 40
BASE_RADIUS = 15
MAX_RADIUS = 80
FIXATION_THRESHOLD = 50  # Pixels in 4K space to consider "staying put"
SMOOTHING = 0.2          # Low-pass filter for the bubble center


def nats_worker(msg_queue):
    """Background asyncio loop to handle NATS subscriptions."""
    async def run():
        try:
            # Connect with low-latency settings
            nc = await nats.connect(
                NATS_HOST,
                max_reconnect_attempts=-1,
                pedantic=False
            )
            
            async def message_handler(msg):
                # Push the raw bytes into the queue for Tkinter to process
                msg_queue.put(msg.data)

            await nc.subscribe(SUBJECT, cb=message_handler)
            
            # Keep the async loop alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"NATS Connection Error: {e}")

    # Set up and run the asyncio loop inside this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())


class GazeInspector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gaze Inspector (NATS + Protobuf)")
        self.attributes("-fullscreen", True)
        self.configure(bg="#0a0a0a")
        self.bind("<Escape>", lambda e: self.destroy())

        # UI Components
        self.canvas = tk.Canvas(self, bg="#0a0a0a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.custom_font = tkfont.Font(family="Consolas", size=12, weight="bold")

        # NATS & Protobuf Setup
        self.proto = GazeScreenPosition()
        self.msg_queue = queue.Queue()
        
        # Start NATS in a background daemon thread so it doesn't block Tkinter
        self.nats_thread = threading.Thread(
            target=nats_worker, 
            args=(self.msg_queue,), 
            daemon=True
        )
        self.nats_thread.start()
        
        # Internal State
        self.trail = collections.deque(maxlen=TRAIL_MAX)
        self.last_msg_time = 0
        self.msg_count = 0
        self.fps = 0
        self.last_fps_calc = time.time()
        
        # Smooth Logic
        self.sm_x, self.sm_y = 0, 0
        self.valid = False
        self.bubble_r = BASE_RADIUS
        self.raw_x, self.raw_y = 0, 0

        self.after(10, self.update_loop)

    def update_loop(self):
        now = time.time()
        
        # 1. Consume all available messages in the queue (Drain buffer)
        while not self.msg_queue.empty():
            try:
                payload = self.msg_queue.get_nowait()
                self.proto.ParseFromString(payload)
                
                self.raw_x = self.proto.x
                self.raw_y = self.proto.y
                self.valid = self.proto.is_valid
                self.last_msg_time = now
                self.msg_count += 1

                if self.valid:
                    # Smoothing
                    self.sm_x += (self.raw_x - self.sm_x) * SMOOTHING
                    self.sm_y += (self.raw_y - self.sm_y) * SMOOTHING
                    
                    # Fixation Bubble Logic
                    if self.trail:
                        dist = math.hypot(self.raw_x - self.trail[-1][0], self.raw_y - self.trail[-1][1])
                        if dist < FIXATION_THRESHOLD:
                            self.bubble_r = min(MAX_RADIUS, self.bubble_r + 2)
                        else:
                            self.bubble_r = max(BASE_RADIUS, self.bubble_r - 5)
                    
                    self.trail.append((self.raw_x, self.raw_y))
            except queue.Empty:
                break

        # 2. Performance Stats
        if now - self.last_fps_calc > 1.0:
            self.fps = self.msg_count
            self.msg_count = 0
            self.last_fps_calc = now

        # 3. Draw
        self.render(now)
        self.after(16, self.update_loop)

    def map_coords(self, x, y):
        """Maps 4K space to current Window size."""
        win_w = self.canvas.winfo_width()
        win_h = self.canvas.winfo_height()
        
        rel_x = x / SOURCE_WIDTH
        rel_y = y / SOURCE_HEIGHT
        
        return rel_x * win_w, rel_y * win_h

    def render(self, now):
        self.canvas.delete("all")
        
        # Determine Connection State
        time_since_last = now - self.last_msg_time
        if time_since_last > 0.5:
            state = "NO_SIGNAL"
            header_color = "#ff4444"
        elif not self.valid:
            state = "INVALID_GAZE"
            header_color = "#ffbb00"
        else:
            state = "TRACKING"
            header_color = "#00ff88"

        # Sidebar Stats
        stats = [
            f"STATE:  {state}",
            f"NATS:   {self.fps} Hz",
            f"RAW X:  {self.raw_x}",
            f"RAW Y:  {self.raw_y}",
            f"NORM X: {self.raw_x/SOURCE_WIDTH:.3f}" if SOURCE_WIDTH else "NORM X: 0.0",
            f"NORM Y: {self.raw_y/SOURCE_HEIGHT:.3f}" if SOURCE_HEIGHT else "NORM Y: 0.0",
            f"VALID:  {self.valid}"
        ]
        
        for i, text in enumerate(stats):
            color = header_color if i == 0 else "white"
            self.canvas.create_text(20, 20 + (i * 22), text=text, fill=color, 
                                    anchor="nw", font=self.custom_font)

        if state == "NO_SIGNAL":
            self.canvas.create_text(
                self.canvas.winfo_width()/2, self.canvas.winfo_height()/2,
                text="NATS FEED DISCONNECTED\nCheck container or port 4222",
                fill="#333333", font=("Consolas", 30), justify="center"
            )
            return

        # Render Trail
        if len(self.trail) > 1:
            points = [self.map_coords(x, y) for x, y in self.trail]
            flat_points = [c for p in points for c in p]
            self.canvas.create_line(flat_points, fill="#222222", width=2, smooth=True)

        # Render Gaze
        if self.valid:
            mx, my = self.map_coords(self.sm_x, self.sm_y)
            
            # The Fixation Bubble
            self.canvas.create_oval(
                mx - self.bubble_r, my - self.bubble_r,
                mx + self.bubble_r, my + self.bubble_r,
                outline=header_color, width=2
            )
            
            # Crosshair
            ch = 10
            self.canvas.create_line(mx-ch, my, mx+ch, my, fill=header_color, width=1)
            self.canvas.create_line(mx, my-ch, mx, my+ch, fill=header_color, width=1)

if __name__ == "__main__":
    app = GazeInspector()
    app.mainloop()