"""
metrics.py
----------
Tracks performance over time: frames per second (FPS) and per-frame latency.

It keeps a rolling window of recent frame times so the FPS number reflects
current performance rather than the whole run. These numbers are what the
visualizer plots and what the benchmark compares.
"""

import time
from collections import deque


class PerformanceTracker:
    def __init__(self, window=30):
        # Store the timestamps of the last `window` frames.
        self.frame_times = deque(maxlen=window)
        self.latencies_ms = []   # full history, for end-of-run plots
        self.last_time = None

    def tick(self):
        """Call once per processed frame. Records timing for that frame."""
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            self.frame_times.append(dt)
            self.latencies_ms.append(dt * 1000.0)
        self.last_time = now

    def fps(self):
        """Current FPS, averaged over the rolling window."""
        if not self.frame_times:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def average_latency_ms(self):
        """Mean per-frame latency over the whole run, in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)
