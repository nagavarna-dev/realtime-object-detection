"""
metrics.py
----------
Tracks two DISTINCT performance quantities that are often confused:

  * Latency  -- how long it takes to process ONE frame (grab -> detect -> done).
                Measured by timing the work itself with a context manager.
  * Throughput (FPS) -- how many frames the pipeline completes per second of
                wall-clock time. This is 1 / (interval between completed frames)
                and is affected by overlap (threading), not just raw work time.

These are not the same number. Threading can raise throughput while per-frame
latency stays constant, because capture overlaps with inference. Keeping them
separate is what makes the benchmark in benchmark.py honest.
"""

import time
from collections import deque
from contextlib import contextmanager


class PerformanceTracker:
    def __init__(self, window=30):
        # Rolling window of recent frame-completion intervals (for live FPS).
        self.intervals = deque(maxlen=window)
        # Full histories, for end-of-run plots / stats.
        self.latencies_ms = []      # per-frame WORK time
        self.intervals_ms = []      # gap between frame completions (throughput)
        self._last_completion = None

    @contextmanager
    def measure(self):
        """
        Context manager that times the actual per-frame work (true latency):

            with tracker.measure():
                detections = detector.detect(frame)

        On exit it records both the work latency and the inter-frame interval.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.latencies_ms.append((t1 - t0) * 1000.0)

            if self._last_completion is not None:
                interval = t1 - self._last_completion
                self.intervals.append(interval)
                self.intervals_ms.append(interval * 1000.0)
            self._last_completion = t1

    def fps(self):
        """Current throughput (FPS), averaged over the rolling window."""
        if not self.intervals:
            return 0.0
        avg = sum(self.intervals) / len(self.intervals)
        return 1.0 / avg if avg > 0 else 0.0

    def average_latency_ms(self):
        """Mean per-frame WORK latency over the whole run, in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def p95_latency_ms(self):
        """95th-percentile per-frame latency -- the tail SREs actually care about."""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]
