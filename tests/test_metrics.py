"""Tests for PerformanceTracker: latency vs throughput, percentiles, edges."""
import time

from src.metrics import PerformanceTracker


def test_empty_tracker_returns_zero():
    t = PerformanceTracker()
    assert t.fps() == 0.0
    assert t.average_latency_ms() == 0.0
    assert t.p95_latency_ms() == 0.0


def test_measure_records_latency():
    t = PerformanceTracker()
    with t.measure():
        time.sleep(0.01)
    assert len(t.latencies_ms) == 1
    # Slept ~10ms; allow generous slack for slow CI runners.
    assert 5.0 <= t.latencies_ms[0] <= 200.0


def test_first_frame_has_no_interval():
    # Interval needs two completions; one measure() yields zero intervals.
    t = PerformanceTracker()
    with t.measure():
        pass
    assert t.intervals_ms == []
    assert t.fps() == 0.0


def test_fps_after_multiple_frames():
    t = PerformanceTracker()
    for _ in range(5):
        with t.measure():
            time.sleep(0.005)
    assert len(t.intervals_ms) == 4   # N frames -> N-1 intervals
    assert t.fps() > 0.0


def test_p95_is_monotonic_in_tail():
    t = PerformanceTracker()
    t.latencies_ms = [float(x) for x in range(1, 101)]  # 1..100
    p95 = t.p95_latency_ms()
    assert 90.0 <= p95 <= 100.0
    assert p95 >= t.average_latency_ms()
