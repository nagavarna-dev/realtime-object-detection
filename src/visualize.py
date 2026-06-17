"""
visualize.py
------------
Two kinds of visualization:

1. draw_detections() -- overlays bounding boxes + labels onto a video frame
   (this is what you see live in the window).

2. save_latency_plot() -- after the run, uses Matplotlib to chart per-frame
   WORK latency (time spent in detection per frame) over time, so you can SEE
   the performance, not just read a number.
"""

import logging
import os

# Select a non-interactive Matplotlib backend before pyplot is imported, so the
# plot can be saved to a file on headless machines (servers, CI, containers)
# with no display attached. Setting it via the environment keeps the imports in
# normal sorted order (no mid-file matplotlib.use() call to special-case).
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def draw_detections(frame, detections, fps=None):
    """Draw boxes, labels, and an FPS readout onto the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def save_latency_plot(latencies_ms, out_path="outputs/latency.png"):
    """Plot per-frame detection latency over the run and save it as an image."""
    if not latencies_ms:
        log.warning("No latency samples to plot; skipping.")
        return

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    avg = sum(latencies_ms) / len(latencies_ms)
    plt.figure(figsize=(10, 4))
    plt.plot(latencies_ms, linewidth=1, label="per-frame latency")
    plt.axhline(avg, linestyle="--", linewidth=1, label=f"mean {avg:.1f} ms")
    plt.title("Per-Frame Detection Latency")
    plt.xlabel("Frame number")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    log.info("Saved latency plot to %s", out_path)
