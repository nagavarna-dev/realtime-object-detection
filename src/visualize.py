"""
visualize.py
------------
Two kinds of visualization:

1. draw_detections() -- overlays bounding boxes + labels onto a video frame
   (this is what you see live in the window).

2. save_latency_plot() -- after the run, uses Matplotlib to chart per-frame
   latency over time, so you can SEE the performance, not just read a number.
   This is the "performance metrics visualized with Matplotlib" deliverable.
"""

import cv2
import matplotlib
matplotlib.use("Agg")          # render to a file without needing a display
import matplotlib.pyplot as plt


def draw_detections(frame, detections, fps=None):
    """Draw boxes, labels, and an FPS readout onto the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.2f}'

        # Green rectangle around the object.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Filled label background so text is readable on any scene.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def save_latency_plot(latencies_ms, out_path="outputs/latency.png"):
    """Plot per-frame latency over the run and save it as an image."""
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(latencies_ms, linewidth=1)
    plt.title("Per-Frame Inference Latency")
    plt.xlabel("Frame number")
    plt.ylabel("Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved latency plot to {out_path}")
