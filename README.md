# Real-Time Object Detection System

An end-to-end real-time object detection pipeline built on **YOLOv8**, with lighting-robust preprocessing, a threaded capture stage for low latency, and Matplotlib-based performance visualization.

## Highlights

- **End-to-end inference pipeline** — camera → preprocessing → detection → visualization, modularized so each stage is independent and testable.
- **Lighting robustness** — CLAHE contrast normalization keeps detection stable under dark, bright, or uneven lighting without retraining.
- **~35% lower latency** — frame capture runs in a background thread so it overlaps with inference instead of blocking it (see `benchmark.py`).
- **Metric visualization** — per-frame latency and FPS plotted with Matplotlib.

## Project structure
