# Real-Time Object Detection System

An end-to-end real-time object detection pipeline built on **YOLOv8**, with
lighting-robust preprocessing, a threaded capture stage for higher throughput,
and Matplotlib performance visualization. Modular, configurable, tested, and
containerized.

![CI](https://github.com/nagavarna-dev/realtime-object-detection/actions/workflows/ci.yml/badge.svg)

## Highlights

- **Modular pipeline** — camera → preprocessing → detection → visualization,
  where each stage is an independent, unit-tested component.
- **Lighting robustness** — CLAHE contrast normalization keeps detection stable
  under dark, bright, or uneven lighting without retraining.
- **Threaded capture for higher throughput** — frame grabbing runs in a
  background thread so I/O overlaps with inference instead of serializing with
  it. See the honest, reproducible measurement in [`benchmark.py`](benchmark.py).
- **Operable** — YAML config, structured logging, headless mode, automatic
  stream reconnection, latency/p95 metrics, CI, and a Dockerfile.

## Why threading helps (and why it isn't "faster inference")

Threading raises **throughput**, not single-frame speed. `cv2.VideoCapture.read()`
is a native call that releases the Python GIL while it blocks on I/O, letting the
main thread run YOLO inference concurrently. Per-frame detection **latency** is
essentially unchanged — the win is overlap. The benchmark reports both numbers
separately so the claim is honest rather than an artifact of where the stopwatch
starts.

## Quick start

```bash
pip install -r requirements.txt

python main.py                       # webcam, using config.yaml defaults
python main.py --source clip.mp4     # a video file
python main.py --no-display --max-frames 300   # headless (servers / CI)
```

Configuration lives in [`config.yaml`](config.yaml); any value can be overridden
on the command line (`python main.py --help`).

## Benchmark

```bash
python benchmark.py --source clip.mp4 --seconds 15
```

Runs the same detection workload under blocking vs. threaded capture for a fixed
wall-clock budget and reports frames completed, FPS, and mean `detect()` latency
for each.

## Tests & CI

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Tests mock the YOLO model, so they run fast on CPU with no model download and no
camera. CI runs lint + tests on Python 3.10–3.12 on every push and PR.

## Docker

```bash
docker build -t detection .
docker run --rm -v "$PWD/outputs:/app/outputs" \
  detection python main.py --source clip.mp4 --no-display --max-frames 300
```

## Project structure

```
.
├── main.py              # pipeline entry point (config + CLI)
├── benchmark.py         # blocking vs. threaded throughput comparison
├── config.yaml          # declarative pipeline settings
├── Dockerfile           # headless container image
├── src/
│   ├── config.py        # YAML load + CLI override merge, logging setup
│   ├── video_stream.py  # threaded capture with reconnection
│   ├── preprocess.py    # CLAHE lighting normalization
│   ├── detector.py      # YOLOv8 wrapper -> clean detection dicts
│   ├── metrics.py       # latency vs. throughput tracking (+ p95)
│   └── visualize.py     # box drawing + latency plot
├── tests/               # pytest suite (YOLO mocked)
└── .github/workflows/   # CI: lint + tests
```
