"""
benchmark.py
------------
Compares blocking vs. threaded capture HONESTLY.

The naive way to benchmark this is biased: if you time `read() + detect()` for
the baseline but only `detect()` for the threaded path (because the frame is
already waiting), you are not measuring the same work -- you are just moving the
I/O cost outside the stopwatch. That inflates the "improvement".

The honest comparison is THROUGHPUT over a fixed wall-clock budget:
how many frames does each approach fully process in N seconds? Threading should
win because capture I/O overlaps with inference instead of serializing with it.
Per-frame WORK latency (the detect() call) is essentially unchanged by
threading -- we report it separately to make that explicit.

Run it:
    python benchmark.py --source clip.mp4 --seconds 15
"""

import argparse
import logging
import time

import cv2

from src.detector import ObjectDetector
from src.video_stream import ThreadedVideoStream

log = logging.getLogger("benchmark")


def run_blocking(source, detector, seconds):
    """Capture and detection serialize: read a frame, then detect, repeat."""
    cap = cv2.VideoCapture(source)
    frames, work_ms = 0, []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        t0 = time.perf_counter()
        detector.detect(frame)
        work_ms.append((time.perf_counter() - t0) * 1000)
        frames += 1
    cap.release()
    return frames, work_ms


def run_threaded(source, detector, seconds):
    """Background thread captures while the main thread runs detection."""
    stream = ThreadedVideoStream(source).start()
    frames, work_ms = 0, []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        frame = stream.read()
        if frame is None:
            break
        t0 = time.perf_counter()
        detector.detect(frame)
        work_ms.append((time.perf_counter() - t0) * 1000)
        frames += 1
    stream.stop()
    return frames, work_ms


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="Wall-clock budget per approach")
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    detector = ObjectDetector(model_path=args.model)

    log.info("Running blocking capture for %.0fs...", args.seconds)
    b_frames, b_work = run_blocking(source, detector, args.seconds)
    log.info("Running threaded capture for %.0fs...", args.seconds)
    t_frames, t_work = run_threaded(source, detector, args.seconds)

    b_fps = b_frames / args.seconds
    t_fps = t_frames / args.seconds
    gain = (t_fps - b_fps) / b_fps * 100 if b_fps else 0.0

    log.info("\n--- RESULTS (over %.0fs each) ---", args.seconds)
    log.info("Blocking : %4d frames | %5.1f FPS | detect() avg %.1f ms",
             b_frames, b_fps, _avg(b_work))
    log.info("Threaded : %4d frames | %5.1f FPS | detect() avg %.1f ms",
             t_frames, t_fps, _avg(t_work))
    log.info("Throughput gain from threading: %+.1f%%", gain)
    log.info("(Per-frame detect() latency is ~unchanged -- threading raises "
             "throughput by overlapping capture with inference, not by making "
             "inference itself faster.)")


if __name__ == "__main__":
    main()
