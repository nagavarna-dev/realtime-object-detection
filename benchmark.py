"""
benchmark.py
------------
Measures the latency improvement from threaded capture -- this is the script
behind the "~35% latency reduction" claim. It runs the SAME detection workload
twice on the same video and compares average latency:

    1. Baseline: read a frame, then detect, then read the next (blocking).
    2. Threaded: a background thread reads frames while detection runs.

Run it:
    python benchmark.py --source clip.mp4 --frames 200
"""

import argparse
import time
import cv2

from src.video_stream import ThreadedVideoStream
from src.detector import ObjectDetector


def run_baseline(source, detector, n_frames):
    """Blocking version: capture and detection take turns."""
    cap = cv2.VideoCapture(source)
    latencies = []
    for _ in range(n_frames):
        t0 = time.time()
        grabbed, frame = cap.read()      # blocks until a frame arrives
        if not grabbed:
            break
        detector.detect(frame)
        latencies.append((time.time() - t0) * 1000)
    cap.release()
    return sum(latencies) / len(latencies)


def run_threaded(source, detector, n_frames):
    """Threaded version: capture overlaps with detection."""
    stream = ThreadedVideoStream(source).start()
    latencies = []
    for _ in range(n_frames):
        t0 = time.time()
        frame = stream.read()            # newest frame is already waiting
        if frame is None:
            break
        detector.detect(frame)
        latencies.append((time.time() - t0) * 1000)
    stream.stop()
    return sum(latencies) / len(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    detector = ObjectDetector(model_path=args.model)

    print("Running baseline (blocking)...")
    base = run_baseline(source, detector, args.frames)
    print("Running threaded...")
    threaded = run_threaded(source, detector, args.frames)

    improvement = (base - threaded) / base * 100
    print("\n--- RESULTS ---")
    print(f"Baseline avg latency : {base:.1f} ms")
    print(f"Threaded avg latency : {threaded:.1f} ms")
    print(f"Improvement          : {improvement:.1f}% lower latency")


if __name__ == "__main__":
    main()
