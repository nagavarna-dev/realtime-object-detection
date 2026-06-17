"""
main.py
-------
The end-to-end real-time object detection pipeline.

Flow per frame:
    threaded camera  ->  lighting normalization  ->  YOLOv8 detection
                     ->  draw boxes + FPS  ->  show on screen

Run it:
    python main.py                      # use config.yaml defaults (webcam)
    python main.py --source clip.mp4    # override the source
    python main.py --no-display         # headless (CI / servers / containers)

Press 'q' in the window to quit. A latency plot is saved when you exit.
"""

import argparse
import logging

import cv2

from src.config import configure_logging, load_config
from src.detector import ObjectDetector
from src.metrics import PerformanceTracker
from src.preprocess import LightingNormalizer
from src.video_stream import ThreadedVideoStream
from src.visualize import draw_detections, save_latency_plot

log = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None,
                   help="Webcam index (0) or path/URL to a video stream")
    p.add_argument("--model", default=None, help="YOLOv8 weights file")
    p.add_argument("--conf", type=float, default=None,
                   help="Confidence threshold")
    p.add_argument("--no-display", action="store_true",
                   help="Run without a GUI window (headless)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (useful for smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        "source": args.source,
        "model": args.model,
        "conf_threshold": args.conf,
    }
    cfg = load_config(args.config, overrides)
    configure_logging(cfg["log_level"])

    source = cfg["source"]
    source = int(source) if str(source).isdigit() else source

    stream = ThreadedVideoStream(source).start()
    normalizer = LightingNormalizer(
        clip_limit=cfg["preprocess"]["clip_limit"],
        tile_grid_size=tuple(cfg["preprocess"]["tile_grid"]),
    )
    detector = ObjectDetector(model_path=cfg["model"],
                              conf_threshold=cfg["conf_threshold"])
    tracker = PerformanceTracker(window=cfg["metrics"]["window"])

    display = not args.no_display
    log.info("Running%s... press 'q' to quit.",
             " (headless)" if not display else "")
    processed = 0
    try:
        while True:
            frame = stream.read()
            if frame is None:
                break

            with tracker.measure():                       # true per-frame latency
                frame = normalizer.apply(frame)
                detections = detector.detect(frame)

            frame = draw_detections(frame, detections, fps=tracker.fps())
            processed += 1

            if display:
                cv2.imshow("Real-Time Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        stream.stop()
        if display:
            cv2.destroyAllWindows()
        log.info("Processed %d frames | mean latency %.1f ms | p95 %.1f ms | %.1f FPS",
                 processed, tracker.average_latency_ms(),
                 tracker.p95_latency_ms(), tracker.fps())
        save_latency_plot(tracker.latencies_ms, cfg["metrics"]["plot_path"])


if __name__ == "__main__":
    main()
