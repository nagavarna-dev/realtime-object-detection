"""
main.py
-------
The end-to-end real-time object detection pipeline.

Flow per frame:
    threaded camera  ->  lighting normalization  ->  YOLOv8 detection
                     ->  draw boxes + FPS  ->  show on screen

Run it:
    python main.py                 # use webcam
    python main.py --source clip.mp4   # use a video file

Press 'q' in the window to quit. A latency plot is saved when you exit.
"""

import argparse
import cv2

from src.video_stream import ThreadedVideoStream
from src.preprocess import LightingNormalizer
from src.detector import ObjectDetector
from src.metrics import PerformanceTracker
from src.visualize import draw_detections, save_latency_plot


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection")
    parser.add_argument("--source", default=0,
                        help="Webcam index (0) or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold")
    args = parser.parse_args()

    # A video file path stays a string; a webcam index must be an int.
    source = int(args.source) if str(args.source).isdigit() else args.source

    # Build the pipeline components once.
    stream = ThreadedVideoStream(source).start()
    normalizer = LightingNormalizer()
    detector = ObjectDetector(model_path=args.model, conf_threshold=args.conf)
    tracker = PerformanceTracker()

    print("Running... press 'q' in the window to quit.")
    try:
        while True:
            frame = stream.read()
            if frame is None:
                break

            frame = normalizer.apply(frame)        # 1. fix lighting
            detections = detector.detect(frame)    # 2. detect objects
            tracker.tick()                         # 3. record timing
            frame = draw_detections(frame, detections, fps=tracker.fps())  # 4. draw

            cv2.imshow("Real-Time Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print(f"Average latency: {tracker.average_latency_ms():.1f} ms "
              f"({tracker.fps():.1f} FPS)")
        save_latency_plot(tracker.latencies_ms)


if __name__ == "__main__":
    main()
