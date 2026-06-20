"""
detector.py
-----------
A thin wrapper around the Ultralytics YOLOv8 model.

It loads the model once (loading is slow, so we do it a single time), then
exposes a detect() method that takes one frame and returns a clean list of
detections. Keeping all YOLO-specific code in this one file means the rest of
the project never has to know the details of the model's output format.

The `ultralytics` import is deferred to construction time rather than module
import time. This keeps the heavy dependency (and its torch backend) out of the
import path for anything that only needs the class definition -- e.g. unit tests
that mock the model, or tooling that imports the module for inspection.
"""

import logging

log = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        # "yolov8n.pt" is the nano model -- smallest and fastest, good for
        # real-time on a laptop. Swap for yolov8s/m/l/x for more accuracy.
        # The file downloads automatically the first time it is used.
        from ultralytics import YOLO  # deferred: imported only on real use
        log.info("Loading model %s (conf>=%.2f)", model_path, conf_threshold)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Maps class IDs (0, 1, 2, ...) to human names ("person", "car", ...).
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Run detection on a single BGR frame.
        Returns a list of dicts: {box, confidence, class_id, label}.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()   # bounding-box corners
            class_id = int(box.cls[0])
            detections.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "label": self.class_names[class_id],
            })
        return detections
