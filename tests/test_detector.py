"""Tests for ObjectDetector output schema, with YOLO fully mocked."""
from unittest.mock import MagicMock, patch

import numpy as np


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.cls = [float(cls)]
        self.conf = [float(conf)]


def _fake_results(boxes):
    r = MagicMock()
    r.boxes = boxes
    return [r]


@patch("src.detector.YOLO")
def test_detect_returns_clean_schema(mock_yolo):
    instance = mock_yolo.return_value
    instance.names = {0: "person", 2: "car"}
    instance.return_value = _fake_results([
        _FakeBox([10.4, 20.6, 110.9, 220.1], cls=0, conf=0.91),
        _FakeBox([5.0, 5.0, 50.0, 60.0], cls=2, conf=0.55),
    ])

    from src.detector import ObjectDetector
    det = ObjectDetector(model_path="fake.pt", conf_threshold=0.4)
    out = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(out) == 2
    first = out[0]
    assert set(first) == {"box", "confidence", "class_id", "label"}
    assert first["box"] == (10, 20, 110, 220)      # ints, truncated
    assert first["label"] == "person"
    assert first["class_id"] == 0
    assert abs(first["confidence"] - 0.91) < 1e-6
    assert out[1]["label"] == "car"


@patch("src.detector.YOLO")
def test_detect_handles_no_boxes(mock_yolo):
    instance = mock_yolo.return_value
    instance.names = {}
    instance.return_value = _fake_results([])

    from src.detector import ObjectDetector
    det = ObjectDetector()
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []
