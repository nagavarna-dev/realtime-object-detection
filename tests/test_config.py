"""Tests for config loading and deep-merge override precedence."""
from src.config import DEFAULTS, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(path=str(tmp_path / "missing.yaml"))
    assert cfg["model"] == DEFAULTS["model"]
    assert cfg["preprocess"]["clip_limit"] == 2.0


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model: yolov8s.pt\npreprocess:\n  clip_limit: 4.0\n")
    cfg = load_config(path=str(p))
    assert cfg["model"] == "yolov8s.pt"
    assert cfg["preprocess"]["clip_limit"] == 4.0
    # Untouched nested keys keep their defaults.
    assert cfg["preprocess"]["tile_grid"] == [8, 8]


def test_cli_overrides_win(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model: yolov8s.pt\n")
    cfg = load_config(path=str(p), overrides={"model": "yolov8m.pt", "conf_threshold": None})
    assert cfg["model"] == "yolov8m.pt"        # override wins
    assert cfg["conf_threshold"] == DEFAULTS["conf_threshold"]  # None ignored
