"""
config.py
---------
Loads pipeline settings from a YAML file and merges CLI overrides on top.
Keeping configuration in one declarative place (rather than scattered argparse
defaults) is what lets the same image/binary run in dev, CI, and a container
without code changes -- it just gets a different config.yaml or env.
"""

import logging
from pathlib import Path

import yaml

DEFAULTS = {
    "source": 0,
    "model": "yolov8n.pt",
    "conf_threshold": 0.4,
    "preprocess": {"clip_limit": 2.0, "tile_grid": [8, 8]},
    "metrics": {"window": 30, "plot_path": "outputs/latency.png"},
    "log_level": "INFO",
}


def _deep_merge(base, override):
    """Recursively merge override into base; override wins on conflicts."""
    out = dict(base)
    for key, val in override.items():
        if val is None:
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path="config.yaml", overrides=None):
    """Return DEFAULTS <- file <- overrides (later sources win)."""
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def configure_logging(level="INFO"):
    """One place to set up structured, timestamped logging for the whole app."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
