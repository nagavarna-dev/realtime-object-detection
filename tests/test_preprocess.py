"""Tests for LightingNormalizer: output contract is preserved."""
import numpy as np

from src.preprocess import LightingNormalizer


def _frame(h=48, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_output_shape_and_dtype_preserved():
    norm = LightingNormalizer()
    out = norm.apply(_frame())
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_dark_frame_is_brightened():
    # A very dark frame should have its mean lightness lifted by CLAHE.
    norm = LightingNormalizer()
    dark = np.full((48, 64, 3), 20, dtype=np.uint8)
    # Add a little texture so histogram equalization has something to work on.
    dark[::4, ::4] = 60
    out = norm.apply(dark)
    assert out.mean() >= dark.mean()


def test_does_not_mutate_input():
    norm = LightingNormalizer()
    frame = _frame()
    before = frame.copy()
    norm.apply(frame)
    assert np.array_equal(frame, before)
