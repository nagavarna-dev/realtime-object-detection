"""
preprocess.py
-------------
Frame preprocessing to keep detection accuracy stable under varying lighting.

THE PROBLEM: A detector trained on normally-lit images loses accuracy when the
scene is too dark, too bright, or has harsh shadows. Rather than retraining for
every condition, we normalize the lighting of each incoming frame first.

THE TECHNIQUE: CLAHE (Contrast Limited Adaptive Histogram Equalization).
Plain histogram equalization brightens an image but blows out already-bright
regions. CLAHE instead equalizes contrast in small tiles across the image and
caps how much any tile can be amplified, so dark areas get lifted without
washing out bright areas. We apply it only to the brightness channel in
LAB color space so colors are not distorted.
"""

import cv2


class LightingNormalizer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # clip_limit caps contrast amplification (higher = stronger effect).
        # tile_grid_size is how many tiles the image is divided into.
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_grid_size)

    def apply(self, frame):
        """Return a lighting-normalized copy of a BGR frame."""
        # Convert to LAB: lightness + two color channels. We only touch lightness.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, chan_a, chan_b = cv2.split(lab)

        # Equalize contrast on the lightness channel only.
        lightness = self.clahe.apply(lightness)

        # Recombine and convert back to BGR for the detector.
        merged = cv2.merge((lightness, chan_a, chan_b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
