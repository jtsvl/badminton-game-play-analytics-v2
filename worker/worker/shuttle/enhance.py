"""Frame-enhancement helpers for the shuttle labeler — make the low-contrast white shuttle easier to spot.

The labeler's strongest spotting aid on a static-tripod white wall is background-difference: subtract the
per-clip median (the static wall + court lines) so the moving shuttle stands out. This module holds the pure
per-frame amplification; the I/O (sampling cached frames, building the median) lives in verify.py. cv2 is
imported lazily so the pure track-assembly tests stay torch/cv2-free. (specs/features/shuttle-tracking.md.)
"""
from __future__ import annotations


def diff_view(img_bgr, median_bgr, gain: float = 3.0):
    """Background-difference view: |img - median| amplified by `gain`, saturating at 255.

    `cv2.convertScaleAbs` does the absolute-value scale with a uint8 saturate_cast, so a static background
    cancels to ~0 and a bright shuttle is amplified without overflow/wraparound. Inputs are BGR uint8 of the
    same shape; returns BGR uint8."""
    import cv2

    return cv2.convertScaleAbs(cv2.absdiff(img_bgr, median_bgr), alpha=gain)


def clahe_view(img_bgr, clip: float = 3.0, grid: int = 8):
    """Local-contrast view via CLAHE on the L (lightness) channel — the cue v1 named for separating the
    white shuttle from the white wall, and the complement to `diff_view` at the apex (where the shuttle
    slows and motion-difference weakens). Equalize only L (keep colour) so a near-uniform bright region's
    faint texture is pulled apart without recolouring. Returns BGR uint8 of the same shape."""
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
