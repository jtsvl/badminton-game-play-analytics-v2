"""Tests for the labeler's background-difference view (worker/shuttle/enhance.py).

The white shuttle is hard to spot on the white wall; subtracting the static per-clip median makes the moving
shuttle pop. `diff_view` is the pure per-frame amplification — its only correctness risks are uint8 overflow
and that an unchanged background really does cancel. (specs/features/shuttle-tracking.md labeler views.)
"""
import unittest

import numpy as np

from worker.shuttle.enhance import clahe_view, diff_view


class TestDiffView(unittest.TestCase):
    def test_identical_frame_cancels_to_zero(self):
        median = np.full((40, 60, 3), 200, dtype=np.uint8)  # uniform "white wall"
        out = diff_view(median.copy(), median, gain=3.0)
        self.assertEqual(out.shape, median.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out.max()), 0)  # static background fully cancels

    def test_difference_is_amplified(self):
        median = np.full((10, 10, 3), 100, dtype=np.uint8)
        img = median.copy()
        img[5, 5] = 110  # a +10 blip (the shuttle) over the background
        out = diff_view(img, median, gain=3.0)
        self.assertEqual(int(out[5, 5, 0]), 30)  # 10 * gain 3 = 30, far brighter than the 0 background
        self.assertEqual(int(out[0, 0, 0]), 0)

    def test_amplification_clips_no_overflow(self):
        median = np.zeros((4, 4, 3), dtype=np.uint8)
        img = np.full((4, 4, 3), 200, dtype=np.uint8)  # 200 * gain 3 = 600 → must saturate at 255, not wrap
        out = diff_view(img, median, gain=3.0)
        self.assertTrue(np.all(out == 255))

    def test_gain_one_is_plain_absdiff(self):
        median = np.full((3, 3, 3), 50, dtype=np.uint8)
        img = np.full((3, 3, 3), 90, dtype=np.uint8)
        out = diff_view(img, median, gain=1.0)
        self.assertTrue(np.all(out == 40))


class TestClaheView(unittest.TestCase):
    """CLAHE local-contrast view — v1's named cue for separating the white shuttle from the white wall.
    It must boost local contrast on a flat-ish bright region without changing shape/dtype, and leave a
    truly uniform frame alone."""

    def test_preserves_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        out = clahe_view(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_uniform_frame_stays_uniform(self):
        img = np.full((64, 64, 3), 230, dtype=np.uint8)  # blank white wall
        out = clahe_view(img)
        self.assertLess(float(out.std()), 1.0)  # nothing to stretch → no fabricated texture

    def test_low_contrast_region_gains_contrast(self):
        # a faint texture in the bright "white wall" band (values ~225-235) → CLAHE pulls it apart.
        rng = np.random.default_rng(1)
        img = (225 + rng.integers(0, 11, size=(128, 128, 3))).astype(np.uint8)
        out = clahe_view(img, clip=3.0, grid=8)
        self.assertGreater(float(out.std()), float(img.std()))


if __name__ == "__main__":
    unittest.main()
