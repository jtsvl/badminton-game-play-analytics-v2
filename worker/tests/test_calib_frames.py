"""Tests for the pure calibration-frame selection (worker/calib_frames.py).

Pure index math only — no cv2/torch/store — so these run under the dev extra. The selector picks
N evenly-spaced frames across a decoded clip so the calibration UI can step to a moment where all
four court corners are unoccluded.
"""
import unittest

from worker import calib_frames as cf


class TestEvenlySpacedIndices(unittest.TestCase):
    def test_empty_clip_returns_nothing(self):
        self.assertEqual(cf.evenly_spaced_indices(0, 12), [])

    def test_non_positive_k_returns_nothing(self):
        self.assertEqual(cf.evenly_spaced_indices(100, 0), [])
        self.assertEqual(cf.evenly_spaced_indices(100, -3), [])

    def test_k_ge_total_returns_all_frames(self):
        self.assertEqual(cf.evenly_spaced_indices(5, 12), [0, 1, 2, 3, 4])
        self.assertEqual(cf.evenly_spaced_indices(12, 12), list(range(12)))

    def test_single_pick_is_the_middle(self):
        self.assertEqual(cf.evenly_spaced_indices(100, 1), [49])

    def test_includes_first_and_last(self):
        idx = cf.evenly_spaced_indices(5060, 12)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 5059)
        self.assertEqual(len(idx), 12)

    def test_even_spacing_clean_division(self):
        self.assertEqual(cf.evenly_spaced_indices(9, 3), [0, 4, 8])
        self.assertEqual(cf.evenly_spaced_indices(10, 4), [0, 3, 6, 9])

    def test_round_half_up_is_deterministic(self):
        # i*(n-1)/(k-1) hits 2.5 and 7.5 — round-half-up (not banker's) → 3 and 8, stays monotonic.
        self.assertEqual(cf.evenly_spaced_indices(11, 5), [0, 3, 5, 8, 10])

    def test_strictly_increasing_and_in_range(self):
        idx = cf.evenly_spaced_indices(1000, 37)
        self.assertTrue(all(b > a for a, b in zip(idx, idx[1:])))
        self.assertTrue(all(0 <= i < 1000 for i in idx))

    def test_dedup_when_rounding_collides(self):
        # k just below n_total: rounding can repeat an index — output must stay strictly increasing.
        idx = cf.evenly_spaced_indices(13, 12)
        self.assertEqual(idx, sorted(set(idx)))
        self.assertTrue(all(b > a for a, b in zip(idx, idx[1:])))


class TestSelectCalibFrames(unittest.TestCase):
    def _manifest(self, n, fps=12.0):
        return [{"frame": i, "t": round(i / fps, 4)} for i in range(n)]

    def test_returns_frame_and_time_dicts_in_order(self):
        picks = cf.select_calib_frames(self._manifest(9), 3)
        self.assertEqual([p["frame"] for p in picks], [0, 4, 8])
        self.assertEqual(picks[0]["t"], 0.0)
        self.assertAlmostEqual(picks[1]["t"], 4 / 12.0, places=4)

    def test_short_clip_returns_all(self):
        picks = cf.select_calib_frames(self._manifest(4), 12)
        self.assertEqual([p["frame"] for p in picks], [0, 1, 2, 3])

    def test_empty_manifest(self):
        self.assertEqual(cf.select_calib_frames([], 12), [])

    def test_uses_manifest_frame_number_not_position(self):
        # ingest writes sequential `frame`, but the selector must read the manifest value, not assume
        # it equals the list index — guard against a future sparse manifest.
        manifest = [{"frame": 100 + i, "t": i * 0.5} for i in range(5)]
        picks = cf.select_calib_frames(manifest, 5)
        self.assertEqual([p["frame"] for p in picks], [100, 101, 102, 103, 104])


if __name__ == "__main__":
    unittest.main()
