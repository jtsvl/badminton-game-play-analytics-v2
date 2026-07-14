"""Tests for the pose-coverage harness (worker/eval_pose_coverage.py).

Pure functions only — coverage = fraction of analysis frames with a skeleton keyframe within
`max_gap_s`, exactly the web's poseAt hide threshold (0.3 s), so thinned exports can't game the
metric. Expected values are hand-computed.
"""
import unittest

from worker import eval_pose_coverage as e


class TestFramesInSpans(unittest.TestCase):
    def test_inclusive_bounds(self):
        # span [1.0, 3.0] keeps 1.0 and 3.0 (mirrors aggregate.filter_to_spans).
        self.assertEqual(e.frames_in_spans([0.0, 1.0, 2.0, 3.0, 4.0], [(1.0, 3.0)]), [1.0, 2.0, 3.0])

    def test_multiple_spans_union(self):
        self.assertEqual(
            e.frames_in_spans([0.0, 1.0, 2.0, 3.0, 4.0], [(0.0, 0.5), (3.5, 4.5)]), [0.0, 4.0]
        )

    def test_no_spans_keeps_nothing(self):
        self.assertEqual(e.frames_in_spans([0.0, 1.0], []), [])


class TestRenderableCoverage(unittest.TestCase):
    def test_counts_frames_with_a_key_within_gap(self):
        # keys 1.05, 2.9 with 0.3 s: frame 1.0 (0.05) and 3.0 (0.1) covered; 0.0 and 2.0 not.
        cov = e.renderable_coverage([1.05, 2.9], [0.0, 1.0, 2.0, 3.0], max_gap_s=0.3)
        self.assertAlmostEqual(cov, 0.5, places=9)

    def test_thinned_6hz_keys_cover_a_12fps_window(self):
        # 6 Hz keys over a 12 fps second: every frame is <= 1/12 s from a key — full coverage.
        frames = [i / 12 for i in range(13)]
        keys = [i / 6 for i in range(7)]
        self.assertAlmostEqual(e.renderable_coverage(keys, frames, max_gap_s=0.3), 1.0, places=9)

    def test_small_gap_acts_as_exact_match(self):
        cov = e.renderable_coverage([0.1], [0.1, 0.2], max_gap_s=0.002)
        self.assertAlmostEqual(cov, 0.5, places=9)

    def test_empty_keys_or_frames_is_zero(self):
        self.assertEqual(e.renderable_coverage([], [0.0, 1.0]), 0.0)
        self.assertEqual(e.renderable_coverage([0.0], []), 0.0)

    def test_unsorted_inputs_are_handled(self):
        cov = e.renderable_coverage([2.9, 1.05], [3.0, 0.0, 1.0, 2.0], max_gap_s=0.3)
        self.assertAlmostEqual(cov, 0.5, places=9)


class TestCoverageReport(unittest.TestCase):
    def test_per_player_wall_and_in_rally_with_half(self):
        frames = [0.0, 1.0, 2.0, 3.0]  # 4 analysis frames
        spans = [(2.0, 3.0)]  # in-rally = frames 2.0, 3.0
        per_player = {
            "A1": [0.0, 1.0, 2.0, 3.0],  # everywhere
            "B2": [3.0],  # only the last frame
        }
        rep = e.coverage_report(per_player, frames, spans, max_gap_s=0.002)
        self.assertAlmostEqual(rep["players"]["A1"]["wall"], 1.0, places=9)
        self.assertAlmostEqual(rep["players"]["A1"]["in_rally"], 1.0, places=9)
        self.assertEqual(rep["players"]["A1"]["half"], "near")
        self.assertAlmostEqual(rep["players"]["B2"]["wall"], 0.25, places=9)
        self.assertAlmostEqual(rep["players"]["B2"]["in_rally"], 0.5, places=9)
        self.assertEqual(rep["players"]["B2"]["half"], "far")

    def test_no_rallies_reports_none_for_in_rally(self):
        rep = e.coverage_report({"A1": [0.0]}, [0.0, 1.0], [], max_gap_s=0.002)
        self.assertAlmostEqual(rep["players"]["A1"]["wall"], 0.5, places=9)
        self.assertIsNone(rep["players"]["A1"]["in_rally"])


if __name__ == "__main__":
    unittest.main()
