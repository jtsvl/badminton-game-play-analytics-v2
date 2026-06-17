"""Tests for the pure insight math. Run: `python -m unittest discover -s tests` (or pytest)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from worker import insights  # noqa: E402


class TestInsights(unittest.TestCase):
    def test_total_distance_known(self):
        # full court width then full length: 6.10 + 13.40 = 19.50 m
        track = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 1.0)]
        self.assertAlmostEqual(insights.total_distance_m(track), 19.50, places=2)

    def test_distance_empty_and_single(self):
        self.assertEqual(insights.total_distance_m([]), 0.0)
        self.assertEqual(insights.total_distance_m([(0.0, 0.5, 0.5)]), 0.0)

    def test_speeds_and_peak(self):
        # move full length (13.40 m) in 1 s -> 48.24 km/h
        track = [(0.0, 0.5, 0.0), (1.0, 0.5, 1.0)]
        speeds = insights.step_speeds_kmh(track)
        self.assertEqual(len(speeds), 1)
        self.assertAlmostEqual(speeds[0], 13.40 * 3.6, places=2)
        avg, top = insights.avg_top_speed_kmh(speeds)
        self.assertAlmostEqual(top, round(13.40 * 3.6, 1), places=1)

    def test_sprint_count_counts_bursts(self):
        speeds = [2, 20, 22, 3, 1, 18, 4]  # two separate bursts >= 15
        self.assertEqual(insights.sprint_count(speeds), 2)

    def test_zones_sum_about_100(self):
        depths = [0.1, 0.2, 0.5, 0.55, 0.9, 0.95]
        laterals = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        z = insights.zones(depths, laterals)
        self.assertAlmostEqual(z["front"] + z["mid"] + z["rear"], 100, delta=2)
        self.assertAlmostEqual(z["left"] + z["right"], 100, delta=2)
        self.assertEqual(z["left"], 50)  # 3 of 6 below 0.5

    def test_zones_empty(self):
        self.assertEqual(insights.zones([], []), {"front": 0, "mid": 0, "rear": 0, "left": 0, "right": 0})

    def test_timeline_length_and_total(self):
        # 10 minutes -> 2 buckets of 5 min. One move of full length in bucket 0.
        track = [(0.0, 0.5, 0.0), (10.0, 0.5, 1.0)]
        buckets = insights.timeline_buckets(track, duration_min=10)
        self.assertEqual(len(buckets), 2)
        self.assertEqual(sum(buckets), round(13.40))

    def test_heatmap_kernels_shape(self):
        pts = [(0.5, 0.55)] * 10 + [(0.3, 0.3)] * 4
        ks = insights.heatmap_kernels(pts, k=4)
        self.assertLessEqual(len(ks), 4)
        self.assertEqual(ks[0]["w"], 1.0)  # densest cluster normalized to 1
        for k in ks:
            self.assertTrue(0.0 <= k["x"] <= 1.0 and 0.0 <= k["y"] <= 1.0)
            self.assertGreater(k["s"], 0)

    def test_heatmap_empty(self):
        self.assertEqual(insights.heatmap_kernels([]), [])


if __name__ == "__main__":
    unittest.main()
