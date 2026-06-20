"""Tests for the shuttle raw-pixel scoreboard (worker/eval_shuttle.score_against_labels).

The fair before/after metric: precision/recall/F1 + median pixel error of the tracker peaks vs. the labeled
shuttle, scored in ONE pinned space (512x288, tol_px=4 ≈ the SIGMA=2.5 label disk). The caller maps both
the parquet peaks and the labels into that space first; this function is pure matching math. A detection is
a TP only if it lands within tol_px of a visibility=1 label on the same frame
(specs/features/shuttle-finetune-plan.md §5).
"""
import unittest

from worker.eval_shuttle import score_against_labels

# dense labels in 512x288 space: frames 0..4, frame 2 occluded (visibility=0).
LABELS = [
    {"frame": 0, "x": 100.0, "y": 100.0, "visibility": 1},
    {"frame": 1, "x": 110.0, "y": 120.0, "visibility": 1},
    {"frame": 2, "visibility": 0},
    {"frame": 3, "x": 130.0, "y": 160.0, "visibility": 1},
    {"frame": 4, "x": 140.0, "y": 180.0, "visibility": 1},
]


class TestScoreAgainstLabels(unittest.TestCase):
    def test_perfect_detection(self):
        dets = [{"frame": l["frame"], "x": l["x"], "y": l["y"]}
                for l in LABELS if l["visibility"] == 1]
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual(s["n"], 4)              # four visible frames
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (4, 0, 0))
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["f1"], 1.0)
        self.assertEqual(s["median_px_error"], 0.0)

    def test_a_miss_lowers_recall(self):
        dets = [{"frame": 0, "x": 100.0, "y": 100.0}, {"frame": 1, "x": 110.0, "y": 120.0},
                {"frame": 3, "x": 130.0, "y": 160.0}]  # frame 4 missed
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (3, 0, 1))
        self.assertEqual(s["precision"], 1.0)
        self.assertAlmostEqual(s["recall"], 0.75)

    def test_detection_on_occluded_frame_is_false_positive(self):
        dets = [{"frame": 2, "x": 50.0, "y": 50.0}]  # shuttle is occluded here → spurious
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (0, 1, 4))
        self.assertEqual(s["precision"], 0.0)

    def test_just_outside_tolerance_is_miss_and_fp(self):
        # a peak 5px away from the visible shuttle (tol=4) → the GT is missed AND the peak is spurious.
        dets = [{"frame": 0, "x": 105.0, "y": 100.0}]  # dx=5 > 4
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual(s["tp"], 0)
        self.assertEqual(s["fp"], 1)
        self.assertGreaterEqual(s["fn"], 1)

    def test_exactly_on_tolerance_is_a_hit(self):
        dets = [{"frame": 0, "x": 104.0, "y": 100.0}]  # dx=4 == tol → TP
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual(s["tp"], 1)
        self.assertAlmostEqual(s["median_px_error"], 4.0)

    def test_detection_outside_labeled_span_is_ignored(self):
        # no ground truth beyond the labeled frames → such peaks are neither TP/FP/FN.
        dets = [{"frame": 99, "x": 10.0, "y": 10.0}]
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (0, 0, 4))

    def test_median_over_matched_pairs(self):
        dets = [{"frame": 0, "x": 101.0, "y": 100.0},   # err 1
                {"frame": 1, "x": 113.0, "y": 120.0},   # err 3
                {"frame": 3, "x": 130.0, "y": 160.0},   # err 0
                {"frame": 4, "x": 140.0, "y": 180.0}]   # err 0
        s = score_against_labels(dets, LABELS, tol_px=4)
        self.assertEqual(s["tp"], 4)
        self.assertAlmostEqual(s["median_px_error"], 0.5)  # median of [0,0,1,3]


if __name__ == "__main__":
    unittest.main()
