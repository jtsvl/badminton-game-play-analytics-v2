"""eval_rally additions for the 12-vs-30 fps measurement (specs/features/analysis-30fps.md).

Pure scoring only: the landing summary over stage-6 events and hit recall against the trainer's stroke
labels (video_id + t_s, cadence-neutral). Precision proper is NOT computable from those labels — they
only judge candidates the old heuristic proposed — so the scorer reports recall, the re-detection rate of
the 43 known false positives (`no_hit`) and the candidate ratio, and says so.
"""
import unittest

from worker.eval_rally import hit_recall, landing_summary


def landing(idx, located, reason=None, call=None):
    payload = {"located": located, "call": call or "uncertain", "line": None, "band_cm": 45.0,
               "confidence": 0.5 if located else None}
    if not located:
        payload["reason"] = reason
    return {"rally_idx": idx, "t_s": 10.0 * idx, "type": "landing", "payload": payload, "confidence": None}


class TestLandingSummary(unittest.TestCase):
    def test_counts_located_and_reason_histogram(self):
        events = [
            landing(0, True, call="in"), landing(1, True, call="out"),
            landing(2, False, "sparse"), landing(3, False, "sparse"), landing(4, False, "no_descent"),
            {"rally_idx": 0, "t_s": 1.0, "type": "serve", "payload": {}, "confidence": 0.7},
        ]
        s = landing_summary(events)
        self.assertEqual(s["rallies"], 5)
        self.assertEqual(s["located"], 2)
        self.assertEqual(s["reasons"], {"sparse": 2, "no_descent": 1})
        self.assertEqual(s["calls"], {"in": 1, "out": 1})

    def test_empty(self):
        self.assertEqual(landing_summary([]), {"rallies": 0, "located": 0, "reasons": {}, "calls": {}})


class TestHitRecall(unittest.TestCase):
    LABELS = [10.0, 11.0, 12.0, 13.0]       # trainer-confirmed hits
    NO_HIT = [20.0, 21.0]                   # stage-6 candidates the trainer rejected

    def test_recall_within_tolerance(self):
        pred = [10.1, 11.3, 12.6, 30.0]      # 10.1 ✓, 11.3 ✗ (0.3 > 0.25), 12.6 ✗ (0.4/0.6), 30 unlabelled
        r = hit_recall(pred, self.LABELS, self.NO_HIT, tol_s=0.25)
        self.assertEqual(r["n_labels"], 4)
        self.assertEqual(r["found"], 1)
        self.assertAlmostEqual(r["recall"], 0.25)
        self.assertEqual(r["tol_s"], 0.25)

    def test_each_prediction_matches_at_most_one_label(self):
        # one candidate near two labels only recovers one of them
        r = hit_recall([10.5], [10.3, 10.6], [], tol_s=0.5)
        self.assertEqual(r["found"], 1)

    def test_no_hit_redetection_rate(self):
        r = hit_recall([20.05, 25.0], self.LABELS, self.NO_HIT, tol_s=0.25)
        self.assertEqual(r["n_no_hit"], 2)
        self.assertEqual(r["no_hit_redetected"], 1)
        self.assertAlmostEqual(r["no_hit_redetected_rate"], 0.5)

    def test_candidate_ratio_and_honesty_note(self):
        r = hit_recall([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], self.LABELS, [], tol_s=0.25)
        self.assertAlmostEqual(r["candidates_per_label"], 1.5)
        self.assertIn("precision", r["note"].lower())

    def test_empty_labels_are_reported_not_divided(self):
        r = hit_recall([1.0], [], [], tol_s=0.25)
        self.assertIsNone(r["recall"])
        self.assertIsNone(r["candidates_per_label"])
        self.assertIsNone(r["no_hit_redetected_rate"])


if __name__ == "__main__":
    unittest.main()
