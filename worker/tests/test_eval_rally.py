"""eval_rally — rally-segmentation scorecard (specs/features/pose-swing-fusion.md).

Pure scoring layer only (no polars/DB — dev-extra testable). Spans are (t_start_s, t_end_s).
The GT regime is drop/merge-only (the review UI can't move boundaries), so kept-rally GT
boundaries equal the machine's own — the scorecard exists to count phantoms (false_rallies),
splits, and the two unrecoverable-error alarms (missed = falsely vetoed, merges = falsely
bridged), which are hard flip gates.
"""
import unittest

from worker.eval_rally import boundary_prf, confusion, evaluate, span_stats


class TestBoundaryPRF(unittest.TestCase):
    def test_identical_rallies_score_perfect_f1(self):
        spans = [(0.0, 5.0), (8.0, 14.0), (20.0, 31.5)]
        rep = boundary_prf(spans, spans)
        self.assertEqual(rep["precision"], 1.0)
        self.assertEqual(rep["recall"], 1.0)
        self.assertEqual(rep["f1"], 1.0)
        self.assertEqual(rep["matched"], 6)  # 3 starts + 3 ends

    def test_phantom_rally_lowers_precision_and_counts_false_rally(self):
        gt = [(0.0, 5.0)]
        pred = [(0.0, 5.0), (7.0, 8.0)]  # phantom between rallies, overlaps nothing
        rep = boundary_prf(pred, gt)
        self.assertEqual(rep["recall"], 1.0)
        self.assertEqual(rep["precision"], 0.5)  # 2 of 4 pred boundaries match
        conf = confusion(pred, gt)
        self.assertEqual(conf["false_rallies"], 1)
        self.assertEqual(conf["missed"], 0)

    def test_boundary_match_respects_tolerance(self):
        gt = [(0.0, 10.0)]
        inside = boundary_prf([(0.4, 10.0)], gt, tol_s=0.5)
        self.assertEqual(inside["matched"], 2)   # start off by 0.4 <= tol, end exact
        outside = boundary_prf([(0.6, 10.0)], gt, tol_s=0.5)
        self.assertEqual(outside["matched"], 1)  # start off by 0.6 > tol

    def test_starts_never_match_ends(self):
        # pred start coincides with the gt END — must not count as a boundary match
        rep = boundary_prf([(5.0, 9.0)], [(0.0, 5.0)])
        self.assertEqual(rep["matched"], 0)
        self.assertEqual(rep["f1"], 0.0)

    def test_greedy_match_is_one_to_one(self):
        # two pred starts within tolerance of the single gt start: only one may match
        gt = [(0.0, 10.0)]
        pred = [(0.1, 4.0), (0.2, 10.0)]
        rep = boundary_prf(pred, gt, tol_s=0.5)
        # starts: 1 of 2 match (1:1); ends: the 10.0 matches → 2 matches of 4 pred boundaries
        self.assertEqual(rep["matched"], 2)
        self.assertEqual(rep["precision"], 0.5)
        self.assertEqual(rep["recall"], 1.0)


class TestConfusion(unittest.TestCase):
    def test_split_fragments_count_as_split_not_false_rally(self):
        gt = [(0.0, 10.0)]
        pred = [(0.0, 4.0), (5.5, 10.0)]  # one real rally split at a detection dropout
        conf = confusion(pred, gt)
        self.assertEqual(conf["splits"], 1)
        self.assertEqual(conf["false_rallies"], 0)
        self.assertEqual(conf["missed"], 0)
        self.assertEqual(conf["merges"], 0)

    def test_pred_spanning_two_gt_counts_merge_error(self):
        gt = [(0.0, 5.0), (7.0, 12.0)]
        pred = [(0.0, 12.0)]  # falsely bridged across two real rallies — the hard-gate error
        conf = confusion(pred, gt)
        self.assertEqual(conf["merges"], 1)
        self.assertEqual(conf["missed"], 0)
        self.assertEqual(conf["false_rallies"], 0)

    def test_missed_gt_rally_counted(self):
        gt = [(0.0, 5.0), (10.0, 15.0)]
        pred = [(0.0, 5.0)]  # second real rally gone — the false-veto alarm
        conf = confusion(pred, gt)
        self.assertEqual(conf["missed"], 1)
        self.assertEqual(conf["matched"], 1)


class TestEvaluateAndStats(unittest.TestCase):
    def test_evaluate_bundles_boundary_and_confusion(self):
        gt = [(0.0, 5.0)]
        pred = [(0.0, 5.0), (7.0, 8.0)]
        rep = evaluate(pred, gt)
        self.assertEqual(rep["n_pred"], 2)
        self.assertEqual(rep["n_gt"], 1)
        self.assertEqual(rep["confusion"]["false_rallies"], 1)
        self.assertLess(rep["boundary"]["f1"], 1.0)

    def test_span_stats_reports_min_median_and_gaps(self):
        spans = [(0.0, 3.0), (5.0, 6.0), (10.0, 18.0)]
        st = span_stats(spans)
        self.assertEqual(st["n"], 3)
        self.assertEqual(st["min_span_s"], 1.0)
        self.assertEqual(st["median_span_s"], 3.0)
        self.assertEqual(st["gaps_s"], [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
