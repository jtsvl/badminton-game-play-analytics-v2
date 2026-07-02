"""eval_rally — rally-segmentation scorecard (specs/features/pose-swing-fusion.md).

Pure scoring layer only (no polars/DB — dev-extra testable). Spans are (t_start_s, t_end_s).
The GT regime is drop/merge-only (the review UI can't move boundaries), so kept-rally GT
boundaries equal the machine's own — the scorecard exists to count phantoms (false_rallies),
splits, and the two unrecoverable-error alarms (missed = falsely vetoed, merges = falsely
bridged), which are hard flip gates.
"""
import unittest

from worker.eval_rally import (boundary_prf, confusion, evaluate, format_report, gt_from_db,
                               serve_window_report, span_stats, split_gaps, swing_strength_split,
                               veto_operating_points)


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


class _FakeStore:
    def __init__(self, match, rallies):
        self._match = match
        self._rallies = rallies

    def get_match_by_slug(self, slug):
        return self._match

    def get_rallies(self, match_id):
        assert match_id == self._match["id"]
        return self._rallies


class TestCLIGlue(unittest.TestCase):
    def test_gt_from_db_reads_current_rally_rows(self):
        store = _FakeStore(
            match={"id": "m1", "slug": "g1", "rally_edited_at": "2026-07-02T10:00:00Z"},
            rallies=[{"idx": 0, "t_start_s": 1.5, "t_end_s": 9.0},
                     {"idx": 1, "t_start_s": 12.0, "t_end_s": 30.25}])
        spans, match = gt_from_db(store, "g1")
        self.assertEqual(spans, [(1.5, 9.0), (12.0, 30.25)])
        self.assertEqual(match["id"], "m1")

    def test_gt_from_db_warns_on_unreviewed_match(self):
        store = _FakeStore(match={"id": "m1", "rally_edited_at": None}, rallies=[])
        with self.assertLogs("worker.eval_rally", level="WARNING"):
            gt_from_db(store, "g1")

    def test_report_lists_confusion_in_trainer_vocabulary(self):
        text = format_report(evaluate([(0.0, 5.0), (7.0, 8.0)], [(0.0, 5.0)]))
        self.assertIn("false rallies", text)
        self.assertIn("drop", text)    # the trainer's op for a phantom
        self.assertIn("splits", text)
        self.assertIn("merge", text)   # the trainer's op for a split
        self.assertIn("hard gate 0", text)

    def test_baseline_gt_is_subset_of_machine_boundaries(self):
        # Flags-off determinism: GT = machine output minus drops, plus merges of adjacent
        # machine rallies — so the baseline can show false_rallies/splits but NEVER a missed
        # real rally or a falsely-bridged pair (missed == 0 and merges == 0).
        machine = [(0.0, 5.0), (6.5, 8.0), (9.0, 14.0), (15.0, 16.0), (20.0, 28.0)]
        gt = [(0.0, 5.0),        # kept
              (6.5, 14.0),       # trainer merged machine[1] + machine[2]
              (20.0, 28.0)]      # kept; machine[3] was dropped as a phantom
        conf = confusion(machine, gt)
        self.assertEqual(conf["missed"], 0)
        self.assertEqual(conf["merges"], 0)
        self.assertEqual(conf["false_rallies"], 1)
        self.assertEqual(conf["splits"], 1)
        rep = boundary_prf(machine, gt)
        self.assertEqual(rep["recall"], 1.0)  # every GT boundary is a machine boundary

    def test_split_gaps_reports_trainer_merged_gaps(self):
        machine = [(0.0, 5.0), (6.5, 8.0), (9.0, 14.0), (20.0, 28.0)]
        gt = [(0.0, 14.0), (20.0, 28.0)]  # trainer merged the first three machine rallies
        self.assertEqual(split_gaps(machine, gt), [1.0, 1.5])


class TestSwingTuning(unittest.TestCase):
    def test_swing_hist_partitions_in_vs_out_of_rally(self):
        gt = [(10.0, 15.0)]
        swings = [
            {"t": 11.0, "track": 1, "strength": 6.0},   # in-rally
            {"t": 14.5, "track": 2, "strength": 8.0},   # in-rally
            {"t": 20.0, "track": 1, "strength": 1.5},   # between rallies
        ]
        rep = swing_strength_split(swings, gt)
        self.assertEqual(rep["in_rally"]["n"], 2)
        self.assertEqual(rep["between"]["n"], 1)
        self.assertEqual(rep["in_rally"]["p50"], 7.0)
        self.assertEqual(rep["between"]["p90"], 1.5)

    def test_swing_hist_splits_by_court_half_when_mapped(self):
        # Near player has their back to the camera — the split must expose a weaker near-side
        # distribution instead of hiding it in the aggregate (back-to-camera check, T6).
        gt = [(10.0, 15.0)]
        swings = [
            {"t": 11.0, "track": 1, "strength": 3.0},
            {"t": 12.0, "track": 2, "strength": 9.0},
        ]
        rep = swing_strength_split(swings, gt, half_by_track={1: "near", 2: "far"})
        self.assertEqual(rep["in_rally_near"]["n"], 1)
        self.assertEqual(rep["in_rally_near"]["p50"], 3.0)
        self.assertEqual(rep["in_rally_far"]["p50"], 9.0)

    def test_veto_sweep_counts_covered_rallies_and_killed_phantoms(self):
        gt = [(0.0, 5.0), (20.0, 24.0)]
        pred = [(0.0, 5.0), (10.0, 11.5), (20.0, 24.0)]  # middle pred is a phantom
        swings = [
            {"t": 2.0, "track": 1, "strength": 6.0},    # inside gt rally 1
            {"t": 20.2, "track": 2, "strength": 2.5},   # weak swing in gt rally 2
        ]
        pts = veto_operating_points(swings, pred, gt, thresholds=[2.0, 4.0], margin_s=0.5)
        # thr=2.0: both real rallies have a swing >= thr (covered 2/2); the phantom has none.
        self.assertEqual(pts[2.0]["rallies_covered"], 2)
        self.assertEqual(pts[2.0]["gt_total"], 2)
        self.assertEqual(pts[2.0]["phantoms_killed"], 1)
        self.assertEqual(pts[2.0]["phantom_total"], 1)
        # thr=4.0: the weak-swing rally is no longer covered -> would be a missed (hard-gate fail).
        self.assertEqual(pts[4.0]["rallies_covered"], 1)

    def test_veto_sweep_margin_saves_swing_just_outside_span(self):
        gt = [(10.0, 11.0)]  # short serve-fault rally; swing sits 0.3 s before the span
        swings = [{"t": 9.7, "track": 1, "strength": 5.0}]
        pts = veto_operating_points(swings, [(10.0, 11.0)], gt, thresholds=[4.0], margin_s=0.5)
        self.assertEqual(pts[4.0]["rallies_covered"], 1)


class TestServeWindow(unittest.TestCase):
    def test_serve_window_reports_per_rally_max_and_attribution(self):
        gt = [(10.0, 15.0), (30.0, 34.0)]
        half = {1: "near", 2: "far"}
        swings = [
            {"t": 10.2, "track": 1, "strength": 4.0},   # rally 1: near serve swing wins
            {"t": 10.3, "track": 2, "strength": 2.0},
            {"t": 29.8, "track": 2, "strength": 5.0},   # rally 2: far swing just before t_start
            {"t": 30.1, "track": 1, "strength": 1.0},
            {"t": 12.5, "track": 2, "strength": 9.0},   # mid-rally clear — outside both windows
        ]
        rep = serve_window_report(swings, gt, half)
        r1, r2 = rep["per_rally"]
        self.assertEqual(r1["n_samples"], 2)
        self.assertEqual(r1["max_strength"], 4.0)
        self.assertEqual(r1["best_half"], "near")
        self.assertAlmostEqual(r1["best_t_rel"], 0.2)
        self.assertEqual(r2["max_strength"], 5.0)
        self.assertEqual(r2["best_half"], "far")
        self.assertAlmostEqual(r2["best_t_rel"], -0.2)
        self.assertEqual(rep["attribution"], {"near": 1, "far": 1, "none": 0})
        self.assertEqual(rep["no_sample_rallies"], 0)
        self.assertEqual(rep["max_by_half"]["near"]["p50"], 4.0)
        self.assertEqual(rep["max_by_half"]["far"]["p50"], 5.0)

    def test_serve_window_counts_rally_with_no_sample(self):
        gt = [(10.0, 15.0)]
        swings = [{"t": 13.0, "track": 1, "strength": 6.0}]  # mid-rally only, serve instant empty
        rep = serve_window_report(swings, gt, {1: "near"})
        r = rep["per_rally"][0]
        self.assertEqual(r["n_samples"], 0)
        self.assertIsNone(r["max_strength"])
        self.assertIsNone(r["best_half"])
        self.assertIsNone(r["best_t_rel"])
        self.assertEqual(rep["no_sample_rallies"], 1)
        self.assertEqual(rep["attribution"], {"near": 0, "far": 0, "none": 1})

    def test_serve_window_boundaries_inclusive(self):
        # Closed window bounds, matching the veto sweep's span±margin test.
        gt = [(10.0, 15.0)]
        swings = [
            {"t": 9.5, "track": 1, "strength": 1.0},    # exactly t_start - pre_s → in
            {"t": 10.5, "track": 1, "strength": 2.0},   # exactly t_start + post_s → in
            {"t": 10.51, "track": 1, "strength": 9.0},  # just past → out
        ]
        rep = serve_window_report(swings, gt, {1: "near"})
        self.assertEqual(rep["per_rally"][0]["n_samples"], 2)
        self.assertEqual(rep["per_rally"][0]["max_strength"], 2.0)

    def test_serve_window_tie_attributes_earliest(self):
        gt = [(10.0, 15.0)]
        swings = [
            {"t": 9.9, "track": 1, "strength": 3.0},
            {"t": 10.1, "track": 2, "strength": 3.0},
        ]
        rep = serve_window_report(swings, gt, {1: "near", 2: "far"})
        self.assertEqual(rep["per_rally"][0]["best_half"], "near")
        self.assertAlmostEqual(rep["per_rally"][0]["best_t_rel"], -0.1)

    def test_serve_window_without_half_map(self):
        gt = [(10.0, 15.0)]
        swings = [{"t": 10.0, "track": 7, "strength": 4.5}]
        rep = serve_window_report(swings, gt, half_by_track=None)
        self.assertEqual(rep["per_rally"][0]["max_strength"], 4.5)
        self.assertIsNone(rep["per_rally"][0]["best_half"])
        self.assertEqual(rep["attribution"], {"near": 0, "far": 0, "none": 1})


if __name__ == "__main__":
    unittest.main()
