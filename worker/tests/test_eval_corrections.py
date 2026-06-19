"""Tests for the correction 'scorecard' (worker/eval_corrections.py).

Pure functions only — turn trainer verify-tool corrections into a near/far accuracy report,
so each detection fix can be proven against a real number (cm on court / px on frame).
Expected values are hand-computed.
"""
import unittest

from worker import eval_corrections as e
from worker.insights import COURT_L_M, COURT_W_M


class TestHalfOf(unittest.TestCase):
    def test_a_labels_are_near_b_labels_are_far(self):
        self.assertEqual(e.half_of("A1"), "near")
        self.assertEqual(e.half_of("A2"), "near")
        self.assertEqual(e.half_of("B1"), "far")
        self.assertEqual(e.half_of("B2"), "far")

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            e.half_of("Z9")


class TestErrors(unittest.TestCase):
    def test_foot_px_error_scales_by_image_size(self):
        # 0.1 of the height on a 1000px-tall frame == 100 px.
        self.assertAlmostEqual(e.foot_px_error((0.5, 0.5), (0.5, 0.6), 1000, 1000), 100.0, places=6)

    def test_court_m_error_uses_court_dimensions(self):
        # full court length and width in metres.
        self.assertAlmostEqual(e.court_m_error((0.0, 0.0), (0.0, 1.0)), COURT_L_M, places=6)
        self.assertAlmostEqual(e.court_m_error((0.0, 0.0), (1.0, 0.0)), COURT_W_M, places=6)


class TestSummarize(unittest.TestCase):
    def test_reports_n_mean_median_p90(self):
        s = e.summarize([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        self.assertEqual(s["n"], 11)
        self.assertAlmostEqual(s["mean"], 50.0, places=6)
        self.assertAlmostEqual(s["median"], 50.0, places=6)
        self.assertAlmostEqual(s["p90"], 90.0, places=6)  # linear-interp percentile

    def test_empty_is_zero_count(self):
        s = e.summarize([])
        self.assertEqual(s["n"], 0)


class TestGroundTruth(unittest.TestCase):
    def test_bbox_correction_yields_corrected_foot_and_uv(self):
        cors = [{"role": "A1", "type": "bbox", "frame_idx": 2,
                 "bbox": [0.5604, 0.4602, 0.0476, 0.1857],
                 "orig_bbox": [0.5604, 0.4602, 0.0476, 0.1608],
                 "foot_norm": [0.5842, 0.6459], "court_uv": [0.7435, 0.695]}]
        gt = e.ground_truth(cors)
        foot, uv = gt[(2, "A1")]["foot"], gt[(2, "A1")]["uv"]
        # foot = bottom-centre of the *corrected* bbox
        self.assertAlmostEqual(foot[0], 0.5604 + 0.0476 / 2, places=6)
        self.assertAlmostEqual(foot[1], 0.4602 + 0.1857, places=6)
        self.assertEqual(uv, (0.7435, 0.695))
        self.assertFalse(gt[(2, "A1")]["miss"])

    def test_added_correction_is_flagged_as_a_miss(self):
        cors = [{"role": "B1", "type": "added", "frame_idx": 284,
                 "bbox": [0.5026, 0.3953, 0.0162, 0.0744],
                 "foot_norm": [0.5107, 0.4697], "court_uv": [0.5612, 0.0]}]
        gt = e.ground_truth(cors)
        self.assertTrue(gt[(284, "B1")]["miss"])


class TestPipelineBaseline(unittest.TestCase):
    def test_bbox_correction_recovers_original_pipeline_foot(self):
        cors = [{"role": "A1", "type": "bbox", "frame_idx": 2,
                 "bbox": [0.5604, 0.4602, 0.0476, 0.1857],
                 "orig_bbox": [0.5604, 0.4602, 0.0476, 0.1608],
                 "foot_norm": [0.5842, 0.6459], "court_uv": [0.7435, 0.695]}]
        base = e.pipeline_baseline(cors)
        foot = base[(2, "A1")]["foot"]
        # pipeline foot = bottom-centre of the *original* (truncated) bbox — higher up the frame.
        self.assertAlmostEqual(foot[1], 0.4602 + 0.1608, places=6)

    def test_added_correction_has_no_baseline_entry(self):
        cors = [{"role": "B1", "type": "added", "frame_idx": 284,
                 "bbox": [0.5, 0.4, 0.02, 0.08], "foot_norm": [0.51, 0.48], "court_uv": [0.5, 0.0]}]
        self.assertNotIn((284, "B1"), e.pipeline_baseline(cors))


class TestTrackLabels(unittest.TestCase):
    def test_singles_roles_map_to_player_labels(self):
        rows = [{"track_id": 7, "role": "near"}, {"track_id": 9, "role": "far"},
                {"track_id": 7, "role": "near"}]  # repeated track rows collapse to one label
        labels = e.track_labels(rows)
        self.assertEqual(labels[7], "A1")  # near -> A1
        self.assertEqual(labels[9], "B1")  # far  -> B1

    def test_doubles_quadrants_map_to_slots(self):
        rows = [{"track_id": 1, "role": "near_left"}, {"track_id": 2, "role": "near_right"},
                {"track_id": 3, "role": "far_left"}, {"track_id": 4, "role": "far_right"}]
        labels = e.track_labels(rows)
        self.assertEqual([labels[i] for i in (1, 2, 3, 4)], ["A1", "A2", "B1", "B2"])


class TestPipelineFromDetections(unittest.TestCase):
    H_ID = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # identity homography: projected uv == pixel foot

    def test_foot_and_uv_from_current_detection(self):
        det = [{"frame": 2, "track": 7, "x": 100.0, "y": 200.0, "w": 40.0, "h": 160.0}]
        pm = e.pipeline_from_detections(det, {7: "A1"}, self.H_ID, foot_k=0.0, w=1000, h=1000)
        foot = pm[(2, "A1")]["foot"]
        self.assertAlmostEqual(foot[0], (100 + 20) / 1000)   # bbox centre-x, normalized
        self.assertAlmostEqual(foot[1], (200 + 160) / 1000)  # bbox bottom-y (k=0), normalized
        uv = pm[(2, "A1")]["uv"]
        self.assertAlmostEqual(uv[0], 120.0)  # identity H over the pixel foot
        self.assertAlmostEqual(uv[1], 360.0)

    def test_foot_bias_k_pushes_foot_down(self):
        det = [{"frame": 2, "track": 7, "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}]
        pm = e.pipeline_from_detections(det, {7: "A1"}, self.H_ID, foot_k=0.16, w=1000, h=1000)
        self.assertAlmostEqual(pm[(2, "A1")]["foot"][1], 116 / 1000)  # bottom y = h*(1+0.16)

    def test_untracked_or_unlabeled_detections_are_skipped(self):
        det = [{"frame": 1, "track": None, "x": 0, "y": 0, "w": 1, "h": 1},
               {"frame": 1, "track": 99, "x": 0, "y": 0, "w": 1, "h": 1}]  # 99 has no label
        self.assertEqual(e.pipeline_from_detections(det, {7: "A1"}, self.H_ID, 0.0, 1000, 1000), {})

    def test_uses_confident_pose_foot_over_bbox(self):
        det = [{"frame": 2, "track": 7, "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}]
        pose_feet = {(2, 7): (50.0, 95.0, 0.9)}  # confident ankle, lower than bbox bottom
        pm = e.pipeline_from_detections(det, {7: "A1"}, self.H_ID, foot_k=0.16, w=1000, h=1000,
                                        pose_feet=pose_feet)
        self.assertAlmostEqual(pm[(2, "A1")]["foot"][0], 50 / 1000)
        self.assertAlmostEqual(pm[(2, "A1")]["foot"][1], 95 / 1000)

    def test_low_conf_pose_foot_falls_back_to_bbox_nudge(self):
        det = [{"frame": 2, "track": 7, "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}]
        pose_feet = {(2, 7): (50.0, 95.0, 0.1)}  # below threshold
        pm = e.pipeline_from_detections(det, {7: "A1"}, self.H_ID, foot_k=0.16, w=1000, h=1000,
                                        pose_feet=pose_feet, pose_conf_thr=0.3)
        self.assertAlmostEqual(pm[(2, "A1")]["foot"][1], 116 / 1000)  # h*(1+0.16)


class TestScore(unittest.TestCase):
    def test_buckets_foot_error_by_half_and_counts_misses(self):
        w = h = 1000
        gt = {
            (1, "A1"): {"foot": (0.5, 0.60), "uv": (0.5, 0.8), "miss": False, "spurious": False},
            (1, "B1"): {"foot": (0.5, 0.50), "uv": (0.5, 0.1), "miss": False, "spurious": False},
            (9, "B1"): {"foot": (0.4, 0.48), "uv": (0.4, 0.0), "miss": True, "spurious": False},
        }
        pipe = {
            (1, "A1"): {"foot": (0.5, 0.58), "uv": (0.5, 0.78)},  # 20 px high (near)
            (1, "B1"): {"foot": (0.5, 0.45), "uv": (0.5, 0.05)},  # 50 px high (far)
        }
        rep = e.score(pipe, gt, w, h)
        self.assertAlmostEqual(rep["near"]["foot_px"]["mean"], 20.0, places=6)
        self.assertAlmostEqual(rep["far"]["foot_px"]["mean"], 50.0, places=6)
        self.assertEqual(rep["counts"]["matched"], 2)
        self.assertEqual(rep["counts"]["misses"], 1)


class TestScoreRecovered(unittest.TestCase):
    """Phase-4: a 'miss' that the pipeline now covers (re-ID / interp fill) counts as recovered, not a
    miss, and its court error is reported in a SEPARATE bucket so the detected near/far stays honest."""

    def test_covered_miss_is_recovered_and_kept_out_of_detected_buckets(self):
        w = h = 1000
        gt = {
            (1, "B1"): {"foot": (0.5, 0.5), "uv": (0.5, 0.10), "miss": False, "spurious": False},
            (9, "B1"): {"foot": (0.4, 0.48), "uv": (0.40, 0.0), "miss": True, "spurious": False},
        }
        pipe = {
            (1, "B1"): {"foot": (0.5, 0.5), "uv": (0.5, 0.10)},     # detected, exact -> 0 error
            (9, "B1"): {"uv": (0.45, 0.0), "filled": True},          # recovered (re-ID/interp), du=0.05
        }
        rep = e.score(pipe, gt, w, h)
        self.assertEqual(rep["counts"]["misses"], 0)
        self.assertEqual(rep["counts"]["recovered"], 1)
        # the detected far court_m sees ONLY the genuinely-detected exact point -> mean 0 (not inflated)
        self.assertAlmostEqual(rep["far"]["court_m"]["mean"], 0.0, places=6)
        # the recovered point's error is bucketed separately (0.05 of court width)
        self.assertAlmostEqual(rep["recovered"]["far"]["mean"], 0.05 * COURT_W_M, places=4)

    def test_uncovered_miss_still_counts_as_a_miss(self):
        gt = {(9, "B1"): {"foot": (0.4, 0.48), "uv": (0.4, 0.0), "miss": True, "spurious": False}}
        rep = e.score({}, gt, 1000, 1000)
        self.assertEqual(rep["counts"]["misses"], 1)
        self.assertEqual(rep["counts"]["recovered"], 0)


class TestOverlayFilled(unittest.TestCase):
    """Phase-4: pull the filled (re-ID/interp) court positions out of tracks.parquet into the pipeline
    map for frames that have no detection of their own — without overriding a genuine detection."""

    def test_adds_filled_uv_without_overriding_detected(self):
        pm = {(2, "B1"): {"foot": (0.5, 0.5), "uv": (0.5, 0.10)}}  # genuinely detected at frame 2
        tracks = [
            {"t": 0.2, "u": 0.55, "v": 0.05, "role": "far_left", "source": "detected"},  # frame 2 -> skip
            {"t": 0.3, "u": 0.60, "v": 0.02, "role": "far_left", "source": "interp"},    # frame 3 -> add
        ]
        n = e.overlay_filled(pm, tracks, {0.2: 2, 0.3: 3})
        self.assertEqual(n, 1)
        self.assertEqual(pm[(2, "B1")]["uv"], (0.5, 0.10))   # detected position untouched
        self.assertEqual(pm[(3, "B1")]["uv"], (0.60, 0.02))  # filled position added
        self.assertTrue(pm[(3, "B1")]["filled"])

    def test_unknown_role_or_unmapped_time_is_ignored(self):
        pm: dict = {}
        tracks = [{"t": 9.9, "u": 0.5, "v": 0.5, "role": "far_left", "source": "interp"},  # t not in map
                  {"t": 0.2, "u": 0.5, "v": 0.5, "role": "", "source": "interp"}]          # no role
        self.assertEqual(e.overlay_filled(pm, tracks, {0.2: 2}), 0)
        self.assertEqual(pm, {})


if __name__ == "__main__":
    unittest.main()
