"""Tests for the pose stage's pure row assembly (worker/pipeline/pose.py).

Only the keypoint->foot derivation + parquet-row shape is unit-tested here (no model download / no
ONNXRuntime). The real RTMPose inference is validated end-to-end on the Mac Studio via the scorecard.
"""
import unittest

from worker.pipeline import pose


def _kp_scores(ankle_conf, head_conf=0.0, hip_conf=0.0, head=(110.0, 0.0), hip=(110.0, 200.0)):
    kp = [(0.0, 0.0)] * 26  # Halpe-26
    kp[15], kp[16] = (100.0, 500.0), (120.0, 510.0)  # left/right ankle
    kp[17], kp[19] = head, hip                       # head top, mid-hip
    scores = [0.0] * 26
    scores[15] = scores[16] = ankle_conf
    scores[17], scores[19] = head_conf, hip_conf
    return kp, scores


class TestPoseRow(unittest.TestCase):
    def test_confident_foot_row_carries_mean_ankle_and_full_keypoints(self):
        kp, scores = _kp_scores(0.8)
        r = pose.pose_row(3, 7, (0.0, 0.0, 200.0, 400.0), kp, scores)
        self.assertEqual(r["frame"], 3)
        self.assertEqual(r["track"], 7)
        self.assertAlmostEqual(r["foot_x"], 110.0)   # mean ankle x
        self.assertAlmostEqual(r["foot_y"], 505.0)   # mean ankle y
        self.assertAlmostEqual(r["foot_conf"], 0.8)
        self.assertEqual(len(r["kp"]), 26)           # full keypoint array kept (Phase 4 head grounding)
        self.assertEqual(len(r["kp_score"]), 26)
        self.assertEqual(r["kp"][15], [100.0, 500.0])

    def test_low_conf_foot_row_marks_zero_conf_so_gate_track_falls_back(self):
        kp, scores = _kp_scores(0.1)  # below the default confidence threshold
        r = pose.pose_row(3, 7, (0.0, 0.0, 200.0, 400.0), kp, scores)
        self.assertEqual(r["foot_conf"], 0.0)

    def test_confident_foot_row_is_sourced_from_the_ankle(self):
        kp, scores = _kp_scores(0.8)
        self.assertEqual(pose.pose_row(3, 7, (0, 0, 200, 400), kp, scores)["foot_source"], "ankle")

    def test_low_conf_foot_row_has_no_ankle_source(self):
        kp, scores = _kp_scores(0.1)
        self.assertNotEqual(pose.pose_row(3, 7, (0, 0, 200, 400), kp, scores)["foot_source"], "ankle")


class TestGroundOccludedFeet(unittest.TestCase):
    """Second pass (Phase-4 occlusion): per track, learn the head->foot drop from confident feet-visible
    frames, then predict the foot of an ankle-occluded frame from its still-visible head + hip."""

    def test_masked_ankle_foot_recovered_from_head_and_hip(self):
        # Two confident frames define foot_y-head_y = 2.5*(hip_y-head_y): head y=0, hip y=200, foot y=500.
        rows = []
        for fr in (1, 2):
            kp, sc = _kp_scores(0.9, head_conf=0.9, hip_conf=0.9, head=(110.0, 0.0), hip=(110.0, 200.0))
            r = pose.pose_row(fr, 7, (0, 0, 200, 520), kp, sc)
            self.assertAlmostEqual(r["foot_y"], 505.0)  # mean ankle y = (500+510)/2
            rows.append(r)
        # Frame 3: ankle hidden (conf 0) but head+hip visible at a different scale (torso 100 -> foot 250).
        kp, sc = _kp_scores(0.0, head_conf=0.8, hip_conf=0.8, head=(300.0, 50.0), hip=(305.0, 150.0))
        masked = pose.pose_row(3, 7, (0, 0, 200, 520), kp, sc)
        rows.append(masked)
        self.assertEqual(masked["foot_source"], "none")  # before grounding: no confident foot

        pose.ground_occluded_feet(rows)

        # ratio learned ≈ (505-0)/(200-0) ≈ 2.525 -> foot_y = 50 + 2.525*(150-50) ≈ 302.5; x under hip.
        self.assertEqual(masked["foot_source"], "head")
        self.assertAlmostEqual(masked["foot_x"], 305.0)
        self.assertAlmostEqual(masked["foot_y"], 50.0 + 2.525 * 100.0, places=6)
        self.assertGreaterEqual(masked["foot_conf"], 0.3)  # usable, but flagged via foot_source

    def test_confident_rows_are_left_untouched(self):
        kp, sc = _kp_scores(0.9, head_conf=0.9, hip_conf=0.9)
        r = pose.pose_row(1, 7, (0, 0, 200, 520), kp, sc)
        before = (r["foot_x"], r["foot_y"], r["foot_source"])
        pose.ground_occluded_feet([r])
        self.assertEqual((r["foot_x"], r["foot_y"], r["foot_source"]), before)

    def test_no_grounding_without_a_fitted_model(self):
        # masked ankle, head/hip visible, but NO confident feet-visible frame to fit from -> unchanged.
        kp, sc = _kp_scores(0.0, head_conf=0.8, hip_conf=0.8)
        r = pose.pose_row(1, 7, (0, 0, 200, 520), kp, sc)
        pose.ground_occluded_feet([r])
        self.assertEqual(r["foot_source"], "none")


if __name__ == "__main__":
    unittest.main()
