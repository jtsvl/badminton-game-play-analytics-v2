"""Tests for the pose stage's pure row assembly (worker/pipeline/pose.py).

Only the keypoint->foot derivation + parquet-row shape is unit-tested here (no model download / no
ONNXRuntime). The real RTMPose inference is validated end-to-end on the Mac Studio via the scorecard.
"""
import unittest

from worker.pipeline import pose


def _kp_scores(ankle_conf):
    kp = [(0.0, 0.0)] * 26  # Halpe-26
    kp[15], kp[16] = (100.0, 500.0), (120.0, 510.0)  # left/right ankle
    scores = [0.0] * 26
    scores[15] = scores[16] = ankle_conf
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


if __name__ == "__main__":
    unittest.main()
