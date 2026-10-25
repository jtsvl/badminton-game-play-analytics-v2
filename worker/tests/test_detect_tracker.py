"""BoT-SORT's track_buffer is a DURATION (worker/pipeline/detect.py) — pure config math, no ultralytics.

The stock botsort.yaml keeps a lost track for 30 FRAMES; ultralytics hard-codes frame_rate=30 inside the
tracker, so that is 3.0 s at the real ~10 fps cadence of the tuned caches but 1.0 s at 30 fps. detect
writes a per-cache yaml copy whose track_buffer follows the manifest's cadence
(specs/features/analysis-30fps.md).
"""
import unittest

from worker.manifest import frames_for_seconds
from worker.pipeline import detect

STOCK = {
    "tracker_type": "botsort", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
    "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True,
    "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5, "appearance_thresh": 0.25, "with_reid": False,
}


class TestTrackBufferDuration(unittest.TestCase):
    def test_reference_cadence_keeps_the_stock_30_frames(self):
        self.assertEqual(frames_for_seconds(detect.TRACK_BUFFER_S, 10.0), 30)

    def test_30fps_triples_the_buffer(self):
        self.assertEqual(frames_for_seconds(detect.TRACK_BUFFER_S, 30.0), 90)


class TestTrackerConfig(unittest.TestCase):
    def test_only_track_buffer_changes(self):
        cfg = detect.tracker_config(STOCK, 90)
        self.assertEqual(cfg["track_buffer"], 90)
        self.assertEqual({k: v for k, v in cfg.items() if k != "track_buffer"},
                         {k: v for k, v in STOCK.items() if k != "track_buffer"})
        self.assertEqual(STOCK["track_buffer"], 30)   # input untouched

    def test_buffer_floor_is_one_frame(self):
        self.assertEqual(detect.tracker_config(STOCK, 0)["track_buffer"], 1)

    def test_stock_yaml_name_is_the_fallback(self):
        self.assertEqual(detect.TRACKER_YAML, "botsort.yaml")


if __name__ == "__main__":
    unittest.main()
