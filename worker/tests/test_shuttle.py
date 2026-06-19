"""Tests for the pure shuttle-track assembly (worker/shuttle/track.py).

Pure math only (no torch/cv2) — the tracker backends produce raw per-frame peaks; this module turns
them into court-space shuttle rows: threshold visibility, map crop->frame->court, reject spikes.
"""
import unittest

from worker import geometry as g
from worker.shuttle.track import build_shuttle_track

# Crop box + model input chosen so map_detection_to_frame is offset-only: px=100+mx, py=100+my.
# RECT_PX maps to the unit court square, so the court coords reduce to u=mx/400, v=my/300.
RECT_PX = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
H = g.homography_from_corners(RECT_PX)
CROP = (100, 100, 500, 400)  # cw=400, ch=300
MW, MH = 400, 300
T_BY_FRAME = {0: 0.0, 1: 0.0833, 2: 0.1667, 3: 0.25}


class TestBuildShuttleTrack(unittest.TestCase):
    def _run(self, raw, conf_thr=0.5, max_step=0.5):
        return build_shuttle_track(raw, H, CROP, MW, MH, T_BY_FRAME,
                                   conf_thr=conf_thr, max_step=max_step)

    def test_drops_low_confidence_peaks_as_real_dropouts(self):
        raw = [(0, 200, 150, 0.9), (1, 200, 150, 0.2), (2, 240, 150, 0.8)]
        rows = self._run(raw)
        self.assertEqual([r["frame"] for r in rows], [0, 2])  # frame 1 below threshold -> absent

    def test_maps_peak_to_court_coords(self):
        rows = self._run([(0, 200, 150, 0.9)])
        r = rows[0]
        self.assertAlmostEqual(r["u"], 0.5, places=6)         # mx/400
        self.assertAlmostEqual(r["v"], 0.5, places=6)         # my/300
        self.assertAlmostEqual(r["visibility"], 0.9, places=6)
        self.assertEqual((r["px"], r["py"]), (300.0, 250.0))  # crop offset 100 + peak

    def test_rows_sorted_by_time_with_contract_fields(self):
        raw = [(2, 240, 150, 0.8), (0, 200, 150, 0.9)]        # out of order in
        rows = self._run(raw)
        self.assertEqual([r["t"] for r in rows], sorted(r["t"] for r in rows))
        for key in ("t", "u", "v", "visibility", "frame", "px", "py", "conf"):
            self.assertIn(key, rows[0])

    def test_isolated_spike_is_rejected(self):
        # frame 1 maps to ~(1.0, 0.97), far from frames 0 & 2 at (0.5,0.5) which match each other.
        raw = [(0, 200, 150, 0.9), (1, 400, 290, 0.9), (2, 200, 150, 0.9)]
        rows = self._run(raw, max_step=0.5)
        self.assertEqual([r["frame"] for r in rows], [0, 2])


if __name__ == "__main__":
    unittest.main()
