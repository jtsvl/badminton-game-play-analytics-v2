"""Tests for the pure shuttle-track assembly (worker/shuttle/track.py).

Pure math only (no torch/cv2) — the tracker backends produce raw per-frame peaks; this module turns
them into court-space shuttle rows: threshold visibility, map crop->frame->court, reject spikes.
"""
import os
import unittest
from unittest import mock

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


class TestStageThresholds(unittest.TestCase):
    """Stage knobs that depend on the backend or the cadence (specs/features/analysis-30fps.md,
    shuttle-tracknetv5.md): the confidence gate follows the backend's published operating point unless
    SHUTTLE_CONF_THR is set, and the spike guard is a rate applied per frame step."""

    def test_env_threshold_wins(self):
        from worker.pipeline.shuttle import resolve_conf_thr

        class Backend:
            conf_thr = 0.35

        self.assertEqual(resolve_conf_thr(Backend(), env_value="0.6"), 0.6)

    def test_backend_declared_threshold_when_env_unset(self):
        from worker.pipeline.shuttle import resolve_conf_thr

        class Backend:
            conf_thr = 0.35

        self.assertEqual(resolve_conf_thr(Backend(), env_value=None), 0.35)
        self.assertEqual(resolve_conf_thr(Backend(), env_value=""), 0.35)

    def test_half_when_nothing_is_declared(self):
        from worker.pipeline.shuttle import resolve_conf_thr

        class Backend:
            pass

        self.assertEqual(resolve_conf_thr(Backend(), env_value=None), 0.5)

    def test_spike_guard_is_a_rate(self):
        from worker.manifest import per_step
        from worker.pipeline.shuttle import MAX_STEP, MAX_STEP_PER_S

        self.assertAlmostEqual(MAX_STEP, 0.5)                       # the tuned per-step value at 10 fps
        self.assertAlmostEqual(per_step(MAX_STEP_PER_S, 30.0), 0.5 / 3)


class TestGetTrackerDefault(unittest.TestCase):
    """Locks the production default: no env, no arg → the fine-tuned TrackNetV3 backend (not WASB).
    The backend classes are patched with sentinels so no weights load — this asserts SELECTION only."""

    def test_no_env_default_is_tracknetv3(self):
        import worker.shuttle.tracknetv3 as tnv3
        from worker.shuttle import get_tracker

        sentinel = object()
        with mock.patch.dict(os.environ), \
                mock.patch.object(tnv3, "TrackNetV3Tracker", lambda **kw: sentinel):
            os.environ.pop("SHUTTLE_TRACKER", None)
            self.assertIs(get_tracker(), sentinel)

    def test_env_still_selects_wasb(self):
        import worker.shuttle.wasb as wasb_mod
        from worker.shuttle import get_tracker

        sentinel = object()
        with mock.patch.dict(os.environ, {"SHUTTLE_TRACKER": "wasb"}), \
                mock.patch.object(wasb_mod, "WASBTracker", lambda **kw: sentinel):
            self.assertIs(get_tracker(), sentinel)


if __name__ == "__main__":
    unittest.main()
