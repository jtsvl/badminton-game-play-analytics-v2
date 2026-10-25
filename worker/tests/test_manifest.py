"""Frame-cache cadence helpers (worker/worker/manifest.py) — pure, stdlib only.

The frames.json manifest records the REQUESTED `analysis_fps`; the real cadence is whatever ingest's
"keep when t >= next_t" rule produced from the source's PTS (on a 30 fps source "12 fps" is really
10 fps). Every seconds→frames conversion in the worker must therefore come from the measured `t`
spacing, never from the requested number (specs/features/analysis-30fps.md).
"""
import unittest

from worker import manifest as mf


def frames_at(fps: float, n: int = 50, t0: float = 0.0) -> list:
    return [{"frame": i, "t": round(t0 + i / fps, 4)} for i in range(n)]


def ingest_like(source_fps: float, requested: float, n_source: int = 300) -> list:
    """Replay ingest.py's keep rule (t + 1e-6 < next_t → skip; next_t = t + 1/requested)."""
    step = 1.0 / requested
    kept, next_t = [], 0.0
    for i in range(n_source):
        t = i / source_fps
        if t + 1e-6 < next_t:
            continue
        next_t = t + step
        kept.append({"frame": len(kept), "t": round(t, 4)})
    return kept


class TestMeasuredFps(unittest.TestCase):
    def test_uniform_12_and_30(self):
        self.assertAlmostEqual(mf.measured_fps(frames_at(12.0)), 12.0, places=1)
        self.assertAlmostEqual(mf.measured_fps(frames_at(30.0)), 30.0, places=1)

    def test_requested_12_on_a_30fps_source_is_really_10(self):
        # the production Session-2 case: every 3rd source frame survives → 10 fps, not 12
        self.assertAlmostEqual(mf.measured_fps(ingest_like(30.0, 12.0)), 10.0, places=1)

    def test_requested_30_on_a_29_97_source_is_native(self):
        fps = mf.measured_fps(ingest_like(30000 / 1001, 30.0))
        self.assertAlmostEqual(fps, 30000 / 1001, delta=0.05)

    def test_requested_12_on_a_59_94_source_is_about_12(self):
        self.assertAlmostEqual(mf.measured_fps(ingest_like(60000 / 1001, 12.0)), 11.99, delta=0.05)

    def test_median_is_robust_to_one_dropped_frame(self):
        frames = frames_at(30.0)
        del frames[10]  # a single gap must not move the cadence estimate
        self.assertAlmostEqual(mf.measured_fps(frames), 30.0, places=1)

    def test_too_few_frames_is_none(self):
        self.assertIsNone(mf.measured_fps([]))
        self.assertIsNone(mf.measured_fps([{"frame": 0, "t": 0.0}]))

    def test_zero_spacing_is_none(self):
        self.assertIsNone(mf.measured_fps([{"frame": 0, "t": 1.0}, {"frame": 1, "t": 1.0}]))


class TestManifestFps(unittest.TestCase):
    def test_prefers_measured_over_requested(self):
        manifest = {"analysis_fps": 12.0, "frames": ingest_like(30.0, 12.0)}
        self.assertAlmostEqual(mf.manifest_fps(manifest), 10.0, places=1)

    def test_falls_back_to_requested_when_unmeasurable(self):
        self.assertEqual(mf.manifest_fps({"analysis_fps": 30.0, "frames": []}), 30.0)
        self.assertEqual(mf.manifest_fps({"analysis_fps": 30.0}), 30.0)

    def test_falls_back_to_12_when_nothing_is_recorded(self):
        # old test fixtures / hand-written manifests carry neither key
        self.assertEqual(mf.manifest_fps({"frames": [{"frame": 0, "t": 0.0}]}), 12.0)
        self.assertEqual(mf.manifest_fps({}), 12.0)


class TestFramesForSeconds(unittest.TestCase):
    def test_reproduces_todays_counts_at_the_real_10fps_cadence(self):
        # the frames×0.1 s convention: 8 frames ≡ 0.8 s, 9 ≡ 0.9 s, 6 ≡ 0.6 s, 2 ≡ 0.2 s
        self.assertEqual(mf.frames_for_seconds(0.8, 10.0), 8)
        self.assertEqual(mf.frames_for_seconds(0.9, 10.0), 9)
        self.assertEqual(mf.frames_for_seconds(0.6, 10.0), 6)
        self.assertEqual(mf.frames_for_seconds(0.2, 10.0), 2)

    def test_scales_to_30fps(self):
        self.assertEqual(mf.frames_for_seconds(0.8, 30.0), 24)
        self.assertEqual(mf.frames_for_seconds(0.2, 30.0), 6)
        self.assertEqual(mf.frames_for_seconds(0.9, 29.97), 27)

    def test_rounds_half_up_not_bankers(self):
        self.assertEqual(mf.frames_for_seconds(0.25, 10.0), 3)   # 2.5 → 3 (round() would give 2)
        self.assertEqual(mf.frames_for_seconds(0.35, 10.0), 4)   # 3.5 → 4

    def test_minimum_clamp(self):
        self.assertEqual(mf.frames_for_seconds(0.01, 10.0), 1)
        self.assertEqual(mf.frames_for_seconds(0.01, 10.0, minimum=2), 2)
        self.assertEqual(mf.frames_for_seconds(0.0, 30.0), 1)

    def test_invalid_fps_raises(self):
        with self.assertRaises(ValueError):
            mf.frames_for_seconds(0.5, 0.0)
        with self.assertRaises(ValueError):
            mf.frames_for_seconds(0.5, -1.0)


class TestSecondsForFrames(unittest.TestCase):
    def test_round_trip(self):
        self.assertAlmostEqual(mf.seconds_for_frames(8, 10.0), 0.8)
        self.assertAlmostEqual(mf.seconds_for_frames(24, 30.0), 0.8)

    def test_invalid_fps_raises(self):
        with self.assertRaises(ValueError):
            mf.seconds_for_frames(1, 0.0)


class TestFpsFromFramesT(unittest.TestCase):
    def test_exact_over_gaps(self):
        # shuttle detections with dropouts still pin the cadence from the (frame, t) span
        pairs = [(0, 0.0), (3, 0.1), (10, 1 / 3), (30, 1.0)]
        self.assertAlmostEqual(mf.fps_from_frames_t(pairs), 30.0, places=6)

    def test_order_independent(self):
        self.assertAlmostEqual(mf.fps_from_frames_t([(30, 3.0), (0, 0.0)]), 10.0)

    def test_degenerate_is_none(self):
        self.assertIsNone(mf.fps_from_frames_t([]))
        self.assertIsNone(mf.fps_from_frames_t([(5, 0.5)]))
        self.assertIsNone(mf.fps_from_frames_t([(5, 0.5), (5, 0.6)]))
        self.assertIsNone(mf.fps_from_frames_t([(5, 0.5), (6, 0.5)]))


class TestReferenceCadence(unittest.TestCase):
    def test_reference_is_the_real_production_cadence(self):
        # the frame-count thresholds were tuned on "12 fps" caches that really run at 10 fps
        self.assertEqual(mf.REFERENCE_FPS, 10.0)
        self.assertEqual(mf.NOMINAL_FPS, 12.0)


class TestPerStepFromRate(unittest.TestCase):
    def test_rate_to_per_step(self):
        # a per-second rate becomes a per-frame step: 5.0 court-units/s at 10 fps = 0.5 per step
        self.assertAlmostEqual(mf.per_step(5.0, 10.0), 0.5)
        self.assertAlmostEqual(mf.per_step(5.0, 30.0), 5.0 / 30.0)

    def test_invalid_fps_raises(self):
        with self.assertRaises(ValueError):
            mf.per_step(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
