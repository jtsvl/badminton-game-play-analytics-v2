"""Landing + in/out call (worker/worker/rally/landing.py) — pure, image-space selection.

The landing-frame search runs in NORMALIZED IMAGE space (x=px/W, y=py/H); court (u,v) is only read
off the chosen floor-contact detection, where the ground homography is geometrically valid. Honest
degradation is the contract: occlusion/mask-exit/frame-straddle/sparse trajectories yield
located=false — never an extrapolated landing pixel. Synthetic fixtures; no torch/cv2.
"""
import unittest

from worker.rally import landing as lnd


def det(t, x, y, conf=0.7, frame=None, u=0.5, v=0.5):
    """One shuttle detection in the shape build_rallies consumes (image-space x,y + court u,v)."""
    return {"t": t, "x": x, "y": y, "conf": conf,
            "frame": int(round(t * 30)) if frame is None else frame, "u": u, "v": v}


def descent_then_flat(n_fall=8, n_flat=6, x=0.5, y0=0.40, dy_fall=0.02, dy_flat=0.001,
                      u=0.5, v=0.9, t0=10.0, dt=1 / 30, conf=0.8):
    """A shuttle falling (vy≈0.6/s) then resting (vy≈0.03/s) — a clean visible landing.

    The first detection of the flat tail is the floor-contact candidate; it carries (u, v). The
    fixture is a 30 fps cache (frame = round(t·30)), so the detection-count thresholds derive from
    that cadence: 14 dets ≈ 0.43 s clears the 0.4 s "sparse" floor (12 dets at 30 fps)."""
    dets = []
    y = y0
    for i in range(n_fall):
        dets.append(det(t0 + i * dt, x, y, conf=conf, u=0.3, v=0.3))
        y += dy_fall
    for j in range(n_flat):
        dets.append(det(t0 + (n_fall + j) * dt, x, y, conf=conf, u=u, v=v))
        y += dy_flat
    return dets


class TestClassifyLanding(unittest.TestCase):
    def test_center_court_is_in_with_positive_margin(self):
        out = lnd.classify_landing(0.5, 0.5)
        self.assertEqual(out["call"], "in")
        self.assertGreater(out["margin_cm"], 0)

    def test_far_outside_singles_sideline_is_out(self):
        # 60 cm outside the singles side line (> the ±45 cm band)
        u = lnd.SINGLES_U_MIN - 0.60 / 6.10
        out = lnd.classify_landing(u, 0.5)
        self.assertEqual(out["call"], "out")
        self.assertEqual(out["line"], "side")
        self.assertLess(out["margin_cm"], 0)

    def test_far_beyond_baseline_is_out(self):
        v = 1.0 + 0.60 / 13.40
        out = lnd.classify_landing(0.5, v)
        self.assertEqual(out["call"], "out")
        self.assertEqual(out["line"], "baseline")

    def test_near_line_calls_are_uncertain_both_sides(self):
        just_out = lnd.SINGLES_U_MIN - 0.20 / 6.10   # 20 cm outside < band
        just_in = lnd.SINGLES_U_MIN + 0.20 / 6.10    # 20 cm inside < band
        self.assertEqual(lnd.classify_landing(just_out, 0.5)["call"], "uncertain")
        self.assertEqual(lnd.classify_landing(just_in, 0.5)["call"], "uncertain")

    def test_default_band_is_honestly_wide(self):
        self.assertGreaterEqual(lnd.LANDING_BAND_CM, 40)
        self.assertLessEqual(lnd.LANDING_BAND_CM, 50)

    def test_doubles_alley_differs_by_discipline(self):
        # mid-alley (~23 cm outside the singles line, ~23 cm inside the doubles line): with a narrow
        # band the calls diverge. The default ±45 cm band covers the whole 46 cm alley → uncertain —
        # that saturation is by design, so the discipline split is tested with an explicit band.
        u = lnd.SINGLES_U_MIN - 0.23 / 6.10
        self.assertEqual(lnd.classify_landing(u, 0.5, band_cm=10)["call"], "out")
        self.assertEqual(lnd.classify_landing(u, 0.5, band_cm=10, discipline="doubles")["call"], "in")
        self.assertEqual(lnd.classify_landing(u, 0.5)["call"], "uncertain")  # default band, singles


class TestDetectLanding(unittest.TestCase):
    def test_descent_then_flat_locates_first_flat_detection(self):
        dets = descent_then_flat()
        out = lnd.detect_landing(dets)
        self.assertTrue(out["located"])
        # the landing det is the first of the flat tail — it carries the distinct (u, v)
        self.assertAlmostEqual(out["u"], 0.5)
        self.assertAlmostEqual(out["v"], 0.9)
        self.assertFalse(out["edge_uncertain"])

    def test_too_few_terminal_detections_is_sparse(self):
        dets = descent_then_flat()[-3:]
        out = lnd.detect_landing(dets)
        self.assertFalse(out["located"])
        self.assertEqual(out["reason"], "sparse")

    def test_terminal_frame_gap_is_straddle(self):
        dets = descent_then_flat()
        # dropout right at contact: a > 0.2 s gap (7 frames at 30 fps) between the last two detections,
        # with the time advanced too so the (frame, t) span still reads as a 30 fps cache
        dets[-1]["frame"] = dets[-2]["frame"] + 7
        dets[-1]["t"] = dets[-2]["t"] + 7 / 30
        out = lnd.detect_landing(dets)
        self.assertFalse(out["located"])
        self.assertEqual(out["reason"], "straddle")

    def test_last_detection_at_frame_border_is_edge_cutoff(self):
        dets = descent_then_flat(x=0.99)
        out = lnd.detect_landing(dets)
        self.assertFalse(out["located"])
        self.assertEqual(out["reason"], "edge_cutoff")

    def test_trajectory_that_stops_while_falling_is_no_contact_and_never_extrapolated(self):
        # constant fast descent to the very last det (occlusion / mask exit / between-frame contact)
        dets = descent_then_flat(n_fall=14, n_flat=0)
        out = lnd.detect_landing(dets)
        self.assertFalse(out["located"])
        self.assertEqual(out["reason"], "no_contact")
        self.assertNotIn("u", out)   # never fabricate a landing position
        self.assertNotIn("v", out)

    def test_flat_only_trajectory_is_no_descent(self):
        dets = descent_then_flat(n_fall=0, n_flat=14)
        out = lnd.detect_landing(dets)
        self.assertFalse(out["located"])
        self.assertEqual(out["reason"], "no_descent")

    def test_landing_near_frame_edge_is_flagged_uncertain(self):
        dets = descent_then_flat(x=0.93)   # inside the hard cutoff (0.96) but in the bow zone (0.92)
        out = lnd.detect_landing(dets)
        self.assertTrue(out["located"])
        self.assertTrue(out["edge_uncertain"])


class TestLandingEvent(unittest.TestCase):
    def test_located_payload_carries_full_spec_shape(self):
        p = lnd.landing_event(descent_then_flat())
        self.assertTrue(p["located"])
        for key in ("call", "line", "u", "v", "px", "py", "margin_cm", "band_cm", "confidence"):
            self.assertIn(key, p)
        self.assertEqual(p["call"], "in")            # (0.5, 0.9) is comfortably inside
        self.assertGreater(p["confidence"], 0)
        self.assertLessEqual(p["confidence"], 1)

    def test_degraded_payload_is_uncertain_without_position(self):
        p = lnd.landing_event(descent_then_flat(n_fall=14, n_flat=0))
        self.assertFalse(p["located"])
        self.assertEqual(p["call"], "uncertain")
        self.assertIsNone(p["confidence"])
        self.assertNotIn("u", p)
        self.assertNotIn("px", p)
        self.assertIn("band_cm", p)

    def test_frame_edge_forces_uncertain_call(self):
        p = lnd.landing_event(descent_then_flat(x=0.93))   # located, but in the barrel-bow zone
        self.assertTrue(p["located"])
        self.assertEqual(p["call"], "uncertain")

    def test_implausibly_far_out_rest_is_degraded_not_called(self):
        # Real-footage false positive (Session2_Game_5 rally 5, 2026-07-03): a shuttle caught mid-air
        # at the net "came to rest" in image space; the ground homography projected the elevated pixel
        # to (u=1.13, v=-0.21) — 2.85 m beyond the baseline. A landing that far out is (almost) never
        # a floor contact → degrade honestly instead of fabricating a confident out call.
        p = lnd.landing_event(descent_then_flat(u=1.13, v=-0.21), discipline="doubles")
        self.assertFalse(p["located"])
        self.assertEqual(p["reason"], "implausible")
        self.assertEqual(p["call"], "uncertain")
        self.assertIsNone(p["confidence"])
        self.assertNotIn("u", p)
        self.assertNotIn("px", p)

    def test_out_landing_within_plausibility_cap_stays_out(self):
        # 60 cm beyond the baseline: > the ±45 cm band (a clear out) but < the implausibility cap.
        v = 1.0 + 0.60 / 13.40
        p = lnd.landing_event(descent_then_flat(u=0.5, v=v), discipline="doubles")
        self.assertTrue(p["located"])
        self.assertEqual(p["call"], "out")

    def test_plausibility_cap_default_is_wider_than_the_band(self):
        self.assertGreater(lnd.LANDING_MAX_OUT_CM, lnd.LANDING_BAND_CM)
        self.assertLessEqual(lnd.LANDING_MAX_OUT_CM, 200)

    def test_doubles_uses_full_width(self):
        # landing mid-alley: uncertain for singles (inside the band), in for doubles with the same
        # geometry only when the band allows — use a clearly-inside doubles point instead.
        u = 0.04  # 24 cm from the doubles side line, 21 cm outside the singles line
        p_s = lnd.landing_event(descent_then_flat(u=u, v=0.5))
        p_d = lnd.landing_event(descent_then_flat(u=u, v=0.5), discipline="doubles")
        self.assertEqual(p_s["call"], "uncertain")
        self.assertEqual(p_d["call"], "uncertain")   # 24 cm inside < band — honest near-line call
        self.assertGreater(p_d["margin_cm"], p_s["margin_cm"])


class TestCadenceAwareCounts(unittest.TestCase):
    """The count thresholds are durations (specs/features/analysis-30fps.md): 4 detections at the real
    10 fps cadence the values were tuned on ≡ 12 detections at 30 fps (both 0.4 s)."""

    def test_reference_cadence_reproduces_the_tuned_counts(self):
        self.assertEqual(lnd.landing_counts(10.0),
                         {"max_window_dets": 8, "min_dets": 4, "max_end_gap_frames": 2})
        self.assertEqual((lnd.LANDING_MAX_WINDOW_DETS, lnd.LANDING_MIN_DETS, lnd.LANDING_MAX_END_GAP_FRAMES),
                         (8, 4, 2))

    def test_30fps_scales_the_counts(self):
        self.assertEqual(lnd.landing_counts(30.0),
                         {"max_window_dets": 24, "min_dets": 12, "max_end_gap_frames": 6})

    def test_nine_detections_are_sparse_at_30fps_but_enough_at_10fps(self):
        dets = descent_then_flat(n_fall=5, n_flat=4)          # 9 dets = 0.27 s on a 30 fps cache
        self.assertEqual(lnd.detect_landing(dets)["reason"], "sparse")   # cadence inferred from frames
        out = lnd.detect_landing(dets, fps=10.0)              # the same 9 dets would be 0.8 s at 10 fps
        self.assertTrue(out["located"])
        self.assertAlmostEqual(out["v"], 0.9)

    def test_landing_event_passes_fps_through(self):
        dets = descent_then_flat(n_fall=5, n_flat=4)
        self.assertFalse(lnd.landing_event(dets)["located"])
        self.assertTrue(lnd.landing_event(dets, fps=10.0)["located"])

    def test_detections_without_frames_fall_back_to_the_reference_cadence(self):
        dets = [{k: v for k, v in d.items() if k != "frame"} for d in descent_then_flat(n_fall=5, n_flat=4)]
        for d in dets:
            d["frame"] = 0   # degenerate frame span → unmeasurable → REFERENCE_FPS (10) counts apply
        out = lnd.detect_landing(dets)
        self.assertTrue(out["located"])


class TestWinnerHalf(unittest.TestCase):
    def test_near_half_landing_wins_for_near(self):
        self.assertEqual(lnd.winner_half(0.9), "near")

    def test_far_half_landing_wins_for_far(self):
        self.assertEqual(lnd.winner_half(0.1), "far")

    def test_near_net_is_ambiguous(self):
        self.assertIsNone(lnd.winner_half(0.52))    # 0.27 m from the net < 1 m margin


if __name__ == "__main__":
    unittest.main()
