"""Tests for the pure geometry helpers (worker/geometry.py).

Pure math only — no numpy/cv2/torch — so these run under the dev extra without the ml stack.
Expected values are hand-computed from clean affine cases.
"""
import unittest

from worker import geometry as g


# An axis-aligned pixel rectangle whose corners map to the unit court square.
# Order matches the canonical confirmed_points: FL, FR, NR, NL (u left->right, v far->near).
RECT_PX = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
UNIT = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
# This rectangle gives the affine map u = (x-100)/400, v = (y-100)/300.


class TestHomography(unittest.TestCase):
    def test_corners_map_to_unit_square(self):
        H = g.homography_from_corners(RECT_PX)
        for (x, y), (eu, ev) in zip(RECT_PX, UNIT):
            u, v = g.apply_homography(H, x, y)
            self.assertAlmostEqual(u, eu, places=6)
            self.assertAlmostEqual(v, ev, places=6)

    def test_center_pixel_maps_to_court_center(self):
        H = g.homography_from_corners(RECT_PX)
        u, v = g.apply_homography(H, 300.0, 250.0)  # rectangle center
        self.assertAlmostEqual(u, 0.5, places=6)
        self.assertAlmostEqual(v, 0.5, places=6)

    def test_h8_normalized_to_one(self):
        H = g.homography_from_corners(RECT_PX)
        self.assertEqual(len(H), 9)
        self.assertAlmostEqual(H[8], 1.0, places=9)

    def test_singular_corners_raise(self):
        # three collinear corners -> degenerate, no valid homography
        with self.assertRaises(Exception):
            g.homography_from_corners([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])


class TestReprojectError(unittest.TestCase):
    def test_exact_corners_zero_error(self):
        H = g.homography_from_corners(RECT_PX)
        err = g.reproject_error_m(H, RECT_PX, UNIT)
        self.assertLess(err, 1e-6)

    def test_known_offset_residual_in_metres(self):
        # center pixel truly maps to (0.5,0.5); claim it is (0.6,0.5) -> du=0.1 of court width.
        H = g.homography_from_corners(RECT_PX)
        err = g.reproject_error_m(H, [(300.0, 250.0)], [(0.6, 0.5)])
        self.assertAlmostEqual(err, 0.1 * g.COURT_W_M, places=4)  # 0.61 m


class TestPointInPoly(unittest.TestCase):
    SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    def test_inside(self):
        self.assertTrue(g.point_in_poly((0.5, 0.5), self.SQUARE))

    def test_outside(self):
        self.assertFalse(g.point_in_poly((1.3, 0.5), self.SQUARE))
        self.assertFalse(g.point_in_poly((-0.1, 0.5), self.SQUARE))

    def test_margin_polygon_admits_just_outside_point(self):
        m = 0.1
        expanded = [(-m, -m), (1 + m, -m), (1 + m, 1 + m), (-m, 1 + m)]
        self.assertTrue(g.point_in_poly((1.05, 0.5), expanded))
        self.assertFalse(g.point_in_poly((1.05, 0.5), self.SQUARE))


class TestAnklePoint(unittest.TestCase):
    def test_bottom_center_of_top_left_bbox(self):
        # bbox = (x, y, w, h) top-left + size
        self.assertEqual(g.ankle_point((100.0, 200.0, 40.0, 80.0)), (120.0, 280.0))


class TestFootPoint(unittest.TestCase):
    def test_default_k_is_identical_to_ankle_point(self):
        bbox = (100.0, 200.0, 40.0, 80.0)
        self.assertEqual(g.foot_point(bbox), g.ankle_point(bbox))

    def test_biases_foot_down_by_k_times_height(self):
        # Correct the detector's bbox-bottom truncation: push the foot down k*height.
        fx, fy = g.foot_point((100.0, 200.0, 40.0, 80.0), 0.25)
        self.assertEqual(fx, 120.0)            # x unchanged (centre)
        self.assertEqual(fy, 280.0 + 0.25 * 80.0)  # 280 + 20 = 300


class TestFootFromKeypoints(unittest.TestCase):
    """Phase-3 durable foot: ankle/foot keypoint when confident, else the Phase-2 bbox nudge."""

    def _kp(self):
        kp = [(0.0, 0.0)] * 26  # Halpe-26
        kp[15], kp[16] = (100.0, 500.0), (120.0, 510.0)  # left/right ankle
        return kp

    def test_confident_ankles_give_mean_ankle_foot(self):
        scores = [0.0] * 26
        scores[15], scores[16] = 0.9, 0.8
        foot = g.foot_from_keypoints(self._kp(), scores, bbox=(0.0, 0.0, 200.0, 400.0),
                                     k_fallback=0.16, conf_thr=0.3)
        self.assertAlmostEqual(foot[0], 110.0)  # mean ankle x
        self.assertAlmostEqual(foot[1], 505.0)  # mean ankle y

    def test_low_confidence_falls_back_to_bbox_nudge(self):
        scores = [0.0] * 26
        scores[15], scores[16] = 0.1, 0.05  # both below conf_thr
        bbox = (0.0, 0.0, 200.0, 400.0)
        foot = g.foot_from_keypoints(self._kp(), scores, bbox=bbox, k_fallback=0.16, conf_thr=0.3)
        self.assertEqual(foot, g.foot_point(bbox, 0.16))

    def test_empty_keypoints_falls_back(self):
        bbox = (0.0, 0.0, 10.0, 20.0)
        self.assertEqual(g.foot_from_keypoints([], [], bbox=bbox, k_fallback=0.0),
                         g.foot_point(bbox, 0.0))

    def test_foot_keypoint_reports_zero_conf_when_all_below_threshold(self):
        scores = [0.0] * 26
        scores[15], scores[16] = 0.1, 0.2
        fk = g.foot_keypoint(self._kp(), scores, conf_thr=0.3)
        self.assertIsNotNone(fk)
        self.assertEqual(fk[2], 0.0)  # conf 0 => caller must fall back


class TestHeadFootGrounding(unittest.TestCase):
    """Phase-4 occlusion layer: when the ankle is hidden but the head + hip are still visible, predict
    the foot from the head using a head->foot drop learned (per player) from their own confident,
    feet-visible frames. Scaled by torso length (hip_y-head_y) — visible under lower-body occlusion —
    not bbox height (which the detector truncates, the very RC1 problem this avoids)."""

    def test_halpe26_head_and_hip_indices(self):
        self.assertEqual(g.HEAD_KP, 17)
        self.assertEqual(g.HIP_KP, 19)

    def test_fit_learns_median_foot_over_torso_ratio(self):
        # each sample: foot_y-head_y is 3x the torso (hip_y-head_y) -> ratio 3.0.
        samples = [(0.0, 100.0, 300.0), (10.0, 60.0, 160.0), (0.0, 50.0, 150.0)]
        self.assertAlmostEqual(g.fit_head_foot_model(samples), 3.0, places=6)

    def test_fit_is_robust_to_an_outlier_via_median(self):
        samples = [(0.0, 100.0, 300.0), (0.0, 100.0, 300.0), (0.0, 100.0, 999.0)]  # one bad foot
        self.assertAlmostEqual(g.fit_head_foot_model(samples), 3.0, places=6)

    def test_fit_returns_none_when_no_valid_torso(self):
        self.assertIsNone(g.fit_head_foot_model([]))
        self.assertIsNone(g.fit_head_foot_model([(100.0, 100.0, 300.0)]))  # hip==head, zero torso

    def test_ground_predicts_foot_below_pelvis_from_head_and_hip(self):
        foot = g.ground_foot_from_head((40.0, 0.0), (50.0, 100.0), 3.0)
        self.assertAlmostEqual(foot[0], 50.0)   # foot x under the pelvis (hip x)
        self.assertAlmostEqual(foot[1], 300.0)  # head_y + 3*(hip_y-head_y)


class TestNetDepth(unittest.TestCase):
    def test_near_role(self):
        self.assertAlmostEqual(g.net_depth(1.0, "near"), 1.0)   # own baseline -> rear
        self.assertAlmostEqual(g.net_depth(0.5, "near"), 0.0)   # net -> front
        self.assertAlmostEqual(g.net_depth(0.75, "near"), 0.5)  # mid

    def test_far_role(self):
        self.assertAlmostEqual(g.net_depth(0.0, "far"), 1.0)
        self.assertAlmostEqual(g.net_depth(0.5, "far"), 0.0)
        self.assertAlmostEqual(g.net_depth(0.25, "far"), 0.5)

    def test_doubles_role_prefix(self):
        self.assertAlmostEqual(g.net_depth(0.9, "near_left"), g.net_depth(0.9, "near"))
        self.assertAlmostEqual(g.net_depth(0.1, "far_right"), g.net_depth(0.1, "far"))

    def test_clamped_across_net(self):
        self.assertAlmostEqual(g.net_depth(0.4, "near"), 0.0)  # wrong half -> clamp 0
        self.assertAlmostEqual(g.net_depth(0.6, "far"), 0.0)


class TestRoleAssign(unittest.TestCase):
    def test_singles_near_far_by_v(self):
        roles = g.role_assign({1: (0.5, 0.8), 2: (0.5, 0.2)}, "singles")
        self.assertEqual(roles, {1: "near", 2: "far"})

    def test_doubles_quadrants(self):
        medians = {1: (0.3, 0.8), 2: (0.7, 0.8), 3: (0.3, 0.2), 4: (0.7, 0.2)}
        roles = g.role_assign(medians, "doubles")
        self.assertEqual(
            roles, {1: "near_left", 2: "near_right", 3: "far_left", 4: "far_right"}
        )


class TestEnforceNTracks(unittest.TestCase):
    def test_keeps_top_n_by_count(self):
        counts = {1: 50, 2: 200, 3: 10, 4: 180, 5: 5}
        self.assertEqual(g.enforce_n_tracks(counts, 2), [2, 4])

    def test_fewer_than_n_returns_all(self):
        self.assertEqual(set(g.enforce_n_tracks({7: 3, 9: 1}, 4)), {7, 9})


class TestResample10Hz(unittest.TestCase):
    def test_linear_single_segment(self):
        track = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        out = g.resample_10hz(track, hz=10)
        self.assertEqual(len(out), 11)  # t = 0.0 .. 1.0 inclusive at 0.1
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[-1][0], 1.0)
        # monotonic time
        for a, b in zip(out, out[1:]):
            self.assertLess(a[0], b[0])
        # midpoint
        t, u, v = out[3]
        self.assertAlmostEqual(t, 0.3, places=6)
        self.assertAlmostEqual(u, 0.3, places=6)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_endpoints_preserved_multi_segment(self):
        track = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.5, 0.5)]
        out = g.resample_10hz(track, hz=10)
        self.assertAlmostEqual(out[0][1], 0.0)
        self.assertAlmostEqual(out[0][2], 0.0)
        # t=0.7 lies in the second segment (0.5->1.0); fraction 0.4 -> u stays 0.5, v=0.2
        at7 = next(s for s in out if abs(s[0] - 0.7) < 1e-9)
        self.assertAlmostEqual(at7[1], 0.5, places=6)
        self.assertAlmostEqual(at7[2], 0.2, places=6)


class TestStitchRoleFragments(unittest.TestCase):
    """Phase-4 role re-ID: re-associate dropped in-court track fragments with the role they continue,
    so a player lost by BoT-SORT and re-detected under a fresh id resumes the same role. Modelled on
    f35d0984: far-left player (B1) lost after frame 282, re-detected as short fragments 64 & 68."""

    PRIMARIES = {
        "near_left": {"median": (0.30, 0.80), "span": (0, 355)},
        "near_right": {"median": (0.70, 0.80), "span": (0, 355)},
        "far_left": {"median": (0.55, 0.05), "span": (0, 282)},   # lost after 282 (the hole)
        "far_right": {"median": (0.50, 0.20), "span": (0, 355)},  # fully detected (no hole)
    }

    def test_far_fragments_stitch_into_the_role_with_the_temporal_hole(self):
        frags = {64: {"median": (0.70, 0.05), "span": (286, 337)},
                 68: {"median": (0.50, 0.02), "span": (342, 355)}}
        self.assertEqual(g.stitch_role_fragments(self.PRIMARIES, frags),
                         {64: "far_left", 68: "far_left"})

    def test_overlapping_role_is_not_claimed_even_if_nearest(self):
        # fragment sits exactly on far_right's median, but far_right is present the whole time
        # (overlap) → it must fall to far_left, the role with the actual hole.
        frags = {64: {"median": (0.50, 0.20), "span": (286, 337)}}
        self.assertEqual(g.stitch_role_fragments(self.PRIMARIES, frags), {64: "far_left"})

    def test_fragment_overlapping_every_same_half_role_is_dropped(self):
        # a far blip while both far roles are present fills no hole → not stitched.
        prim = dict(self.PRIMARIES, far_left={"median": (0.55, 0.05), "span": (0, 355)})
        frags = {64: {"median": (0.70, 0.05), "span": (286, 337)}}
        self.assertEqual(g.stitch_role_fragments(prim, frags), {})

    def test_other_half_roles_are_ignored(self):
        # a near-half fragment is never assigned to a far role and vice-versa.
        frags = {64: {"median": (0.70, 0.05), "span": (286, 337)}}  # far (v<0.5)
        out = g.stitch_role_fragments(self.PRIMARIES, frags)
        self.assertNotIn(out.get(64), ("near_left", "near_right"))

    def test_spatially_distant_fragment_is_rejected(self):
        # far-half but nowhere near either far role's position → guarded out (not a continuation).
        frags = {99: {"median": (0.02, 0.45), "span": (286, 337)}}
        self.assertEqual(g.stitch_role_fragments(self.PRIMARIES, frags, max_court_dist=0.4), {})


class TestFillTrackGaps(unittest.TestCase):
    """Phase-4 bounded short-hole interpolation: fill interior gaps of <= max_gap missing frames
    between two confident detections; never extrapolate past the ends (no second anchor → leave absent)."""

    def test_fills_short_interior_gap_with_linear_interp(self):
        # frames 0 and 4 present, 1..3 missing (a 3-frame hole, <= max_gap) -> filled linearly.
        out = g.fill_track_gaps([(0, 0.0, 0.0), (4, 4.0, 0.0)], max_gap=8)
        frames = [f for f, _, _, _ in out]
        self.assertEqual(frames, [0, 1, 2, 3, 4])
        filled = {f: (u, v, is_filled) for f, u, v, is_filled in out}
        self.assertEqual(filled[2], (2.0, 0.0, True))   # midpoint interp, flagged filled
        self.assertFalse(filled[0][2])                  # anchors not flagged
        self.assertFalse(filled[4][2])

    def test_gap_larger_than_max_is_left_unfilled(self):
        # 20-frame hole > max_gap -> no fabricated positions across the void.
        out = g.fill_track_gaps([(0, 0.0, 0.0), (20, 20.0, 0.0)], max_gap=8)
        self.assertEqual([f for f, _, _, _ in out], [0, 20])

    def test_never_extrapolates_before_first_or_after_last(self):
        # only interior frames are produced; ends are the existing anchors, nothing beyond.
        out = g.fill_track_gaps([(5, 0.0, 0.0), (7, 2.0, 0.0)], max_gap=8)
        frames = [f for f, _, _, _ in out]
        self.assertEqual(min(frames), 5)
        self.assertEqual(max(frames), 7)

    def test_single_or_empty_track_unchanged(self):
        self.assertEqual(g.fill_track_gaps([(3, 0.1, 0.2)], max_gap=8), [(3, 0.1, 0.2, False)])
        self.assertEqual(g.fill_track_gaps([], max_gap=8), [])


class TestSmoothTrack(unittest.TestCase):
    def test_single_spike_is_damped(self):
        track = [(0, 0.5, 0.5), (1, 0.5, 0.5), (2, 0.9, 0.5), (3, 0.5, 0.5), (4, 0.5, 0.5)]
        out = g.smooth_track(track, window=5)
        self.assertLess(out[2][1], 0.7)  # the 0.9 spike averaged down
        self.assertAlmostEqual(out[2][0], 2)  # t preserved

    def test_constant_track_unchanged(self):
        track = [(i, 0.4, 0.6) for i in range(6)]
        for s in g.smooth_track(track, window=5):
            self.assertAlmostEqual(s[1], 0.4)
            self.assertAlmostEqual(s[2], 0.6)

    def test_linear_interior_preserved(self):
        track = [(i, 0.1 * i, 0.0) for i in range(7)]  # ramp in u
        out = g.smooth_track(track, window=3)
        self.assertAlmostEqual(out[3][1], 0.3, places=6)  # centred avg of a ramp = the centre value

    def test_short_track_returned_as_is(self):
        track = [(0, 0.2, 0.3), (1, 0.4, 0.5)]
        self.assertEqual(g.smooth_track(track, window=5), track)


class TestCourtCropBox(unittest.TestCase):
    """Stage-5 multi-court isolation + tiny-shuttle resolution: crop to the target court's polygon,
    expanded UPWARD (image up = smaller y) to keep the shuttle's arc over the net, side margin for
    width, small bottom margin; clamped to the frame."""

    POLY = [(100.0, 200.0), (300.0, 200.0), (300.0, 260.0), (100.0, 260.0)]  # w=200, h=60

    def test_expands_up_and_sideways_from_polygon_bbox(self):
        box = g.court_crop_box(self.POLY, 1000, 1000, up=0.5, side=0.1, down=0.0)
        # x: 100-20 .. 300+20 ; y: 200-30 (headroom above) .. 260 (no bottom margin)
        self.assertEqual(box, (80, 170, 320, 260))

    def test_headroom_is_upward_not_downward(self):
        box = g.court_crop_box(self.POLY, 1000, 1000, up=1.0, side=0.0, down=0.0)
        x0, y0, x1, y1 = box
        self.assertLess(y0, 200)        # top extended up for the arc
        self.assertEqual(y1, 260)       # bottom unchanged (down=0)

    def test_clamps_to_frame_bounds(self):
        poly = [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)]  # w=40, h=20
        box = g.court_crop_box(poly, 100, 100, up=2.0, side=2.0, down=0.0)
        x0, y0, x1, y1 = box
        self.assertEqual(x0, 0)         # 10 - 80 -> clamp 0
        self.assertEqual(y0, 0)         # 10 - 40 -> clamp 0
        self.assertEqual(x1, 100)       # 50 + 80 -> clamp 100


class TestCourtAirspacePolygon(unittest.TestCase):
    """Stage-5 isolation mask: the target court's AIRSPACE = the ground quad with its far (top) edge
    raised upward (image up = smaller y) to cover where the shuttle actually flies, bounded by the
    court's own sidelines so the neighbouring courts are excluded. Robust to corner input order."""

    # FL, FR, NR, NL (canonical) — far baseline at y=200 (higher in image), near at y=400.
    QUAD = [(100.0, 200.0), (300.0, 200.0), (320.0, 400.0), (80.0, 400.0)]

    def test_raises_far_edge_and_widens(self):
        poly = g.court_airspace_polygon(self.QUAD, 1000, 1000, up=0.5, side=0.1)
        # height=200, width=240 -> up 100px, side 24px; near corners unchanged.
        self.assertEqual(poly, [(76.0, 100.0), (324.0, 100.0), (320.0, 400.0), (80.0, 400.0)])

    def test_robust_to_corner_order(self):
        shuffled = [self.QUAD[2], self.QUAD[0], self.QUAD[3], self.QUAD[1]]
        self.assertEqual(g.court_airspace_polygon(shuffled, 1000, 1000, up=0.5, side=0.1),
                         g.court_airspace_polygon(self.QUAD, 1000, 1000, up=0.5, side=0.1))

    def test_clamps_to_frame_top(self):
        poly = g.court_airspace_polygon(self.QUAD, 1000, 1000, up=2.0, side=0.0)
        self.assertEqual(poly[0][1], 0.0)   # raised far above the frame -> clamped to 0
        self.assertEqual(poly[1][1], 0.0)


class TestSplitOnGaps(unittest.TestCase):
    """Overlay must not draw a straight line across a real shuttle dropout (white-wall camouflage):
    split a time-sorted track wherever consecutive samples are more than max_dt apart, so each segment
    is resampled on its own and the gap stays a gap."""

    def test_splits_where_time_gap_exceeds_max_dt(self):
        track = [(0.0, 0.1, 0.1), (0.1, 0.2, 0.1), (1.5, 0.8, 0.5), (1.6, 0.9, 0.5)]
        segs = g.split_on_gaps(track, max_dt=0.5)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], [(0.0, 0.1, 0.1), (0.1, 0.2, 0.1)])
        self.assertEqual(segs[1], [(1.5, 0.8, 0.5), (1.6, 0.9, 0.5)])

    def test_contiguous_track_is_one_segment(self):
        track = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0), (0.2, 0.2, 0.0)]
        self.assertEqual(g.split_on_gaps(track, max_dt=0.5), [track])

    def test_empty_track(self):
        self.assertEqual(g.split_on_gaps([], max_dt=0.5), [])


class TestMapDetectionToFrame(unittest.TestCase):
    """A tracker peak is in model-input pixels over the crop; map it back to full-frame pixels
    (direct resize: independent x/y scale, then add the crop offset) before the homography."""

    BOX = (100, 200, 300, 260)  # cw=200, ch=60

    def test_model_centre_maps_to_crop_centre(self):
        px, py = g.map_detection_to_frame(256, 144, self.BOX, 512, 288)
        self.assertAlmostEqual(px, 200.0)  # crop centre x = (100+300)/2
        self.assertAlmostEqual(py, 230.0)  # crop centre y = (200+260)/2

    def test_corners_map_to_box_corners(self):
        self.assertEqual(g.map_detection_to_frame(0, 0, self.BOX, 512, 288), (100.0, 200.0))
        self.assertEqual(g.map_detection_to_frame(512, 288, self.BOX, 512, 288), (300.0, 260.0))

    def test_frame_to_model_is_exact_inverse(self):
        # round-trip identity: a label click -> 512x288 -> back to full-frame must return the click.
        for px, py in [(100.0, 200.0), (300.0, 260.0), (200.0, 230.0), (157.3, 241.8)]:
            mx, my = g.frame_to_model_xy(px, py, self.BOX, 512, 288)
            bx, by = g.map_detection_to_frame(mx, my, self.BOX, 512, 288)
            self.assertAlmostEqual(bx, px)
            self.assertAlmostEqual(by, py)

    def test_frame_to_model_corners(self):
        self.assertEqual(g.frame_to_model_xy(100.0, 200.0, self.BOX, 512, 288), (0.0, 0.0))
        self.assertEqual(g.frame_to_model_xy(300.0, 260.0, self.BOX, 512, 288), (512.0, 288.0))


class TestRejectShuttleOutliers(unittest.TestCase):
    """Light spike removal in court space: drop a point only when it juts away from BOTH neighbours
    while those neighbours stay close to each other (an isolated false positive, e.g. another court).
    A genuine fast smash (consecutive large, collinear steps) must survive — no arc flattening."""

    def test_smooth_arc_is_unchanged(self):
        track = [(0, 0.10, 0.10), (1, 0.20, 0.15), (2, 0.30, 0.18), (3, 0.40, 0.19)]
        self.assertEqual(g.reject_shuttle_outliers(track, max_step=0.3), track)

    def test_isolated_spike_is_dropped(self):
        track = [(0, 0.10, 0.10), (1, 0.12, 0.10), (2, 0.90, 0.80), (3, 0.16, 0.10), (4, 0.18, 0.10)]
        out = g.reject_shuttle_outliers(track, max_step=0.3)
        self.assertEqual(len(out), 4)
        self.assertNotIn((2, 0.90, 0.80), out)

    def test_fast_collinear_motion_is_kept(self):
        # each step 0.4 > max_step, but prev~next are also far apart -> not an isolated spike.
        track = [(0, 0.0, 0.0), (1, 0.4, 0.0), (2, 0.8, 0.0), (3, 1.2, 0.0)]
        self.assertEqual(g.reject_shuttle_outliers(track, max_step=0.3), track)

    def test_short_track_returned_as_is(self):
        track = [(0, 0.1, 0.2), (1, 0.4, 0.5)]
        self.assertEqual(g.reject_shuttle_outliers(track, max_step=0.3), track)


if __name__ == "__main__":
    unittest.main()
