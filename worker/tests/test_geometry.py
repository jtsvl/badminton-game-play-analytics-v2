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

    def test_horizon_point_raises_value_error(self):
        # denom = 0*x + 1*y - 100 -> zero along y=100 (the camera horizon of this H).
        # A raw ZeroDivisionError here would be cryptic; the guard names the problem.
        H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -100.0]
        with self.assertRaises(ValueError):
            g.apply_homography(H, 50.0, 100.0)
        # ...and stays a working projection off the horizon
        u, v = g.apply_homography(H, 50.0, 200.0)
        self.assertAlmostEqual(u, 0.5, places=9)
        self.assertAlmostEqual(v, 2.0, places=9)


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


class TestSelectRolePrimaries(unittest.TestCase):
    """Half-aware primary selection (successor of global enforce_n_tracks + role_assign): pick
    each half's anchor track by presence, and its partner by PROVEN COEXISTENCE — two tracks that
    exist at once are two people. The global top-N-by-count broke on Session2 Game 3: the far
    players fragmented so heavily that all four biggest tracks were near-half, and role_assign
    labelled two NEAR tracks as the far roles (identity corruption, not just lost coverage)."""

    SPF = 1 / 12

    @staticmethod
    def _t(f0, f1, u, v):
        return [(f, u, v) for f in range(f0, f1 + 1)]

    def test_doubles_two_primaries_per_half_even_when_near_dominates(self):
        # The Game-3 shape: near pair fragments into four big tracks (each bigger than any far
        # track); far players only have small fragments. All four near tracks outrank the far
        # ones globally — but each half must still get its own two primaries.
        by_tid = {
            1: self._t(0, 500, 0.30, 0.80),    # near pair, first period
            2: self._t(0, 500, 0.70, 0.85),
            3: self._t(600, 1000, 0.30, 0.82),  # near pair, second period (same people, new ids)
            4: self._t(600, 1000, 0.70, 0.80),
            5: self._t(0, 300, 0.40, 0.20),     # far pair — far smaller row counts
            6: self._t(0, 280, 0.60, 0.25),
        }
        roles = g.select_role_primaries(by_tid, n=4, seconds_per_frame=self.SPF)
        self.assertEqual(roles[1], "near_left")
        self.assertEqual(roles[2], "near_right")
        self.assertEqual(roles[5], "far_left")
        self.assertEqual(roles[6], "far_right")
        self.assertNotIn(3, roles)  # second-period near tracks stay fragments (stitchable)
        self.assertNotIn(4, roles)

    def test_partner_must_coexist_not_just_be_big(self):
        # Track 3 is bigger than track 2 but never coexists with track 1 — it's the SAME person
        # re-tracked, not the partner. The coexisting track 2 is the partner.
        by_tid = {
            1: self._t(0, 400, 0.30, 0.80),
            3: self._t(401, 900, 0.35, 0.82),   # bigger, but zero coexistence with 1
            2: self._t(100, 350, 0.70, 0.85),   # coexists 250 frames with 1
            5: self._t(0, 400, 0.40, 0.20),
            6: self._t(0, 400, 0.60, 0.25),
        }
        roles = g.select_role_primaries(by_tid, n=4, seconds_per_frame=self.SPF)
        self.assertEqual(roles[1], "near_left")
        self.assertEqual(roles[2], "near_right")
        self.assertNotIn(3, roles)

    def test_singles_one_per_half(self):
        by_tid = {
            1: self._t(0, 500, 0.50, 0.80),
            2: self._t(0, 500, 0.50, 0.20),
            9: self._t(0, 100, 0.90, 0.85),  # smaller near track: not a primary in singles
        }
        roles = g.select_role_primaries(by_tid, n=2, seconds_per_frame=self.SPF)
        self.assertEqual(roles, {1: "near", 2: "far"})

    def test_empty_half_yields_no_roles_for_it(self):
        by_tid = {1: self._t(0, 500, 0.30, 0.80), 2: self._t(0, 500, 0.70, 0.85)}
        roles = g.select_role_primaries(by_tid, n=4, seconds_per_frame=self.SPF)
        self.assertEqual(set(roles.values()), {"near_left", "near_right"})


class TestStitchFragmentsFramewise(unittest.TestCase):
    """Framewise re-ID (stitching v2, successor of stitch_role_fragments): fragments attach to the
    role whose per-frame HOLES they fill, gated by boundary continuity (velocity-bounded distance
    to the role's nearest anchors in meters) and by ambiguity rejection (two same-half roles nearly
    equally close → honest hole beats a partner swap). Frames @ ~12 fps (SPF = 1/12 s).

    Old-suite mapping: temporal-hole stitch → test_fragment_continues_the_role_with_the_hole;
    overlapping-role-not-claimed / blip-dropped → the hole test (test_no_new_frames_for_any_role…);
    other-half ignored → test_other_half_never_claimed; distant-fragment rejected →
    test_velocity_bound_rejects_teleport. NEW vs v1: span overlap no longer rejects
    (test_fragment_overlapping_primary_span…), chaining, ambiguity trap."""

    SPF = 1 / 12

    @staticmethod
    def _path(f0, f1, u, v):
        return [(f, u, v) for f in range(f0, f1 + 1)]

    def _roles(self, far_left_end=282, far_right_end=355):
        return {
            "near_left": self._path(0, 355, 0.30, 0.80),
            "near_right": self._path(0, 355, 0.70, 0.80),
            "far_left": self._path(0, far_left_end, 0.35, 0.15),
            "far_right": self._path(0, far_right_end, 0.75, 0.15),
        }

    def test_fragment_continues_the_role_with_the_hole(self):
        # far_left lost after 282; fragment resumes 14 cm away 4 frames later → stitched.
        frags = {64: self._path(286, 337, 0.36, 0.16)}
        out = g.stitch_fragments_framewise(self._roles(), frags, self.SPF)
        self.assertEqual(out, {64: "far_left"})

    def test_fragment_overlapping_primary_span_still_fills_its_hole_frames(self):
        # far_left detected 0-282 and 320-355 (span 0-355). Fragment 278-310 overlaps the span —
        # v1 rejected ANY span overlap — and briefly double-tracks frames 278-282 (15%, under the
        # collision threshold); frames 283-310 are real holes → must stitch.
        roles = self._roles()
        roles["far_left"] = self._path(0, 282, 0.35, 0.15) + self._path(320, 355, 0.37, 0.17)
        frags = {64: self._path(278, 310, 0.36, 0.16)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "far_left"})

    def test_no_new_frames_for_any_role_is_dropped(self):
        # a far blip while both far roles are fully present fills no hole → not stitched.
        frags = {64: self._path(286, 337, 0.70, 0.14)}
        out = g.stitch_fragments_framewise(self._roles(far_left_end=355), frags, self.SPF)
        self.assertEqual(out, {})

    def test_other_half_never_claimed(self):
        # near-half fragment (v>0.5) never attaches to a far role, whatever the holes.
        frags = {64: self._path(286, 337, 0.36, 0.80)}
        roles = self._roles()
        del roles["near_left"], roles["near_right"]  # only far roles offered
        self.assertEqual(g.stitch_fragments_framewise(roles, frags, self.SPF), {})

    def test_velocity_bound_rejects_teleport_when_both_roles_open(self):
        # Both far roles lost after 282; the fragment starts 4 m from far_left's anchor and 4.7 m
        # from far_right's, 4 frames later (allow = 1 + 6·(1/3) = 3 m): continuous with NEITHER,
        # could be either → honest drop. (With the partner accounted for instead, elimination
        # would accept — see test_long_fragment_accepted_by_elimination….)
        frags = {99: self._path(286, 337, 0.36, 0.45)}
        out = g.stitch_fragments_framewise(self._roles(far_right_end=282), frags, self.SPF)
        self.assertEqual(out, {})

    def test_corpse_anchor_in_the_wrong_half_is_not_veto_evidence(self):
        # A dying near-half track often drifts its box across the net before BoT-SORT kills it —
        # the role's last samples then sit in the FAR half. Such an anchor is self-evidently not
        # the player's position: discarded, near_right passes via its sane re-acquire anchor and
        # wins clearly over near_left. (Vetoing on the corpse would hand the fragment to the
        # WRONG partner — a swap, not just a hole.)
        roles = {
            "near_left": self._path(0, 282, 0.30, 0.80),
            "near_right": (self._path(0, 270, 0.70, 0.80)
                           + self._path(271, 282, 0.60, 0.38)     # box drifted across the net
                           + self._path(350, 355, 0.56, 0.76)),   # re-acquired
        }
        frags = {64: self._path(286, 340, 0.55, 0.75)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "near_right"})

    def test_end_anchor_must_also_be_continuous(self):
        # Start matches far_left, but 4 frames before far_left's re-acquire anchor the fragment
        # sits 4 m up-court (allow 3 m) — a different person crossed; reject.
        roles = self._roles()
        roles["far_left"] = self._path(0, 282, 0.35, 0.15) + self._path(312, 355, 0.35, 0.15)
        frags = {64: [(f, 0.36, 0.16 if f < 300 else 0.45) for f in range(286, 309)]}
        self.assertEqual(g.stitch_fragments_framewise(roles, frags, self.SPF), {})

    def test_partner_swap_trap_ambiguous_fragment_is_dropped(self):
        # Both far roles lost after 282; the fragment starts exactly between their last anchors
        # (0.305 m to each) → ambiguous, dropped: an honest hole beats a possible partner swap.
        roles = self._roles(far_left_end=282, far_right_end=282)
        roles["far_left"] = self._path(0, 282, 0.45, 0.15)
        roles["far_right"] = self._path(0, 282, 0.55, 0.15)
        frags = {64: self._path(286, 300, 0.50, 0.15)}
        self.assertEqual(g.stitch_fragments_framewise(roles, frags, self.SPF), {})

    def test_clearly_nearer_role_wins_when_both_pass(self):
        # Same double-hole, but the fragment is 0.12 m from far_left vs 0.79 m from far_right —
        # margin (0.5 m) exceeded → far_left wins.
        roles = self._roles(far_left_end=282, far_right_end=282)
        roles["far_left"] = self._path(0, 282, 0.35, 0.15)
        roles["far_right"] = self._path(0, 282, 0.48, 0.15)
        frags = {64: self._path(286, 300, 0.37, 0.15)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "far_left"})

    def test_chained_fragments_extend_the_role(self):
        # Both far roles lost after 282. Fragment A clearly continues far_left and drifts to
        # (0.40, 0.20). Fragment B starts at A's tail: with chaining its far_left distance is
        # 0.15 m (unambiguous win); without chaining it would be ~0.88 m vs far_right's ~1.17 m —
        # inside the ambiguity margin and wrongly dropped.
        roles = self._roles(far_left_end=282, far_right_end=282)
        roles["far_left"] = self._path(0, 282, 0.35, 0.15)
        roles["far_right"] = self._path(0, 282, 0.55, 0.15)
        a = [(f, 0.36 + 0.04 * (f - 286) / 24, 0.16 + 0.04 * (f - 286) / 24) for f in range(286, 311)]
        b = self._path(314, 340, 0.41, 0.21)
        out = g.stitch_fragments_framewise(roles, {64: a, 68: b}, self.SPF)
        self.assertEqual(out, {64: "far_left", 68: "far_left"})

    def test_collision_with_a_roles_frames_excludes_that_role(self):
        # Mutual exclusion: one person can't be two places. near_left is DETECTED for 64% of the
        # fragment's frames (286-320 of 286-340), so the fragment cannot be near_left even though
        # it still fills some near_left holes (321-340) and near_left's anchor is equally close.
        # Collision removes near_left from the candidates → no ambiguity drop → near_right.
        roles = {
            "near_left": self._path(0, 320, 0.45, 0.80),   # covers most of the fragment
            "near_right": self._path(0, 282, 0.55, 0.80),  # lost after 282
        }
        frags = {64: self._path(286, 340, 0.50, 0.80)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "near_right"})

    def test_small_collision_does_not_exclude(self):
        # BoT-SORT briefly double-tracks around an id switch: a few overlapping frames (< the
        # collision threshold) must not disqualify the role the fragment genuinely continues.
        roles = self._roles()
        roles["far_left"] = self._path(0, 290, 0.35, 0.15)  # 5 frames overlap with the fragment
        frags = {64: self._path(286, 340, 0.36, 0.16)}      # 55 frames, ~9% collision
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "far_left"})

    def test_long_fragment_accepted_by_elimination_despite_boundary_glitch(self):
        # near_right is detected during 100% of the (long) fragment → the fragment must be
        # near_left by elimination (only two near players exist). BoT-SORT id switches often drift
        # the box in the handover frames, so the 1-frame-gap boundary check fails (1.67 m > 1.5 m
        # allowance) — elimination evidence must outrank that veto.
        roles = {
            "near_left": self._path(0, 99, 0.30, 0.80),
            "near_right": self._path(0, 550, 0.65, 0.80),
        }
        frags = {64: self._path(100, 500, 0.55, 0.85)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "near_left"})

    def test_elimination_accepts_on_sustained_partner_coexistence(self):
        # near_left coexists with the fragment for 151 frames (~12.6 s) — 25% of the fragment,
        # but mutual exclusion is about absolute coexistence, not ratio: 12.6 s of simultaneous
        # tracks proves the fragment is NOT near_left. near_right's only anchor is a corpse
        # (discarded) → no continuity evidence either way → elimination must still accept.
        roles = {
            "near_left": self._path(200, 350, 0.40, 0.85),
            "near_right": self._path(0, 100, 0.60, 0.38),  # died drifting across the net
        }
        frags = {64: self._path(200, 800, 0.60, 0.80)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "near_right"})

    def test_boundary_check_uses_median_not_single_glitch_sample(self):
        # The fragment's very first box is a glitch 1.8 m off; the median of its first samples
        # sits right at near_left's tail. near_right is excluded by collision (21%) but not
        # "accounted for" (<50%), so elimination doesn't apply — the median boundary must pass.
        roles = {
            "near_left": self._path(0, 99, 0.30, 0.80),
            "near_right": self._path(0, 120, 0.75, 0.80),
        }
        frag = [(100, 0.60, 0.80)] + self._path(101, 200, 0.32, 0.80)
        out = g.stitch_fragments_framewise(roles, {64: frag}, self.SPF)
        self.assertEqual(out, {64: "near_left"})

    def test_ambiguous_fragment_resolves_after_partner_timeline_fills_in(self):
        # Pass 1: X (100-300) is ambiguous between the stale near anchors → not attached yet.
        # Y (250-400) clearly continues near_left. Pass 2: X now coexists with near_left's
        # extended coverage (frames 250-300, ~4 s) → excluded there → near_right by elimination.
        # A single in-order pass would drop X forever; the fixpoint loop must recover it.
        roles = {
            "near_left": self._path(0, 99, 0.35, 0.85),
            "near_right": self._path(0, 99, 0.65, 0.85),
        }
        x = self._path(100, 300, 0.50, 0.85)   # dead-center: ambiguous vs both anchors
        y = self._path(250, 400, 0.35, 0.85)   # unambiguously near_left
        out = g.stitch_fragments_framewise(roles, {64: x, 68: y}, self.SPF)
        self.assertEqual(out, {68: "near_left", 64: "near_right"})

    def test_fragment_before_the_primary_uses_the_after_anchor(self):
        # Role acquired late (frames 100+); a fragment covering the start continues INTO it.
        roles = self._roles()
        roles["far_left"] = self._path(100, 355, 0.35, 0.15)
        frags = {64: self._path(0, 80, 0.36, 0.16)}
        out = g.stitch_fragments_framewise(roles, frags, self.SPF)
        self.assertEqual(out, {64: "far_left"})


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

    def test_wider_window_damps_jitter_more(self):
        # zig-zag jitter around a fixed point: a wider window must reduce the residual amplitude more.
        track = [(i, 0.5 + (0.1 if i % 2 == 0 else -0.1), 0.5) for i in range(20)]
        amp = lambda out: max(abs(s[1] - 0.5) for s in out[2:-2])  # ignore edge windows
        self.assertLess(amp(g.smooth_track(track, window=9)), amp(g.smooth_track(track, window=5)))

    def test_wider_window_preserves_monotone_trend(self):
        # a steady ramp (genuine motion) must still be monotonically increasing after window=9.
        track = [(i, 0.02 * i, 0.5) for i in range(20)]
        us = [s[1] for s in g.smooth_track(track, window=9)]
        self.assertTrue(all(b >= a for a, b in zip(us, us[1:])))
        self.assertGreater(us[-1], us[0])


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

    def test_near_edge_margins_lower_and_widen_near_corners(self):
        # down=0.1, out=0.05 on height 200 / width 240 -> near corners drop 20px and spread 12px each
        # (NR right + down, NL left + down); the far edge is untouched by down/out.
        poly = g.court_airspace_polygon(self.QUAD, 1000, 1000, up=0.5, side=0.1, down=0.1, out=0.05)
        self.assertEqual(poly, [(76.0, 100.0), (324.0, 100.0), (332.0, 420.0), (68.0, 420.0)])

    def test_down_out_zero_matches_legacy_output(self):
        # Backward-compat lock: the new params default to 0.0 and reproduce the old near corners verbatim.
        legacy = g.court_airspace_polygon(self.QUAD, 1000, 1000, up=0.5, side=0.1)
        explicit = g.court_airspace_polygon(self.QUAD, 1000, 1000, up=0.5, side=0.1, down=0.0, out=0.0)
        self.assertEqual(explicit, legacy)
        self.assertEqual(explicit[2:], [(320.0, 400.0), (80.0, 400.0)])  # near corners unchanged

    def test_near_margins_clamp_to_frame(self):
        # Frame 400x410: out 0.5*240=120 pushes NR to x=440 (clamp 400) and NL to x=-40 (clamp 0);
        # down 0.1*200=20 pushes both near corners to y=420 (clamp 410).
        poly = g.court_airspace_polygon(self.QUAD, 400, 410, up=0.0, side=0.0, down=0.1, out=0.5)
        self.assertEqual(poly[2], (400.0, 410.0))   # near-right clamped to bottom-right corner
        self.assertEqual(poly[3], (0.0, 410.0))     # near-left clamped to bottom-left corner


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
