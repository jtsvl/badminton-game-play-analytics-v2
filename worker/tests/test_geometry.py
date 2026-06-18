"""Tests for the pure geometry helpers (worker/geometry.py).

Pure math only — no numpy/cv2/torch — so these run under the dev extra without the ml stack.
Expected values are hand-computed from clean affine cases.
"""
import math
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


if __name__ == "__main__":
    unittest.main()
