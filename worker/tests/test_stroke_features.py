"""stroke.features — pure stage-7 sample builders (train == inference by construction).

REVISED design (spec 2026-07-05): 16-frame windows at the 12 fps analysis cadence and 4 fixed
quadrant slots (near_left, near_right, far_left, far_right) — no hitter-picking. Slots fill from
time-aware role tracks, then pose recovery (feet projected through the calibration homography)
for whatever the court tracks lost. The trajectory-flip half signal and the nearest-per-half
selector were measured dead at the Phase-1 gate (48 % / coverage 55 %) and are gone.
All inputs are plain rows/dicts — no I/O here; the exporter CLI is smoke-tested on real footage.
"""
import unittest

from worker.stroke.features import (SLOTS, anchor_frame, assign_slots, build_sample,
                                    half_of_role, normalize_joints, position_stream,
                                    quadrant, recover_slots, shuttle_stream, slot_of_role,
                                    window_frames)

FRAMES = [{"frame": i, "t": i / 12.0} for i in range(240)]  # 12 fps analysis cadence


class TestAnchorFrame(unittest.TestCase):
    def test_picks_nearest_frame_to_t_s(self):
        self.assertEqual(anchor_frame(FRAMES, 1.0), 12)

    def test_rounds_to_nearest_not_floor(self):
        # 1.06 s sits between frame 12 (1.000 s) and 13 (1.083 s) — closer to 13
        self.assertEqual(anchor_frame(FRAMES, 1.06), 13)

    def test_empty_manifest_returns_none(self):
        self.assertIsNone(anchor_frame([], 1.0))


class TestWindowFrames(unittest.TestCase):
    def test_sixteen_frames_centered_on_anchor(self):
        w = window_frames(100)
        self.assertEqual(len(w), 16)
        self.assertEqual(w[0], 92)     # anchor − 8
        self.assertEqual(w[-1], 107)   # anchor + 7
        self.assertIn(100, w)


class TestShuttleStream(unittest.TestCase):
    """(px/W, py/H) per window frame; gaps ≤ max_gap interpolated within an arc, NEVER across
    the anchor (the racket contact breaks the arc); everything else honest zeros."""

    W, H = 1920, 1080

    def _stream(self, dets, anchor=100):
        return shuttle_stream(dets, window_frames(anchor), anchor, self.W, self.H)

    def test_detected_frames_are_resolution_normalized(self):
        s = self._stream({100: (960.0, 540.0)})
        self.assertEqual(len(s), 16)
        self.assertEqual(s[8], [0.5, 0.5])  # window index of the anchor frame

    def test_undetected_frames_are_zero(self):
        s = self._stream({100: (960.0, 540.0)})
        self.assertEqual(s[0], [0.0, 0.0])

    def test_small_gap_interpolated_linearly(self):
        # dets at 102 and 106 (gap of 3 missing frames) after the anchor — same arc
        s = self._stream({102: (192.0, 108.0), 106: (960.0, 540.0)})
        self.assertAlmostEqual(s[12][0], (0.1 + 0.5) / 2)  # frame 104, halfway
        self.assertAlmostEqual(s[12][1], (0.1 + 0.5) / 2)

    def test_gap_across_anchor_not_interpolated(self):
        # dets at 98 and 102 bracket the anchor (100): contact breaks the arc → zeros between
        s = self._stream({98: (192.0, 108.0), 102: (960.0, 540.0)})
        self.assertEqual(s[8], [0.0, 0.0])   # frame 100
        self.assertEqual(s[7], [0.0, 0.0])   # frame 99

    def test_detected_anchor_as_gap_endpoint_is_within_one_arc(self):
        # anchor itself detected at 100, next det at 104 — 101–103 lie inside the OUTGOING
        # arc (contact is the endpoint, not inside the gap) → interpolate
        s = self._stream({100: (192.0, 108.0), 104: (960.0, 540.0)})
        self.assertAlmostEqual(s[10][0], (0.1 + 0.5) / 2)  # frame 102, halfway

    def test_gap_wider_than_max_gap_not_interpolated(self):
        # dets at 93 and 102: 8 missing frames > max_gap 6 → zeros between
        s = self._stream({93: (192.0, 108.0), 102: (960.0, 540.0)})
        self.assertEqual(s[5], [0.0, 0.0])   # frame 97


def _body26(pts17, scores17):
    """Pad a 17-joint COCO spec out to body26 rows (the 9 extras low-confidence)."""
    kp = [list(p) for p in pts17] + [[999.0, 999.0]] * 9
    score = list(scores17) + [0.9] * 9
    return kp, score


class TestNormalizeJoints(unittest.TestCase):
    """bbox-relative, center-aligned, diagonal-normalized COCO-17 joints (BST convention);
    the bbox envelope comes from confident joints only, and body26 extras never leak in."""

    def test_centered_and_diagonal_scaled(self):
        pts = [[0.0, 0.0], [3.0, 4.0]] + [[1.5, 2.0]] * 15   # bbox diag = 5, center (1.5, 2)
        kp, score = _body26(pts, [0.9, 0.9] + [0.9] * 15)
        j = normalize_joints(kp, score)
        self.assertEqual(len(j), 17)
        self.assertAlmostEqual(j[0][0], -0.3)
        self.assertAlmostEqual(j[0][1], -0.4)
        self.assertAlmostEqual(j[1][0], 0.3)
        self.assertAlmostEqual(j[1][1], 0.4)

    def test_body26_extras_do_not_affect_bbox(self):
        pts = [[0.0, 0.0], [3.0, 4.0]] + [[1.5, 2.0]] * 15
        kp, score = _body26(pts, [0.9] * 17)
        self.assertAlmostEqual(normalize_joints(kp, score)[0][0], -0.3)

    def test_low_confidence_joint_excluded_from_bbox_but_still_normalized(self):
        pts = [[0.0, 0.0], [3.0, 4.0], [30.0, 40.0]] + [[1.5, 2.0]] * 14
        kp, score = _body26(pts, [0.9, 0.9, 0.1] + [0.9] * 14)
        j = normalize_joints(kp, score)
        self.assertAlmostEqual(j[0][0], -0.3)                 # bbox unchanged
        self.assertAlmostEqual(j[2][0], (30.0 - 1.5) / 5.0)   # still normalized

    def test_fewer_than_three_confident_joints_is_zeros(self):
        pts = [[0.0, 0.0], [3.0, 4.0]] + [[1.5, 2.0]] * 15
        kp, score = _body26(pts, [0.9, 0.9] + [0.1] * 15)
        self.assertEqual(normalize_joints(kp, score), [[0.0, 0.0]] * 17)

    def test_zero_size_bbox_is_zeros(self):
        pts = [[2.0, 2.0]] * 17
        kp, score = _body26(pts, [0.9] * 17)
        self.assertEqual(normalize_joints(kp, score), [[0.0, 0.0]] * 17)


class TestHalfOfRole(unittest.TestCase):
    def test_maps_singles_and_doubles_roles(self):
        self.assertEqual(half_of_role("near"), "near")
        self.assertEqual(half_of_role("near_left"), "near")
        self.assertEqual(half_of_role("far_right"), "far")
        self.assertIsNone(half_of_role("bench"))
        self.assertIsNone(half_of_role(None))


class TestQuadrantAndSlots(unittest.TestCase):
    def test_quadrant_absolute_rule(self):
        # near = larger court v (geometry.role_assign sorts v desc for near); left = smaller u
        self.assertEqual(quadrant(0.2, 0.8), "near_left")
        self.assertEqual(quadrant(0.9, 0.7), "near_right")
        self.assertEqual(quadrant(0.1, 0.2), "far_left")
        self.assertEqual(quadrant(0.8, 0.3), "far_right")

    def test_slot_of_role_maps_singles_to_left_slots(self):
        self.assertEqual(slot_of_role("near"), "near_left")
        self.assertEqual(slot_of_role("far"), "far_left")
        self.assertEqual(slot_of_role("far_right"), "far_right")
        self.assertIsNone(slot_of_role("bench"))
        self.assertIsNone(slot_of_role(None))


def _trow(t, tid, role, u=0.5, v=0.5):
    return {"t": t, "track_id": tid, "u": u, "v": v, "role": role, "source": "detected"}


class TestAssignSlots(unittest.TestCase):
    """Time-aware: only role segments with rows near the anchor time fill a slot — track ids
    churn (33 segments for 4 players in game 1), so whole-video role lookups are wrong."""

    def test_active_segment_fills_slot(self):
        rows = [_trow(9.9, 47, "near_left"), _trow(10.1, 47, "near_left"),
                _trow(10.0, 1, "near_right")]
        slots = assign_slots(rows, 10.0)
        self.assertEqual(slots["near_left"], 47)
        self.assertEqual(slots["near_right"], 1)
        self.assertIsNone(slots["far_left"])

    def test_stale_segment_with_same_role_is_ignored(self):
        rows = [_trow(3.0, 3, "near_left"),                     # early segment, long gone
                _trow(9.9, 330, "near_left"), _trow(10.05, 330, "near_left")]
        self.assertEqual(assign_slots(rows, 10.0)["near_left"], 330)

    def test_no_active_segment_leaves_slot_empty(self):
        rows = [_trow(3.0, 3, "near_left")]
        self.assertIsNone(assign_slots(rows, 10.0)["near_left"])

    def test_singles_roles_land_in_left_slots(self):
        rows = [_trow(10.0, 1, "near"), _trow(10.0, 2, "far")]
        slots = assign_slots(rows, 10.0)
        self.assertEqual(slots["near_left"], 1)
        self.assertEqual(slots["far_left"], 2)


IDENTITY_H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # image px == court (u,v) in tests


def _prow(frame, track, x, y, foot_u=None, foot_v=None, foot_conf=0.9):
    """Pose row with confident joints around (x, y) and a foot point in 'court' coords
    (identity homography makes foot pixels == court units in tests)."""
    kp = [[x + dx, y] for dx in (-1.0, 0.0, 1.0)] + [[x, y]] * 23
    return {"frame": frame, "track": track, "kp": kp, "kp_score": [0.9] * 26,
            "foot_x": foot_u if foot_u is not None else x,
            "foot_y": foot_v if foot_v is not None else y,
            "foot_conf": foot_conf}


class TestRecoverSlots(unittest.TestCase):
    """Pose recovery: unassigned pose tracks whose feet project into the court fill empty
    slots by quadrant — this is where the lost far pair comes back."""

    def test_in_court_pose_track_fills_its_quadrant(self):
        rows = [_prow(f, 26, 100.0, 100.0, foot_u=0.7, foot_v=0.2) for f in range(96, 105)]
        rec = recover_slots(rows, taken={}, homography=IDENTITY_H,
                            empty_slots=["far_right"])
        self.assertEqual(rec["far_right"]["track"], 26)
        self.assertAlmostEqual(rec["far_right"]["uv"][100][0], 0.7)

    def test_off_court_bystander_is_ignored(self):
        rows = [_prow(f, 99, 100.0, 100.0, foot_u=1.8, foot_v=0.2) for f in range(96, 105)]
        rec = recover_slots(rows, taken={}, homography=IDENTITY_H, empty_slots=["far_right"])
        self.assertNotIn("far_right", rec)

    def test_track_already_taken_by_a_slot_is_excluded(self):
        rows = [_prow(f, 47, 100.0, 100.0, foot_u=0.7, foot_v=0.2) for f in range(96, 105)]
        rec = recover_slots(rows, taken={"near_left": 47}, homography=IDENTITY_H,
                            empty_slots=["far_right"])
        self.assertNotIn("far_right", rec)

    def test_most_present_candidate_wins_the_slot(self):
        rows = ([_prow(f, 26, 100.0, 100.0, foot_u=0.7, foot_v=0.2) for f in range(96, 105)]
                + [_prow(f, 27, 100.0, 100.0, foot_u=0.8, foot_v=0.3) for f in range(99, 102)])
        rec = recover_slots(rows, taken={}, homography=IDENTITY_H, empty_slots=["far_right"])
        self.assertEqual(rec["far_right"]["track"], 26)

    def test_low_confidence_feet_do_not_recover(self):
        rows = [_prow(f, 26, 100.0, 100.0, foot_u=0.7, foot_v=0.2, foot_conf=0.1)
                for f in range(96, 105)]
        rec = recover_slots(rows, taken={}, homography=IDENTITY_H, empty_slots=["far_right"])
        self.assertNotIn("far_right", rec)


class TestPositionStream(unittest.TestCase):
    def test_nearest_t_join_within_tolerance(self):
        samples = [(1.0, 0.2, 0.3), (1.5, 0.4, 0.5)]
        s = position_stream(samples, [1.01, 1.49])
        self.assertEqual(s, [[0.2, 0.3], [0.4, 0.5]])

    def test_no_sample_within_tolerance_is_zero(self):
        s = position_stream([(1.0, 0.2, 0.3)], [2.0])
        self.assertEqual(s, [[0.0, 0.0]])


class TestBuildSample(unittest.TestCase):
    """The assembled sample: per-slot joint/position streams, shuttle stream, honest meta."""

    W, H = 3840.0, 2160.0

    def _fixture(self):
        frames = [{"frame": i, "t": i / 12.0} for i in range(240)]
        t_s = 100 / 12.0  # anchor lands exactly on frame 100
        dets = {f: (900.0, 500.0 + 25.0 * (f - 95)) for f in range(93, 108)}
        # near_left as a role track; far_right only recoverable from pose
        pose_rows = ([_prow(f, 47, 900.0, 1800.0, foot_u=0.3, foot_v=0.8) for f in range(90, 111)]
                     + [_prow(f, 26, 900.0, 400.0, foot_u=0.7, foot_v=0.2) for f in range(90, 111)])
        track_rows = [_trow(f / 12.0, 47, "near_left", u=0.3, v=0.8) for f in range(90, 111)]
        return pose_rows, track_rows, dets, frames, t_s

    def test_slots_fill_from_tracks_and_recovery(self):
        pose_rows, track_rows, dets, frames, t_s = self._fixture()
        s = build_sample(pose_rows, track_rows, dets, frames, t_s, self.W, self.H,
                         homography=IDENTITY_H)
        self.assertEqual(len(s["joints"]), 4)                 # one per slot, SLOTS order
        self.assertEqual(len(s["joints"][0]), 16)
        self.assertEqual(len(s["joints"][0][0]), 17)
        self.assertEqual(len(s["shuttle"]), 16)
        i_nl, i_fr = SLOTS.index("near_left"), SLOTS.index("far_right")
        self.assertEqual(s["meta"]["track_near_left"], 47)
        self.assertEqual(s["meta"]["source_near_left"], "track")
        self.assertEqual(s["meta"]["track_far_right"], 26)
        self.assertEqual(s["meta"]["source_far_right"], "recovery")
        self.assertIsNone(s["meta"]["track_near_right"])
        # near_left positions from the role track; far_right from projected feet
        self.assertEqual(s["positions"][i_nl][8], [0.3, 0.8])
        self.assertEqual(s["positions"][i_fr][8], [0.7, 0.2])
        # joints of filled slots are real at the anchor
        self.assertNotEqual(s["joints"][i_nl][8], [[0.0, 0.0]] * 17)
        self.assertNotEqual(s["joints"][i_fr][8], [[0.0, 0.0]] * 17)
        self.assertEqual(s["meta"]["anchor"], 100)
        self.assertEqual(s["meta"]["pad"], 0)
        self.assertGreater(s["meta"]["shuttle_coverage"], 0.9)

    def test_no_homography_skips_recovery_but_keeps_tracks(self):
        pose_rows, track_rows, dets, frames, t_s = self._fixture()
        s = build_sample(pose_rows, track_rows, dets, frames, t_s, self.W, self.H)
        self.assertEqual(s["meta"]["track_near_left"], 47)
        self.assertIsNone(s["meta"]["track_far_right"])

    def test_empty_manifest_returns_none(self):
        pose_rows, track_rows, dets, _, t_s = self._fixture()
        self.assertIsNone(build_sample(pose_rows, track_rows, dets, [], t_s, self.W, self.H))

    def test_missing_everything_degrades_to_zeros(self):
        _, _, dets, frames, t_s = self._fixture()
        s = build_sample([], [], dets, frames, t_s, self.W, self.H, homography=IDENTITY_H)
        self.assertEqual(s["joints"][0][8], [[0.0, 0.0]] * 17)
        self.assertIsNone(s["meta"]["track_near_left"])
        self.assertEqual(s["positions"][1][0], [0.0, 0.0])

    def test_window_at_clip_edge_reports_pad(self):
        pose_rows, track_rows, dets, frames, _ = self._fixture()
        s = build_sample(pose_rows, track_rows, dets, frames[:105], 103 / 12.0, self.W, self.H)
        self.assertEqual(s["meta"]["pad"], 6)  # frames 105–110 missing from the manifest


if __name__ == "__main__":
    unittest.main()
