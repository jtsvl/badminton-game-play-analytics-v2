"""rally_hit stage glue — _player_energy, the winner_side estimate from out landings, and swings.

_player_energy turns the player tracks parquet into the [(t_s, energy_kmh)] series find_serve_frame
consumes. _side_by_role/_apply_out_winners map an out landing's court half to a match side via the
track_label↔role binding — an ESTIMATE decoupled from the human-entered matches.score.
_swings_from_frames turns cached pose/tracks parquets into the swing-activity events the T8 gap
bridge consumes. The rest of run() is parquet + DB I/O, exercised by the standalone stage run on
game1 (not here).
"""
import unittest

import polars as pl

from worker.geometry import HEAD_KP, HIP_KP, WRIST_KP
from worker.pipeline import rally_hit
from worker.pipeline.rally_hit import _apply_out_winners, _player_energy, _side_by_role

_SCHEMA = {"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64, "v": pl.Float64,
           "role": pl.Utf8, "source": pl.Utf8}
_POSE_SCHEMA = {"frame": pl.Int64, "track": pl.Int64,
               "kp": pl.List(pl.List(pl.Float64)), "kp_score": pl.List(pl.Float64)}


def _tracks(rows):
    return pl.DataFrame(rows, schema=_SCHEMA)


def _pose_row(frame, track, wrist_x):
    """One Halpe-26 pose row: right wrist at (wrist_x, 0), unit torso (head→hip = 1.0)."""
    kp = [[0.0, 0.0] for _ in range(26)]
    score = [0.0] * 26
    kp[HEAD_KP], kp[HIP_KP] = [0.0, 0.0], [0.0, 1.0]
    score[HEAD_KP] = score[HIP_KP] = 0.9
    kp[WRIST_KP[1]] = [wrist_x, 0.0]
    score[WRIST_KP[1]] = 0.9
    return {"frame": frame, "track": track, "kp": kp, "kp_score": score}


class TestPlayerEnergy(unittest.TestCase):
    def test_constant_speed_pairs_speed_with_segment_end_time(self):
        df = _tracks([{"t": 0.0, "track_id": 1, "u": 0.0, "v": 0.0, "role": "a", "source": "d"},
                      {"t": 1.0, "track_id": 1, "u": 0.0, "v": 1.0, "role": "a", "source": "d"}])
        e = _player_energy(df)
        self.assertEqual(len(e), 1)
        self.assertAlmostEqual(e[0][0], 1.0)               # end time of the segment
        self.assertAlmostEqual(e[0][1], 13.40 * 3.6, places=1)   # full court length covered in 1s

    def test_merges_players_and_sorts_by_time(self):
        df = _tracks([
            {"t": 0.0, "track_id": 1, "u": 0.0, "v": 0.0, "role": "a", "source": "d"},
            {"t": 2.0, "track_id": 1, "u": 0.0, "v": 0.0, "role": "a", "source": "d"},   # still
            {"t": 0.0, "track_id": 2, "u": 0.0, "v": 0.0, "role": "b", "source": "d"},
            {"t": 1.0, "track_id": 2, "u": 1.0, "v": 0.0, "role": "b", "source": "d"},   # moving
        ])
        e = _player_energy(df)
        self.assertEqual([round(t, 1) for t, _ in e], [1.0, 2.0])   # merged + time-sorted


def _row(t, tid, role):
    return {"t": t, "track_id": tid, "u": 0.5, "v": 0.5, "role": role, "source": "d"}


def _landing(rally_idx, call="out", located=True, v=0.95):
    payload = {"located": located, "call": call, "band_cm": 45.0}
    if located:
        payload.update({"u": 0.5, "v": v})
    return {"rally_idx": rally_idx, "t_s": 1.0, "type": "landing", "payload": payload,
            "confidence": 0.5}


def _rally(idx):
    return {"idx": idx, "t_start_s": 0.0, "t_end_s": 1.0, "shot_count": 3,
            "winner_side": None, "end_shot_type": None, "confidence": 0.8}


class TestSideByRole(unittest.TestCase):
    def test_singles_roles_map_to_sides(self):
        df = _tracks([_row(0.0, 1, "near"), _row(0.0, 2, "far")])
        mps = [{"track_label": 1, "side": "A"}, {"track_label": 2, "side": "B"}]
        self.assertEqual(_side_by_role(df, mps), {"near": "A", "far": "B"})

    def test_doubles_quadrant_roles_collapse_to_halves(self):
        df = _tracks([_row(0.0, 1, "near_left"), _row(0.0, 2, "near_right"),
                      _row(0.0, 3, "far_left"), _row(0.0, 4, "far_right")])
        mps = [{"track_label": 1, "side": "A"}, {"track_label": 2, "side": "A"},
               {"track_label": 3, "side": "B"}, {"track_label": 4, "side": "B"}]
        self.assertEqual(_side_by_role(df, mps), {"near": "A", "far": "B"})

    def test_conflicting_half_binding_is_dropped_not_guessed(self):
        # two tracks of the same half bound to different sides (a broken remap) → skip that half
        df = _tracks([_row(0.0, 1, "near_left"), _row(0.0, 2, "near_right")])
        mps = [{"track_label": 1, "side": "A"}, {"track_label": 2, "side": "B"}]
        self.assertEqual(_side_by_role(df, mps), {})

    def test_unset_track_labels_yield_empty_mapping(self):
        df = _tracks([_row(0.0, 1, "near")])
        self.assertEqual(_side_by_role(df, [{"track_label": None, "side": "A"}]), {})


class TestApplyOutWinners(unittest.TestCase):
    SIDES = {"near": "A", "far": "B"}

    def test_out_landing_on_near_half_awards_near_side(self):
        rallies = [_rally(0)]
        _apply_out_winners(rallies, [_landing(0, v=0.95)], self.SIDES)
        self.assertEqual(rallies[0]["winner_side"], "A")   # far player hit out → near side wins

    def test_out_landing_on_far_half_awards_far_side(self):
        rallies = [_rally(0)]
        _apply_out_winners(rallies, [_landing(0, v=0.05)], self.SIDES)
        self.assertEqual(rallies[0]["winner_side"], "B")

    def test_non_out_and_degraded_landings_leave_none(self):
        rallies = [_rally(0), _rally(1), _rally(2)]
        events = [_landing(0, call="in"), _landing(1, call="uncertain"),
                  _landing(2, located=False)]
        _apply_out_winners(rallies, events, self.SIDES)
        self.assertTrue(all(r["winner_side"] is None for r in rallies))

    def test_near_net_landing_is_ambiguous(self):
        rallies = [_rally(0)]
        _apply_out_winners(rallies, [_landing(0, v=0.52)], self.SIDES)   # may not have crossed
        self.assertIsNone(rallies[0]["winner_side"])

    def test_missing_side_mapping_means_no_attribution(self):
        rallies = [_rally(0)]
        _apply_out_winners(rallies, [_landing(0, v=0.95)], {})
        self.assertIsNone(rallies[0]["winner_side"])


class TestSwingsFromFrames(unittest.TestCase):
    def test_fast_wrist_motion_becomes_a_swing_event_for_a_kept_track(self):
        # 12fps spacing; wrist steps 0.5 torso-lengths in one frame = 6 tl/s.
        pose_df = pl.DataFrame([_pose_row(0, 7, 0.0), _pose_row(1, 7, 0.5)], schema=_POSE_SCHEMA)
        tracks_df = _tracks([{"t": 0.0, "track_id": 7, "u": 0.0, "v": 0.0,
                              "role": "near", "source": "d"}])
        frame_t = {0: 0.0, 1: 1 / 12}
        events = rally_hit._swings_from_frames(pose_df, tracks_df, frame_t)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["track"], 7)
        self.assertAlmostEqual(events[0]["strength"], 6.0, places=3)

    def test_pose_rows_for_untracked_ids_are_ignored(self):
        # track 99 never appears in tracks_df (not a gate_track-kept player) — its rows are dropped.
        pose_df = pl.DataFrame([_pose_row(0, 99, 0.0), _pose_row(1, 99, 0.5)], schema=_POSE_SCHEMA)
        tracks_df = _tracks([{"t": 0.0, "track_id": 7, "u": 0.0, "v": 0.0,
                              "role": "near", "source": "d"}])
        frame_t = {0: 0.0, 1: 1 / 12}
        self.assertEqual(rally_hit._swings_from_frames(pose_df, tracks_df, frame_t), [])


if __name__ == "__main__":
    unittest.main()
