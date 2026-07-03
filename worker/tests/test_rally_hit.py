"""rally_hit stage glue — the one piece of real logic is _player_energy (the serve-cue motion series).

It turns the player tracks parquet into the [(t_s, energy_kmh)] series find_serve_frame consumes, reusing
insights.step_speeds_kmh and aligning each speed to its segment's end time. The rest of run() is parquet +
DB I/O, exercised by the standalone stage run on game1 (not here).
"""
import unittest

import polars as pl

from worker.geometry import HEAD_KP, HIP_KP, WRIST_KP
from worker.pipeline import rally_hit
from worker.pipeline.rally_hit import _player_energy

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
