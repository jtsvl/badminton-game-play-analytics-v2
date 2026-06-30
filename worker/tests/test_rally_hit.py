"""rally_hit stage glue — the one piece of real logic is _player_energy (the serve-cue motion series).

It turns the player tracks parquet into the [(t_s, energy_kmh)] series find_serve_frame consumes, reusing
insights.step_speeds_kmh and aligning each speed to its segment's end time. The rest of run() is parquet +
DB I/O, exercised by the standalone stage run on game1 (not here).
"""
import unittest

import polars as pl

from worker.pipeline.rally_hit import _player_energy

_SCHEMA = {"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64, "v": pl.Float64,
           "role": pl.Utf8, "source": pl.Utf8}


def _tracks(rows):
    return pl.DataFrame(rows, schema=_SCHEMA)


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


if __name__ == "__main__":
    unittest.main()
