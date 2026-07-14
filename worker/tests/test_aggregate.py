"""Tests for the pure aggregation wiring (worker/pipeline/aggregate.compute_player_stats).

Guards the insights.zones() depth-orientation fix: raw v is NOT distance-from-net (v=0 is the far
baseline, net is at v=0.5), so aggregate must apply geometry.net_depth before zones(). Also pins
the identity join in run(): stats consume the WHOLE role timeline (re-ID fragment rows included),
not just the single labeled track_id.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from worker import insights as ins
from worker.pipeline.aggregate import compute_player_stats, out_error_counts, run as aggregate_run

class _FakeStore:
    """Just enough store for aggregate.run's player-stats half (doubles skips the landing part)."""

    def __init__(self, match_players):
        self.match_players = match_players
        self.stats_rows = []

    def get_video(self, video_id):
        return {"duration_s": 60.0, "match_id": "m-1"}

    def get_match(self, match_id):
        return {"discipline": "doubles"}

    def get_match_players(self, match_id):
        return self.match_players

    def upsert_player_match_stats(self, row):
        self.stats_rows.append(row)


class TestRunIdentityJoin(unittest.TestCase):
    def test_reid_fragment_rows_flow_into_player_stats(self):
        # Player labeled track 1; frames after t=1 belong to re-ID fragment 42. Each step is one
        # metre of lateral movement (Δu = 1/COURT_W_M): label-only would see 1 m, the role
        # timeline sees 3 m.
        import polars as pl

        du = 1.0 / ins.COURT_W_M
        rows = [(0.0, 1, "detected"), (1.0, 1, "detected"), (2.0, 42, "reid"), (3.0, 42, "reid")]
        df = pl.DataFrame(
            {
                "t": [r[0] for r in rows],
                "track_id": [r[1] for r in rows],
                "u": [0.2 + i * du for i in range(len(rows))],
                "v": [0.75] * len(rows),
                "role": ["near_left"] * len(rows),
                "source": [r[2] for r in rows],
            }
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tracks_path = Path(tmp.name) / "tracks.parquet"
        df.write_parquet(tracks_path)

        store = _FakeStore([{"id": "mp-1", "player_id": "p-1", "side": "A", "slot": 0, "track_label": 1}])
        ctx = {
            "store": store,
            "cfg": SimpleNamespace(frame_cache_dir=tmp.name),  # no rallies.parquet → whole clip
            "job": {"video_id": "v-1", "match_id": "m-1"},
            "tracks_path": str(tracks_path),
        }
        aggregate_run(ctx)

        self.assertEqual(len(store.stats_rows), 1)
        row = store.stats_rows[0]
        self.assertEqual(row["player_id"], "p-1")
        self.assertEqual(row["distance_m"], 3)  # all 4 samples, not just the labeled track's 2


# near player camped deep (own baseline, v≈0.9) — must read as REAR, not front.
NEAR_REAR = [(0.0, 0.5, 0.90), (1.0, 0.55, 0.92), (2.0, 0.48, 0.88), (3.0, 0.51, 0.91)]
# near player at the net (v≈0.55) — must read as FRONT.
NEAR_NET = [(0.0, 0.5, 0.55), (1.0, 0.52, 0.56), (2.0, 0.49, 0.54)]
# far player deep (own baseline, v≈0.1) — must read as REAR (raw v would wrongly say front).
FAR_DEEP = [(0.0, 0.5, 0.10), (1.0, 0.52, 0.08), (2.0, 0.48, 0.12)]


class TestComputePlayerStats(unittest.TestCase):
    def test_distance_and_speed_wire_to_insights(self):
        st = compute_player_stats(NEAR_REAR, "near", duration_min=1)
        self.assertEqual(st["distance_m"], round(ins.total_distance_m(NEAR_REAR)))
        speeds = ins.step_speeds_kmh(NEAR_REAR)
        avg, top = ins.avg_top_speed_kmh(speeds)
        self.assertEqual(st["avg_kmh"], avg)
        self.assertEqual(st["top_kmh"], top)
        self.assertEqual(st["sprints"], ins.sprint_count(speeds))

    def test_near_player_deep_reads_as_rear(self):
        st = compute_player_stats(NEAR_REAR, "near", duration_min=1)
        self.assertEqual(st["zones"]["rear"], 100)
        self.assertEqual(st["zones"]["front"], 0)
        self.assertEqual(st["zones"]["front"] + st["zones"]["mid"] + st["zones"]["rear"], 100)

    def test_near_player_at_net_reads_as_front(self):
        st = compute_player_stats(NEAR_NET, "near", duration_min=1)
        self.assertEqual(st["zones"]["front"], 100)
        self.assertEqual(st["zones"]["rear"], 0)

    def test_far_player_deep_reads_as_rear_not_front(self):
        # Without the net_depth transform, raw v=0.1 would land in 'front' — the latent bug.
        st = compute_player_stats(FAR_DEEP, "far", duration_min=1)
        self.assertEqual(st["zones"]["rear"], 100)
        self.assertEqual(st["zones"]["front"], 0)

    def test_no_fabricated_shuttle_speed(self):
        st = compute_player_stats(NEAR_REAR, "near", duration_min=1)
        self.assertIsNone(st["fastest"])

    def test_timeline_length_and_heat_shape(self):
        st = compute_player_stats(NEAR_REAR, "near", duration_min=12)
        self.assertEqual(len(st["timeline"]), 3)  # ceil(12/5)
        self.assertTrue(st["heat"])
        for k in st["heat"]:
            self.assertEqual(set(k.keys()), {"x", "y", "w", "s"})


def _rally(idx, winner=None):
    return {"idx": idx, "winner_side": winner}


def _landing(rally_idx, call="out", located=True):
    p = {"located": located, "call": call, "band_cm": 45.0}
    return {"rally_idx": rally_idx, "type": "landing", "payload": p}


class TestOutErrorCounts(unittest.TestCase):
    def test_out_landing_counts_against_the_hitter(self):
        # winner A (out landed on A's half) ⇒ the error was B's
        counts = out_error_counts([_rally(0, winner="A"), _rally(1)],
                                  [_landing(0), _landing(1, call="in")])
        self.assertEqual(counts, {"B": 1})

    def test_uncertain_unlocated_and_unattributed_never_count(self):
        rallies = [_rally(0, winner="A"), _rally(1, winner="B"), _rally(2)]
        events = [_landing(0, call="uncertain"), _landing(1, located=False),
                  _landing(2)]                     # out but winner_side unset → no attribution
        self.assertEqual(out_error_counts(rallies, events), {})

    def test_dropped_rally_loses_its_error(self):
        # trainer dropped rally 0: its row is gone from rallies.parquet, its landing is an orphan
        counts = out_error_counts([_rally(1, winner="B")], [_landing(0), _landing(1)])
        self.assertEqual(counts, {"A": 1})

    def test_no_landing_events_means_none_written(self):
        # pre-landing cache (feature never ran) → None sentinel, the stage writes nothing
        self.assertIsNone(out_error_counts([_rally(0, winner="A")],
                                           [{"rally_idx": 0, "type": "rally_end", "payload": {}}]))


if __name__ == "__main__":
    unittest.main()
