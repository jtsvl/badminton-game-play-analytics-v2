"""Tests for the pure aggregation wiring (worker/pipeline/aggregate.compute_player_stats).

Guards the insights.zones() depth-orientation fix: raw v is NOT distance-from-net (v=0 is the far
baseline, net is at v=0.5), so aggregate must apply geometry.net_depth before zones().
"""
import unittest

from worker import insights as ins
from worker.pipeline.aggregate import compute_player_stats, out_error_counts

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
