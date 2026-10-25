"""identity — the label→role→timeline binding shared by overlay, pose and aggregate.

The bug this module exists for: match_players.track_label names ONE raw tracker id, but a
player's timeline in tracks.parquet spans many ids (re-ID fragments keep their own track_id).
Consumers must bind a player to the labeled id's ROLE and then consume every row of that role.
Pure functions over plain dict rows — no polars, no DB.
"""
import unittest

from worker.pipeline import identity


def _row(t, tid, role, source="detected"):
    return {"t": t, "track_id": tid, "role": role, "source": source}


DOUBLES_ROWS = [
    _row(0.0, 118, "near_right"),
    _row(0.5, 429, "near_right", "reid"),
    _row(1.0, 118, "near_right", "interp"),
    _row(0.0, 579, "near_left"),
    _row(0.0, 788, "far_left"),
    _row(0.5, 927, "far_right"),
]


def _mp(pid, side, slot, label):
    return {"player_id": pid, "side": side, "slot": slot, "track_label": label}


class TestTrackRoles(unittest.TestCase):
    def test_maps_every_tid_to_its_role(self):
        roles = identity.track_roles(DOUBLES_ROWS)
        self.assertEqual(roles[118], "near_right")
        self.assertEqual(roles[429], "near_right")
        self.assertEqual(roles[579], "near_left")

    def test_first_role_wins_on_conflict(self):
        rows = DOUBLES_ROWS + [_row(2.0, 118, "far_left")]  # impossible today; defensive
        self.assertEqual(identity.track_roles(rows)[118], "near_right")

    def test_rows_without_role_are_skipped(self):
        self.assertEqual(identity.track_roles([{"t": 0.0, "track_id": 5, "role": None}]), {})


class TestRolePlayers(unittest.TestCase):
    def test_label_binds_the_role_not_the_tid(self):
        players = identity.role_players([_mp("p1", "A", 1, 118)], DOUBLES_ROWS)
        self.assertEqual(players["near_right"]["player_id"], "p1")

    def test_overridden_label_on_a_reid_fragment_binds_that_fragments_role(self):
        # FR-35: the trainer can point track_label at ANY tid — binding follows that tid's role.
        players = identity.role_players([_mp("p1", "A", 1, 429)], DOUBLES_ROWS)
        self.assertEqual(players["near_right"]["player_id"], "p1")

    def test_label_missing_from_tracks_binds_nothing(self):
        self.assertEqual(identity.role_players([_mp("p1", "A", 1, 999)], DOUBLES_ROWS), {})

    def test_unlabeled_player_binds_nothing(self):
        self.assertEqual(identity.role_players([_mp("p1", "A", 1, None)], DOUBLES_ROWS), {})

    def test_role_collision_keeps_the_most_present_label(self):
        # Both labels resolve to near_right: 118 has 2 non-interp rows, 429 has 1 → p1 wins
        # even though p2 sorts first (interp rows don't count toward presence).
        rows = DOUBLES_ROWS + [_row(2.0, 118, "near_right")]
        mps = [_mp("p1", "A", 1, 118), _mp("p2", "A", 0, 429)]
        players = identity.role_players(mps, rows)
        self.assertEqual(players["near_right"]["player_id"], "p1")
        self.assertEqual(len(players), 1)

    def test_singles_roles(self):
        rows = [_row(0.0, 1, "near"), _row(0.0, 2, "far")]
        mps = [_mp("p1", "A", 0, 1), _mp("p2", "B", 0, 2)]
        players = identity.role_players(mps, rows)
        self.assertEqual(players["near"]["player_id"], "p1")
        self.assertEqual(players["far"]["player_id"], "p2")


class TestRoleFrameIndex(unittest.TestCase):
    T_TO_FRAME = {0.0: 0, 0.5: 6, 1.0: 12}

    def test_indexes_non_interp_rows_by_frame_and_tid(self):
        idx = identity.role_frame_index(DOUBLES_ROWS, self.T_TO_FRAME)
        self.assertEqual(idx[(0, 118)], "near_right")
        self.assertEqual(idx[(6, 429)], "near_right")
        self.assertEqual(idx[(0, 788)], "far_left")

    def test_interp_rows_are_excluded(self):
        # interp rows are synthetic positions — no pose row exists for them; the join must not
        # fabricate a skeleton there.
        idx = identity.role_frame_index(DOUBLES_ROWS, self.T_TO_FRAME)
        self.assertNotIn((12, 118), idx)

    def test_missing_source_treated_as_detected(self):
        # Old caches predate the source column — their rows are all real detections.
        rows = [{"t": 0.0, "track_id": 7, "role": "near"}]
        self.assertEqual(identity.role_frame_index(rows, {0.0: 0}), {(0, 7): "near"})

    def test_t_not_in_manifest_is_skipped(self):
        rows = [_row(99.0, 118, "near_right")]
        self.assertEqual(identity.role_frame_index(rows, self.T_TO_FRAME), {})


if __name__ == "__main__":
    unittest.main()
