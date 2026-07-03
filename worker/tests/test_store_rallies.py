"""Store.replace_rallies — the idempotent stage-6 writer (delete-then-insert).

`events` has no upsert key (unlike `rallies`' unique(match_id,idx)), so a re-run must clear the match's
rows before re-inserting or events would pile up. PostgREST refuses an unfiltered delete, so the
.eq("match_id", …) filter is load-bearing. We test the pure row-mapping helpers directly and the
orchestration (delete order + filter + rally_idx→rally_id resolution) against a fake PostgREST client —
no real Supabase.
"""
import unittest
from types import SimpleNamespace

from worker.store import Store, _event_insert_rows, _rally_insert_rows


class TestRowMapping(unittest.TestCase):
    def test_rally_rows_carry_match_id_and_columns(self):
        rallies = [{"idx": 0, "t_start_s": 1.0, "t_end_s": 2.0, "shot_count": 4,
                    "winner_side": None, "end_shot_type": None, "confidence": 0.8}]
        rows = _rally_insert_rows("M1", rallies)
        self.assertEqual(rows, [{"match_id": "M1", "idx": 0, "t_start_s": 1.0, "t_end_s": 2.0,
                                 "shot_count": 4, "winner_side": None, "end_shot_type": None,
                                 "confidence": 0.8}])

    def test_event_rows_resolve_rally_idx_to_rally_id(self):
        events = [{"rally_idx": 0, "t_s": 1.0, "type": "serve", "payload": {"settled": True},
                   "confidence": 0.7},
                  {"rally_idx": 1, "t_s": 9.0, "type": "rally_end", "payload": {}, "confidence": None}]
        rows = _event_insert_rows("M1", events, {0: "rid-0", 1: "rid-1"})
        self.assertEqual([r["rally_id"] for r in rows], ["rid-0", "rid-1"])
        self.assertEqual(rows[0]["match_id"], "M1")
        self.assertEqual(rows[0]["type"], "serve")


class _FakeTable:
    def __init__(self, name, recorder, select_data):
        self.name, self.rec = name, recorder
        self.select_data = select_data
        self.op = None
        self.filters = {}
        self.rows = None
        self.ordered = None

    def delete(self):
        self.op = "delete"
        return self

    def insert(self, rows):
        self.op, self.rows = "insert", rows
        return self

    def upsert(self, rows, on_conflict=None):
        self.op, self.rows = f"upsert:{on_conflict}", rows
        return self

    def select(self, cols="*"):
        self.op, self.rows = "select", cols
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def order(self, col):
        self.ordered = col
        return self

    def execute(self):
        self.rec.append((self.name, self.op, dict(self.filters), self.rows, self.ordered))
        if self.op == "insert" and self.name == "rallies":
            return SimpleNamespace(data=[{**r, "id": f"rid-{r['idx']}"} for r in self.rows])
        if self.op == "select":
            return SimpleNamespace(data=self.select_data.get(self.name, []))
        return SimpleNamespace(data=[])


class _FakeSB:
    def __init__(self, select_data=None):
        self.calls = []
        self.select_data = select_data or {}

    def table(self, name):
        return _FakeTable(name, self.calls, self.select_data)


def _store_with_fake(select_data=None):
    s = Store.__new__(Store)        # bypass __init__ (no real supabase client)
    s._sb = _FakeSB(select_data)
    s._worker_id = "test"
    return s


class TestReplaceRalliesOrchestration(unittest.TestCase):
    def _rallies(self):
        return [{"idx": 0, "t_start_s": 1.0, "t_end_s": 2.0, "shot_count": 3,
                 "winner_side": None, "end_shot_type": None, "confidence": 0.9},
                {"idx": 1, "t_start_s": 5.0, "t_end_s": 7.0, "shot_count": 5,
                 "winner_side": None, "end_shot_type": None, "confidence": 0.8}]

    def _events(self):
        return [{"rally_idx": 0, "t_s": 1.0, "type": "serve", "payload": {"settled": True},
                 "confidence": 0.7},
                {"rally_idx": 1, "t_s": 7.0, "type": "rally_end", "payload": {"u": 0.5, "v": 0.95},
                 "confidence": None}]

    def test_deletes_with_match_filter_before_inserting(self):
        s = _store_with_fake()
        s.replace_rallies("M1", self._rallies(), self._events())
        deletes = [c for c in s._sb.calls if c[1] == "delete"]
        # events deleted first (FK), then rallies, both filtered by match_id
        self.assertEqual(deletes[0][:3], ("events", "delete", {"match_id": "M1"}))
        self.assertEqual(deletes[1][:3], ("rallies", "delete", {"match_id": "M1"}))

    def test_inserts_rallies_then_events_with_resolved_ids(self):
        s = _store_with_fake()
        n = s.replace_rallies("M1", self._rallies(), self._events())
        self.assertEqual(n, 2)
        inserts = [c for c in s._sb.calls if c[1] == "insert"]
        self.assertEqual(inserts[0][0], "rallies")
        self.assertEqual(inserts[1][0], "events")
        event_rows = inserts[1][3]
        self.assertEqual([r["rally_id"] for r in event_rows], ["rid-0", "rid-1"])

    def test_empty_rallies_still_clears_then_writes_nothing(self):
        s = _store_with_fake()
        n = s.replace_rallies("M1", [], [])
        self.assertEqual(n, 0)
        ops = [(c[0], c[1]) for c in s._sb.calls if c[1] != "select"]
        self.assertEqual(ops, [("events", "delete"), ("rallies", "delete")])   # cleared, no inserts

    def test_warns_before_clobbering_trainer_edits(self):
        # FR-35 rally review: a full re-run regenerates rallies from scratch — trainer drop/merge
        # edits (matches.rally_edited_at set on every review save) must at minimum be surfaced.
        s = _store_with_fake(select_data={"matches": [{"rally_edited_at": "2026-07-02T10:00:00Z"}]})
        with self.assertLogs("worker.store", level="WARNING") as logs:
            n = s.replace_rallies("M1", self._rallies(), self._events())
        self.assertEqual(n, 2)                       # still replaces — warn, don't block
        self.assertTrue(any("rally" in m.lower() for m in logs.output))

    def test_no_warning_without_trainer_edits(self):
        s = _store_with_fake(select_data={"matches": [{"rally_edited_at": None}]})
        with self.assertNoLogs("worker.store", level="WARNING"):
            s.replace_rallies("M1", self._rallies(), self._events())


class TestInsertRallyCorrections(unittest.TestCase):
    def test_plain_insert_of_given_rows(self):
        # append-only by design (0011): a plain insert, NOT upsert — the recorded op asserts it.
        rows = [{"op": "drop", "rally_id": "r-old", "match_id": "M1"}]
        s = _store_with_fake()
        s.insert_rally_corrections(rows)
        self.assertEqual(s._sb.calls, [("rally_corrections", "insert", {}, rows, None)])

    def test_empty_list_makes_no_call(self):
        s = _store_with_fake()
        s.insert_rally_corrections([])
        self.assertEqual(s._sb.calls, [])


class TestUpsertPlayerShotErrors(unittest.TestCase):
    def test_upserts_on_the_shot_stats_unique_key(self):
        # PostgREST merge-duplicates only touches the sent columns, so an existing row's `count`
        # (stage 7's column, once it exists) survives an errors-only upsert.
        s = _store_with_fake()
        s.upsert_player_shot_errors("M1", "P1", "unknown", 3)
        self.assertEqual(s._sb.calls, [(
            "player_shot_stats", "upsert:match_id,player_id,shot_type", {},
            {"match_id": "M1", "player_id": "P1", "shot_type": "unknown", "errors": 3}, None)])


class TestRallyReaders(unittest.TestCase):
    def test_get_rallies_selects_by_match_ordered_by_idx(self):
        rows = [{"id": "r0", "idx": 0}, {"id": "r1", "idx": 1}]
        s = _store_with_fake(select_data={"rallies": rows})
        self.assertEqual(s.get_rallies("M1"), rows)
        call = next(c for c in s._sb.calls if c[0] == "rallies")
        self.assertEqual(call[1], "select")
        self.assertEqual(call[2], {"match_id": "M1"})
        self.assertEqual(call[4], "idx")             # deterministic order for idx↔id mapping

    def test_get_events_selects_by_match(self):
        rows = [{"id": "e0", "rally_id": "r0"}]
        s = _store_with_fake(select_data={"events": rows})
        self.assertEqual(s.get_events("M1"), rows)
        call = next(c for c in s._sb.calls if c[0] == "events")
        self.assertEqual(call[1], "select")
        self.assertEqual(call[2], {"match_id": "M1"})


if __name__ == "__main__":
    unittest.main()
