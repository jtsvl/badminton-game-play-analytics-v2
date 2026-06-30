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
    def __init__(self, name, recorder):
        self.name, self.rec = name, recorder
        self.op = None
        self.filters = {}
        self.rows = None

    def delete(self):
        self.op = "delete"
        return self

    def insert(self, rows):
        self.op, self.rows = "insert", rows
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def execute(self):
        self.rec.append((self.name, self.op, dict(self.filters), self.rows))
        if self.op == "insert" and self.name == "rallies":
            return SimpleNamespace(data=[{**r, "id": f"rid-{r['idx']}"} for r in self.rows])
        return SimpleNamespace(data=[])


class _FakeSB:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return _FakeTable(name, self.calls)


def _store_with_fake():
    s = Store.__new__(Store)        # bypass __init__ (no real supabase client)
    s._sb = _FakeSB()
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
        calls = s._sb.calls
        # events deleted first (FK), then rallies, both filtered by match_id
        self.assertEqual(calls[0][:3], ("events", "delete", {"match_id": "M1"}))
        self.assertEqual(calls[1][:3], ("rallies", "delete", {"match_id": "M1"}))

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
        ops = [(c[0], c[1]) for c in s._sb.calls]
        self.assertEqual(ops, [("events", "delete"), ("rallies", "delete")])   # cleared, no inserts


if __name__ == "__main__":
    unittest.main()
