"""hit_review stage — persists params.corrections into hit_corrections (0013). Pure-logic +
stub-store tests, no Supabase. Mirrors the rally_sync correction-write pattern."""
from __future__ import annotations

import unittest

from worker.pipeline.hit_review import _hit_correction_rows, run


class StubStore:
    def __init__(self):
        self.inserted: list[dict] | None = None

    def insert_hit_corrections(self, rows):
        self.inserted = rows


def _ctx(store, corrections):
    job = {"id": "job-1", "match_id": "m-1", "video_id": "v-1", "params": {"corrections": corrections}}
    return {"store": store, "job": job, "cfg": object()}


class TestHitCorrectionRows(unittest.TestCase):
    def test_stamps_match_video_job(self):
        cors = [{"op": "add", "event_id": "e1", "new_t_s": 12.0},
                {"op": "delete", "event_id": "e2", "orig_t_s": 5.0}]
        rows = _hit_correction_rows("m-1", "v-1", "job-1", cors)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["match_id"], "m-1")
        self.assertEqual(rows[0]["video_id"], "v-1")
        self.assertEqual(rows[0]["job_id"], "job-1")
        self.assertEqual(rows[0]["op"], "add")
        self.assertEqual(rows[0]["event_id"], "e1")
        self.assertEqual(rows[1]["op"], "delete")


class TestRun(unittest.TestCase):
    def test_persists_corrections(self):
        store = StubStore()
        run(_ctx(store, [{"op": "nudge", "event_id": "e1", "orig_t_s": 1.0, "new_t_s": 1.2}]))
        self.assertIsNotNone(store.inserted)
        self.assertEqual(len(store.inserted), 1)
        self.assertEqual(store.inserted[0]["op"], "nudge")
        self.assertEqual(store.inserted[0]["match_id"], "m-1")

    def test_no_corrections_is_noop(self):
        store = StubStore()
        run(_ctx(store, []))
        self.assertIsNone(store.inserted)

    def test_missing_params_is_noop(self):
        store = StubStore()
        run({"store": store, "job": {"id": "j", "match_id": "m", "video_id": "v"}, "cfg": object()})
        self.assertIsNone(store.inserted)

    def test_insert_failure_does_not_raise(self):
        class Boom:
            def insert_hit_corrections(self, rows):
                raise RuntimeError("db down")
        run(_ctx(Boom(), [{"op": "add", "event_id": "e1", "new_t_s": 3.0}]))  # must not raise


if __name__ == "__main__":
    unittest.main()
