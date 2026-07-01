"""Store.claim_job — queued-first claiming + stale-running reclaim (specs/40-worker-contract.md).

A worker killed mid-job (OOM/SIGKILL) leaves its job status='running' forever; the spec promises the
claim is reclaimable once `claimed_at` goes stale. `set_stage` doubles as the heartbeat (it refreshes
`claimed_at`), so the timeout only has to outlast the longest single stage. Fake PostgREST client in
the style of test_store_rallies — no real Supabase.
"""
import unittest
from types import SimpleNamespace

from worker.store import Store

OLD = "2000-01-01T00:00:00+00:00"     # far past → always stale
FRESH = "2999-01-01T00:00:00+00:00"   # far future → never stale (ISO strings compare lexically)


class _FakeJobsTable:
    """Row-backed fake supporting the select/update + eq/lt/order/limit chains claim_job uses."""

    def __init__(self, rows, opts):
        self.rows, self.opts = rows, opts
        self.op, self.patch, self.filters = None, None, []
        self._limit = None

    def select(self, _cols):
        self.op = "select"
        return self

    def update(self, patch):
        self.op, self.patch = "update", patch
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def lt(self, col, val):
        self.filters.append(("lt", col, val))
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _match(self, r):
        for kind, col, val in self.filters:
            if kind == "eq" and r.get(col) != val:
                return False
            if kind == "lt" and not (r.get(col) is not None and r[col] < val):
                return False
        return True

    def execute(self):
        hit = [r for r in self.rows if self._match(r)]
        if self._limit is not None:
            hit = hit[: self._limit]
        if self.op == "update":
            if self.opts.get("race"):      # simulate losing the conditional update
                return SimpleNamespace(data=[])
            for r in hit:
                r.update(self.patch)
        return SimpleNamespace(data=[dict(r) for r in hit])


class _FakeSB:
    def __init__(self, rows):
        self.rows, self.opts = rows, {}

    def table(self, name):
        assert name == "processing_jobs"
        return _FakeJobsTable(self.rows, self.opts)


def _store_with(rows, stale_timeout_s=3600.0):
    s = Store.__new__(Store)  # bypass __init__ (no real supabase client)
    s._sb = _FakeSB(rows)
    s._worker_id = "test-worker"
    s._stale_job_timeout_s = stale_timeout_s
    return s


def _job(id, status, claimed_at=None, **extra):
    return {"id": id, "match_id": "m-1", "video_id": None, "stage": "calibrate",
            "params": {}, "status": status, "claimed_at": claimed_at,
            "claimed_by": "other-worker" if status == "running" else None,
            "created_at": "2026-01-01T00:00:00+00:00", **extra}


class TestClaimJob(unittest.TestCase):
    def test_queued_job_wins_over_stale_running(self):
        rows = [_job("stale", "running", OLD), _job("fresh-q", "queued")]
        job = _store_with(rows).claim_job()
        self.assertEqual(job["id"], "fresh-q")
        self.assertEqual(rows[1]["status"], "running")
        self.assertEqual(rows[0]["claimed_by"], "other-worker")  # stale one untouched

    def test_stale_running_job_is_reclaimed(self):
        rows = [_job("stale", "running", OLD)]
        job = _store_with(rows).claim_job()
        self.assertEqual(job["id"], "stale")
        self.assertEqual(rows[0]["claimed_by"], "test-worker")
        self.assertEqual(rows[0]["status"], "running")
        self.assertNotEqual(rows[0]["claimed_at"], OLD)  # heartbeat reset on reclaim

    def test_fresh_running_job_is_not_touched(self):
        rows = [_job("busy", "running", FRESH)]
        self.assertIsNone(_store_with(rows).claim_job())
        self.assertEqual(rows[0]["claimed_by"], "other-worker")

    def test_lost_race_returns_none(self):
        rows = [_job("stale", "running", OLD)]
        s = _store_with(rows)
        s._sb.opts["race"] = True
        self.assertIsNone(s.claim_job())

    def test_no_jobs_returns_none(self):
        self.assertIsNone(_store_with([]).claim_job())


class TestHeartbeat(unittest.TestCase):
    def test_set_stage_refreshes_claimed_at(self):
        rows = [_job("j1", "running", OLD)]
        _store_with(rows).set_stage("j1", "detect", 0.4)
        self.assertEqual(rows[0]["stage"], "detect")
        self.assertNotEqual(rows[0]["claimed_at"], OLD)


if __name__ == "__main__":
    unittest.main()
