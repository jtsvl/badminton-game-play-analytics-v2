"""Tests for __main__.process_job — the claim→run→update wiring around run_pipeline.

Guards the specs/40-worker-contract.md contract: every run logs to `job_runs` (success AND
failure), the job row is finished, and the match status flips to the slice's end status
('failed' on error). Fake store, fake stages — no ml deps.
"""
import unittest
from unittest.mock import patch

from worker.__main__ import RESUME_STAGES, process_job
from worker.pipeline import STAGES, run_pipeline


class _Store:
    """Records every call process_job makes; stands in for Store/JobStore."""

    def __init__(self):
        self.set_stages: list[tuple[str, float]] = []
        self.finished: tuple[bool, str | None] | None = None
        self.match_status: tuple[str, str, dict] | None = None
        self.job_runs: list[dict] = []

    def set_stage(self, job_id, stage, progress):
        self.set_stages.append((stage, progress))

    def finish(self, job_id, ok, error=None):
        self.finished = (ok, error)

    def update_match_status(self, match_id, status, **extra):
        self.match_status = (match_id, status, extra)

    def log_job_run(self, job, match_id, is_ok, detail=""):
        self.job_runs.append({"job": job, "match_id": match_id, "is_ok": is_ok, "detail": detail})


class TestProcessJob(unittest.TestCase):
    def test_success_finishes_flips_status_and_logs_job_run(self):
        store = _Store()
        job = {"id": "job-1", "match_id": "m-1", "stage": "calibrate"}
        with patch("worker.__main__.run_pipeline") as rp:
            process_job(store, cfg=None, job=job)
        rp.assert_called_once()
        self.assertEqual(store.finished, (True, None))
        self.assertEqual(store.match_status, ("m-1", "review", {}))
        self.assertEqual(len(store.job_runs), 1)
        run = store.job_runs[0]
        self.assertEqual((run["job"], run["match_id"], run["is_ok"]), ("calibrate", "m-1", True))
        self.assertIn("job-1", run["detail"])

    def test_failure_records_error_and_logs_job_run(self):
        store = _Store()
        job = {"id": "job-2", "match_id": "m-2", "stage": "calibrate"}
        with patch("worker.__main__.run_pipeline", side_effect=RuntimeError("boom")):
            process_job(store, cfg=None, job=job)
        self.assertEqual(store.finished, (False, "boom"))
        self.assertEqual(store.match_status, ("m-2", "failed", {"error": "boom"}))
        self.assertEqual(len(store.job_runs), 1)
        run = store.job_runs[0]
        self.assertEqual((run["job"], run["match_id"], run["is_ok"]), ("calibrate", "m-2", False))
        self.assertIn("boom", run["detail"])


class TestRunPipelineOrchestration(unittest.TestCase):
    def test_resume_stages_run_in_order_with_progress(self):
        ran: list[str] = []
        progress: list[tuple[str, float]] = []
        fakes = {name: (lambda n: lambda ctx: ran.append(n))(name) for name in RESUME_STAGES}
        with patch.dict(STAGES, fakes):
            run_pipeline({}, RESUME_STAGES, on_stage=lambda n, f: progress.append((n, f)))
        self.assertEqual(ran, RESUME_STAGES)
        self.assertEqual(progress[-1], ("done", 1.0))
        self.assertEqual([n for n, _ in progress[:-1]], RESUME_STAGES)
        fractions = [f for _, f in progress]
        self.assertEqual(fractions, sorted(fractions))  # monotonically advancing


if __name__ == "__main__":
    unittest.main()
