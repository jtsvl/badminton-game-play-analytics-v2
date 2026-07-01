"""Job store: claim and update processing_jobs.

Claiming uses `FOR UPDATE SKIP LOCKED` so multiple workers never grab the same job
(specs/40-worker-contract.md). psycopg is imported lazily so the package imports without it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("worker.db")


class JobStore:
    def __init__(self, db_url: str, worker_id: str, stale_job_timeout_s: float = 3600.0) -> None:
        self._db_url = db_url
        self._worker_id = worker_id
        self._stale_job_timeout_s = stale_job_timeout_s
        self._conn: Any = None

    def connect(self) -> None:
        import psycopg  # lazy: only needed when actually running against a DB

        self._conn = psycopg.connect(self._db_url, autocommit=True)
        log.info("connected to database")

    def claim_job(self) -> Optional[dict]:
        """Atomically claim the oldest queued job; with the queue empty, reclaim a stale running one
        (dead worker — specs/40). `claimed_at` is the heartbeat set_stage refreshes."""
        sql = (
            "update processing_jobs j set status='running', claimed_by=%s, "
            "claimed_at=now(), started_at=now() "
            "where j.id = (select id from processing_jobs where status='queued' "
            "order by created_at for update skip locked limit 1) "
            "returning j.id, j.match_id, j.video_id, j.stage, j.params"
        )
        job = self._claim(sql, (self._worker_id,))
        if job is not None:
            return job
        stale_sql = (
            "update processing_jobs j set status='running', claimed_by=%s, "
            "claimed_at=now(), started_at=now() "
            "where j.id = (select id from processing_jobs where status='running' "
            "and claimed_at < now() - make_interval(secs => %s) "
            "order by claimed_at for update skip locked limit 1) "
            "returning j.id, j.match_id, j.video_id, j.stage, j.params"
        )
        job = self._claim(stale_sql, (self._worker_id, self._stale_job_timeout_s))
        if job is not None:
            log.warning("reclaimed stale job %s (no heartbeat for %.0fs)",
                        job["id"], self._stale_job_timeout_s)
        return job

    def _claim(self, sql: str, params: tuple) -> Optional[dict]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))

    def set_stage(self, job_id: str, stage: str, progress: float) -> None:
        # claimed_at doubles as the heartbeat (see claim_job) — refresh it on every stage.
        with self._conn.cursor() as cur:
            cur.execute(
                "update processing_jobs set stage=%s, progress=%s, claimed_at=now() where id=%s",
                (stage, progress, job_id),
            )

    def finish(self, job_id: str, ok: bool, error: Optional[str] = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "update processing_jobs set status=%s, error=%s, finished_at=now(), "
                "progress=case when %s then 1 else progress end where id=%s",
                ("succeeded" if ok else "failed", error, ok, job_id),
            )

    def log_job_run(
        self, job: str, match_id: Optional[str], is_ok: bool, detail: str = ""
    ) -> None:
        """Audit row per run (specs/40). started_at defaults to insert time ≈ finished_at —
        good enough for an audit trail. Must never break a job, hence the swallow."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "insert into job_runs (job, match_id, is_ok, detail, finished_at) "
                    "values (%s, %s, %s, %s, now())",
                    (job, match_id, is_ok, detail),
                )
        except Exception as e:  # noqa: BLE001 — audit log must never break a job
            log.warning("job_runs insert failed: %s", e)
