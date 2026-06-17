"""Worker entry point: poll → claim → run pipeline → update. Idles cleanly if not configured."""
from __future__ import annotations

import logging
import time

from .config import load_config
from .pipeline import STAGE_ORDER, SIDE_ANGLE_STAGES, StageContext, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def process_job(store, cfg, job: dict) -> None:
    job_id = job["id"]
    log.info("claimed job %s (match=%s stage=%s)", job_id, job.get("match_id"), job.get("stage"))
    ctx = StageContext(job=job, cfg=cfg, store=store)
    stages = SIDE_ANGLE_STAGES if (job.get("params") or {}).get("angle") == "side" else STAGE_ORDER
    try:
        run_pipeline(ctx, stages, on_stage=lambda name, frac: store.set_stage(job_id, name, frac))
        store.finish(job_id, ok=True)
        log.info("job %s succeeded", job_id)
    except Exception as e:  # noqa: BLE001 — record failure, keep the worker alive
        log.exception("job %s failed", job_id)
        store.finish(job_id, ok=False, error=str(e))


def main() -> None:
    cfg = load_config()
    log.info("worker %s starting (configured=%s)", cfg.worker_id, cfg.configured)
    if not cfg.configured:
        log.warning("SUPABASE_URL / SERVICE_ROLE_KEY not set — nothing to do. Fill .env and restart.")
        return

    if not cfg.db_url:
        log.info("SUPABASE_DB_URL not set — job claiming via service_role RPC lands in P1. Idle for now.")
        return

    from .db import JobStore

    store = JobStore(cfg.db_url, cfg.worker_id)
    store.connect()
    log.info("polling every %.1fs", cfg.poll_interval_s)
    while True:
        job = store.claim_job()
        if job is None:
            time.sleep(cfg.poll_interval_s)
            continue
        process_job(store, cfg, job)


if __name__ == "__main__":
    main()
