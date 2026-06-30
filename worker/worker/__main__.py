"""Worker entry point: poll → claim → run pipeline → update. `python -m worker enqueue ...` to seed.

Job `stage` selects the slice to run, honoring the gates (specs/40-worker-contract.md):
  ingest    → [ingest]                                 then match.status='calibrating'
  calibrate → [calibrate,detect,pose,gate_track,aggregate,overlay]  then match.status='review'
  remap     → [remap,aggregate,overlay]                then match.status='done'
The trainer confirms corners (calibrate) then the track→player mapping (remap, FR-35) in the web UI
between jobs; remap reuses the cached tracks.parquet, so it recomputes stats/overlays without re-detecting.
"""
from __future__ import annotations

import logging
import sys
import time

from .config import load_config
from .pipeline import StageContext, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

RESUME_STAGES = ["calibrate", "detect", "pose", "gate_track", "shuttle", "rally_hit", "aggregate", "overlay"]


def _slice_for(job: dict) -> tuple[list[str], str | None]:
    stage = job.get("stage") or "ingest"
    if stage == "ingest":
        return ["ingest"], "calibrating"
    if stage in ("calibrate", "resume_after_calibration"):
        return RESUME_STAGES, "review"
    if stage == "remap":
        # FR-35: trainer corrected the track→player mapping → cheap recompute (no re-detect).
        return ["remap", "aggregate", "overlay"], "done"
    return [stage], None


def process_job(store, cfg, job: dict) -> None:
    job_id = job["id"]
    log.info("claimed job %s (match=%s stage=%s)", job_id, job.get("match_id"), job.get("stage"))
    stages, end_status = _slice_for(job)
    ctx = StageContext(job=job, cfg=cfg, store=store)
    try:
        run_pipeline(ctx, stages, on_stage=lambda name, frac: store.set_stage(job_id, name, frac))
        store.finish(job_id, ok=True)
        if end_status and hasattr(store, "update_match_status"):
            store.update_match_status(job["match_id"], end_status)
        log.info("job %s succeeded (match->%s)", job_id, end_status)
    except Exception as e:  # noqa: BLE001 — record failure, keep the worker alive
        log.exception("job %s failed", job_id)
        store.finish(job_id, ok=False, error=str(e))
        if hasattr(store, "update_match_status"):
            store.update_match_status(job["match_id"], "failed", error=str(e))


def _make_store(cfg):
    if cfg.db_url:
        from .db import JobStore  # direct Postgres path (not used on Coolify API-only)

        store = JobStore(cfg.db_url, cfg.worker_id)
        store.connect()
        return store
    from .store import Store

    log.info("using Supabase API store (no direct SUPABASE_DB_URL)")
    return Store(cfg.supabase_url, cfg.service_role_key).with_worker(cfg.worker_id)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "enqueue":
        from .enqueue import enqueue_cli

        enqueue_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        from .verify import verify_cli

        verify_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "import-corrections":
        from .import_corrections import import_cli

        import_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "import-shuttle-labels":
        from .import_shuttle_labels import import_cli

        import_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "export-shuttle-dataset":
        from .shuttle.dataset_export import export_cli

        export_cli(sys.argv[2:])
        return

    once = "--once" in sys.argv[1:]  # drain queued jobs then exit (dev / cron)
    cfg = load_config()
    log.info("worker %s starting (configured=%s, once=%s)", cfg.worker_id, cfg.configured, once)
    if not cfg.configured:
        log.warning("SUPABASE_URL / SERVICE_ROLE_KEY not set — nothing to do. Fill .env and restart.")
        return

    store = _make_store(cfg)
    log.info("polling every %.1fs", cfg.poll_interval_s)
    while True:
        job = store.claim_job()
        if job is None:
            if once:
                log.info("no queued jobs — exiting (--once)")
                return
            time.sleep(cfg.poll_interval_s)
            continue
        process_job(store, cfg, job)


if __name__ == "__main__":
    main()
