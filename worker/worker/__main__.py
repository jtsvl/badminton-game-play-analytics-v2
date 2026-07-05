"""Worker entry point: poll → claim → run pipeline → update. `python -m worker enqueue ...` to seed.

Job `stage` selects the slice to run, honoring the gates (specs/40-worker-contract.md):
  ingest       → [ingest,proxy]                           then match.status='calibrating'
  calibrate    → RESUME_STAGES (calibrate…overlay, incl. shuttle+rally_hit)  then match.status='review'
  remap        → [remap,rally_sync,aggregate,overlay]     then match.status='done'
  rally_review → [remap,rally_sync,aggregate]             then params.return_status (default 'done')
  proxy        → [proxy] standalone (backfill; match status untouched, stage never raises)
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
        # proxy rides along so every new upload gets a playback proxy (video-proxy spec);
        # it never raises, so a proxy hiccup can't fail the ingest.
        return ["ingest", "proxy"], "calibrating"
    if stage in ("calibrate", "resume_after_calibration"):
        return RESUME_STAGES, "review"
    if stage == "remap":
        # FR-35: trainer corrected the track→player mapping → cheap recompute (no re-detect).
        # rally_sync keeps the DB-authoritative rally spans (incl. trainer edits) in the cache.
        return ["remap", "rally_sync", "aggregate", "overlay"], "done"
    if stage == "rally_review":
        # FR-35: trainer dropped/merged rallies in the web UI → re-gate the stats. No overlay
        # (it never reads rallies). End status = where the match was before the edit ('review'
        # must not skip the PlayerReview confirmation; 'done' stays done).
        end = (job.get("params") or {}).get("return_status") or "done"
        return ["remap", "rally_sync", "aggregate"], end
    return [stage], None


def process_job(store, cfg, job: dict) -> None:
    job_id = job["id"]
    stage = job.get("stage") or "ingest"
    log.info("claimed job %s (match=%s stage=%s)", job_id, job.get("match_id"), stage)
    stages, end_status = _slice_for(job)
    ctx = StageContext(job=job, cfg=cfg, store=store)
    try:
        run_pipeline(ctx, stages, on_stage=lambda name, frac: store.set_stage(job_id, name, frac))
        store.finish(job_id, ok=True)
        if end_status and hasattr(store, "update_match_status"):
            store.update_match_status(job["match_id"], end_status)
        store.log_job_run(stage, job.get("match_id"), is_ok=True,
                          detail=f"job={job_id} stages={stages} -> {end_status or 'n/a'}")
        log.info("job %s succeeded (match->%s)", job_id, end_status)
    except Exception as e:  # noqa: BLE001 — record failure, keep the worker alive
        log.exception("job %s failed", job_id)
        store.finish(job_id, ok=False, error=str(e))
        if hasattr(store, "update_match_status"):
            store.update_match_status(job["match_id"], "failed", error=str(e))
        store.log_job_run(stage, job.get("match_id"), is_ok=False,
                          detail=f"job={job_id} stages={stages}: {e}")


def _make_store(cfg):
    if cfg.db_url:
        from .db import JobStore  # direct Postgres path (not used on Coolify API-only)

        store = JobStore(cfg.db_url, cfg.worker_id, stale_job_timeout_s=cfg.stale_job_timeout_s)
        store.connect()
        return store
    from .store import Store

    log.info("using Supabase API store (no direct SUPABASE_DB_URL)")
    return Store(
        cfg.supabase_url, cfg.service_role_key, stale_job_timeout_s=cfg.stale_job_timeout_s
    ).with_worker(cfg.worker_id)


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

    if len(sys.argv) > 1 and sys.argv[1] == "export-stroke-clips":
        from .stroke_clips import export_cli

        export_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "import-stroke-labels":
        from .import_stroke_labels import import_cli

        import_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "export-stroke-dataset":
        from .stroke.dataset import export_cli

        export_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "train-stroke":
        from .stroke.train import train_cli

        train_cli(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backfill-calib-frames":
        from .backfill_calib_frames import backfill_cli

        backfill_cli(sys.argv[2:])
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
