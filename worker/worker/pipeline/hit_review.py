"""hit_review — persist the trainer's hit-list edits (add/delete/nudge) as the learned
hit-detector's eval + training corpus (stroke-classification-v2 §5 Stage 1 / §9 gate G0).

The web already wrote the corrected hits DIRECTLY to `events` (type='hit') — INSERT a missed
hit, UPDATE t_s to nudge, DELETE a false one — so Postgres `events` is the authoritative hit
list the detector's dataset builder reads. This stage's only job is to persist the append-only
CORRECTION signal (what changed, and from what) into `hit_corrections` (0013), mirroring how
rally_sync persists rally_corrections: the web packs already-DB-column-shaped rows into
params.corrections (hitCorrections.ts), and this service-role stage stamps match/video/job and
inserts them. Wrapped so a broken corpus write never fails the job.

Deliberately does NOT re-derive hits (that would clobber the manual edits) and does NOT export a
hit-list parquet yet — the Phase-3 detector dataset builder reads `events` + the 30 fps cache.
"""
from __future__ import annotations

import logging

log = logging.getLogger("worker.pipeline.hit_review")


def _hit_correction_rows(match_id, video_id, job_id, corrections: list[dict]) -> list[dict]:
    """Stamp match/video/job onto the web's already DB-column-shaped params.corrections payload
    (hitCorrections.ts) -> hit_corrections insert rows."""
    return [{"match_id": match_id, "video_id": video_id, "job_id": job_id, **c} for c in corrections]


def run(ctx) -> None:
    store = ctx["store"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job["match_id"]

    corrections = (job.get("params") or {}).get("corrections")
    if not corrections:
        log.info("hit_review: no corrections in params — nothing to persist (match=%s)", match_id)
        return
    try:
        store.insert_hit_corrections(
            _hit_correction_rows(match_id, video_id, job.get("id"), corrections)
        )
        log.info("hit_review: persisted %d hit correction(s) (match=%s)", len(corrections), match_id)
    except Exception:
        log.exception("hit_review: failed to persist hit corrections (match=%s) — continuing", match_id)
