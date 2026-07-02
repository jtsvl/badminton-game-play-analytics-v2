"""rally_sync — resync cache rallies/events.parquet from the (trainer-edited) DB rows.

`aggregate._rally_spans` reads `rallies.parquet` from the cache, not the DB — so after the
trainer drops/merges rallies in the web UI (FR-35), this stage rewrites the parquet from the
authoritative Postgres rows before `aggregate` re-gates the stats. Also runs in the `remap`
slice so identity recomputes keep the DB spans (a fresh cache used to silently fall back to
whole-clip gating). Mirrors remap.py: no re-detection, just artifact resolution. Artifacts are
re-uploaded so cache-miss recovery stays consistent with the DB.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .rally_hit import _event_schema, _rally_schema

log = logging.getLogger("worker.pipeline.rally_sync")

_RALLY_COLS = ("idx", "t_start_s", "t_end_s", "shot_count", "winner_side",
               "end_shot_type", "confidence")


def _rally_correction_rows(match_id, video_id, job_id, corrections: list[dict]) -> list[dict]:
    """Stamp match/video/job onto the web's already DB-column-shaped params.corrections payload
    (rallyCorrections.ts) -> rally_corrections insert rows."""
    return [{"match_id": match_id, "video_id": video_id, "job_id": job_id, **c} for c in corrections]


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job["match_id"]

    import polars as pl  # lazy, matches the other stages

    rallies = store.get_rallies(match_id)
    events = store.get_events(match_id)

    rally_rows = [{col: r.get(col) for col in _RALLY_COLS} for r in rallies]
    idx_by_id = {r["id"]: r["idx"] for r in rallies}
    event_rows, skipped = [], 0
    for e in events:
        idx = idx_by_id.get(e.get("rally_id"))
        if idx is None:  # orphaned by a drop, or pointing at a row that no longer exists
            skipped += 1
            continue
        payload = e.get("payload")
        event_rows.append({"rally_idx": idx, "t_s": e["t_s"], "type": e["type"],
                           "payload": json.dumps(payload) if payload is not None else None,
                           "confidence": e.get("confidence")})
    if skipped:
        log.warning("rally_sync: skipped %d event(s) with no matching rally (match=%s)",
                    skipped, match_id)

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    cache.mkdir(parents=True, exist_ok=True)
    rpath = cache / "rallies.parquet"
    epath = cache / "events.parquet"
    pl.DataFrame(rally_rows, schema=_rally_schema()).write_parquet(rpath)
    pl.DataFrame(event_rows, schema=_event_schema()).write_parquet(epath)
    ctx["rallies_path"] = str(rpath)
    ctx["events_path"] = str(epath)
    log.info("rally_sync: %d rallies, %d events → cache (match=%s)",
             len(rally_rows), len(event_rows), match_id)

    store.upload(cfg.artifact_bucket, f"{video_id}/rallies.parquet", rpath.read_bytes(),
                 "application/octet-stream")
    store.upload(cfg.artifact_bucket, f"{video_id}/events.parquet", epath.read_bytes(),
                 "application/octet-stream")

    # FR-35 eval corpus: params.corrections is only present on rally_review-triggered saves (the
    # web's remap enqueue sends params={}), so this is naturally a no-op on the remap slice — no
    # explicit job["stage"] check needed. Wrapped: a broken write must never block the rallies/
    # stats resync the trainer is waiting on.
    corrections = (job.get("params") or {}).get("corrections")
    if corrections:
        try:
            store.insert_rally_corrections(
                _rally_correction_rows(match_id, video_id, job.get("id"), corrections)
            )
            log.info("rally_sync: persisted %d rally correction(s) (match=%s)",
                     len(corrections), match_id)
        except Exception:
            log.exception("rally_sync: failed to persist rally corrections (match=%s) — continuing",
                          match_id)
