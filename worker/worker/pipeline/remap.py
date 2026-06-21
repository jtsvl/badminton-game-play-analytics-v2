"""Stage: remap — resolve the cached tracks.parquet for a cheap identity-recompute job.

After the trainer corrects the track→player mapping in the review UI (FR-35), the save enqueues a
`remap` job: __main__._slice_for runs [remap, aggregate, overlay] → status 'done'. This stage does NO
re-detection — it only makes the already-computed tracks available to aggregate/overlay (both of which
require ctx["tracks_path"]). gate_track wrote tracks.parquet to the frame cache AND uploaded it to the
artifact bucket, so we prefer the local cache and fall back to an S3 download (robust on a fresh worker).
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("worker.pipeline.remap")


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    local = cache / "tracks.parquet"

    if local.exists():
        ctx["tracks_path"] = str(local)
        log.info("remap: using cached tracks.parquet (%s)", local)
        return

    # Cache miss (fresh worker / evicted cache) → restore from the artifact bucket gate_track uploaded to.
    key = f"{video_id}/tracks.parquet"
    log.info("remap: cache miss — downloading %s/%s", cfg.artifact_bucket, key)
    data = store.download(cfg.artifact_bucket, key)
    if not data:
        raise RuntimeError(
            f"remap: no tracks.parquet in cache or {cfg.artifact_bucket}/{key} — run the full pipeline first"
        )
    cache.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    ctx["tracks_path"] = str(local)
    log.info("remap: restored tracks.parquet → %s (%d bytes)", local, len(data))
