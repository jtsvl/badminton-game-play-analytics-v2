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


def resolve_artifact(store, cfg, video_id: str, cache: Path, name: str) -> str | None:
    """Local cache path for a per-video artifact, restoring it from the artifact bucket on a
    cache miss (fresh worker / evicted cache). None when it exists in neither place. Shared by
    remap (tracks.parquet), retrack's gate_track (detections.parquet) and overlay (pose.parquet)."""
    local = cache / name
    if local.exists():
        return str(local)
    key = f"{video_id}/{name}"
    log.info("artifact: cache miss — downloading %s/%s", cfg.artifact_bucket, key)
    data = store.download(cfg.artifact_bucket, key)
    if not data:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    log.info("artifact: restored %s → %s (%d bytes)", key, local, len(data))
    return str(local)


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    path = resolve_artifact(store, cfg, video_id, cache, "tracks.parquet")
    if not path:
        raise RuntimeError(
            f"remap: no tracks.parquet in cache or {cfg.artifact_bucket}/{video_id}/tracks.parquet"
            " — run the full pipeline first"
        )
    ctx["tracks_path"] = path
    log.info("remap: tracks.parquet resolved (%s)", path)
