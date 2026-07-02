"""Stage: proxy — 720p playback proxy for the browser (specs/features/video-proxy.md).

The 4K originals never reach storage (ingest decodes them in place from videos/), so the web
app had no playable source. This stage transcodes the source to a small streamable MP4 via
the ffmpeg CLI (host dependency; VideoToolbox hw encode, ~5x realtime on the M4 Max) and
uploads it to `analysis-artifacts/<video_id>/proxy_720p.mp4` with a TUS resumable upload —
the Kong gateway kills any single request at ~60 s (spiked 2026-07-02), which a plain POST
of the larger games (~70-105 MB) would hit. `match_videos.proxy_path` records the result.

This stage NEVER raises: the proxy is a supplementary playback artifact, and process_job's
failure path sets matches.status='failed' — a missing proxy must not fail an ingest or knock
a backfilled match out of review. All ML stages keep reading the 4K frame cache.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .ingest import resolve_source

log = logging.getLogger("worker.pipeline.proxy")

TRANSCODE_TIMEOUT_S = 1800  # generous: a 11-min 4K game transcodes in ~2 min


def _ffmpeg_cmd(src: Path, dst: Path) -> list[str]:
    """Spike-proven recipe: 720p/30fps H.264 (yuv420p, browser-safe), audio kept when present
    (hit sounds help verify rally boundaries), +faststart for progressive streaming."""
    return [
        "ffmpeg", "-y", "-hwaccel", "videotoolbox", "-i", str(src),
        "-vf", "scale=-2:720", "-r", "30",
        "-c:v", "h264_videotoolbox", "-b:v", "2500k", "-maxrate", "3000k", "-bufsize", "5000k",
        "-g", "60",
        "-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(dst),
    ]


def run(ctx) -> None:
    try:
        _run(ctx)
    except Exception:  # noqa: BLE001 — supplementary artifact, never fail the job
        log.warning("proxy stage failed (non-fatal, no proxy for this video)", exc_info=True)


def _run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    params = job.get("params") or {}
    video_id = job["video_id"]

    video = store.get_video(video_id)
    if not video:
        log.warning("proxy: no match_videos row for video_id=%s — skipping", video_id)
        return
    if video.get("proxy_path") and not params.get("force"):
        log.info("proxy already exists (%s) — skipping (params.force regenerates)",
                 video["proxy_path"])
        return
    if shutil.which("ffmpeg") is None:
        log.warning("proxy: ffmpeg not on PATH — skipping (brew install ffmpeg)")
        return

    cache_dir = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source(cfg, video["storage_path"], cache_dir, store)
    dst = cache_dir / "proxy_720p.mp4"

    log.info("transcoding %s -> %s", src.name, dst)
    res = subprocess.run(_ffmpeg_cmd(src, dst), capture_output=True,
                         timeout=TRANSCODE_TIMEOUT_S)
    if res.returncode != 0:
        tail = (res.stderr or b"")[-500:].decode(errors="replace")
        log.warning("proxy: ffmpeg exited %d — skipping. stderr tail: %s", res.returncode, tail)
        return

    proxy_path = f"{video_id}/proxy_720p.mp4"
    store.upload_resumable(cfg.artifact_bucket, proxy_path, dst, "video/mp4")
    store.set_video_proxy(video_id, proxy_path)
    log.info("proxy uploaded: %s/%s (%.1f MB)", cfg.artifact_bucket, proxy_path,
             dst.stat().st_size / 1e6)
