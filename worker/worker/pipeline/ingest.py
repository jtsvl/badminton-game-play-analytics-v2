"""Stage: ingest — download, probe, decode-once → frame cache + still, seed calibration.

Decode once and cache frames at a fixed analysis fps so every later model reads the cache
(CLAUDE.md). Uses PyAV (+ PIL for JPEG) only — NOT cv2 — to avoid the libavdevice dylib clash.
DJI ultrawide undistortion is a no-op in P1 (no intrinsics): we assume rectilinear footage and
record the caveat; barrel distortion would bias edge (u,v). See specs/30-ml-pipeline.md stage 0.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from ..calib_frames import select_calib_frames, upload_calib_frames

log = logging.getLogger("worker.pipeline.ingest")

DEFAULT_ANALYSIS_FPS = 12.0


def resolve_source(cfg, storage_path: str, cache_dir: Path, store) -> Path:
    """Locate the source video: local videos/ file (decode in place — no copy, no extra disk)
    → cached download → fetch from the video bucket. Shared with the proxy stage."""
    local = Path(cfg.local_video_dir).expanduser() / storage_path
    if local.exists() and local.stat().st_size > 0:
        log.info("using local file %s (in place)", local)
        return local
    ext = Path(storage_path).suffix or ".mp4"
    src = cache_dir / f"source{ext}"
    if not src.exists() or src.stat().st_size == 0:
        log.info("downloading %s/%s", cfg.video_bucket, storage_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        src.write_bytes(store.download(cfg.video_bucket, storage_path))
    return src


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    params = job.get("params") or {}
    video_id = job["video_id"]
    match_id = job["match_id"]

    video = store.get_video(video_id)
    if not video:
        raise RuntimeError(f"ingest: no match_videos row for video_id={video_id}")

    cache_dir = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    src = resolve_source(cfg, video["storage_path"], cache_dir, store)
    log.info("source = %.1f MB", src.stat().st_size / 1e6)

    import av  # lazy (ml extra)

    analysis_fps = float(params.get("analysis_fps", DEFAULT_ANALYSIS_FPS))
    max_frames = params.get("max_frames")  # dev cap; None = whole clip
    step = 1.0 / analysis_fps

    container = av.open(str(src))
    vstream = container.streams.video[0]
    vstream.thread_type = "AUTO"
    width = int(vstream.codec_context.width)
    height = int(vstream.codec_context.height)
    fps = float(vstream.average_rate) if vstream.average_rate else 30.0
    duration_s = float(container.duration) / 1_000_000 if container.duration else None
    still_target = min(5.0, (duration_s or 10.0) / 2)

    manifest: list[dict] = []
    still_bytes: bytes | None = None
    still_t: float | None = None
    still_locked = False
    next_t = 0.0
    kept = 0

    for frame in container.decode(video=0):
        t = float(frame.time) if frame.time is not None else kept * step
        if t + 1e-6 < next_t:
            continue
        next_t = t + step
        img = frame.to_image()  # PIL RGB; undistort here once intrinsics exist
        img.save(frames_dir / f"{kept:06d}.jpg", "JPEG", quality=88)
        manifest.append({"frame": kept, "t": round(t, 4)})
        if still_bytes is None or (not still_locked and t >= still_target):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            still_bytes, still_t = buf.getvalue(), t
            still_locked = t >= still_target
        kept += 1
        if max_frames and kept >= int(max_frames):
            break
    container.close()

    if not manifest:
        raise RuntimeError("ingest: decoded 0 frames")
    if duration_s is None:
        duration_s = manifest[-1]["t"]

    (cache_dir / "frames.json").write_text(
        json.dumps(
            {
                "analysis_fps": analysis_fps,
                "source_fps": fps,
                "width": width,
                "height": height,
                "duration_s": duration_s,
                "undistorted": False,  # P1 caveat: rectilinear assumed (no DJI intrinsics)
                "frames": manifest,
            }
        )
    )
    store.update_video_meta(video_id, round(fps, 3), round(duration_s, 2), width, height)

    still_path = f"{video_id}/calib_frame.jpg"
    store.upload(cfg.artifact_bucket, still_path, still_bytes, "image/jpeg")

    # N evenly-spaced snapshots so the calibration screen can switch frames when a player blocks a
    # corner in one of them. Reuse the already-cached JPEGs on disk (no re-encode); the court is
    # static so the corner handles are frame-independent. See 0007_calibration_frames.sql.
    picks = select_calib_frames(manifest, cfg.calib_frame_count)
    frame_uris = upload_calib_frames(store, cfg.artifact_bucket, video_id, frames_dir, picks)

    still_fields = {
        "frame_uri": still_path,
        "frame_s": round(still_t or 0.0, 3),
        "image_size": {"w": width, "h": height},
        "frame_uris": frame_uris,
    }
    existing = store.get_calibration(video_id)
    if existing:
        # Re-ingest: refresh the still/dims but PRESERVE confirmed corners + status (don't clobber
        # a calibration the trainer already finalized).
        store.update_calibration(existing["id"], still_fields)
    else:
        store.insert_calibration(
            {"video_id": video_id, "match_id": match_id, "status": "needs_review",
             "source": "ingest", **still_fields}
        )

    ctx["frame_cache"] = str(cache_dir)
    ctx["fps"] = fps
    ctx["duration_s"] = duration_s
    ctx["width"] = width
    ctx["height"] = height
    log.info(
        "ingest done: %d frames @ %.1f fps cached, %dx%d, %.1fs, still@%.1fs, %d calib snapshots",
        kept,
        analysis_fps,
        width,
        height,
        duration_s,
        still_t or 0.0,
        len(frame_uris),
    )
