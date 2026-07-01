"""Backfill calibration snapshots for videos ingested before 0007_calibration_frames.

Ingest now uploads N evenly-spaced stills so the calibration screen can switch frames when a player
blocks a corner. Videos ingested earlier only have the single `calib_frame.jpg`; this reuses their
already-cached frames on disk (no re-decode) to upload the N snapshots and set `frame_uris`.

    python -m worker backfill-calib-frames <video_id> [<video_id> …] [--count 12]

Reuses the exact same pure selection + upload the ingest stage uses (worker/calib_frames.py) →
analysis-artifacts/<video_id>/calib_frames/NNN.jpg. Idempotent (re-uploads + overwrites frame_uris).
See supabase/migrations/0007_calibration_frames.sql.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from .calib_frames import select_calib_frames, upload_calib_frames

log = logging.getLogger("worker.backfill_calib_frames")


def backfill_one(store, cfg, video_id: str, count: int) -> int:
    """Upload N calibration snapshots for one video from its cache; set `frame_uris`. Returns N."""
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    manifest_path = cache / "frames.json"
    if not manifest_path.exists():
        log.warning("no frame cache for %s (%s) — skipping", video_id, manifest_path)
        return 0
    calib = store.get_calibration(video_id)
    if not calib:
        log.warning("no court_calibrations row for video_id=%s — skipping", video_id)
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["frames"]
    picks = select_calib_frames(manifest, count)
    frame_uris = upload_calib_frames(store, cfg.artifact_bucket, video_id, cache / "frames", picks)
    store.update_calibration(calib["id"], {"frame_uris": frame_uris})
    log.info("backfilled %d calib snapshots for %s", len(frame_uris), video_id)
    print(f"backfilled {len(frame_uris)} calibration snapshots for {video_id}")
    return len(frame_uris)


def backfill_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .__main__ import _make_store
    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="worker backfill-calib-frames",
        description="Upload N evenly-spaced calibration snapshots for already-ingested videos.",
    )
    ap.add_argument("video_ids", nargs="+", help="match_videos.id(s) whose cached frames to snapshot")
    ap.add_argument("--count", type=int, default=None, help="snapshots per video (default: CALIB_FRAME_COUNT / 12)")
    args = ap.parse_args(argv)

    cfg = load_config()
    count = args.count or cfg.calib_frame_count
    store = _make_store(cfg)
    for video_id in args.video_ids:
        backfill_one(store, cfg, video_id, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(backfill_cli())
