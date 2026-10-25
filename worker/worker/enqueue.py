"""`python -m worker enqueue` — wire an uploaded clip into a match + ingest job (P1 upload-UI stand-in).

Creates placeholder players + a match + a match_videos row pointing at the storage path + the
match_players roster + a queued `ingest` processing_job. Names are placeholders the trainer edits later.
Example:
  python -m worker enqueue --path doubles_match.MOV --discipline doubles --slug 2026-06-14-doppel
"""
from __future__ import annotations

import argparse
import logging

from .config import load_config
from .store import Store

log = logging.getLogger("worker.enqueue")

# side/slot roster per discipline; placeholder display names.
ROSTERS = {
    "singles": [("A", 0, "Spieler A", "A"), ("B", 0, "Spieler B", "B")],
    "doubles": [
        ("A", 0, "Spieler A1", "A1"),
        ("A", 1, "Spieler A2", "A2"),
        ("B", 0, "Spieler B1", "B1"),
        ("B", 1, "Spieler B2", "B2"),
    ],
}


def enqueue_cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="worker enqueue")
    ap.add_argument("--path", required=True, help="storage path within the video bucket")
    ap.add_argument("--slug", required=True, help="match slug (unique)")
    ap.add_argument("--discipline", choices=["singles", "doubles"], default="doubles")
    ap.add_argument("--bucket", default=None, help="override video bucket (default from env)")
    ap.add_argument("--label", default=None, help="video label")
    ap.add_argument("--angle", choices=["overhead", "side"], default="overhead")
    ap.add_argument("--played-on", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--max-frames", type=int, default=None, help="dev cap on decoded frames")
    from .pipeline.ingest import DEFAULT_ANALYSIS_FPS

    ap.add_argument("--analysis-fps", type=float, default=None,
                    help=f"requested frame-cache fps (default {DEFAULT_ANALYSIS_FPS:g}; the real cadence "
                         "snaps to the source's frames and is recorded as frames.json cadence_fps)")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not cfg.configured:
        raise SystemExit("not configured — fill SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env")
    store = Store(cfg.supabase_url, cfg.service_role_key).with_worker(cfg.worker_id)

    if store.get_match_by_slug(args.slug):
        raise SystemExit(f"match slug already exists: {args.slug}")

    # matches.played_on is a NOT NULL date with no default — supply a concrete date.
    from datetime import date

    extra = {"played_on": args.played_on or date.today().isoformat()}
    match = store.create_match(args.slug, args.discipline, **extra)
    match_id = match["id"]
    log.info("created match %s (%s)", args.slug, match_id)

    video = store.create_video(
        match_id,
        args.path,
        angle=args.angle,
        label=args.label or args.path,
    )
    video_id = video["id"]
    log.info("created match_videos %s -> %s", video_id, args.path)

    for side, slot, name, first in ROSTERS[args.discipline]:
        player = store.create_player(name, first)
        store.create_match_player(match_id, player["id"], side, slot)
    log.info("created %d placeholder players/roster", len(ROSTERS[args.discipline]))

    params: dict = {}
    if args.max_frames is not None:
        params["max_frames"] = args.max_frames
    if args.analysis_fps is not None:
        params["analysis_fps"] = args.analysis_fps
    job = store.enqueue(match_id, video_id, stage="ingest", params=params)
    log.info("queued ingest job %s", job.get("id"))
    print(f"OK: match={args.slug} video_id={video_id} job={job.get('id')} stage=ingest")
    print(f"Run the worker (python -m worker), then calibrate at /match/{args.slug}/calibrate")
