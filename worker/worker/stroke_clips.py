"""Export stage-6 hit events as short real-video clips + a Label Studio task list.

Stroke classification (stage 7, specs/features/stroke-classification.md) is gated on labels, and a
human tells shot types apart from swing shape, shuttle arc AND sound — so the labeler gets real
video with audio cut from the existing 720p proxy (specs/features/video-proxy.md keeps the audio
for exactly this kind of use), not a skeleton render. One clip per provisional hit event, plus a
`tasks_<game>.json` in Label Studio import format whose task data carries the DB ids
(event/video/match) so the later import step joins labels back to rows without guesswork.

Clips are cut with a re-encode: the proxy has a keyframe only every 2 s (`-g 60` at 30 fps), so a
stream-copy cut would snap up to 2 s away from the hit. `-ss` before `-i` keeps the seek fast and,
because we re-encode, frame-accurate. ~740 clips take a few minutes on the M4 Max.

Pure mapping (window clamping, rally filtering, filenames, task dicts, ffmpeg argv) is TDD'd in
tests/test_stroke_clips.py; the I/O shell below is smoke-tested on the real proxies.
(specs/features/stroke-labeling.md)
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger("worker.stroke_clips")

DEFAULT_PRE_S = 0.7    # wider than rally_hit's ±0.5 s search margin: the labeler needs the full
DEFAULT_POST_S = 0.7   # backswing-through-follow-through, not just the contact instant
CUT_TIMEOUT_S = 120
SERVE_HINT = "Erster Schlag im Ballwechsel (vermutlich Aufschlag)"


# --- pure mapping helpers (TDD) -------------------------------------------------------------------

def clip_window(t_s: float, pre: float, post: float, duration_s: Optional[float]) -> Tuple[float, float]:
    """(start, duration) of the cut around a hit, clamped to [0, video duration].

    duration_s comes from match_videos.duration_s and is nullable — without it only the start
    can be clamped (ffmpeg tolerates -t past EOF anyway; the clamp just keeps metadata honest)."""
    start = max(0.0, t_s - pre)
    end = t_s + post
    if duration_s is not None:
        end = min(end, float(duration_s))
    return start, max(0.0, end - start)


def hits_in_rallies(events: Sequence[dict], rallies: Sequence[dict]) -> List[dict]:
    """Hit events inside a current rally span, time-sorted, annotated for labeling.

    Rally-review edits may have dropped rallies since stage 6 wrote the events — a hit outside
    every remaining rally would become a junk clip, so it is filtered here (boundaries inclusive).
    Returns [{event, rally_idx, hit_idx (1-based within rally), first_of_rally}]; only the
    earliest hit of a rally carries the serve hint downstream."""
    hits: List[dict] = []
    per_rally: dict = {}
    for ev in sorted((e for e in events if e.get("type") == "hit"), key=lambda e: float(e["t_s"])):
        t = float(ev["t_s"])
        rally = next((r for r in rallies
                      if float(r["t_start_s"]) <= t <= float(r["t_end_s"])), None)
        if rally is None:
            continue
        idx = int(rally["idx"])
        per_rally[idx] = per_rally.get(idx, 0) + 1
        hits.append({"event": ev, "rally_idx": idx, "hit_idx": per_rally[idx],
                     "first_of_rally": per_rally[idx] == 1})
    return hits


def sanitize_label(label: str) -> str:
    """Video label → filesystem-safe stem ('Session2_Game_1.mp4' → 'Session2_Game_1')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(label).stem).strip("_")


def clip_filename(game_label: str, *, rally_idx: int, hit_idx: int, t_s: float, event_id: str) -> str:
    """Sortable, self-describing clip name; zero-padded so lexicographic == chronological."""
    return (f"{sanitize_label(game_label)}_r{rally_idx:02d}_h{hit_idx:02d}"
            f"_{t_s:07.2f}s_{event_id[:8]}.mp4")


def task_dict(hit: dict, *, game: str, clip_relpath: str, video_id: str, match_id: str) -> dict:
    """One Label Studio import task. The serve hint is display-only, never an auto-filled answer
    (pre-filling would bias the labeler toward 'Aufschlag' on every rally-start false positive)."""
    ev = hit["event"]
    return {"data": {
        "video": f"/data/local-files/?d={clip_relpath}",
        "game": game,
        "rally_idx": hit["rally_idx"],
        "hit_idx": hit["hit_idx"],
        "hit_t_s": float(ev["t_s"]),
        "first_of_rally": hit["first_of_rally"],
        "serve_hint": SERVE_HINT if hit["first_of_rally"] else "",
        "event_id": ev["id"],
        "video_id": video_id,
        "match_id": match_id,
    }}


def ffmpeg_cmd(src, dst, *, start: float, dur: float) -> list[str]:
    """Frame-accurate cut with audio. Re-encode (house VideoToolbox recipe, pipeline/proxy.py) —
    NOT -c copy, which would snap the cut to the proxy's 2 s keyframe grid."""
    return [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{dur:.3f}",
        "-c:v", "h264_videotoolbox", "-b:v", "2500k",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(dst),
    ]


def sample_evenly(items: Sequence, n: int) -> list:
    """Deterministic even spread across the list (reproducible pilot batches), ends included."""
    items = list(items)
    if n <= 0:
        return []
    if n >= len(items):
        return items
    if n == 1:
        return [items[0]]
    picked = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in picked]


# --- export shell (DB + ffmpeg I/O; smoke-tested on the real proxies) ------------------------------

def export_video(*, store, cfg, video_id: str, out_dir, pre: float, post: float,
                 sample: Optional[int] = None, limit: Optional[int] = None) -> dict:
    """Cut one game's hit clips into {out}/stroke/<game>/ and write {out}/tasks_<game>.json."""
    video = store.get_video(video_id)
    if not video:
        raise SystemExit(f"no match_videos row for video_id={video_id}")
    proxy = Path(cfg.frame_cache_dir).expanduser() / str(video_id) / "proxy_720p.mp4"
    if not proxy.exists():
        raise SystemExit(f"no local proxy at {proxy} — run the proxy stage for this video first")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH (brew install ffmpeg)")

    match_id = video["match_id"]
    game = sanitize_label(video.get("label") or str(video_id))
    events = store.get_events(match_id)
    rallies = store.get_rallies(match_id)
    hits = hits_in_rallies(events, rallies)
    n_hit_events = sum(1 for e in events if e.get("type") == "hit")
    skipped = n_hit_events - len(hits)
    if skipped:
        log.info("%s: %d of %d hit events fall outside current rally spans — skipped",
                 game, skipped, n_hit_events)
    if sample:
        hits = sample_evenly(hits, sample)
    if limit:
        hits = hits[:limit]

    duration = video.get("duration_s")
    duration = float(duration) if duration is not None else None
    game_dir = Path(out_dir) / "stroke" / game
    game_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[dict] = []
    failed = 0
    for hit in hits:
        ev = hit["event"]
        start, dur = clip_window(float(ev["t_s"]), pre, post, duration)
        fname = clip_filename(game, rally_idx=hit["rally_idx"], hit_idx=hit["hit_idx"],
                              t_s=float(ev["t_s"]), event_id=str(ev["id"]))
        res = subprocess.run(ffmpeg_cmd(proxy, game_dir / fname, start=start, dur=dur),
                             capture_output=True, timeout=CUT_TIMEOUT_S)
        if res.returncode != 0:
            failed += 1
            tail = (res.stderr or b"")[-300:].decode(errors="replace")
            log.warning("ffmpeg failed for %s (exit %d): %s", fname, res.returncode, tail)
            continue
        tasks.append(task_dict(hit, game=game, clip_relpath=f"stroke/{game}/{fname}",
                               video_id=str(video_id), match_id=str(match_id)))

    tasks_path = Path(out_dir) / f"tasks_{game}.json"
    tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    summary = {"game": game, "clips": len(tasks), "failed": failed,
               "skipped_outside_rallies": skipped,
               "clip_dir": str(game_dir), "tasks_json": str(tasks_path)}
    log.info("exported %d clips (%d failed) → %s", len(tasks), failed, game_dir)
    return summary


def export_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="worker export-stroke-clips",
        description="Cut a labeling clip (real video + audio, from the 720p proxy) around every "
                    "stage-6 hit event and write a Label Studio task list.")
    ap.add_argument("--video-id", required=True, action="append",
                    help="match_videos.id (repeatable for several games)")
    ap.add_argument("--out", help="labeling root (default: <frame_cache>/../.labeling; "
                                  "point LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT here)")
    ap.add_argument("--pre", type=float, default=DEFAULT_PRE_S, help="seconds before the hit")
    ap.add_argument("--post", type=float, default=DEFAULT_POST_S, help="seconds after the hit")
    ap.add_argument("--sample", type=int, help="cut only N evenly spread hits (pilot batches)")
    ap.add_argument("--limit", type=int, help="cut at most N hits")
    args = ap.parse_args(argv)

    cfg = load_config()
    out_dir = Path(args.out) if args.out else Path(cfg.frame_cache_dir).expanduser().parent / ".labeling"

    from .__main__ import _make_store

    store = _make_store(cfg)
    total = 0
    for vid in args.video_id:
        s = export_video(store=store, cfg=cfg, video_id=vid, out_dir=out_dir,
                         pre=args.pre, post=args.post, sample=args.sample, limit=args.limit)
        print(f"{s['game']}: {s['clips']} clips ({s['failed']} failed) → {s['clip_dir']}")
        print(f"  tasks: {s['tasks_json']}")
        total += s["clips"]
    print(f"total: {total} clips under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_cli())
