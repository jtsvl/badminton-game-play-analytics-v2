"""Shuttle bench — run a tracker over a cached clip and report auto-stats (no ground truth needed).

Slice-1 judges shuttle tracking by eye + automatic statistics (there's no shuttle answer key yet): the
flight-path overlay shows whether the dot follows the shuttle, and these numbers show WHERE it drops and
whether the detections are physically plausible / on the target court. `--compare` runs every backend
side by side so we can keep the winner. This is also the dev entrypoint (no job queue):

    uv run python -m worker.eval_shuttle --video-id <id> --tracker fake
    uv run python -m worker.eval_shuttle --video-id <id> --compare

Honest reminder: court (u,v) for an airborne shuttle is a ground-projection estimate — % inside court is
a multi-court-leakage check, not an accuracy score (a high shuttle legitimately projects past a line).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

from .insights import COURT_L_M, COURT_W_M

TRACKERS = ("wasb", "tracknetv3", "fake")


def run_stage(video_id: str, tracker: str | None = None) -> str:
    """Run the shuttle stage on the cached clip with the chosen backend; return shuttle.parquet path.

    Fetches the calibration itself and passes H + polygon in ctx (no `store` key) so the bench neither
    uploads artifacts nor needs a match_id — it just writes the local parquet."""
    if tracker:
        os.environ["SHUTTLE_TRACKER"] = tracker
    from .__main__ import _make_store
    from .config import load_config
    from .pipeline.shuttle import run as shuttle_run

    cfg = load_config()
    cal = _make_store(cfg).get_calibration(video_id)
    if not cal or not cal.get("homography"):
        raise RuntimeError(f"eval_shuttle: no homography for {video_id} — calibrate must run first")
    ctx = {
        "cfg": cfg,
        "job": {"video_id": video_id},
        "H": cal["homography"],
        "mask_poly_px": cal.get("mask_polygon_px") or cal.get("confirmed_points"),
    }
    shuttle_run(ctx)
    return ctx["shuttle_path"]


def _runs_of_missing(present_frames: Sequence[int], all_frames: Sequence[int]) -> List[int]:
    """Lengths of consecutive-missing runs between the first and last detection (interior dropouts)."""
    have = set(present_frames)
    inside = [f for f in all_frames if present_frames and min(present_frames) <= f <= max(present_frames)]
    runs, cur = [], 0
    for f in inside:
        if f in have:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return runs


def stats(parquet_path: str, manifest: dict, margin: float = 0.5) -> dict:
    """Auto-stats for one tracker run: coverage, dropout structure, motion plausibility, court-leakage."""
    import math

    import polars as pl

    frames = [int(f["frame"]) for f in manifest["frames"]]
    fps = float(manifest.get("analysis_fps") or 12.0)
    df = pl.read_parquet(parquet_path).sort("frame")
    rows = df.to_dicts()
    present = [int(r["frame"]) for r in rows]

    runs = _runs_of_missing(present, frames)
    longest = max(runs) if runs else 0

    disp = []
    for a, b in zip(rows, rows[1:]):
        disp.append(math.hypot((a["u"] - b["u"]) * COURT_W_M, (a["v"] - b["v"]) * COURT_L_M))
    disp.sort()

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0.0

    inside = sum(1 for r in rows if -margin <= r["u"] <= 1 + margin and -margin <= r["v"] <= 1 + margin)
    return {
        "tracker": rows[0]["tracker"] if rows else "?",
        "total_frames": len(frames),
        "kept": len(rows),
        "coverage_pct": 100.0 * len(rows) / max(len(frames), 1),
        "n_gaps": len(runs),
        "longest_gap_frames": longest,
        "longest_gap_s": round(longest / fps, 2),
        "disp_median_m": round(pct(disp, 50), 2),
        "disp_p90_m": round(pct(disp, 90), 2),
        "disp_max_m": round(disp[-1], 2) if disp else 0.0,
        "inside_court_pct": round(100.0 * inside / max(len(rows), 1), 1),
    }


def format_stats(s: dict) -> str:
    return (
        f"[{s['tracker']:>10}] coverage {s['coverage_pct']:5.1f}%  "
        f"({s['kept']}/{s['total_frames']})  gaps={s['n_gaps']}  "
        f"longest={s['longest_gap_frames']}f/{s['longest_gap_s']}s  "
        f"step m(med/p90/max)={s['disp_median_m']}/{s['disp_p90_m']}/{s['disp_max_m']}  "
        f"on-court={s['inside_court_pct']}%"
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Shuttle tracker bench (auto-stats, no ground truth).")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--tracker", choices=TRACKERS, help="single backend to run")
    ap.add_argument("--compare", action="store_true", help="run every backend and tabulate")
    ap.add_argument("--render", metavar="DIR",
                    help="draw raw detections in image space (scatter + key frames + overlay.mp4) to DIR")
    args = ap.parse_args(argv)

    if args.render:
        from .shuttle.render import render_overlay

        s = render_overlay(args.video_id, args.tracker or "tracknetv3", args.render)
        print(f"rendered {s['detections']}/{s['frames']} detections → {s['out_dir']}")
        print(f"  scatter: {s['scatter']}\n  video:   {s['video']}")
        return 0

    from .config import load_config

    cfg = load_config()
    cache = Path(cfg.frame_cache_dir).expanduser() / args.video_id
    manifest = json.loads((cache / "frames.json").read_text())

    backends = list(TRACKERS) if args.compare else [args.tracker or os.environ.get("SHUTTLE_TRACKER", "wasb")]
    results: Dict[str, dict] = {}
    print(f"shuttle bench — video={args.video_id}  frames={len(manifest['frames'])}")
    for name in backends:
        try:
            path = run_stage(args.video_id, tracker=name)
            s = stats(path, manifest)
            results[name] = s
            print(format_stats(s))
        except Exception as e:  # noqa: BLE001 — a missing-weights backend shouldn't abort the others
            print(f"[{name:>10}] FAILED: {e}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
