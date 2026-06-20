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


def score_against_labels(detections: Sequence[dict], labels: Sequence[dict], tol_px: float = 4.0) -> dict:
    """Raw-pixel precision/recall of tracker peaks vs. the labeled shuttle — the fair before/after metric.

    Both `detections` ({frame, x, y}) and `labels` ({frame, x, y, visibility}) must ALREADY be in the same
    pinned space (512x288). `labels` is the DENSE answer key (every frame in the labeled span, visibility 0
    or 1). A detection is a TP only if it lands within `tol_px` of a visibility=1 label on its frame; a peak
    on a visible frame but beyond tol is both a miss (FN) and a spurious peak (FP); a peak on an occluded
    (visibility=0) frame is a FP; a visible frame with no peak is a FN. Peaks OUTSIDE the labeled span have
    no ground truth and are ignored. `n` = visible-frame count (report it beside every rate — a small `n`
    makes a few-point delta meaningless). Pure; TDD'd in tests/test_eval_shuttle_score.py."""
    import math
    import statistics

    det = {int(d["frame"]): (float(d["x"]), float(d["y"])) for d in detections}
    lab = {int(row["frame"]): row for row in labels}
    visible = [f for f, row in lab.items() if int(row.get("visibility", 0)) == 1]

    tp = fp = fn = 0
    errors: List[float] = []
    for f, row in lab.items():
        has_det = f in det
        if int(row.get("visibility", 0)) == 1:
            if has_det:
                d = math.hypot(det[f][0] - float(row["x"]), det[f][1] - float(row["y"]))
                if d <= tol_px:
                    tp += 1
                    errors.append(d)
                else:
                    fn += 1   # the labeled shuttle was not found...
                    fp += 1   # ...and the peak that fired is spurious
            else:
                fn += 1
        elif has_det:             # occluded frame, yet a peak fired → false positive
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(visible),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_px_error": round(statistics.median(errors), 3) if errors else None,
    }


def format_score(name: str, s: dict) -> str:
    err = "n/a" if s["median_px_error"] is None else f"{s['median_px_error']:.2f}px"
    return (
        f"[{name:>10}] P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  "
        f"med-err {err}  (TP {s['tp']} FP {s['fp']} FN {s['fn']}; n={s['n']} visible)"
    )


def format_stats(s: dict) -> str:
    return (
        f"[{s['tracker']:>10}] coverage {s['coverage_pct']:5.1f}%  "
        f"({s['kept']}/{s['total_frames']})  gaps={s['n_gaps']}  "
        f"longest={s['longest_gap_frames']}f/{s['longest_gap_s']}s  "
        f"step m(med/p90/max)={s['disp_median_m']}/{s['disp_p90_m']}/{s['disp_max_m']}  "
        f"on-court={s['inside_court_pct']}%"
    )


def _score_backend(video_id: str, cfg, manifest: dict, parquet_path: str, tol_px: float):
    """Map a backend's parquet peaks + the clip's shuttle_labels into the pinned 512x288 space and score.

    Returns a score dict, or None if the clip has no shuttle_labels.json yet. The parquet's px,py are
    full-frame pixels (track.py), and the labels are normalized clicks — both go through
    frame_to_model_xy with the recomputed crop_box so the metric lives in ONE space (512x288, tol=4)."""
    import polars as pl

    from .__main__ import _make_store
    from .geometry import frame_to_model_xy
    from .shuttle.dataset_export import MODEL_H, MODEL_W, crop_geometry
    from .shuttle.label import interpolate_anchors

    labels_path = Path(cfg.frame_cache_dir).expanduser() / video_id / "shuttle_labels.json"
    if not labels_path.exists():
        return None

    cal = _make_store(cfg).get_calibration(video_id)
    poly = (cal or {}).get("mask_polygon_px") or (cal or {}).get("confirmed_points")
    if not poly:
        raise RuntimeError("eval_shuttle: no calibration polygon — calibrate must run first")
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])
    _, crop_box = crop_geometry(poly, frame_w, frame_h)
    fnum_by_pos = {i: int(f["frame"]) for i, f in enumerate(manifest["frames"])}

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = []
    for pos, px, py, vis, _interp in interpolate_anchors(raw.get("anchors", []), raw.get("boundaries", [])):
        frame = fnum_by_pos.get(int(pos), int(pos))
        if vis == 1 and px is not None and py is not None:
            X, Y = frame_to_model_xy(px * frame_w, py * frame_h, crop_box, MODEL_W, MODEL_H)
            labels.append({"frame": frame, "x": X, "y": Y, "visibility": 1})
        else:
            labels.append({"frame": frame, "visibility": 0})

    dets = []
    for r in pl.read_parquet(parquet_path).iter_rows(named=True):
        X, Y = frame_to_model_xy(float(r["px"]), float(r["py"]), crop_box, MODEL_W, MODEL_H)
        dets.append({"frame": int(r["frame"]), "x": X, "y": Y})

    return score_against_labels(dets, labels, tol_px=tol_px)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Shuttle tracker bench (auto-stats + optional GT scoreboard).")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--tracker", choices=TRACKERS, help="single backend to run")
    ap.add_argument("--compare", action="store_true", help="run every backend and tabulate")
    ap.add_argument("--labels", action="store_true",
                    help="also score precision/recall/F1 vs. shuttle_labels.json (512x288, tol=4px)")
    ap.add_argument("--tol-px", type=float, default=4.0,
                    help="match tolerance in the pinned 512x288 space (default 4, the train.py default)")
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
    scores: Dict[str, dict] = {}
    print(f"shuttle bench — video={args.video_id}  frames={len(manifest['frames'])}")
    for name in backends:
        try:
            path = run_stage(args.video_id, tracker=name)
            s = stats(path, manifest)
            results[name] = s
            print(format_stats(s))
            if args.labels:
                sc = _score_backend(args.video_id, cfg, manifest, path, args.tol_px)
                if sc is None:
                    print(f"[{name:>10}] no shuttle_labels.json — label a rally first (verify --shuttle)")
                else:
                    scores[name] = sc
                    print(format_score(name, sc))
        except Exception as e:  # noqa: BLE001 — a missing-weights backend shouldn't abort the others
            print(f"[{name:>10}] FAILED: {e}")
    if args.labels and scores:
        best = max(scores, key=lambda k: scores[k]["f1"])
        n = next(iter(scores.values()))["n"]
        print(f"\nscoreboard (tol={args.tol_px:g}px, 512x288): best F1 = {best}  ·  n={n} visible frames")
        print("  (a meaningful win is recall +~10pp at this tol; a few-point delta on a small n is noise)")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
