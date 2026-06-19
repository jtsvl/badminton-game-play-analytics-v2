"""Correction 'scorecard' — turn verify-tool corrections into a near/far accuracy report.

Pure scoring functions (no numpy/cv2/torch, so they test under the dev extra). The trainer's
`corrections.json` is the answer key: every `bbox`/`position` correction carries BOTH the pipeline's
original foot (`orig_bbox`/`orig_foot`/`orig_uv`) and the corrected foot (`foot_norm`/`court_uv`), so
the error at each flagged point is computable directly — no model re-run needed for a baseline.

Errors are reported split **near** (label `A*`) vs **far** (label `B*`): far-court foot error is
geometrically amplified by the homography (see specs/fix-later.md), so the split is where the signal is.

Run a baseline straight from a corrections file:
    uv run python -m worker.eval_corrections --corrections /path/corrections.json --width 3840 --height 2160
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .geometry import ankle_point, apply_homography, foot_point
from .insights import COURT_L_M, COURT_W_M

Pt = Tuple[float, float]
Key = Tuple[int, str]

# Track role (gate_track) -> player label the corrections use (A* = near, B* = far). Mirrors
# gate_track.ROLE_TO_SIDESLOT (side + slot): label = side + str(slot + 1).
ROLE_TO_LABEL: Dict[str, str] = {
    "near": "A1", "far": "B1",
    "near_left": "A1", "near_right": "A2",
    "far_left": "B1", "far_right": "B2",
}


def half_of(label: str) -> str:
    """Court half a player label belongs to: 'A*' = near, 'B*' = far."""
    if label.startswith("A"):
        return "near"
    if label.startswith("B"):
        return "far"
    raise ValueError(f"unknown player label: {label!r}")


def foot_px_error(pf: Pt, gf: Pt, w: int, h: int) -> float:
    """Euclidean distance (frame pixels) between a pipeline foot and the ground-truth foot,
    both normalized (x,y) in [0,1]."""
    return math.hypot((pf[0] - gf[0]) * w, (pf[1] - gf[1]) * h)


def court_m_error(puv: Pt, guv: Pt, w_m: float = COURT_W_M, l_m: float = COURT_L_M) -> float:
    """Euclidean distance (metres) between two court positions (u,v) in [0,1]."""
    return math.hypot((puv[0] - guv[0]) * w_m, (puv[1] - guv[1]) * l_m)


def _percentile(sorted_xs: List[float], p: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    rank = (p / 100.0) * (len(sorted_xs) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_xs[int(lo)]
    frac = rank - lo
    return sorted_xs[int(lo)] * (1 - frac) + sorted_xs[int(hi)] * frac


def summarize(values: Sequence[float]) -> Dict[str, object]:
    """n / mean / median / p90 of a list of errors (None stats when empty)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    xs = sorted(values)
    return {"n": n, "mean": sum(xs) / n, "median": _percentile(xs, 50.0), "p90": _percentile(xs, 90.0)}


def _foot_from_bbox(bbox: Sequence[float]) -> Pt:
    """Bottom-centre foot of a (normalized) bbox — ankle_point is linear so units don't matter."""
    return ankle_point((bbox[0], bbox[1], bbox[2], bbox[3]))


def ground_truth(corrections: Sequence[dict]) -> Dict[Key, dict]:
    """Map (frame_idx, label) -> corrected truth: foot/uv (normalized) + miss/spurious flags.

    `miss` marks an 'added' correction (the pipeline detected nobody there). Precedence mirrors the
    verify tool's resolve_cors: an explicit corrected box wins, else the latest foot_norm.
    """
    out: Dict[Key, dict] = {}
    for c in corrections:
        label = c.get("role")
        if not label:
            continue
        key = (int(c["frame_idx"]), label)
        o = out.setdefault(key, {"foot": None, "uv": None, "miss": False, "spurious": False})
        ctype = c.get("type")
        if ctype == "spurious":
            o["spurious"] = True
        if ctype == "added":
            o["miss"] = True
        if c.get("bbox"):
            o["foot"] = _foot_from_bbox(c["bbox"])
        elif c.get("foot_norm"):
            o["foot"] = tuple(c["foot_norm"])
        if c.get("court_uv"):
            o["uv"] = tuple(c["court_uv"])
    return out


def pipeline_baseline(corrections: Sequence[dict]) -> Dict[Key, dict]:
    """Map (frame_idx, label) -> the pipeline's ORIGINAL position from the orig_* fields a
    correction carries. 'added' (a miss) yields no baseline entry — the pipeline produced nothing."""
    out: Dict[Key, dict] = {}
    for c in corrections:
        label = c.get("role")
        if not label or c.get("type") == "added":
            continue
        key = (int(c["frame_idx"]), label)
        o = out.setdefault(key, {"foot": None, "uv": None})
        if c.get("orig_bbox"):
            o["foot"] = _foot_from_bbox(c["orig_bbox"])
        elif c.get("orig_foot"):
            o["foot"] = tuple(c["orig_foot"])
        if c.get("orig_uv"):
            o["uv"] = tuple(c["orig_uv"])
    return out


def track_labels(track_rows: Sequence[dict]) -> Dict[int, str]:
    """Map each pipeline track_id -> player label (A1/A2/B1/B2) via its assigned role, so detections
    (keyed by track) can be matched to the trainer's role-labelled corrections. Rows for a track repeat
    its role; the first known role wins. Tracks with an unknown/empty role are skipped."""
    out: Dict[int, str] = {}
    for r in track_rows:
        tid = r.get("track_id")
        if tid is None or tid in out:
            continue
        label = ROLE_TO_LABEL.get(r.get("role") or "")
        if label:
            out[int(tid)] = label
    return out


def pipeline_from_detections(
    det_rows: Sequence[dict], labels: Dict[int, str], H: Sequence[float],
    foot_k: float, w: int, h: int,
    pose_feet: Dict[Tuple[int, int], Tuple] | None = None, pose_conf_thr: float = 0.3,
) -> Dict[Key, dict]:
    """Build the CURRENT pipeline position map from the real detections + DB homography, mirroring
    gate_track's foot resolution: the confident pose ankle (`pose_feet[(frame,track)] = (x,y,conf)`)
    when available, else `foot_point(bbox, foot_k)`. This is the 'after' the baseline (orig_* in the
    JSON) must be compared to.

    Returns (frame, label) -> {foot: normalized (nx,ny), uv: court (u,v)}. Detections with no track id
    or a track not mapped to a label (off-court / dropped) are skipped."""
    out: Dict[Key, dict] = {}
    for r in det_rows:
        tid = r.get("track")
        if tid is None:
            continue
        label = labels.get(int(tid))
        if not label:
            continue
        pf = pose_feet.get((int(r["frame"]), int(tid))) if pose_feet else None
        if pf is not None and pf[0] is not None and pf[2] >= pose_conf_thr:
            ax, ay = pf[0], pf[1]  # real ankle keypoint
        else:
            ax, ay = foot_point((r["x"], r["y"], r["w"], r["h"]), foot_k)  # bbox-nudge fallback
        out[(int(r["frame"]), label)] = {"foot": (ax / w, ay / h), "uv": apply_homography(H, ax, ay)}
    return out


def score(pipeline_map: Dict[Key, dict], gt_map: Dict[Key, dict], w: int, h: int) -> dict:
    """Compare a pipeline position map against the ground-truth map; bucket errors near/far.

    foot_px needs a foot in both maps; court_m needs a uv in both. Misses/spurious come from gt flags.
    """
    buckets = {"near": {"foot_px": [], "court_m": []}, "far": {"foot_px": [], "court_m": []}}
    matched = misses = spurious = 0
    for key, g in gt_map.items():
        half = half_of(key[1])
        if g.get("spurious"):
            spurious += 1
        if g.get("miss"):
            misses += 1
        p = pipeline_map.get(key)
        if not p:
            continue
        if g.get("foot") and p.get("foot"):
            buckets[half]["foot_px"].append(foot_px_error(p["foot"], g["foot"], w, h))
            matched += 1
        if g.get("uv") and p.get("uv"):
            buckets[half]["court_m"].append(court_m_error(p["uv"], g["uv"]))
    rep: dict = {}
    all_foot: List[float] = []
    all_court: List[float] = []
    for half in ("near", "far"):
        rep[half] = {"foot_px": summarize(buckets[half]["foot_px"]),
                     "court_m": summarize(buckets[half]["court_m"])}
        all_foot += buckets[half]["foot_px"]
        all_court += buckets[half]["court_m"]
    rep["overall"] = {"foot_px": summarize(all_foot), "court_m": summarize(all_court)}
    rep["counts"] = {"matched": matched, "misses": misses, "spurious": spurious, "gt_total": len(gt_map)}
    return rep


def _fmt(stat: Dict[str, object], unit: str) -> str:
    if not stat["n"]:
        return f"n=0 (no {unit} data)"
    return (f"n={stat['n']:>4}  mean={stat['mean']:6.1f}{unit}"
            f"  median={stat['median']:6.1f}{unit}  p90={stat['p90']:6.1f}{unit}")


def format_report(rep: dict) -> str:
    """Human-readable scorecard for the CLI."""
    lines = ["Detection scorecard (pipeline vs trainer corrections)", "=" * 54]
    for half in ("near", "far", "overall"):
        lines.append(f"[{half.upper():7}] foot  {_fmt(rep[half]['foot_px'], 'px')}")
        lines.append(f"          court {_fmt(rep[half]['court_m'], 'm')}")
    c = rep["counts"]
    lines.append("-" * 54)
    lines.append(f"matched={c['matched']}  misses(added)={c['misses']}  "
                 f"spurious={c['spurious']}  gt_total={c['gt_total']}")
    return "\n".join(lines)


def baseline_from_file(path: str, w: int, h: int) -> dict:
    """Load a corrections.json and score the pipeline's original output against the corrected truth —
    no model re-run, no DB. This is the baseline the later fixes must beat."""
    import json

    data = json.loads(open(path, encoding="utf-8").read())
    cors = data.get("corrections", [])
    return score(pipeline_baseline(cors), ground_truth(cors), w, h)


def full_from_pipeline(video_id: str, corrections_path: str | None = None,
                       foot_k: float | None = None) -> dict:
    """Score the CURRENT pipeline output against the corrected ground truth — the real-data scorecard.

    Loads the live `detections.parquet` + DB homography from the analytics Supabase and re-derives the
    foot exactly as the pipeline does now (`foot_point(bbox, FOOT_BIAS_K)` today; the pose foot once
    Phase 3 lands), so re-running this after each fix gives a real near/far cm number — unlike
    `baseline_from_file`, which only replays the orig_* fields frozen into the corrections JSON.
    `foot_k` defaults to the live `FOOT_BIAS_K` env so the score matches the running pipeline.
    """
    import json
    import os
    from pathlib import Path

    import polars as pl  # lazy: keeps the pure functions (+ their tests) numpy/polars-free

    from .__main__ import _make_store
    from .config import load_config

    cfg = load_config()
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    manifest = json.loads((cache / "frames.json").read_text())
    w, h = int(manifest["width"]), int(manifest["height"])

    cpath = Path(corrections_path) if corrections_path else cache / "corrections.json"
    cors = json.loads(cpath.read_text(encoding="utf-8")).get("corrections", [])
    gt = ground_truth(cors)

    cal = _make_store(cfg).get_calibration(video_id)
    if not cal or not cal.get("homography"):
        raise RuntimeError(f"eval_corrections full: no homography for {video_id} — calibrate must run first")
    H = cal["homography"]

    if foot_k is None:
        foot_k = float(os.environ.get("FOOT_BIAS_K", "0"))
    pose_conf_thr = float(os.environ.get("POSE_CONF_THR", "0.3"))
    use_pose = os.environ.get("USE_POSE_FOOT", "0").lower() not in ("0", "", "false", "no")

    # Pose foot (Phase 3) when enabled and the stage has run — mirrors gate_track's foot choice.
    pose_path = cache / "pose.parquet"
    pose_feet = None
    if use_pose and pose_path.exists():
        pose_feet = {(pr["frame"], pr["track"]): (pr["foot_x"], pr["foot_y"], pr["foot_conf"])
                     for pr in pl.read_parquet(pose_path).iter_rows(named=True)}

    labels = track_labels(pl.read_parquet(cache / "tracks.parquet").to_dicts())
    pm = pipeline_from_detections(pl.read_parquet(cache / "detections.parquet").to_dicts(),
                                  labels, H, foot_k, w, h, pose_feet, pose_conf_thr)
    rep = score(pm, gt, w, h)
    rep["meta"] = {"video_id": video_id, "foot_k": foot_k, "w": w, "h": h, "labels": labels,
                   "pose": bool(pose_feet), "pose_conf_thr": pose_conf_thr}
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Detection correction scorecard (near/far error).")
    ap.add_argument("--mode", choices=["baseline", "full"], default="baseline",
                    help="baseline = replay orig_* in the JSON (no DB); full = score the live pipeline output")
    ap.add_argument("--corrections", help="corrections.json answer key (baseline) / GT override (full)")
    ap.add_argument("--width", type=int, help="source frame width in px (baseline)")
    ap.add_argument("--height", type=int, help="source frame height in px (baseline)")
    ap.add_argument("--video-id", help="video to score against its cache + DB homography (full mode)")
    args = ap.parse_args(argv)

    if args.mode == "full":
        if not args.video_id:
            ap.error("--mode full requires --video-id")
        rep = full_from_pipeline(args.video_id, corrections_path=args.corrections)
        m = rep["meta"]
        foot_src = f"pose(thr={m['pose_conf_thr']})+bbox(k={m['foot_k']})" if m["pose"] else f"bbox(k={m['foot_k']})"
        print(f"[full mode] video={m['video_id']}  foot={foot_src}  "
              f"frame={m['w']}x{m['h']}  roles={m['labels']}")
        print(format_report(rep))
        return 0

    if not (args.corrections and args.width and args.height):
        ap.error("--mode baseline requires --corrections, --width, --height")
    print(format_report(baseline_from_file(args.corrections, args.width, args.height)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
