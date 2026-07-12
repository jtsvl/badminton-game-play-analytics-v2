"""stroke.traj — hit-to-hit trajectory-geometry features for the GBDT stroke spike.

Spec: features/stroke-classification.md, "Alternative candidate: trajectory-geometry GBDT".
Per labeled hit one flat feature row from the cache artifacts: 24 core shuttle/timing features
plus 3 optional label-free player-position features (``ppos_``, ablation-only). Segment
boundaries are the *detected* hits inside the rally span (``hits_in_rallies`` semantics,
inference-consistent — the 43 no_hit false positives stay in); the last hit falls back to
``rally.t_end_s``. Court coords are projected at anchor points only (the shuttle parquet's
in-flight u/v are airborne-projection garbage); no interpolation anywhere — endpoints and apex
are read off real detections. Degradation is per-feature (None -> parquet null -> XGBoost NaN
routing), never per-row: all 695 label rows emit a vector. The human ``hitter`` label is
meta-only and never feeds a feature.

Pure stdlib math by design (torch- and numpy-free at module top, like ``stroke/train.py``):
polars appears only inside the exporter shell.

    python -m worker export-stroke-traj [--labels P] [--out FILE] [--cache DIR] [--no-homography]
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence

from ..geometry import apply_homography
from ..stroke_clips import hits_in_rallies

log = logging.getLogger("worker.stroke.traj")

ANCHOR_MAX_DT = 0.1    # nearest-t tolerance, same convention as features.POSITION_MAX_DT
LABEL_MATCH_DT = 0.05  # label t_s -> detected hit event (float-exact today; tolerant for re-runs)
UV_VALID = (-1.0, 2.0)  # court-projection blow-up guard for elevated/degenerate points

FEATURE_COLS = (
    # A rally context
    "first_of_rally", "hit_idx_in_rally", "out_end_is_rally_end",
    # B timing
    "flight_time_out", "flight_time_in",
    # C position at hit (court)
    "hit_dist_net", "hit_u_center_dist", "hit_v_raw",
    # D outgoing displacement (court)
    "out_end_dist_net", "out_dv_abs", "out_du_abs", "out_crossed_net", "out_speed_uv",
    # E incoming displacement (court)
    "in_dv_abs", "in_speed_uv",
    # D'/E' image-space displacement (user-approved 2026-07-12: floor projection of elevated
    # contact points is garbage for ~63% of hits — picture units are always available)
    "out_dpxn_abs", "out_dpyn_abs", "out_speed_img", "in_speed_img",
    # F image-y arc
    "hit_pxn", "hit_pyn", "out_apex_rise", "out_apex_t_frac", "out_end_drop",
    "out_py_slope_early", "in_apex_rise",
    # G segment quality
    "n_dets_out", "n_dets_in",
)
PPOS_COLS = ("ppos_min_dist_same_half", "ppos_min_dist_other_half", "ppos_n_slots_filled")

_META_SCHEMA = {"video_id": "Utf8", "t_s": "Float64", "event_id": "Utf8", "match_id": "Utf8",
                "shot_type": "Utf8", "hitter": "Utf8", "rally_idx": "Int64", "hit_idx": "Int64",
                "is_calibration": "Boolean"}
_PROV_SCHEMA = {"prov_homography_ok": "Boolean", "prov_anchor_dt": "Float64",
                "prov_out_boundary": "Utf8", "prov_rally_found": "Boolean",
                "prov_hit_idx_recomputed": "Int64"}


# --- pure helpers (TDD) ---------------------------------------------------------------------------

def _dedupe(shuttle: Sequence[dict]) -> List[dict]:
    """Time-sorted detections, one row per t (max conf wins)."""
    best: dict = {}
    for d in shuttle:
        t = float(d["t"])
        if t not in best or float(d["conf"]) > float(best[t]["conf"]):
            best[t] = d
    return [best[t] for t in sorted(best)]


def _nearest(rows: Sequence[dict], t: float, max_dt: float = ANCHOR_MAX_DT) -> Optional[dict]:
    cand = min(rows, key=lambda d: abs(float(d["t"]) - t), default=None)
    if cand is None or abs(float(cand["t"]) - t) > max_dt:
        return None
    return cand


def _project(homography, px: float, py: float):
    """Court (u,v) of an image point, or None — horizon and blow-up degrade, never raise."""
    if homography is None:
        return None
    try:
        u, v = apply_homography(homography, px, py)
    except ValueError:
        return None
    if not (UV_VALID[0] <= u <= UV_VALID[1] and UV_VALID[0] <= v <= UV_VALID[1]):
        return None
    return u, v


def _boundaries(t0: float, events: Sequence[dict], rallies: Sequence[dict]) -> Optional[dict]:
    """This hit's segment boundaries among the *detected* hits of its rally (span lookup —
    rally_idx renumbers under review edits). None when no detected hit matches the label."""
    hits = hits_in_rallies(events, rallies)
    mine = min(hits, key=lambda h: abs(float(h["event"]["t_s"]) - t0), default=None)
    if mine is None or abs(float(mine["event"]["t_s"]) - t0) > LABEL_MATCH_DT:
        return None
    rally = next(r for r in rallies if int(r["idx"]) == mine["rally_idx"])
    same = [h for h in hits if h["rally_idx"] == mine["rally_idx"]]
    prev = next((h for h in same if h["hit_idx"] == mine["hit_idx"] - 1), None)
    nxt = next((h for h in same if h["hit_idx"] == mine["hit_idx"] + 1), None)
    return {
        "hit_idx": mine["hit_idx"],
        "first_of_rally": mine["first_of_rally"],
        "t_prev": float(prev["event"]["t_s"]) if prev else None,
        "t1": float(nxt["event"]["t_s"]) if nxt else float(rally["t_end_s"]),
        "out_boundary": "hit" if nxt else "rally_end",
    }


def build_feature_row(label: dict, *, events: Sequence[dict], rallies: Sequence[dict],
                      shuttle: Sequence[dict], tracks: Optional[Sequence[dict]] = None,
                      frame_w: float, frame_h: float, homography=None) -> dict:
    """One feature row per labeled hit. Every FEATURE_COLS/PPOS_COLS key is always present;
    a value is None wherever its inputs are missing (per-feature degradation, never per-row)."""
    t0 = float(label["t_s"])
    row: dict = {k: label.get(k) for k in
                 ("video_id", "t_s", "event_id", "match_id", "shot_type", "hitter",
                  "rally_idx", "hit_idx")}
    row["is_calibration"] = label.get("shot_type") == "unknown"
    row.update({c: None for c in FEATURE_COLS + PPOS_COLS})

    b = _boundaries(t0, events, rallies)
    sh = _dedupe(shuttle)
    anchor = _nearest(sh, t0)
    t1 = b["t1"] if b else None
    t_prev = b["t_prev"] if b else None
    end_row = _nearest(sh, t1) if t1 is not None else None
    prev_row = _nearest(sh, t_prev) if t_prev is not None else None

    row["prov_rally_found"] = b is not None
    row["prov_hit_idx_recomputed"] = b["hit_idx"] if b else None
    row["prov_out_boundary"] = b["out_boundary"] if b else None
    row["prov_homography_ok"] = homography is not None
    row["prov_anchor_dt"] = abs(float(anchor["t"]) - t0) if anchor else None

    # A + B — mechanical, homography- and detection-free
    if b:
        row["first_of_rally"] = 1.0 if b["first_of_rally"] else 0.0
        row["hit_idx_in_rally"] = float(b["hit_idx"])
        row["out_end_is_rally_end"] = 1.0 if b["out_boundary"] == "rally_end" else 0.0
        row["flight_time_out"] = t1 - t0
        row["flight_time_in"] = (t0 - t_prev) if t_prev is not None else None
        row["n_dets_out"] = float(sum(1 for d in sh if t0 <= float(d["t"]) <= t1))
        if t_prev is not None:
            row["n_dets_in"] = float(sum(1 for d in sh if t_prev <= float(d["t"]) < t0))

    # C + image position at the hit
    pt0 = None
    if anchor:
        px0, py0 = float(anchor["px"]), float(anchor["py"])
        row["hit_pxn"], row["hit_pyn"] = px0 / frame_w, py0 / frame_h
        pt0 = _project(homography, px0, py0)
        if pt0:
            u0, v0 = pt0
            row["hit_dist_net"] = abs(v0 - 0.5)
            row["hit_u_center_dist"] = abs(u0 - 0.5)
            row["hit_v_raw"] = v0

    # D — outgoing displacement, anchor -> endpoint detection
    pt1 = (_project(homography, float(end_row["px"]), float(end_row["py"]))
           if end_row else None)
    if pt0 and pt1:
        (u0, v0), (u1, v1) = pt0, pt1
        row["out_end_dist_net"] = abs(v1 - 0.5)
        row["out_dv_abs"] = abs(v1 - v0)
        row["out_du_abs"] = abs(u1 - u0)
        row["out_crossed_net"] = 1.0 if (v0 - 0.5) * (v1 - 0.5) < 0 else 0.0
        if t1 is not None and t1 > t0:
            row["out_speed_uv"] = math.hypot(u1 - u0, v1 - v0) / (t1 - t0)

    # E — incoming displacement, previous anchor -> this anchor
    pt_prev = (_project(homography, float(prev_row["px"]), float(prev_row["py"]))
               if prev_row else None)
    if pt0 and pt_prev and t_prev is not None and t0 > t_prev:
        (u0, v0), (up, vp) = pt0, pt_prev
        row["in_dv_abs"] = abs(v0 - vp)
        row["in_speed_uv"] = math.hypot(u0 - up, v0 - vp) / (t0 - t_prev)

    # D'/E' — image-space displacement, homography-free (picture units, trees don't need meters)
    if anchor and end_row:
        dxn = (float(end_row["px"]) - float(anchor["px"])) / frame_w
        dyn = (float(end_row["py"]) - float(anchor["py"])) / frame_h
        row["out_dpxn_abs"], row["out_dpyn_abs"] = abs(dxn), abs(dyn)
        if t1 is not None and t1 > t0:
            row["out_speed_img"] = math.hypot(dxn, dyn) / (t1 - t0)
    if anchor and prev_row and t_prev is not None and t0 > t_prev:
        row["in_speed_img"] = math.hypot(
            (float(anchor["px"]) - float(prev_row["px"])) / frame_w,
            (float(anchor["py"]) - float(prev_row["py"])) / frame_h) / (t0 - t_prev)

    # F — image-y arc over the outgoing segment (real detections only, first-argmin ties)
    if anchor and t1 is not None:
        t_a = float(anchor["t"])
        upper = float(end_row["t"]) if end_row else t1
        seg = [d for d in sh if t_a <= float(d["t"]) <= upper]
        if seg:
            apex = min(seg, key=lambda d: float(d["py"]))  # min is first-argmin on ties
            min_py, t_min = float(apex["py"]), float(apex["t"])
            row["out_apex_rise"] = (float(anchor["py"]) - min_py) / frame_h
            if len(seg) >= 2 and t1 > t0:
                row["out_apex_t_frac"] = (t_min - t0) / (t1 - t0)
            if end_row:
                row["out_end_drop"] = (float(end_row["py"]) - min_py) / frame_h
            interior = next((d for d in seg if float(d["t"]) > t_a), None)
            if interior:
                dt = float(interior["t"]) - t_a
                row["out_py_slope_early"] = ((float(interior["py"]) - float(anchor["py"]))
                                             / dt / frame_h)
    if prev_row:
        upper_in = float(anchor["t"]) if anchor else t0
        seg_in = [d for d in sh if float(prev_row["t"]) <= float(d["t"]) <= upper_in]
        if seg_in:
            min_py_in = min(float(d["py"]) for d in seg_in)
            row["in_apex_rise"] = (float(prev_row["py"]) - min_py_in) / frame_h

    # P — label-free scene geometry (no who-hit decision anywhere)
    if tracks is not None:
        near_rows = [r for r in tracks if abs(float(r["t"]) - t0) <= ANCHOR_MAX_DT]
        row["ppos_n_slots_filled"] = float(len({r["role"] for r in near_rows}))
        if pt0:
            u0, v0 = pt0
            same, other = [], []
            for r in near_rows:
                d = math.hypot(float(r["u"]) - u0, float(r["v"]) - v0)
                (same if (float(r["v"]) > 0.5) == (v0 > 0.5) else other).append(d)
            row["ppos_min_dist_same_half"] = min(same) if same else None
            row["ppos_min_dist_other_half"] = min(other) if other else None
    return row


# --- exporter shell -------------------------------------------------------------------------------

def build_video_rows(cache: Path, labels: Sequence[dict], homography=None) -> List[dict]:
    """All feature rows for one video from its cache artifacts. Missing parquets degrade
    per-feature; a missing frames.json skips the video loudly (no W/H to normalize with)."""
    import polars as pl  # lazy, matches the pipeline stages

    manifest_path = cache / "frames.json"
    if not manifest_path.exists():
        log.warning("stroke traj: no frames.json under %s — skipping %d labels",
                    cache, len(labels))
        return []
    manifest = json.loads(manifest_path.read_text())
    frame_w, frame_h = float(manifest["width"]), float(manifest["height"])

    def _rows(name: str):
        p = cache / name
        return list(pl.read_parquet(p).iter_rows(named=True)) if p.exists() else None

    events = _rows("events.parquet") or []
    rallies = _rows("rallies.parquet") or []
    shuttle = _rows("shuttle.parquet") or []
    tracks = _rows("tracks.parquet")  # None = file absent -> ppos_ stays None

    return [build_feature_row(lab, events=events, rallies=rallies, shuttle=shuttle,
                              tracks=tracks, frame_w=frame_w, frame_h=frame_h,
                              homography=homography)
            for lab in labels]


def rows_to_frame(rows: Sequence[dict]):
    """Feature rows -> typed polars frame (the _META_SCHEMA convention of stroke/dataset.py)."""
    import polars as pl

    schema = {**_META_SCHEMA, **{c: "Float64" for c in FEATURE_COLS + PPOS_COLS},
              **_PROV_SCHEMA}
    return pl.DataFrame(list(rows), schema={k: getattr(pl, v) for k, v in schema.items()},
                        strict=False)


def export_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from ..config import load_config
    from .dataset import _fetch_homographies

    ap = argparse.ArgumentParser(prog="worker export-stroke-traj",
                                 description="Hit-to-hit trajectory-geometry features "
                                             "for the GBDT stroke spike.")
    ap.add_argument("--labels", help="stroke_labels.parquet (default: worker/.labeling next to the cache)")
    ap.add_argument("--out", help="output parquet (default: <labels dir>/stroke_traj/features.parquet)")
    ap.add_argument("--cache", help="frame cache root (default: WORKER_FRAME_CACHE_DIR)")
    ap.add_argument("--no-homography", action="store_true",
                    help="skip the DB calibration fetch (court features become null)")
    args = ap.parse_args(argv)

    import polars as pl

    cfg = load_config()
    cache_root = Path(args.cache or cfg.frame_cache_dir).expanduser()
    labels_path = (Path(args.labels).expanduser() if args.labels
                   else cache_root.parent / ".labeling" / "stroke_labels.parquet")
    out_path = (Path(args.out).expanduser() if args.out
                else labels_path.parent / "stroke_traj" / "features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(pl.read_parquet(labels_path).iter_rows(named=True))
    video_ids = sorted({lab["video_id"] for lab in labels})
    homographies = {} if args.no_homography else _fetch_homographies(video_ids)

    rows: List[dict] = []
    for video_id in video_ids:
        vid_labels = [lab for lab in labels if lab["video_id"] == video_id]
        vid_rows = build_video_rows(cache_root / str(video_id), vid_labels,
                                    homography=homographies.get(video_id))
        rows.extend(vid_rows)
        print(f"{video_id}: {len(vid_rows)}/{len(vid_labels)} rows")

    if not rows:
        print("no rows built — check cache dirs")
        return 1

    df = rows_to_frame(rows)
    df.write_parquet(out_path)

    train = df.filter(~pl.col("is_calibration"))
    print(f"\n{df.height} rows -> {out_path} "
          f"({train.height} train, {df.height - train.height} calibration)")
    found = df.filter(pl.col("prov_rally_found")).height
    by_boundary = dict(df.group_by("prov_out_boundary").len().iter_rows())
    print(f"rally found: {found}/{df.height}  boundaries: {by_boundary}")
    print(f"homography ok: {df.filter(pl.col('prov_homography_ok')).height}/{df.height}")
    print("null rates:")
    for col in FEATURE_COLS + PPOS_COLS:
        n = df[col].null_count()
        if n:
            print(f"  {col}: {n}/{df.height} ({n / df.height:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_cli())
