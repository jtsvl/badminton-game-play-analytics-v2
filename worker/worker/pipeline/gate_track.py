"""Stage: gate_track — project ankle points, gate to the court, enforce N tracks, assign roles.

Reads detections.parquet, projects each foot point through the calibration homography (image-px→u,v),
keeps only points inside the court polygon (+margin) — decisive when adjacent courts are in frame —
keeps the N most-present tracks, assigns persistent roles, and auto-maps tracks→players by role
(is_test clips auto-advance; doubles identity stays the estimate). Writes tracks.parquet
(t, track_id, u, v, role). specs/30-ml-pipeline.md stage 3.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from statistics import median

from ..geometry import apply_homography, enforce_n_tracks, foot_point, role_assign, smooth_track

log = logging.getLogger("worker.pipeline.gate_track")

MARGIN = 0.06  # court-normalized gating margin (~37 cm wide / 80 cm long)
# Interim foot-truncation correction: push the foot down k*bbox-height before projecting (RC1).
# 0 = off (identical to the old bbox-bottom). k≈0.16 fitted on f35d0984 (see eval_corrections).
# Used as the FALLBACK when the pose stage has no confident ankle (Phase-3 three-layer foot).
FOOT_BIAS_K = float(os.environ.get("FOOT_BIAS_K", "0"))
# Phase-3 pose foot: use the real ankle keypoint for the court position. Default OFF (like FOOT_BIAS_K),
# enabled after the trainer reviews the numbers — on f35d0984 the un-fitted pose ankle beats the raw
# bbox bottom big (far 1.6→0.49 m) but a clip-fitted FOOT_BIAS_K=0.16 still edges it (far 0.30 m); see
# eval_corrections / specs/fix-later.md. The pose stage always runs (keypoints feed Phase 4 + stroke).
USE_POSE_FOOT = os.environ.get("USE_POSE_FOOT", "0").lower() not in ("0", "", "false", "no")
POSE_CONF_THR = float(os.environ.get("POSE_CONF_THR", "0.3"))  # min ankle confidence to trust the pose foot

ROLE_TO_SIDESLOT = {
    "near": ("A", 0),
    "far": ("B", 0),
    "near_left": ("A", 0),
    "near_right": ("A", 1),
    "far_left": ("B", 0),
    "far_right": ("B", 1),
}


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job["match_id"]

    H = ctx.get("H")
    if H is None:
        cal = store.get_calibration(video_id)
        if not cal or not cal.get("homography"):
            raise RuntimeError("gate_track: no homography — calibrate must run first")
        H = cal["homography"]

    detections_path = ctx.get("detections_path")
    if not detections_path:
        raise RuntimeError("gate_track: no detections_path — detect must run first")

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frame_t = {f["frame"]: f["t"] for f in manifest["frames"]}

    match = store.get_match(match_id) or {}
    discipline = match.get("discipline", "doubles")
    n = 2 if discipline == "singles" else 4

    import polars as pl  # lazy

    # Pose foot (Phase 3): real ankle per (frame, track) when confident; else the bbox nudge below.
    # Only consulted when USE_POSE_FOOT is set (default off — no silent change to court positions).
    pose_path = ctx.get("pose_path") or (cache / "pose.parquet")
    pose_feet: dict[tuple, tuple] = {}
    if USE_POSE_FOOT and Path(pose_path).exists():
        for pr in pl.read_parquet(pose_path).iter_rows(named=True):
            pose_feet[(pr["frame"], pr["track"])] = (pr["foot_x"], pr["foot_y"], pr["foot_conf"])

    det = pl.read_parquet(detections_path)
    samples: list[tuple] = []  # (t, track_id, u, v)
    counts: dict[int, int] = {}
    n_pose = 0
    for r in det.iter_rows(named=True):
        tid = r["track"]
        if tid is None:
            continue
        pf = pose_feet.get((r["frame"], tid))
        used_pose = pf is not None and pf[0] is not None and pf[2] >= POSE_CONF_THR
        if used_pose:
            ax, ay = pf[0], pf[1]  # real ankle keypoint
        else:
            ax, ay = foot_point((r["x"], r["y"], r["w"], r["h"]), FOOT_BIAS_K)  # fallback (Phase 2)
        u, v = apply_homography(H, ax, ay)
        if not (-MARGIN <= u <= 1 + MARGIN and -MARGIN <= v <= 1 + MARGIN):
            continue  # off-court (adjacent court / spectator) → drop
        samples.append((frame_t.get(r["frame"], 0.0), int(tid), u, v))
        counts[int(tid)] = counts.get(int(tid), 0) + 1
        n_pose += int(used_pose)

    if not samples:
        raise RuntimeError("gate_track: no in-court detections after gating (check calibration)")

    keep = set(enforce_n_tracks(counts, n))
    samples = [s for s in samples if s[1] in keep]

    medians = {
        tid: (median([s[2] for s in samples if s[1] == tid]), median([s[3] for s in samples if s[1] == tid]))
        for tid in keep
    }
    roles = role_assign(medians, "singles" if n == 2 else "doubles")

    # Per-track jitter smoothing before persisting (keeps speeds/distance honest, overlay smooth).
    out_rows = []
    for tid in keep:
        seq = sorted([(s[0], s[2], s[3]) for s in samples if s[1] == tid])  # (t,u,v)
        for t, u, v in smooth_track(seq):
            out_rows.append({"t": t, "track_id": tid, "u": u, "v": v, "role": roles.get(tid, "")})

    out = cache / "tracks.parquet"
    pl.DataFrame(
        out_rows,
        schema={"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64, "v": pl.Float64, "role": pl.Utf8},
    ).sort(["track_id", "t"]).write_parquet(out)
    ctx["tracks_path"] = str(out)
    store.upload(cfg.artifact_bucket, f"{video_id}/tracks.parquet", out.read_bytes(), "application/octet-stream")

    _auto_map(store, match_id, roles)
    log.info(
        "gate_track: kept %d tracks (%s), %d in-court samples (%d pose-foot / %d bbox-fallback), roles=%s",
        len(keep), discipline, len(samples), n_pose, len(samples) - n_pose, roles,
    )


def _auto_map(store, match_id: str, roles: dict) -> None:
    """Seed match_players.track_label from role (trainer confirms/overrides later; doubles = estimate)."""
    by_sideslot = {(mp["side"], mp["slot"]): mp for mp in store.get_match_players(match_id)}
    for tid, role in roles.items():
        ss = ROLE_TO_SIDESLOT.get(role)
        mp = by_sideslot.get(ss) if ss else None
        if mp:
            store.set_track_label(mp["id"], int(tid))
