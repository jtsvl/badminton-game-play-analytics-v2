"""Stage: overlay — resample each track to ~10 Hz → track_keyframes (the overlay's input).

Web-facing geometry: subject 'player:<player_id>' and (stage 5) subject 'shuttle', data [{t,u,v}]
(t in seconds). The shuttle track is split on real dropouts (`split_on_gaps`) before resampling so the
flight path is NOT bridged across a white-wall camouflage gap with a straight line (specs/40-worker-contract.md).

Also writes pose_keyframes (skeleton overlay): frame-normalized image coords (x,y ∈ [0,1] of the
source frame), NOT court u,v — elevated joints sit above the court plane where the ground homography
is geometrically wrong (specs/features/pose-skeleton-overlay.md).

Player binding is label → role → the whole role timeline (identity.role_players): a player's
movement spans many raw track ids (re-ID fragments keep their own id), so filtering by the single
track_label would drop every fragment row — the coverage bug this replaced.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..geometry import resample_10hz, split_on_gaps
from . import identity
from .aggregate import _rally_spans

log = logging.getLogger("worker.pipeline.overlay")

# A shuttle gap longer than this (seconds) is left as a gap, not interpolated across (~3 frames @12fps).
SHUTTLE_GAP_S = float(os.environ.get("SHUTTLE_OVERLAY_GAP_S", "0.3"))


def _shuttle_keyframes(ctx, cache: Path):
    """Read shuttle.parquet (if the stage ran) → ~10 Hz [{t,u,v}], gap-aware. Returns None if absent."""
    shuttle_path = ctx.get("shuttle_path") or str(cache / "shuttle.parquet")
    if not Path(shuttle_path).exists():
        return None
    import polars as pl  # lazy

    df = pl.read_parquet(shuttle_path).sort("t")
    track = [(r["t"], r["u"], r["v"]) for r in df.iter_rows(named=True)]
    data = []
    for seg in split_on_gaps(track, SHUTTLE_GAP_S):
        data += [{"t": round(t, 3), "u": round(u, 4), "v": round(v, 4)}
                 for (t, u, v) in resample_10hz(seg)]
    return data


# Payload safety valve for very long recordings: over the cap a player's keyframes are thinned
# by a uniform stride (never skipped — a sparse skeleton beats none). Normal games stay under it.
POSE_MAX_KEYFRAMES_PER_PLAYER = int(os.environ.get("POSE_MAX_KEYFRAMES_PER_PLAYER", "10000"))


def _pose_keyframes(ctx, cache: Path, tracks_df, match_players, spans):
    """Read pose.parquet → {subject: [{t, kp:[[x,y,score]×26]}]} (Halpe-26). Returns None if
    pose.parquet or frames.json is missing (old matches / evicted caches just skip).

    t and width/height come from the frames.json manifest, never frame/fps (PTS drift). Identity
    is the label→role join (identity.role_players): a pose row ships iff its (frame, track) is a
    non-interp row of a bound role's timeline in tracks.parquet — re-ID fragments export under
    the role's player, bystanders and gap-fill interp frames never do. Whole-clip export
    (skeletons between rallies too — user decision 2026-07-13); `spans` feeds the out-of-rally
    thinning. Payload is bounded by POSE_MAX_KEYFRAMES_PER_PLAYER (thin, never skip).
    """
    pose_path = ctx.get("pose_path") or str(cache / "pose.parquet")
    manifest_path = cache / "frames.json"
    if not Path(pose_path).exists() or not manifest_path.exists():
        return None

    import polars as pl  # lazy

    manifest = json.loads(manifest_path.read_text())
    width, height = manifest["width"], manifest["height"]
    frame_t = {f["frame"]: f["t"] for f in manifest["frames"]}

    df = pl.read_parquet(pose_path)
    # Inner join drops rows whose frame is missing from the manifest (a default t=0.0 would
    # teleport keypoints to the clip start).
    tmap = pl.DataFrame({"frame": list(frame_t.keys()), "t": [float(t) for t in frame_t.values()]})
    df = df.join(tmap.with_columns(pl.col("frame").cast(df["frame"].dtype)), on="frame", how="inner")

    track_rows = tracks_df.to_dicts()
    players = identity.role_players(match_players, track_rows)
    t_to_frame = {round(float(t), 6): f for f, t in frame_t.items()}
    frame_role = identity.role_frame_index(track_rows, t_to_frame)
    if not frame_role or not players:
        return {}

    idx = pl.DataFrame(
        {
            "frame": [f for f, _ in frame_role],
            "track": [tid for _, tid in frame_role],
            "role": list(frame_role.values()),
        }
    ).with_columns(pl.col("frame").cast(df["frame"].dtype), pl.col("track").cast(df["track"].dtype))
    df = df.join(idx, on=["frame", "track"], how="inner")

    clamp = lambda v: min(1.0, max(0.0, v))  # RTMPose can place edge-cropped joints outside the frame
    out: dict[str, list] = {}
    for role, mp in players.items():
        sub = df.filter(pl.col("role") == role).sort("t")
        data = []
        for r in sub.iter_rows(named=True):
            kp = [[round(clamp(x / width), 4), round(clamp(y / height), 4), round(s, 4)]
                  for (x, y), s in zip(r["kp"], r["kp_score"])]
            data.append({"t": round(r["t"], 3), "kp": kp})
        if len(data) > POSE_MAX_KEYFRAMES_PER_PLAYER:
            stride = -(-len(data) // POSE_MAX_KEYFRAMES_PER_PLAYER)  # ceil
            log.warning("overlay: %d pose keyframes for %s — thinning by stride %d (cap %d)",
                        len(data), role, stride, POSE_MAX_KEYFRAMES_PER_PLAYER)
            data = data[::stride]
        out[f"player:{mp['player_id']}"] = data
    return out


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job["match_id"]
    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)

    tracks_path = ctx.get("tracks_path")
    if not tracks_path:
        raise RuntimeError("overlay: no tracks_path — gate_track must run first")

    import polars as pl  # lazy

    df = pl.read_parquet(tracks_path)
    match_players = store.get_match_players(match_id)
    # label → role → the whole role timeline (detected + reid + interp rows), NOT the single
    # labeled track_id — re-ID fragments keep their own id and would silently drop out.
    players = identity.role_players(match_players, df.to_dicts())

    written = 0
    for role, mp in players.items():
        sub = df.filter(pl.col("role") == role).sort("t")
        track = [(r["t"], r["u"], r["v"]) for r in sub.iter_rows(named=True)]
        kf = resample_10hz(track)
        data = [{"t": round(t, 3), "u": round(u, 4), "v": round(v, 4)} for (t, u, v) in kf]
        store.upsert_track_keyframes(video_id, f"player:{mp['player_id']}", data)
        written += 1

    shuttle = _shuttle_keyframes(ctx, cache)
    if shuttle is not None:
        store.upsert_track_keyframes(video_id, "shuttle", shuttle)
        log.info("overlay: wrote %d player + shuttle (%d keyframes) track_keyframes (~10 Hz)",
                 written, len(shuttle))
    else:
        log.info("overlay: wrote %d track_keyframes (~10 Hz; no shuttle.parquet)", written)

    # Pose skeletons are an optional visualization — a failure here must never fail the stage
    # (tracks/stats/track_keyframes above are already written and correct without it).
    try:
        pose = _pose_keyframes(ctx, cache, df, match_players, _rally_spans(ctx, cache))
        if pose is not None:
            for subject, data in pose.items():
                store.upsert_pose_keyframes(video_id, subject, data)
            # A re-label can move a player to a track without pose rows — drop the stale skeleton
            # (the web would keep drawing it at the OLD track's positions in the player's color).
            store.delete_pose_keyframes(video_id, keep_subjects=list(pose.keys()))
            log.info("overlay: wrote pose_keyframes for %d players", len(pose))
        elif job.get("stage") == "remap":
            # Identity correction on an evicted cache: tracks were re-mapped but pose can't be
            # recomputed (pose.parquet/frames.json gone) — better no skeleton than one that keeps
            # the old, wrong track→player binding.
            store.delete_pose_keyframes(video_id, keep_subjects=[])
            log.info("overlay: pose inputs missing on remap — cleared pose_keyframes (stale identity risk)")
        else:
            log.info("overlay: no pose.parquet — skipped pose_keyframes")
    except Exception:
        log.exception("overlay: pose_keyframes failed — skeleton skipped, stage continues")
