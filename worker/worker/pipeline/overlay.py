"""Stage: overlay — resample each track to ~10 Hz → track_keyframes (the overlay's input).

Web-facing geometry: subject 'player:<player_id>' and (stage 5) subject 'shuttle', data [{t,u,v}]
(t in seconds). The shuttle track is split on real dropouts (`split_on_gaps`) before resampling so the
flight path is NOT bridged across a white-wall camouflage gap with a straight line (specs/40-worker-contract.md).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..geometry import resample_10hz, split_on_gaps

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
    by_label = {
        int(mp["track_label"]): mp
        for mp in store.get_match_players(match_id)
        if mp.get("track_label") is not None
    }

    written = 0
    for tid in df["track_id"].unique().to_list():
        mp = by_label.get(int(tid))
        if not mp:
            continue
        sub = df.filter(pl.col("track_id") == tid).sort("t")
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
