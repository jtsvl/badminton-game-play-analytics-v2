"""Stage: overlay — resample each player's track to ~10 Hz → track_keyframes (the overlay's input).

The only P1 web-facing geometry: subject 'player:<player_id>', data [{t,u,v}] (t in seconds). No
shuttle subject in P1. specs/40-worker-contract.md.
"""
from __future__ import annotations

import logging

from ..geometry import resample_10hz

log = logging.getLogger("worker.pipeline.overlay")


def run(ctx) -> None:
    store = ctx["store"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job["match_id"]

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
    log.info("overlay: wrote %d track_keyframes (~10 Hz)", written)
