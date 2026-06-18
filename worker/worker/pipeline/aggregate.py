"""Stage: aggregate — per-player tracks → player_match_stats. Reuses insights.py verbatim.

`compute_player_stats` is pure (no polars) and unit-tested. The stage glue reads tracks.parquet,
maps each track to its player via match_players.track_label, and upserts. Honesty: `fastest` is the
estimated *shuttle* speed (needs the P2 shuttle stage) — left null, never fabricated from foot speed.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from .. import insights as ins
from ..geometry import net_depth
from ..insights import Sample

log = logging.getLogger("worker.pipeline.aggregate")


def compute_player_stats(track: List[Sample], role: str, duration_min: float) -> dict:
    """Pure: a (t,u,v) track + role → the player_match_stats fields the dashboard consumes."""
    speeds = ins.step_speeds_kmh(track)
    avg, top = ins.avg_top_speed_kmh(speeds)
    depths = [net_depth(v, role) for (_, _, v) in track]  # the zones orientation fix
    laterals = [u for (_, u, _) in track]
    return {
        "distance_m": round(ins.total_distance_m(track)),
        "avg_kmh": avg,
        "top_kmh": top,
        "sprints": ins.sprint_count(speeds),
        "zones": ins.zones(depths, laterals),
        "timeline": ins.timeline_buckets(track, duration_min),
        "heat": ins.heatmap_kernels([(u, v) for (_, u, v) in track]),
        "fastest": None,  # estimated shuttle speed — P2 shuttle stage, never faked from foot speed
    }


def run(ctx) -> None:
    store = ctx["store"]
    job = ctx["job"]
    match_id = job["match_id"]
    video_id = job["video_id"]

    tracks_path = ctx.get("tracks_path")
    if not tracks_path:
        raise RuntimeError("aggregate: no tracks_path — gate_track must run first")

    import polars as pl  # lazy

    df = pl.read_parquet(tracks_path)
    if df.is_empty():
        log.warning("aggregate: tracks.parquet empty — nothing to write")
        return

    video = store.get_video(video_id) or {}
    duration_min = (float(video.get("duration_s") or 0.0)) / 60 or _span_min(df)

    players = store.get_match_players(match_id)
    by_label: Dict[int, dict] = {
        int(mp["track_label"]): mp for mp in players if mp.get("track_label") is not None
    }
    if not by_label:
        raise RuntimeError("aggregate: no match_players.track_label set — track→player mapping missing")

    written = 0
    for tid in sorted(df["track_id"].unique().to_list()):
        mp = by_label.get(int(tid))
        if not mp:
            continue
        sub = df.filter(pl.col("track_id") == tid).sort("t")
        track: List[Sample] = [(r["t"], r["u"], r["v"]) for r in sub.iter_rows(named=True)]
        role = str(sub["role"][0])
        st = compute_player_stats(track, role, duration_min)
        store.upsert_player_match_stats({"match_id": match_id, "player_id": mp["player_id"], **st})
        written += 1
    log.info("aggregate: wrote %d player_match_stats (duration=%.1f min)", written, duration_min)


def _span_min(df) -> float:
    import polars as pl  # noqa: F401

    t = df["t"]
    return max(1.0, (float(t.max()) - float(t.min())) / 60)
