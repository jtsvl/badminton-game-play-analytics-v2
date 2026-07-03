"""Stage: aggregate — per-player tracks → player_match_stats (+ out-landing errors). Reuses insights.py.

`compute_player_stats` and `out_error_counts` are pure (no polars) and unit-tested. The stage glue
reads tracks.parquet, maps each track to its player via match_players.track_label, and upserts.
Honesty: `fastest` is the estimated *shuttle* speed (needs the P2 shuttle stage) — left null, never
fabricated from foot speed. „Aus-Schläge": rally-ending OUT landings (stage 6 `landing` events) count
against the hitting side and land in `player_shot_stats.errors` under `shot_type='unknown'` until
stage 7 attributes real shot types — singles only (who of a doubles pair hit needs pose/stroke).
Counting here keeps trainer edits honest: rally_sync rewrites the parquets from the edited DB before
this stage runs, so a dropped rally loses its error attribution on the recompute.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import insights as ins
from ..geometry import net_depth
from ..insights import Sample

log = logging.getLogger("worker.pipeline.aggregate")


def filter_to_spans(df, spans: List[Tuple[float, float]]):
    """Keep only track rows whose seconds timestamp `t` falls inside any `[t0, t1]` span (inclusive).

    Used to gate player movement to in-rally time: tracks.parquet `t` and rallies' `t_start_s/t_end_s`
    share the frames.json seconds timebase, so no conversion is needed."""
    import polars as pl

    mask = pl.lit(False)
    for t0, t1 in spans:
        mask = mask | ((pl.col("t") >= t0) & (pl.col("t") <= t1))
    return df.filter(mask)


def _rally_spans(ctx, cache: Path) -> Optional[List[Tuple[float, float]]]:
    """Rally [t_start_s, t_end_s] spans written by the rally_hit stage (ctx path or cache fallback).

    Returns None when no rallies are available (file missing or empty) so the caller falls back to the
    whole clip — the report never zeroes out if segmentation produced nothing."""
    import polars as pl

    path = ctx.get("rallies_path") or str(cache / "rallies.parquet")
    if not Path(path).exists():
        return None
    rdf = pl.read_parquet(path)
    if rdf.is_empty():
        return None
    return [(float(r["t_start_s"]), float(r["t_end_s"])) for r in rdf.iter_rows(named=True)]


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


def out_error_counts(rally_rows: List[dict], event_rows: List[dict]) -> Optional[Dict[str, int]]:
    """Pure: rally-ending OUT landings → errors per hitting side ('A'/'B', the schema's side domain).

    A rally counts only when its landing is located AND called out AND `winner_side` is set (the
    winner is the receiving side, so the hitter = the other side). Landings whose rally row is gone
    (trainer drop) are orphans and never count. Returns None when there are no landing events at all
    (a pre-landing cache) so the caller writes nothing instead of zeroing real data."""
    landings = {e["rally_idx"]: e["payload"] for e in event_rows if e["type"] == "landing"}
    if not landings:
        return None
    counts: Dict[str, int] = {}
    by_idx = {r["idx"]: r for r in rally_rows}
    for idx, p in landings.items():
        rally = by_idx.get(idx)
        if rally is None or not p.get("located") or p.get("call") != "out":
            continue
        winner = rally.get("winner_side")
        if winner not in ("A", "B"):
            continue
        hitter = "B" if winner == "A" else "A"
        counts[hitter] = counts.get(hitter, 0) + 1
    return counts


def _landing_rows(ctx, cache: Path) -> Tuple[List[dict], List[dict]]:
    """rallies + events rows from the stage-6 parquets (ctx paths or cache fallback), payloads parsed."""
    import polars as pl

    rpath = ctx.get("rallies_path") or str(cache / "rallies.parquet")
    epath = ctx.get("events_path") or str(cache / "events.parquet")
    if not (Path(rpath).exists() and Path(epath).exists()):
        return [], []
    rallies = list(pl.read_parquet(rpath).iter_rows(named=True))
    events = [{**e, "payload": json.loads(e["payload"]) if e.get("payload") else {}}
              for e in pl.read_parquet(epath).iter_rows(named=True)]
    return rallies, events


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
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

    # `duration_min` from the FULL clip (the timeline axis spans the whole match), computed before gating.
    video = store.get_video(video_id) or {}
    duration_min = (float(video.get("duration_s") or 0.0)) / 60 or _span_min(df)

    # Gate movement to in-rally time (so distance/speed/zones reflect live points, not standing around).
    # Whole-clip fallback when there are no rallies, so the report never zeroes out.
    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    spans = _rally_spans(ctx, cache)
    if spans:
        gated = filter_to_spans(df, spans)
        if gated.is_empty():
            log.warning("aggregate: no track rows in %d rally spans — whole-clip fallback", len(spans))
        else:
            log.info("aggregate: gated tracks to %d rally spans (%d → %d rows)",
                     len(spans), df.height, gated.height)
            df = gated
    else:
        log.info("aggregate: no rallies — movement computed over the whole clip")

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

    # „Aus-Schläge" from out landings → player_shot_stats.errors (shot_type 'unknown' until stage 7).
    match = store.get_match(match_id) or {}
    if (match.get("discipline") or "singles") != "singles":
        log.info("aggregate: doubles — per-player Aus-Schläge deferred (pair attribution needs stage 7)")
        return
    counts = out_error_counts(*_landing_rows(ctx, cache))
    if counts is None:
        log.info("aggregate: no landing events — player_shot_stats.errors untouched")
        return
    for mp in players:
        if mp.get("player_id") and mp.get("side"):
            # 0 included: overwrites a stale count after the trainer drops the rally behind it
            store.upsert_player_shot_errors(match_id, mp["player_id"], "unknown",
                                            counts.get(str(mp["side"]), 0))
    log.info("aggregate: wrote out-landing errors per side %s", counts or "{}")


def _span_min(df) -> float:
    import polars as pl  # noqa: F401

    t = df["t"]
    return max(1.0, (float(t.max()) - float(t.min())) / 60)
