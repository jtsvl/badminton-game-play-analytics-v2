"""Stage 6 — rally segmentation + hit detection (`rally_hit`).

Reads the shuttle spotter's per-frame detections (`shuttle.parquet`) + player tracks (`tracks.parquet`),
segments the clip into rallies in NORMALIZED IMAGE space (the ground homography can't place an airborne
shuttle — see worker/rally/segment.py), and writes:
  • Postgres `rallies` + `events` (via store.replace_rallies) — the report's rally aggregates light up;
  • debug `rallies.parquet` / `events.parquet` in the cache — `aggregate` reads the rally spans to gate
    player movement to in-rally time.
Empty shuttle.parquet ⇒ 0 rallies, no crash. Mirrors pipeline/shuttle.py + gate_track.py. See
specs/30-ml-pipeline.md stage 6 and specs/40-worker-contract.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..rally import segment as seg

log = logging.getLogger("worker.pipeline.rally_hit")


def _rally_schema():
    import polars as pl
    return {"idx": pl.Int64, "t_start_s": pl.Float64, "t_end_s": pl.Float64,
            "shot_count": pl.Int64, "winner_side": pl.Utf8, "end_shot_type": pl.Utf8,
            "confidence": pl.Float64}


def _event_schema():
    import polars as pl
    return {"rally_idx": pl.Int64, "t_s": pl.Float64, "type": pl.Utf8, "payload": pl.Utf8,
            "confidence": pl.Float64}


def load_dets(shuttle_path, frame_w: int, frame_h: int) -> list:
    """Shuttle detections → normalized image-space points (px/W, py/H), carrying court (u,v)
    for the rally-end candidate. Missing parquet ⇒ [] (0 rallies, no crash). Module-level so
    eval_rally can predict through the pipeline's own loading path (not a duplicate)."""
    import polars as pl

    if not Path(shuttle_path).exists():
        log.warning("rally_hit: no shuttle.parquet at %s — 0 rallies", shuttle_path)
        return []
    sdf = pl.read_parquet(shuttle_path).sort("frame")
    return [{"t": float(r["t"]), "x": float(r["px"]) / frame_w, "y": float(r["py"]) / frame_h,
             "conf": float(r["conf"]), "frame": int(r["frame"]),
             "u": float(r["u"]), "v": float(r["v"])}
            for r in sdf.iter_rows(named=True)]


def _player_energy(tracks_df):
    """Player motion series [(t_s, energy_kmh)] for the serve cue, reusing insights.step_speeds_kmh.

    step_speeds_kmh drops non-positive-dt segments, so we align each returned speed to its segment's END
    time with the same dt>0 filter (1:1). Both players' series are merged and time-sorted, so the serve
    detector's window-max picks up whichever player is still moving."""
    import polars as pl

    from ..insights import step_speeds_kmh

    energy = []
    for tid in tracks_df["track_id"].unique().to_list():
        sub = tracks_df.filter(pl.col("track_id") == tid).sort("t")
        track = [(r["t"], r["u"], r["v"]) for r in sub.select(["t", "u", "v"]).iter_rows(named=True)]
        speeds = step_speeds_kmh(track)
        seg_end_times = [t1 for (t0, _, _), (t1, _, _) in zip(track, track[1:]) if t1 - t0 > 0]
        energy.extend(zip(seg_end_times, speeds))
    energy.sort()
    return energy


def run(ctx) -> None:
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]
    match_id = job.get("match_id")

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])

    import polars as pl  # lazy

    # Detections → normalized image-space points (px/W, py/H); carry court (u,v) for the rally-end candidate.
    shuttle_path = ctx.get("shuttle_path") or str(cache / "shuttle.parquet")
    dets = load_dets(shuttle_path, frame_w, frame_h)

    player_energy = None
    tracks_path = ctx.get("tracks_path") or str(cache / "tracks.parquet")
    if Path(tracks_path).exists():
        tdf = pl.read_parquet(tracks_path)
        if not tdf.is_empty():
            player_energy = _player_energy(tdf)

    out = seg.build_rallies(dets, player_energy)
    rallies, events = out["rallies"], out["events"]
    in_play = sum(max(0.0, r["t_end_s"] - r["t_start_s"]) for r in rallies)
    clip_s = manifest["frames"][-1]["t"] if manifest.get("frames") else 0.0
    log.info("rally_hit: %d detections → %d rallies (%.1fs in-play / %.0fs clip = %.0f%%), %d events",
             len(dets), len(rallies), in_play, clip_s,
             100.0 * in_play / clip_s if clip_s else 0.0, len(events))

    # Debug parquet: aggregate reads rallies.parquet to gate movement to in-rally spans.
    rpath = cache / "rallies.parquet"
    epath = cache / "events.parquet"
    pl.DataFrame(rallies, schema=_rally_schema()).write_parquet(rpath)
    ev_rows = [{"rally_idx": e["rally_idx"], "t_s": e["t_s"], "type": e["type"],
                "payload": json.dumps(e["payload"]), "confidence": e["confidence"]} for e in events]
    pl.DataFrame(ev_rows, schema=_event_schema()).write_parquet(epath)
    ctx["rallies_path"] = str(rpath)
    ctx["events_path"] = str(epath)

    store = ctx.get("store")
    if store is not None:
        store.upload(cfg.artifact_bucket, f"{video_id}/rallies.parquet", rpath.read_bytes(),
                     "application/octet-stream")
        store.upload(cfg.artifact_bucket, f"{video_id}/events.parquet", epath.read_bytes(),
                     "application/octet-stream")
        if match_id:
            store.replace_rallies(match_id, rallies, events)
