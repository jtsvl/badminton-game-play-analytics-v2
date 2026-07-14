"""Pose-coverage harness — how much of the video does each player's skeleton actually cover?

Pure scoring functions (no polars/DB, so they test under the dev extra). Coverage = fraction of
analysis frames with a pose keyframe within `max_gap_s` — 0.3 s is exactly the web's `poseAt`
hide threshold, so a thinned export (Phase D ~6 Hz) scores the same as what the viewer sees and
thinning can't game the metric. Wall-clock is the primary basis (user decision 2026-07-13);
the in-rally split is reported alongside.

Per player the CLI prints three columns (the mitigation-phase gates):
  DB       — what pose_keyframes currently serves the web,
  export   — what overlay._pose_keyframes would write from the cache right now,
  ceiling  — the role-stitched timeline (tracks.parquet non-interp rows), Phase B's target.

Run from worker/:
    uv run python -m worker.eval_pose_coverage --slug 2026-06-30-session2-game1
    uv run python -m worker.eval_pose_coverage --video-id <uuid>
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .eval_corrections import ROLE_TO_LABEL, half_of

Span = Tuple[float, float]

# poseAt (web/src/lib/pose.ts) hides the skeleton when the nearest keyframe is > 0.3 s away.
RENDER_MAX_GAP_S = 0.3


def frames_in_spans(frame_ts: Sequence[float], spans: Sequence[Span]) -> List[float]:
    """Analysis-frame timestamps inside any [t0, t1] span (inclusive — mirrors filter_to_spans)."""
    return [t for t in frame_ts if any(t0 <= t <= t1 for t0, t1 in spans)]


def renderable_coverage(
    keyframe_ts: Sequence[float], frame_ts: Sequence[float], max_gap_s: float = RENDER_MAX_GAP_S
) -> float:
    """Fraction of `frame_ts` with a keyframe within `max_gap_s` (poseAt's nearest-key semantics)."""
    if not keyframe_ts or not frame_ts:
        return 0.0
    keys = sorted(keyframe_ts)
    covered = 0
    for t in frame_ts:
        i = bisect_left(keys, t)
        nearest = min(
            abs(keys[j] - t) for j in (i - 1, i) if 0 <= j < len(keys)
        )
        if nearest <= max_gap_s:
            covered += 1
    return covered / len(frame_ts)


def coverage_report(
    per_player_ts: Mapping[str, Sequence[float]],
    frame_ts: Sequence[float],
    spans: Sequence[Span],
    max_gap_s: float = RENDER_MAX_GAP_S,
) -> dict:
    """Per-player wall-clock + in-rally coverage. `per_player_ts` keys are labels (A1/B2/…).

    in_rally is None when there are no spans (nothing to gate against, not zero coverage)."""
    rally_ts = frames_in_spans(frame_ts, spans) if spans else None
    players = {}
    for label, key_ts in per_player_ts.items():
        players[label] = {
            "wall": renderable_coverage(key_ts, frame_ts, max_gap_s),
            "in_rally": renderable_coverage(key_ts, rally_ts, max_gap_s) if rally_ts else None,
            "half": half_of(label),
        }
    return {"players": players, "n_frames": len(frame_ts), "n_rally_frames": len(rally_ts or [])}


# --- CLI glue below: real cache + DB reads (lazy imports, no writes) --------------------------


def _label_of(mp: dict) -> Optional[str]:
    """match_players row -> A1/A2/B1/B2 (side + slot), the label the report tables use."""
    side, slot = mp.get("side"), mp.get("slot")
    if side in ("A", "B") and slot in (0, 1):
        return f"{side}{slot + 1}"
    return None


def _recompute_tracks(store, cache, video_id: str, match_id: str):
    """Rebuild the role timelines from cached detections with the CURRENT gate_track code —
    measures a stitching change offline, before any pipeline re-run or DB write."""
    import json

    import polars as pl

    from .pipeline.gate_track import build_role_tracks, court_samples

    cal = store.get_calibration(video_id)
    if not cal or not cal.get("homography"):
        raise RuntimeError(f"eval_pose_coverage: no homography for {video_id}")
    manifest = json.loads((cache / "frames.json").read_text())
    frame_t = {f["frame"]: f["t"] for f in manifest["frames"]}
    det = pl.read_parquet(cache / "detections.parquet")
    samples, counts, _ = court_samples(det.iter_rows(named=True), cal["homography"], frame_t)
    discipline = (store.get_match(match_id) or {}).get("discipline", "doubles")
    n = 2 if discipline == "singles" else 4
    rows, _ = build_role_tracks(samples, counts, frame_t, n, discipline)
    return pl.DataFrame(rows, schema={"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64,
                                      "v": pl.Float64, "role": pl.Utf8, "source": pl.Utf8})


def report_for_video(video_id: str, recompute: bool = False) -> dict:
    """Assemble the three coverage columns + payload sizes for one video (read-only)."""
    import json
    from pathlib import Path

    import polars as pl  # lazy: keeps the pure functions polars-free

    from .__main__ import _make_store
    from .config import load_config
    from .pipeline.overlay import _pose_keyframes
    from .pipeline.aggregate import _rally_spans

    cfg = load_config()
    store = _make_store(cfg)
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id

    manifest = json.loads((cache / "frames.json").read_text())
    frame_ts = [float(f["t"]) for f in manifest["frames"]]

    spans = _rally_spans({}, cache) or []

    video = store.get_video(video_id) or {}
    match_id = video.get("match_id")
    if not match_id:
        raise RuntimeError(f"eval_pose_coverage: video {video_id} has no match")
    mps = store.get_match_players(match_id)
    label_by_pid = {mp["player_id"]: _label_of(mp) for mp in mps if mp.get("player_id")}

    # Column 1 — DB: what the web is served today.
    db_ts: Dict[str, List[float]] = {}
    db_bytes: Dict[str, int] = {}
    for row in store.get_pose_keyframes(video_id):
        pid = row["subject"].removeprefix("player:")
        label = label_by_pid.get(pid)
        if not label:
            continue
        db_ts[label] = [k["t"] for k in row["data"]]
        db_bytes[label] = len(json.dumps(row["data"], separators=(",", ":")))

    # Column 2 — export: what overlay._pose_keyframes would write from the cache right now.
    # With --recompute the role timelines are rebuilt in-memory by the CURRENT gate_track code
    # (stitching changes measurable before any pipeline run); tracks.parquet is never touched.
    export_ts: Dict[str, List[float]] = {}
    export_bytes: Dict[str, int] = {}
    tracks_df = None
    if recompute:
        tracks_df = _recompute_tracks(store, cache, video_id, match_id)
    elif (cache / "tracks.parquet").exists():
        tracks_df = pl.read_parquet(cache / "tracks.parquet")
    pose = None
    if tracks_df is not None:
        pose = _pose_keyframes({}, cache, tracks_df, mps, spans or None)
    for subject, data in (pose or {}).items():
        label = label_by_pid.get(subject.removeprefix("player:"))
        if not label:
            continue
        export_ts[label] = [k["t"] for k in data]
        export_bytes[label] = len(json.dumps(data, separators=(",", ":")))

    # Column 3 — ceiling: the role-stitched timeline (non-interp rows have a pose row per frame).
    ceil_ts: Dict[str, List[float]] = {}
    if tracks_df is not None:
        tr = tracks_df
        if "source" in tr.columns:
            tr = tr.filter(pl.col("source") != "interp")
        for role, sub in tr.group_by("role"):
            label = ROLE_TO_LABEL.get(str(role[0]))
            if label:
                ceil_ts[label] = sorted(set(sub["t"].to_list()))

    return {
        "video_id": video_id,
        "frame_ts": frame_ts,
        "spans": spans,
        "db": coverage_report(db_ts, frame_ts, spans),
        "export": coverage_report(export_ts, frame_ts, spans),
        "ceiling": coverage_report(ceil_ts, frame_ts, spans),
        "db_bytes": db_bytes,
        "export_bytes": export_bytes,
    }


def format_report(rep: dict) -> str:
    """Human table: one row per player label, the three coverage columns (wall% / in-rally%)."""
    def pct(v: Optional[float]) -> str:
        return f"{100 * v:5.1f}%" if v is not None else "    —"

    labels = sorted(
        set(rep["db"]["players"]) | set(rep["export"]["players"]) | set(rep["ceiling"]["players"])
    )
    lines = [
        f"video {rep['video_id']}  frames={rep['db']['n_frames']}  "
        f"rallies={len(rep['spans'])}  in-rally frames={rep['db']['n_rally_frames']}",
        f"{'player':<8}{'DB wall/rally':<18}{'export wall/rally':<20}{'ceiling wall/rally':<20}"
        f"{'DB KB':>7}{'export KB':>11}",
    ]
    for label in labels:
        cols = []
        for key in ("db", "export", "ceiling"):
            p = rep[key]["players"].get(label)
            cols.append(f"{pct(p['wall']) if p else pct(None)}/{pct(p['in_rally']) if p else pct(None)}")
        db_kb = rep["db_bytes"].get(label)
        ex_kb = rep["export_bytes"].get(label)
        lines.append(
            f"{label:<8}{cols[0]:<18}{cols[1]:<20}{cols[2]:<20}"
            f"{db_kb // 1024 if db_kb else 0:>6}K{ex_kb // 1024 if ex_kb else 0:>10}K"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Pose skeleton coverage report (per player, wall/in-rally).")
    ap.add_argument("--video-id", help="video uuid (cache dir name)")
    ap.add_argument("--slug", help="match slug (e.g. 2026-06-30-session2-game1)")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild role timelines from cached detections with the current gate_track "
                         "code (measure a stitching change offline; no writes)")
    args = ap.parse_args(argv)
    if not args.video_id and not args.slug:
        ap.error("need --video-id or --slug")

    video_id = args.video_id
    if not video_id:
        from .__main__ import _make_store
        from .config import load_config

        store = _make_store(load_config())
        match = store.get_match_by_slug(args.slug)
        if not match:
            raise RuntimeError(f"eval_pose_coverage: no match with slug {args.slug!r}")
        video = store.get_video_by_match(match["id"])
        if not video:
            raise RuntimeError(f"eval_pose_coverage: match {args.slug!r} has no video row")
        video_id = video["id"]

    print(format_report(report_for_video(video_id, recompute=args.recompute)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
