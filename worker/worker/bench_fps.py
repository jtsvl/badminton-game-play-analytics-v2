"""`python -m worker bench-fps` — run the cadence-dependent stages on ONE video at a chosen analysis fps
into an ISOLATED frame cache, writing nothing to Supabase (specs/features/analysis-30fps.md).

The production cache and the DB rows are live data. This driver re-runs
[ingest → detect → pose → gate_track → shuttle → rally_hit] through a `ReadOnlyStore`: reads (video row,
calibration, match, roster) come from the real store, every write (bucket uploads, replace_rallies,
calibration/meta updates, track labels) is a counted no-op. The result is a complete set of per-stage
parquets in a separate cache dir that the eval tools score with `WORKER_FRAME_CACHE_DIR` pointing at it,
plus `bench.json` with per-stage runtime, cache size and the MEASURED cadence. Promoted from PR #19's
scratch `reingest_30fps.py`, which embedded the homographies by hand.

    cd worker
    uv run python -m worker bench-fps --video-id <id> --analysis-fps 12 --cache-dir ../worker/.cache-12fps-bench
    uv run python -m worker bench-fps --video-id <id> --analysis-fps 30 --cache-dir ../worker/.cache-30fps --reuse-frames

`--reuse-frames` skips ingest when `<cache-dir>/<video>/frames.json` already carries the requested fps
(the 30 fps frames PR #19 decoded are code-independent — only the stages after ingest need re-running).
Smoke-test first with `--max-frames 60`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

log = logging.getLogger("worker.bench_fps")

DEFAULT_STAGES = ["ingest", "detect", "pose", "gate_track", "shuttle", "rally_hit"]
# Only stages whose DB side effects are fully stubbed below and whose outputs stay in the cache.
# aggregate / overlay / remap / rally_sync / proxy write stats, keyframes or proxies for the REAL match.
ALLOWED_STAGES = tuple(DEFAULT_STAGES)

# Every mutating method of worker.store.Store (and db.JobStore). A new write method on the store must be
# added here — tests/test_bench_fps.py derives the list from Store's public names and fails otherwise.
WRITE_METHODS = frozenset({
    "claim_job", "set_stage", "finish", "enqueue", "update_match_status", "update_video_meta",
    "set_video_proxy", "insert_calibration", "update_calibration", "set_track_label",
    "upsert_player_match_stats", "upsert_player_shot_errors", "upsert_player_shot_stats",
    "upsert_track_keyframes", "upsert_pose_keyframes", "delete_pose_keyframes",
    "upsert_detection_corrections", "upsert_shuttle_labels", "insert_rally_corrections",
    "insert_hit_corrections",
    "replace_rallies", "log_job_run", "create_player", "create_match", "create_video",
    "create_match_player", "upload", "upload_resumable",
})
_WRITE_RETURNS = {"insert_calibration": {"id": "bench-stub"}, "replace_rallies": 0, "enqueue": {}}


class ReadOnlyStore:
    """Reads delegate to the real store; every write is a counted no-op (`blocked`)."""

    def __init__(self, real) -> None:
        self._real = real
        self.blocked: Dict[str, int] = {}

    def __getattr__(self, name: str):
        if name in WRITE_METHODS:
            def noop(*_a, **_k):
                self.blocked[name] = self.blocked.get(name, 0) + 1
                log.debug("bench: blocked store write %s", name)
                return _WRITE_RETURNS.get(name)
            return noop
        return getattr(self._real, name)


def plan_stages(requested: Sequence[str], reuse_frames: bool, frames_ok: bool) -> List[str]:
    """Validate the stage list (bench-safe stages only) and drop ingest when a matching cache exists."""
    bad = [s for s in requested if s not in ALLOWED_STAGES]
    if bad:
        raise ValueError(f"stages not allowed in a bench run: {bad} (allowed: {list(ALLOWED_STAGES)})")
    stages = list(requested)
    if reuse_frames and frames_ok and "ingest" in stages:
        stages.remove("ingest")
    return stages


def frames_match(video_cache_dir: Path, analysis_fps: float) -> bool:
    """True when <video_cache_dir>/frames.json exists and was decoded at this requested fps."""
    manifest_path = Path(video_cache_dir) / "frames.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return bool(m.get("frames")) and abs(float(m.get("analysis_fps", -1.0)) - analysis_fps) < 1e-6


def is_production_cache(path: Path, configured_cache: Path) -> bool:
    """Refuse the configured cache and anything named `.cache` (the production layout) outright."""
    p = Path(path).expanduser()
    try:
        same = p.resolve() == Path(configured_cache).expanduser().resolve()
    except OSError:
        same = False
    return same or p.name == ".cache"


def dir_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def bench_record(video_id: str, analysis_fps: float, stages: Sequence[str], stage_seconds: Dict[str, float],
                 cache_bytes: int, n_frames: int, cadence_fps: float | None) -> dict:
    return {
        "video_id": video_id,
        "analysis_fps": analysis_fps,
        "cadence_fps": cadence_fps,
        "stages": list(stages),
        "stage_seconds": dict(stage_seconds),
        "total_seconds": round(sum(stage_seconds.values()), 1),
        "cache_bytes": int(cache_bytes),
        "cache_gb": round(cache_bytes / 1e9, 3),
        "n_frames": int(n_frames),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="worker bench-fps",
        description="Non-destructive fps bench: run the cadence-dependent stages for one video into an "
                    "isolated cache through a read-only store (no DB/bucket writes).")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--analysis-fps", type=float, required=True, help="requested decode fps (12 or 30)")
    ap.add_argument("--cache-dir", required=True,
                    help="ISOLATED frame-cache root, e.g. ../worker/.cache-30fps (never the production .cache)")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma list (default {','.join(DEFAULT_STAGES)}; only these are allowed)")
    ap.add_argument("--reuse-frames", action="store_true",
                    help="skip ingest when <cache-dir>/<video>/frames.json already has this analysis_fps")
    ap.add_argument("--max-frames", type=int, default=None, help="dev cap on decoded frames (smoke test)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from .__main__ import _make_store
    from .config import load_config
    from .manifest import manifest_fps
    from .pipeline import STAGES, StageContext

    base_cfg = load_config()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    if is_production_cache(cache_root, Path(base_cfg.frame_cache_dir)):
        raise SystemExit(f"REFUSING to bench into a production cache path: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(base_cfg, frame_cache_dir=str(cache_root))
    store = ReadOnlyStore(_make_store(base_cfg))

    video = store.get_video(args.video_id)
    if not video:
        raise SystemExit(f"no match_videos row for {args.video_id}")
    video_dir = cache_root / args.video_id
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    stages = plan_stages(requested, args.reuse_frames, frames_match(video_dir, args.analysis_fps))
    if "ingest" not in stages and "ingest" in requested:
        log.info("bench: reusing the existing %.0f fps frames in %s (ingest skipped)", args.analysis_fps, video_dir)

    params: dict = {"analysis_fps": args.analysis_fps}
    if args.max_frames:
        params["max_frames"] = args.max_frames
    job = {"id": f"bench-{args.video_id[:8]}", "video_id": args.video_id,
           "match_id": video.get("match_id"), "params": params, "stage": "bench"}
    ctx = StageContext(job=job, cfg=cfg, store=store)
    log.info("bench: video %s @ %.0f fps → %s | stages %s | production cache untouched (%s)",
             args.video_id, args.analysis_fps, video_dir, stages, base_cfg.frame_cache_dir)

    timings: Dict[str, float] = {}
    t_all = time.time()
    for name in stages:
        t0 = time.time()
        STAGES[name](ctx)
        timings[name] = round(time.time() - t0, 1)
        log.info("bench: [%s] done in %.1fs", name, timings[name])

    manifest = json.loads((video_dir / "frames.json").read_text()) if (video_dir / "frames.json").exists() else {}
    rec = bench_record(args.video_id, args.analysis_fps, stages, timings, dir_size_bytes(video_dir),
                       len(manifest.get("frames") or []), manifest_fps(manifest) if manifest else None)
    rec["blocked_writes"] = dict(store.blocked)
    rec["cache_dir"] = str(video_dir)
    (video_dir / "bench.json").write_text(json.dumps(rec, indent=2))

    print(f"bench-fps done in {time.time() - t_all:.0f}s → {video_dir / 'bench.json'}")
    print(f"  requested {args.analysis_fps:g} fps → measured cadence "
          f"{rec['cadence_fps']:.2f} fps, {rec['n_frames']} frames, cache {rec['cache_gb']:.2f} GB"
          if rec["cadence_fps"] else "  (no frames.json — ingest did not run)")
    for name, secs in timings.items():
        print(f"  {name:11} {secs:8.1f} s")
    print(f"  blocked store writes: {rec['blocked_writes'] or 'none'}")
    print(f"Score it with: WORKER_FRAME_CACHE_DIR={cache_root} uv run python -m worker.eval_shuttle "
          f"--video-id {args.video_id} --tracker tracknetv3   (and eval_rally --landings --hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
