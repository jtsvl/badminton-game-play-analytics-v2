"""Import verify-tool SHUTTLE labels (cache `shuttle_labels.json`) into Postgres `shuttle_labels`.

The `verify --shuttle` tool writes sparse anchors + segment boundaries to the frame cache; this idempotent
step expands them to a DENSE per-frame answer key (via shuttle.label.interpolate_anchors) and upserts it so
the TrackNetV3 fine-tune and the eval_shuttle scoreboard re-run against persisted ground truth.

    python -m worker import-shuttle-labels <video_id> [--file /path/shuttle_labels.json]

Pure `rows_from_shuttle_labels` (DB-free) does the expansion + schema mapping and is unit-tested; the CLI
wraps it with config, the frames.json timestamps, and the service-role store. See
supabase/migrations/0005_shuttle_labels.sql.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .shuttle.label import interpolate_anchors

log = logging.getLogger("worker.import_shuttle_labels")


def rows_from_shuttle_labels(
    video_id: str, labels: Mapping, t_by_frame: Optional[Dict[int, float]] = None
) -> List[dict]:
    """Map a shuttle_labels.json record → dense `shuttle_labels` rows.

    Expands the sparse anchors + boundaries to one row per frame. `source_id` carries the frame index so
    `unique(video_id, source_id)` makes re-import idempotent per frame (re-labeling replaces). `t_s` is
    looked up from `t_by_frame` (the clip's frames.json timestamps) so it is exact, never interpolated;
    frames without a timestamp get None. A visibility=0 frame carries null px/py."""
    t_by_frame = t_by_frame or {}
    dense = interpolate_anchors(labels.get("anchors", []), labels.get("boundaries", []))
    rows: List[dict] = []
    for frame, px, py, visibility, interpolated in dense:
        row = {
            "video_id": video_id,
            "source_id": frame,        # = frame_idx → the per-frame idempotent upsert key
            "frame_idx": frame,
            "t_s": t_by_frame.get(frame),
            "px": px,
            "py": py,
            "visibility": visibility,
            "interpolated": interpolated,
        }
        assert row["source_id"] == row["frame_idx"]  # lock the upsert key to the frame
        rows.append(row)
    return rows


def _labels_path(cfg, video_id: str) -> Path:
    return Path(cfg.frame_cache_dir).expanduser() / video_id / "shuttle_labels.json"


def _t_by_frame(cfg, video_id: str) -> Dict[int, float]:
    """frame_idx (frames.json position) → timestamp, matching the labeler's `c.t_s = FRAMES[i].t`."""
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    manifest = json.loads((cache / "frames.json").read_text(encoding="utf-8"))
    return {i: float(f["t"]) for i, f in enumerate(manifest["frames"])}


def import_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(prog="worker import-shuttle-labels",
                                 description="Import verify-tool shuttle_labels.json into Postgres.")
    ap.add_argument("video_id", help="match_videos.id the shuttle labels belong to")
    ap.add_argument("--file", help="shuttle_labels.json path (default: <frame_cache>/<video_id>/shuttle_labels.json)")
    args = ap.parse_args(argv)

    cfg = load_config()
    path = Path(args.file) if args.file else _labels_path(cfg, args.video_id)
    labels = json.loads(path.read_text(encoding="utf-8"))
    rows = rows_from_shuttle_labels(args.video_id, labels, _t_by_frame(cfg, args.video_id))

    from .__main__ import _make_store

    store = _make_store(cfg)
    store.upsert_shuttle_labels(rows)
    n_anchor = sum(1 for r in rows if not r["interpolated"])
    log.info("imported %d shuttle-label rows (%d anchors) for %s from %s", len(rows), n_anchor, args.video_id, path)
    print(f"imported {len(rows)} shuttle-label rows ({n_anchor} anchors) for {args.video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(import_cli())
