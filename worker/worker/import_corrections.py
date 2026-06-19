"""Import verify-tool corrections (cache `corrections.json`) into Postgres `detection_corrections`.

The verify tool keeps writing the JSON in the frame cache (write-only today); this is a separate,
idempotent step that loads that answer key into the DB so the scorecard (eval_corrections full mode)
and future fixes can re-run against a persisted ground truth.

    python -m worker import-corrections <video_id> [--file /path/corrections.json]

Pure `rows_from_corrections` (DB-free) does the schema mapping and is unit-tested; the CLI wraps it
with config + the service-role store. See supabase/migrations/0004_detection_corrections.sql.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence

log = logging.getLogger("worker.import_corrections")

# Always-present fields (every correction carries these). t_s/frame_idx orient the answer key in time.
_CORE = ("frame_idx", "t_s", "role", "type")
# Optional per type — included only when present so the DB column stays NULL otherwise.
_OPTIONAL = ("bbox", "orig_bbox", "foot_norm", "court_uv", "orig_foot", "orig_uv",
             "correct_role", "edited_in", "note")


def rows_from_corrections(video_id: str, cors: Sequence[dict]) -> List[dict]:
    """Map verify-tool correction records → `detection_corrections` rows.

    `_id` (the verify tool's per-correction id) becomes `source_id`; together with `video_id` it is the
    unique key the import upserts on, so re-importing the same JSON is idempotent. Absent optional fields
    are omitted (not written as NULL) to keep rows minimal.
    """
    rows: List[dict] = []
    for c in cors:
        row = {"video_id": video_id, "source_id": c.get("_id")}
        for k in _CORE:
            row[k] = c.get(k)
        for k in _OPTIONAL:
            if k in c:
                row[k] = c[k]
        rows.append(row)
    return rows


def _corrections_path(cfg, video_id: str) -> Path:
    return Path(cfg.frame_cache_dir).expanduser() / video_id / "corrections.json"


def import_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(prog="worker import-corrections",
                                 description="Import verify-tool corrections.json into Postgres.")
    ap.add_argument("video_id", help="match_videos.id the corrections belong to")
    ap.add_argument("--file", help="corrections.json path (default: <frame_cache>/<video_id>/corrections.json)")
    args = ap.parse_args(argv)

    cfg = load_config()
    path = Path(args.file) if args.file else _corrections_path(cfg, args.video_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    cors = data.get("corrections", [])
    rows = rows_from_corrections(args.video_id, cors)

    from .__main__ import _make_store

    store = _make_store(cfg)
    store.upsert_detection_corrections(rows)
    log.info("imported %d corrections for %s from %s", len(rows), args.video_id, path)
    print(f"imported {len(rows)} corrections for {args.video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(import_cli())
