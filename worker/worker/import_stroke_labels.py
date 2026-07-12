"""Import a Label Studio stroke-label export into the stage-7 training parquet.

Reads the JSON export of the „Schlagarten" project (tasks produced by export-stroke-clips, one
choice annotation each), maps the German UI values → canonical tokens per the table in
specs/features/stroke-labeling.md, and writes two parquets with identical schema:
`stroke_labels.parquet` (training, `unknown` included — it is the taxonomy's honesty option) and
`hit_detection_eval.parquet` (the `no_hit` rows — stage-6 false positives, free eval data).

    python -m worker import-stroke-labels <ls-export.json> [--out DIR]

Deliberately DB-free: the task `data` already carries event_id/match_id/video_id/t_s verbatim from
the `events` rows the exporter read, so there is nothing a round-trip would validate. Corruption is
guarded where it can actually enter: unmapped values, missing shot_type, and duplicate event_ids
are hard errors. Pure `rows_from_export`/`split_rows` are unit-tested; the CLI wraps them.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger("worker.import_stroke_labels")

SHOT_TYPE = {"Clear": "clear", "Smash": "smash", "Drop": "drop", "Netzschlag": "net",
             "Drive": "drive", "Lift": "lift", "Aufschlag": "serve",
             "Unsicher": "unknown", "Kein Schlag": "no_hit"}
# canonical → German for anything that writes back INTO Label Studio (pre-labels, review
# queues). Derived, never hand-written: a rename on one side must break loudly on both.
GERMAN_BY_CANONICAL = {v: k for k, v in SHOT_TYPE.items()}
HITTER = {"vorne-links": "near_left", "vorne-rechts": "near_right",
          "hinten-links": "far_left", "hinten-rechts": "far_right"}


def pick_annotation(task: Mapping) -> Optional[Mapping]:
    """The task's current answer: latest non-cancelled annotation (a relabel supersedes), or None."""
    usable = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not usable:
        return None
    return max(usable, key=lambda a: str(a.get("updated_at") or a.get("created_at") or ""))


def _choice(annotation: Mapping, from_name: str) -> Optional[str]:
    for r in annotation.get("result", []):
        if r.get("from_name") == from_name:
            choices = r.get("value", {}).get("choices") or []
            return choices[0] if choices else None
    return None


def rows_from_export(tasks: Sequence[Mapping]) -> List[dict]:
    """Map Label Studio tasks → canonical label rows (one per labeled task).

    Unlabeled tasks are skipped; a missing shot_type, an unmapped German value, or a duplicate
    event_id raises ValueError — bad labels must never slip into training data silently."""
    rows: List[dict] = []
    seen: set = set()
    for task in tasks:
        data = task["data"]
        ann = pick_annotation(task)
        if ann is None:
            continue
        event_id = data["event_id"]
        if event_id in seen:
            raise ValueError(f"duplicate event_id {event_id!r} — conflicting labels for one hit")
        seen.add(event_id)
        shot = _choice(ann, "shot_type")
        if shot is None:
            raise ValueError(f"missing shot_type on event_id {event_id!r} (required in the labeling config)")
        if shot not in SHOT_TYPE:
            raise ValueError(f"unmapped shot_type value {shot!r} on event_id {event_id!r}")
        hitter = _choice(ann, "hitter")
        if hitter is not None and hitter not in HITTER:
            raise ValueError(f"unmapped hitter value {hitter!r} on event_id {event_id!r}")
        rows.append({
            "event_id": event_id,
            "match_id": data["match_id"],
            "video_id": data["video_id"],
            "t_s": data["hit_t_s"],
            "shot_type": SHOT_TYPE[shot],
            "hitter": HITTER[hitter] if hitter is not None else None,
            "rally_idx": data["rally_idx"],
            "hit_idx": data["hit_idx"],
        })
    return rows


def split_rows(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """(training rows, no_hit eval rows). `unknown` stays in training — honesty, not noise."""
    training = [r for r in rows if r["shot_type"] != "no_hit"]
    eval_rows = [r for r in rows if r["shot_type"] == "no_hit"]
    return training, eval_rows


def merge_rows(base: Sequence[dict], fixes: Sequence[dict]) -> List[dict]:
    """Relabel merge: fixes win by event_id, base order stays stable, unseen fixes append
    at the end. Operates on the COMBINED row set (training + no_hit) so a relabel can flip
    a row across the split — the caller re-splits afterwards."""
    pending = {r["event_id"]: r for r in fixes}
    merged = [pending.pop(r["event_id"], None) or dict(r) for r in base]
    merged.extend(f for f in fixes if f["event_id"] in pending)
    return merged


def write_labels_parquet(rows: Sequence[dict], path: Path) -> None:
    import polars as pl  # lazy

    pl.DataFrame(rows, schema={
        "event_id": pl.Utf8, "match_id": pl.Utf8, "video_id": pl.Utf8,
        "t_s": pl.Float64, "shot_type": pl.Utf8, "hitter": pl.Utf8,
        "rally_idx": pl.Int64, "hit_idx": pl.Int64,
    }).write_parquet(path)


def import_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="worker import-stroke-labels",
                                 description="Import a Label Studio stroke-label export into training parquets.")
    ap.add_argument("export", help="Label Studio JSON export (list of tasks with annotations)")
    ap.add_argument("--out", help="output directory (default: next to the export file)")
    ap.add_argument("--merge-into", metavar="DIR",
                    help="merge this export into an existing labeling dir instead of "
                         "writing fresh parquets: fixes win by event_id, rows re-split "
                         "(a relabel can flip training <-> no_hit)")
    args = ap.parse_args(argv)

    export_path = Path(args.export).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else export_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    rows = rows_from_export(tasks)
    n_labeled = len(rows)
    n_changed = None
    if args.merge_into:
        import polars as pl  # lazy

        base_dir = Path(args.merge_into).expanduser()
        base: List[dict] = []
        for fname in ("stroke_labels.parquet", "hit_detection_eval.parquet"):
            p = base_dir / fname
            if not p.exists():
                raise SystemExit(f"--merge-into: {p} not found — point at the labeling dir "
                                 "holding both parquets")
            base.extend(pl.read_parquet(p).to_dicts())
        prior = {r["event_id"]: r["shot_type"] for r in base}
        n_changed = sum(1 for r in rows
                        if r["event_id"] in prior and prior[r["event_id"]] != r["shot_type"])
        rows = merge_rows(base, rows)
        out_dir = base_dir
    training, eval_rows = split_rows(rows)

    train_path = out_dir / "stroke_labels.parquet"
    eval_path = out_dir / "hit_detection_eval.parquet"
    write_labels_parquet(training, train_path)
    write_labels_parquet(eval_rows, eval_path)

    counts = Counter(r["shot_type"] for r in rows)
    no_hitter = sum(1 for r in rows if r["hitter"] is None)
    merged_note = (f"merged {n_labeled} relabels into {out_dir} "
                   f"({n_changed} rows changed class); "
                   if n_changed is not None else "")
    summary = (f"{merged_note}imported {n_labeled}/{len(tasks)} labeled tasks "
               f"({len(tasks) - n_labeled} unlabeled skipped): "
               f"{len(training)} training rows -> {train_path}, "
               f"{len(eval_rows)} no_hit rows -> {eval_path}; "
               f"{no_hitter} without hitter; shot_type counts: {dict(counts)}")
    log.info(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(import_cli())
