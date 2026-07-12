"""Labeling-assist report: what the GBDT's OOF predictions are worth to a human labeler.

Answers three questions from an existing `oof_predictions.parquet` (no new labeling, no
model): how often the model's suggestion would be right (top-1/top-2, per class), how a
pre-fill confidence threshold trades coverage vs correctness, and how strongly sorting by a
rare class's probability enriches it (net/drive mining). High-confidence disagreements with
the human labels double as a mislabel review queue: `--relabel-queue` re-emits the matching
original Label Studio tasks with display-only prior/model fields — deliberately NO
`predictions` array, so the re-labeler is not anchored toward the model on exactly the rows
where model and human already disagree. (specs/features/stroke-labeling.md)

    python -m worker stroke-assist-report [--oof PATH] [--min-p 0.8]
                                          [--relabel-queue OUT.json --tasks GLOB]

Pure functions on OOF row dicts (TDD, tests/test_assist_report.py); polars only in the CLI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from ..import_stroke_labels import GERMAN_BY_CANONICAL
from .train import CLASSES

log = logging.getLogger("worker.stroke.assist_report")

DEFAULT_TAUS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_KS = (20, 50, 100)
MINE_CLASSES = ("net", "drive")  # the starved classes (specs/features/stroke-classification.md)


def _confident(rows: Sequence[Mapping]) -> List[Mapping]:
    """Human-confident rows only — Unsicher calibration rows have no trusted label to agree with."""
    return [r for r in rows if not r["is_calibration"]]


def agreement(rows: Sequence[Mapping]) -> dict:
    """Top-1/top-2 agreement with the human labels, overall and per true class."""
    conf = _confident(rows)
    top1 = top2 = 0
    per: dict = {}
    for r in conf:
        ranked = sorted(CLASSES, key=lambda c: -float(r[f"prob_{c}"]))
        hit1 = r["pred"] == r["shot_type"]
        top1 += hit1
        top2 += r["shot_type"] in ranked[:2]
        slot = per.setdefault(r["shot_type"], {"n": 0, "hits": 0})
        slot["n"] += 1
        slot["hits"] += hit1
    n = len(conf)
    return {"n": n,
            "top1": top1 / n if n else 0.0,
            "top2": top2 / n if n else 0.0,
            "per_class": {c: {"n": s["n"], "top1": s["hits"] / s["n"]}
                          for c, s in sorted(per.items())}}


def threshold_table(rows: Sequence[Mapping],
                    taus: Sequence[float] = DEFAULT_TAUS) -> List[dict]:
    """Per pre-fill threshold: how many clips would come pre-selected, and how often right."""
    conf = _confident(rows)
    out = []
    for tau in taus:
        covered = [r for r in conf if float(r["p_max"]) >= tau]
        correct = sum(1 for r in covered if r["pred"] == r["shot_type"])
        out.append({"tau": tau, "n": len(covered),
                    "coverage": len(covered) / len(conf) if conf else 0.0,
                    "covered_accuracy": correct / len(covered) if covered else None})
    return out


def enrichment(rows: Sequence[Mapping], cls: str,
               ks: Sequence[int] = DEFAULT_KS) -> List[dict]:
    """True-`cls` count in the top-k rows by prob_<cls> — validates rare-class mining order."""
    conf = _confident(rows)
    ranked = sorted(conf, key=lambda r: -float(r[f"prob_{cls}"]))
    base = sum(1 for r in conf if r["shot_type"] == cls)
    out = []
    for k in ks:
        k_eff = min(int(k), len(ranked))
        out.append({"k": k_eff,
                    "hits": sum(1 for r in ranked[:k_eff] if r["shot_type"] == cls),
                    "base_rate": base / len(conf) if conf else 0.0})
    return out


def disagreements(rows: Sequence[Mapping], min_p: float = 0.8) -> List[dict]:
    """Confident rows where a confident model contradicts the human — candidate mislabels,
    most-confident first. Only the review columns survive."""
    out = [{"event_id": r["event_id"], "video_id": r["video_id"], "t_s": r["t_s"],
            "shot_type": r["shot_type"], "pred": r["pred"], "p_max": float(r["p_max"])}
           for r in _confident(rows)
           if r["pred"] != r["shot_type"] and float(r["p_max"]) >= min_p]
    return sorted(out, key=lambda r: -r["p_max"])


def relabel_tasks(tasks: Sequence[Mapping],
                  disags: Sequence[Mapping]) -> Tuple[List[dict], int]:
    """Join disagreements back to the ORIGINAL exporter tasks by event_id (they carry the
    resolved clip path) and re-emit them as review tasks with display-only German
    prior_label/model_pred/model_score. Disagreements without a matching task are counted,
    not raised — the tasks JSON may come from a --sample export."""
    by_event = {t["data"]["event_id"]: t for t in tasks}
    queue: List[dict] = []
    missing = 0
    for d in disags:
        task = by_event.get(d["event_id"])
        if task is None:
            missing += 1
            continue
        data = dict(task["data"])
        data["prior_label"] = GERMAN_BY_CANONICAL[d["shot_type"]]
        data["model_pred"] = GERMAN_BY_CANONICAL[d["pred"]]
        data["model_score"] = round(d["p_max"], 4)
        queue.append({"data": data})
    if missing:
        log.warning("relabel queue: %d disagreement(s) had no matching task", missing)
    return queue, missing


def newest_oof(labeling: Path) -> Optional[Path]:
    """Newest GBDT run's OOF parquet — UTC-stamped dir names sort lexicographically."""
    runs = sorted((labeling / "stroke_runs").glob("*_gbdt_traj/oof_predictions.parquet"))
    return runs[-1] if runs else None


def report_cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import glob as globmod
    import json

    import polars as pl

    ap = argparse.ArgumentParser(
        prog="worker stroke-assist-report",
        description="Simulate the GBDT as a labeling assistant against existing OOF "
                    "predictions; optionally emit a mislabel review queue for Label Studio.")
    ap.add_argument("--oof", help="oof_predictions.parquet "
                                  "(default: newest <labeling>/stroke_runs/*_gbdt_traj run)")
    ap.add_argument("--min-p", type=float, default=0.8,
                    help="disagreement confidence floor (default 0.8)")
    ap.add_argument("--relabel-queue", help="write the review tasks JSON here")
    ap.add_argument("--tasks", help="glob of the original tasks_*.json "
                                    "(required with --relabel-queue)")
    args = ap.parse_args(argv)
    if args.relabel_queue and not args.tasks:
        ap.error("--relabel-queue requires --tasks <glob of the original tasks_*.json>")

    if args.oof:
        oof_path = Path(args.oof).expanduser()
    else:
        from ..config import load_config

        labeling = Path(load_config().frame_cache_dir).expanduser().parent / ".labeling"
        oof_path = newest_oof(labeling)
        if oof_path is None:
            raise SystemExit(
                f"no stroke_runs/*_gbdt_traj/oof_predictions.parquet under {labeling} — "
                "run `worker train-stroke-gbdt` first or pass --oof")
    rows = pl.read_parquet(oof_path).to_dicts()

    agg = agreement(rows)
    print(f"OOF: {oof_path}")
    print(f"top-1 agreement {agg['top1']:.3f}, top-2 {agg['top2']:.3f} "
          f"(n={agg['n']} confident rows)")
    for c, s in agg["per_class"].items():
        print(f"  {c:<6} n={s['n']:<4d} top1={s['top1']:.3f}")
    print("pre-fill threshold: tau  coverage  covered-acc      n")
    for t in threshold_table(rows):
        acc = f"{t['covered_accuracy']:.3f}" if t["covered_accuracy"] is not None else "n/a"
        print(f"                    {t['tau']:.1f} {t['coverage']:9.3f} {acc:>12} {t['n']:>6d}")
    for cls in MINE_CLASSES:
        ranks = enrichment(rows, cls)
        parts = ", ".join(f"top-{e['k']}: {e['hits']}" for e in ranks)
        print(f"mine {cls}: {parts} true {cls} (base rate {ranks[0]['base_rate']:.3f})")

    disags = disagreements(rows, min_p=args.min_p)
    print(f"{len(disags)} high-confidence disagreements (p_max >= {args.min_p:.2f}) "
          "— candidate mislabels")
    if args.relabel_queue:
        tasks: List[dict] = []
        for path in sorted(globmod.glob(args.tasks)):
            tasks.extend(json.loads(Path(path).read_text(encoding="utf-8")))
        queue, missing = relabel_tasks(tasks, disags)
        Path(args.relabel_queue).write_text(json.dumps(queue, indent=2), encoding="utf-8")
        print(f"relabel queue: {len(queue)} review tasks -> {args.relabel_queue}"
              + (f" ({missing} without a matching task)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(report_cli())
