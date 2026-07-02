"""Rally-segmentation 'scorecard' — measure build_rallies against trainer-reviewed rally lists.

Pure scoring functions (no polars/DB — they test under the dev extra), mirroring
eval_corrections.py. The trainer's post-review rally list IS the ground truth: the review UI
only drops and merges (it cannot move boundaries), so every kept rally's GT boundaries equal
the machine's own. Consequences (specs/features/pose-swing-fusion.md, "GT regime"):

- Boundary F1 @ ±tol matches starts to starts and ends to ends SEPARATELY (greedy 1:1) — else a
  phantom's end would match a real rally's start and inflate precision.
- The confusion speaks the trainer's own vocabulary via max-overlap assignment (NOT IoU≥0.5 —
  split fragments fail IoU and would miscount): `false_rallies` (= trainer drops), `splits`
  (= trainer merges), and the two unrecoverable-error alarms `missed` (a real rally the pipeline
  no longer emits — false veto) and `merges` (one prediction spanning two real rallies — false
  bridge), both hard flip gates (must stay 0).

Run against a reviewed match (CLI, lazy imports):
    uv run python -m worker.eval_rally --slug 2026-06-30-session2-game1
"""
from __future__ import annotations

from statistics import median
from typing import Dict, List, Sequence, Tuple

Span = Tuple[float, float]


def _overlap_s(a: Span, b: Span) -> float:
    """Length (seconds) of the intersection of two spans; 0 when disjoint."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _greedy_match(pred_ts: Sequence[float], gt_ts: Sequence[float], tol_s: float) -> int:
    """Number of 1:1 matches between two boundary lists, closest pairs first, within tol_s."""
    pairs = sorted(
        (abs(p - g), i, j)
        for i, p in enumerate(pred_ts) for j, g in enumerate(gt_ts)
        if abs(p - g) <= tol_s
    )
    used_p: set = set()
    used_g: set = set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p)


def boundary_prf(pred: Sequence[Span], gt: Sequence[Span], tol_s: float = 0.5) -> Dict[str, float]:
    """Boundary precision/recall/F1 @ ±tol_s. Starts match starts, ends match ends (separately)."""
    matched = (_greedy_match([s for s, _ in pred], [s for s, _ in gt], tol_s)
               + _greedy_match([e for _, e in pred], [e for _, e in gt], tol_s))
    n_pred, n_gt = 2 * len(pred), 2 * len(gt)
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched,
            "pred_boundaries": n_pred, "gt_boundaries": n_gt}


def confusion(pred: Sequence[Span], gt: Sequence[Span], min_overlap_s: float = 0.2) -> Dict[str, int]:
    """Segment-level confusion in the trainer's vocabulary (max-overlap assignment).

    Per prediction: no GT overlap ≥ min_overlap_s → `false_rally` (what the trainer drops);
    overlap with ≥2 GT rallies → `merges` (falsely bridged — unrecoverable, hard gate);
    else assigned to its single GT rally. Per GT rally: no overlapping prediction → `missed`
    (falsely vetoed — unrecoverable, hard gate); ≥2 assigned predictions → `splits` (what the
    trainer merges). `matched` = GT rallies covered by at least one prediction."""
    assigned: Dict[int, int] = {}  # gt index -> count of single-hit preds
    covered: set = set()           # gt indices touched by any pred (incl. merge-preds)
    false_rallies = merges = 0
    for p in pred:
        hits = [j for j, g in enumerate(gt) if _overlap_s(p, g) >= min_overlap_s]
        covered.update(hits)
        if not hits:
            false_rallies += 1
        elif len(hits) >= 2:
            merges += 1
        else:
            assigned[hits[0]] = assigned.get(hits[0], 0) + 1
    missed = len(gt) - len(covered)
    splits = sum(1 for n in assigned.values() if n >= 2)
    return {"false_rallies": false_rallies, "splits": splits, "missed": missed,
            "merges": merges, "matched": len(covered)}


def evaluate(pred: Sequence[Span], gt: Sequence[Span], tol_s: float = 0.5) -> dict:
    """Full scorecard: boundary F1 + trainer-vocabulary confusion + counts."""
    return {"n_pred": len(pred), "n_gt": len(gt),
            "boundary": boundary_prf(pred, gt, tol_s),
            "confusion": confusion(pred, gt)}


def span_stats(spans: Sequence[Span]) -> dict:
    """Span-length + inter-span-gap distribution of a rally list (time-sorted internally).

    Feeds the two empirical constants: the veto span guard (RALLY_SWING_VETO_MAX_S must sit
    below the shortest kept rally) and the bridge cap (RALLY_SWING_BRIDGE_S just above the
    largest trainer-merged gap)."""
    ordered = sorted(spans)
    lengths = [e - s for s, e in ordered]
    gaps = [b[0] - a[1] for a, b in zip(ordered, ordered[1:])]
    return {"n": len(ordered),
            "min_span_s": min(lengths) if lengths else None,
            "median_span_s": median(lengths) if lengths else None,
            "gaps_s": gaps}


def format_report(rep: dict) -> str:
    """Human-readable scorecard for the CLI — confusion lines in the trainer's vocabulary."""
    b, c = rep["boundary"], rep["confusion"]
    lines = [
        "Rally-segmentation scorecard (prediction vs trainer-reviewed rallies)",
        "=" * 68,
        f"rallies: predicted={rep['n_pred']}  ground-truth={rep['n_gt']}",
        (f"boundaries (±tol): F1={b['f1']:.3f}  precision={b['precision']:.3f}"
         f"  recall={b['recall']:.3f}  ({b['matched']}/{b['pred_boundaries']} pred,"
         f" {b['gt_boundaries']} gt)"),
        "-" * 68,
        f"false rallies (trainer would drop) : {c['false_rallies']}",
        f"splits        (trainer would merge): {c['splits']}",
        f"missed        (REAL RALLY LOST — falsely vetoed,  hard gate 0): {c['missed']}",
        f"merges        (REAL RALLIES GLUED — falsely bridged, hard gate 0): {c['merges']}",
        f"matched gt rallies                 : {c['matched']}/{rep['n_gt']}",
    ]
    return "\n".join(lines)
