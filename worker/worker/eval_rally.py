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
from typing import Dict, List, Optional, Sequence, Tuple

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


def split_gaps(pred: Sequence[Span], gt: Sequence[Span], min_overlap_s: float = 0.2) -> List[float]:
    """Interior gaps (seconds) of every split GT rally: for each GT rally covered by ≥2
    predictions, the time gaps between its consecutive prediction fragments. On a flags-off
    baseline these are exactly the gaps the trainer bridged by merging — the empirical basis
    for the bridge cap RALLY_SWING_BRIDGE_S."""
    frags: Dict[int, List[Span]] = {}
    for p in pred:
        hits = [j for j, g in enumerate(gt) if _overlap_s(p, g) >= min_overlap_s]
        if len(hits) == 1:
            frags.setdefault(hits[0], []).append(p)
    gaps: List[float] = []
    for spans in frags.values():
        if len(spans) >= 2:
            ordered = sorted(spans)
            gaps.extend(b[0] - a[1] for a, b in zip(ordered, ordered[1:]))
    return sorted(gaps)


def _pctiles(values: Sequence[float]) -> dict:
    """n + p10/p25/p50/p75/p90 of a strength list (None stats when empty)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    xs = sorted(values)

    def p(q: float) -> float:
        rank = q / 100.0 * (n - 1)
        lo, hi = int(rank), min(int(rank) + 1, n - 1)
        frac = rank - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {"n": n, "p10": p(10), "p25": p(25), "p50": p(50), "p75": p(75), "p90": p(90)}


def swing_strength_split(swings: Sequence[dict], gt_spans: Sequence[Span],
                         half_by_track: Optional[Dict[int, str]] = None) -> dict:
    """Swing-strength percentiles inside vs between trainer-approved rallies — the measured
    separation the three strength thresholds are picked from (never guessed).

    With ``half_by_track`` (track id → 'near'/'far') the in-rally distribution is additionally
    split by court half: the near player has their back to the camera, so a systematically
    weaker near distribution (foreshortened swings) must be visible here BEFORE thresholds are
    chosen — the back-to-camera check (user concern 2026-07-02)."""
    in_r = [s for s in swings if any(a <= s["t"] <= b for a, b in gt_spans)]
    out_r = [s for s in swings if not any(a <= s["t"] <= b for a, b in gt_spans)]
    rep = {"in_rally": _pctiles([s["strength"] for s in in_r]),
           "between": _pctiles([s["strength"] for s in out_r])}
    if half_by_track:
        for half in ("near", "far"):
            vals = [s["strength"] for s in in_r if half_by_track.get(s["track"]) == half]
            rep[f"in_rally_{half}"] = _pctiles(vals)
    return rep


def _strongest_in(swings: Sequence[dict], lo_t: float, hi_t: float,
                  half_by_track: Optional[Dict[int, str]] = None) -> Optional[dict]:
    """Strongest swing sample with ``lo_t <= t <= hi_t`` (closed bounds, matching the veto
    sweep's span±margin test) as ``{"t", "strength", "half"}`` — half None when the track is
    unmapped. Ties resolve to the earliest sample; None when the window holds no sample."""
    best = None
    for s in sorted(swings, key=lambda s: s["t"]):
        if lo_t <= s["t"] <= hi_t and (best is None or s["strength"] > best["strength"]):
            best = s
    if best is None:
        return None
    return {"t": best["t"], "strength": best["strength"],
            "half": (half_by_track or {}).get(best["track"])}


def serve_window_report(swings: Sequence[dict], gt_spans: Sequence[Span],
                        half_by_track: Optional[Dict[int, str]] = None,
                        pre_s: float = 0.5, post_s: float = 0.5) -> dict:
    """Serve-instant swing report: per GT rally, the strongest swing within
    ``[t_start − pre_s, t_start + post_s]`` and which court half it belongs to — the
    back-to-camera check with serve swings ISOLATED (the per-sample in-rally split lumps every
    shot type and diluted the answer).

    Honest caveat: the server's identity is unlabeled. Attribution = strongest swing in the
    window, and serves alternate ≈50/50, so a strong attribution skew toward 'far' (or
    systematically weaker near-attributed maxima) is the back-to-camera signal — not per-rally
    truth. ``no_sample_rallies`` separately exposes rallies with NO swing sample at the serve
    instant (pose dropout or a fully missed swing); ``attribution['none']`` additionally counts
    best swings on unmapped tracks."""
    per_rally: List[dict] = []
    for t0, _ in gt_spans:
        lo, hi = t0 - pre_s, t0 + post_s
        inw = [s for s in swings if lo <= s["t"] <= hi]
        best = _strongest_in(inw, lo, hi, half_by_track)
        per_rally.append({"t_start": t0, "n_samples": len(inw),
                          "max_strength": best["strength"] if best else None,
                          "best_half": best["half"] if best else None,
                          "best_t_rel": (best["t"] - t0) if best else None})
    attribution = {"near": 0, "far": 0, "none": 0}
    for r in per_rally:
        attribution[r["best_half"] if r["best_half"] in ("near", "far") else "none"] += 1
    return {"pre_s": pre_s, "post_s": post_s, "per_rally": per_rally,
            "no_sample_rallies": sum(1 for r in per_rally if r["n_samples"] == 0),
            "attribution": attribution,
            "max_by_half": {h: _pctiles([r["max_strength"] for r in per_rally
                                         if r["best_half"] == h])
                            for h in ("near", "far")}}


def veto_operating_points(swings: Sequence[dict], pred: Sequence[Span], gt: Sequence[Span],
                          thresholds: Sequence[float], margin_s: float = 0.5,
                          min_overlap_s: float = 0.2) -> Dict[float, dict]:
    """Direct veto operating curve per candidate threshold: how many trainer-approved rallies
    carry a swing ≥ thr within ±margin_s (must be ALL — anything less is a hard-gate `missed`),
    and how many machine phantoms (predictions with no GT overlap) carry none (the kill rate).
    Uses the same GT/prediction data as the baseline, so the numbers translate 1:1 to T7."""
    phantoms = [p for p in pred
                if not any(_overlap_s(p, g) >= min_overlap_s for g in gt)]
    out: Dict[float, dict] = {}
    for thr in thresholds:
        ts = [s["t"] for s in swings if s["strength"] >= thr]

        def hit(span: Span) -> bool:
            return any(span[0] - margin_s <= t <= span[1] + margin_s for t in ts)

        out[thr] = {"rallies_covered": sum(1 for g in gt if hit(g)), "gt_total": len(gt),
                    "phantoms_killed": sum(1 for p in phantoms if not hit(p)),
                    "phantom_total": len(phantoms)}
    return out


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


# ── CLI bottom half (lazy imports: DB + parquet glue, mirrors eval_corrections.py) ──────────


def gt_from_db(store, slug: str) -> Tuple[List[Span], dict]:
    """Trainer-reviewed rally spans of a match, straight from the DB (the post-review rally
    list IS the ground truth). Warns when the match was never reviewed (rally_edited_at null) —
    then the 'GT' is just the machine's own output and every score is trivially perfect."""
    import logging

    match = store.get_match_by_slug(slug)
    if not match:
        raise RuntimeError(f"eval_rally: no match with slug {slug!r}")
    if not match.get("rally_edited_at"):
        logging.getLogger("worker.eval_rally").warning(
            "eval_rally: match %s has rally_edited_at=NULL — rallies are UNREVIEWED machine "
            "output, scores against it are meaningless as GT", slug)
    spans = [(float(r["t_start_s"]), float(r["t_end_s"])) for r in store.get_rallies(match["id"])]
    return spans, match


def predict_from_cache(video_id: str, cfg) -> List[Span]:
    """Re-run build_rallies from the cached parquets through rally_hit's own loaders — NOT a
    duplicate loading path, else the 'flags-off baseline reproduces the pre-review machine
    output' claim silently breaks. Knobs come from env exactly like the pipeline."""
    import json
    from pathlib import Path

    from .pipeline import rally_hit
    from .rally import segment as seg

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])

    dets = rally_hit.load_dets(str(cache / "shuttle.parquet"), frame_w, frame_h)

    player_energy = None
    tracks_path = cache / "tracks.parquet"
    if tracks_path.exists():
        import polars as pl  # lazy

        tdf = pl.read_parquet(tracks_path)
        if not tdf.is_empty():
            player_energy = rally_hit._player_energy(tdf)

    out = seg.build_rallies(dets, player_energy)
    return [(float(r["t_start_s"]), float(r["t_end_s"])) for r in out["rallies"]]


def load_swing_samples(video_id: str, cfg) -> Tuple[List[dict], Dict[int, str]]:
    """Per-sample swing strengths (emission floor 0, no coalescing — the raw distribution the
    thresholds are picked from) + track→court-half map. Pose rows join to tracks.parquet kept
    track ids; frame→t via frames.json; `source=\"interp\"` rows have no pose — harmless."""
    import json
    from pathlib import Path

    import polars as pl  # lazy

    from .rally.swing import swing_events

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frame_t = {int(f["frame"]): float(f["t"]) for f in manifest["frames"]}

    half_by_track: Dict[int, str] = {}
    for r in pl.read_parquet(cache / "tracks.parquet").iter_rows(named=True):
        tid = r.get("track_id")
        half = (r.get("role") or "").split("_")[0]
        if tid is not None and half in ("near", "far"):
            half_by_track.setdefault(int(tid), half)

    rows = [{"t": frame_t[int(r["frame"])], "track": int(r["track"]),
             "kp": r["kp"], "kp_score": r["kp_score"]}
            for r in pl.read_parquet(cache / "pose.parquet").iter_rows(named=True)
            if int(r["track"]) in half_by_track and int(r["frame"]) in frame_t]
    samples = swing_events(rows, min_tlps=0.0, merge_s=0.0)
    return samples, half_by_track


SWEEP_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def _fmt_pct(st: dict) -> str:
    if not st["n"]:
        return "n=0"
    return (f"n={st['n']:>6}  p10={st['p10']:5.2f}  p25={st['p25']:5.2f}  p50={st['p50']:5.2f}"
            f"  p75={st['p75']:5.2f}  p90={st['p90']:5.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    from .__main__ import _make_store
    from .config import load_config

    ap = argparse.ArgumentParser(
        description="Rally-segmentation scorecard: build_rallies (cached parquets, env knobs) "
                    "vs the trainer-reviewed rally list in the DB.")
    ap.add_argument("--slug", required=True, help="match slug (e.g. 2026-06-30-session2-game1)")
    ap.add_argument("--tol", type=float, default=0.5, help="boundary match tolerance ±s")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--swing-hist", action="store_true",
                    help="swing-strength tuning report (in/between rally + near/far + veto sweep)")
    args = ap.parse_args(argv)

    cfg = load_config()
    store = _make_store(cfg)
    gt, match = gt_from_db(store, args.slug)
    video = store.get_video_by_match(match["id"])
    if not video:
        raise RuntimeError(f"eval_rally: match {args.slug!r} has no video row")
    pred = predict_from_cache(video["id"], cfg)

    if args.swing_hist:
        samples, half_by_track = load_swing_samples(video["id"], cfg)
        split = swing_strength_split(samples, gt, half_by_track)
        sweep = veto_operating_points(samples, pred, gt, SWEEP_THRESHOLDS)
        if args.json:
            print(json.dumps({"slug": args.slug, "split": split,
                              "sweep": {str(k): v for k, v in sweep.items()}}, indent=2))
            return 0
        print(f"[{args.slug}] swing-strength tuning report (tl/s, per-sample, floor 0)")
        print("=" * 68)
        for key in ("in_rally", "in_rally_near", "in_rally_far", "between"):
            if key in split:
                print(f"{key:14}: {_fmt_pct(split[key])}")
        print("-" * 68)
        print("veto sweep (swing >= thr within ±0.5 s saves a rally / kills a phantom):")
        for thr, pt in sweep.items():
            gate = "OK  " if pt["rallies_covered"] == pt["gt_total"] else "MISS"
            print(f"  thr={thr:4.1f}  rallies {pt['rallies_covered']:3}/{pt['gt_total']:<3} {gate}"
                  f"  phantoms killed {pt['phantoms_killed']:3}/{pt['phantom_total']}")
        return 0

    rep = evaluate(pred, gt, tol_s=args.tol)
    rep["meta"] = {"slug": args.slug, "video_id": video["id"], "tol_s": args.tol,
                   "rally_edited_at": match.get("rally_edited_at")}
    # The two empirical-constant feeds (spec: veto span guard + bridge cap).
    rep["gt_span_stats"] = span_stats(gt)
    rep["split_gaps_s"] = split_gaps(pred, gt)

    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"[{args.slug}] video={video['id']}  tol=±{args.tol}s  "
          f"reviewed={match.get('rally_edited_at') or 'NEVER'}")
    print(format_report(rep))
    st = rep["gt_span_stats"]
    print(f"kept-rally spans: n={st['n']}  min={st['min_span_s']:.2f}s  "
          f"median={st['median_span_s']:.2f}s")
    gaps = rep["split_gaps_s"]
    print("trainer-merged gaps (s): " + (", ".join(f"{g:.2f}" for g in gaps) if gaps else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
