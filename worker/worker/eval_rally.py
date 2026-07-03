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

from .rally.segment import (BRIDGE_S, RALLY_SERVE_EXCLUSION_S, RALLY_SWING_BRIDGE_S, SETTLE_KMH,
                            SETTLE_WINDOW_S, players_settled_before)

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


def split_gap_spans(pred: Sequence[Span], gt: Sequence[Span],
                    min_overlap_s: float = 0.2) -> List[Span]:
    """(gap_start, gap_end) of every interior gap of a split GT rally: for each GT rally covered
    by ≥2 predictions, the span between consecutive prediction fragments — the "should bridge"
    population (T8): on a flags-off baseline these are exactly the gaps the trainer bridged by
    merging. Unlike ``split_gaps`` (durations only) this keeps the timestamps a swing search
    needs (``bridge_gap_strengths``)."""
    frags: Dict[int, List[Span]] = {}
    for p in pred:
        hits = [j for j, g in enumerate(gt) if _overlap_s(p, g) >= min_overlap_s]
        if len(hits) == 1:
            frags.setdefault(hits[0], []).append(p)
    spans: List[Span] = []
    for fspans in frags.values():
        if len(fspans) >= 2:
            ordered = sorted(fspans)
            spans.extend((a[1], b[0]) for a, b in zip(ordered, ordered[1:]))
    return sorted(spans)


def split_gaps(pred: Sequence[Span], gt: Sequence[Span], min_overlap_s: float = 0.2) -> List[float]:
    """Interior gaps (seconds) of every split GT rally — durations only; see ``split_gap_spans``
    for the (start, end) spans. The empirical basis for the bridge cap RALLY_SWING_BRIDGE_S."""
    return sorted(e - s for s, e in split_gap_spans(pred, gt, min_overlap_s))


def real_gap_spans(gt: Sequence[Span]) -> List[Span]:
    """(gap_start, gap_end) between every pair of consecutive real (GT) rallies — the "must NOT
    bridge" population (T8): genuine silences between distinct real rallies. Needs no
    predictions, only the trainer-approved rally list."""
    ordered = sorted(gt)
    return [(a[1], b[0]) for a, b in zip(ordered, ordered[1:])]


def bridge_gap_strengths(swings: Sequence[dict], gap_spans: Sequence[Span],
                         serve_exclusion_s: float = 0.75) -> List[dict]:
    """Per-gap max swing strength in ``[gap_start, gap_end − serve_exclusion_s]`` — the exact
    quantity ``segment.py``'s swing-bridge rule (``_swing_in_gap``) thresholds on. Run on
    ``split_gap_spans`` (should-bridge) vs ``real_gap_spans`` (must-not) it is the bridge-side
    analogue of ``rally_max_strengths``. ``max_strength=None`` = no qualifying sample (a pose
    dropout, or the gap is too short to leave room outside the exclusion zone)."""
    records: List[dict] = []
    for gs, ge in gap_spans:
        hi = ge - serve_exclusion_s
        inw = [s for s in swings if gs <= s["t"] <= hi] if hi > gs else []
        best = max(inw, key=lambda s: s["strength"], default=None)
        records.append({"gap_start": gs, "gap_end": ge, "gap_s": ge - gs, "n_samples": len(inw),
                        "max_strength": best["strength"] if best else None})
    return records


def bridge_operating_points(swings: Sequence[dict], split_gap_spans_: Sequence[Span],
                            real_gap_spans_: Sequence[Span], thresholds: Sequence[float],
                            serve_exclusion_s: float = 0.75) -> Dict[float, dict]:
    """Bridge operating curve per candidate threshold: how many trainer-merged (split) gaps carry
    a swing ≥ thr in their bridgeable interior (recovered splits), and how many genuine
    inter-rally gaps in the same candidate range would be incorrectly glued (``false_merges`` —
    the hard gate that must stay 0 before RALLY_SWING_BRIDGE_MIN_STRENGTH is ever set)."""
    def covers(gs: float, ge: float, thr: float) -> bool:
        hi = ge - serve_exclusion_s
        return hi > gs and any(gs <= s["t"] <= hi and s["strength"] >= thr for s in swings)

    return {thr: {"splits_bridged": sum(1 for gs, ge in split_gap_spans_ if covers(gs, ge, thr)),
                 "splits_total": len(split_gap_spans_),
                 "false_merges": sum(1 for gs, ge in real_gap_spans_ if covers(gs, ge, thr)),
                 "real_gaps_total": len(real_gap_spans_)}
            for thr in thresholds}


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


def phantom_spans(pred: Sequence[Span], gt: Sequence[Span],
                  min_overlap_s: float = 0.2) -> List[Span]:
    """THE phantom definition (shared by the sweep, the per-rally-max report and `confusion`'s
    `false_rallies` semantics): predictions with no GT overlap ≥ ``min_overlap_s``."""
    return [p for p in pred if not any(_overlap_s(p, g) >= min_overlap_s for g in gt)]


def rally_max_strengths(swings: Sequence[dict], spans: Sequence[Span], margin_s: float = 0.5,
                        half_by_track: Optional[Dict[int, str]] = None) -> List[dict]:
    """Per-span max swing strength over ``[t_start − margin_s, t_end + margin_s]`` — the exact
    quantity the veto thresholds against. Run on GT rallies it shows which real rallies pin the
    threshold down (``max_strength=None`` = unsavable by ANY swing threshold — pose dropout);
    run on ``phantom_spans(pred, gt)`` it shows the kill-side distribution. Per-rally, unlike
    the per-sample ``swing_strength_split`` — this is the decision-relevant separation."""
    records: List[dict] = []
    for s0, e0 in spans:
        lo, hi = s0 - margin_s, e0 + margin_s
        inw = [s for s in swings if lo <= s["t"] <= hi]
        best = _strongest_in(inw, lo, hi, half_by_track)
        records.append({"t_start": s0, "span_s": e0 - s0, "n_samples": len(inw),
                        "max_strength": best["strength"] if best else None,
                        "best_half": best["half"] if best else None,
                        "best_t_rel": (best["t"] - s0) if best else None})
    return records


def veto_operating_points(swings: Sequence[dict], pred: Sequence[Span], gt: Sequence[Span],
                          thresholds: Sequence[float], margin_s: float = 0.5,
                          min_overlap_s: float = 0.2,
                          player_energy: Optional[Sequence] = None) -> Dict[float, dict]:
    """Direct veto operating curve per candidate threshold: how many trainer-approved rallies
    carry a swing ≥ thr within ±margin_s (must be ALL — anything less is a hard-gate `missed`),
    and how many machine phantoms (predictions with no GT overlap) carry none (the kill rate).
    Uses the same GT/prediction data as the baseline, so the numbers translate 1:1 to T7.

    With ``player_energy`` (``[(t_s, kmh)]``, the serve cue's series) each row additionally
    scores the FUSED rule `swing ≥ thr OR players_settled_before(t_start)` at the production
    serve-cue defaults: a real serve follows a settled instant, while the measured phantom
    classes (pickup toss, walk-back) mostly don't — so settledness can protect quiet real
    rallies and let thr rise above the naive missed==0 pin point (4/76 kills)."""
    phantoms = phantom_spans(pred, gt, min_overlap_s)
    settled_gt = settled_ph = None
    if player_energy is not None:  # threshold-independent — computed once, outside the sweep
        settled_gt = [players_settled_before(g[0], player_energy) for g in gt]
        settled_ph = [players_settled_before(p[0], player_energy) for p in phantoms]
    out: Dict[float, dict] = {}
    for thr in thresholds:
        ts = [s["t"] for s in swings if s["strength"] >= thr]

        def hit(span: Span) -> bool:
            return any(span[0] - margin_s <= t <= span[1] + margin_s for t in ts)

        row = {"rallies_covered": sum(1 for g in gt if hit(g)), "gt_total": len(gt),
               "phantoms_killed": sum(1 for p in phantoms if not hit(p)),
               "phantom_total": len(phantoms)}
        if player_energy is not None:
            row["fused_rallies_covered"] = sum(
                1 for i, g in enumerate(gt) if hit(g) or settled_gt[i])
            row["fused_phantoms_killed"] = sum(
                1 for i, p in enumerate(phantoms) if not (hit(p) or settled_ph[i]))
        out[thr] = row
    return out


def settled_counts(pred: Sequence[Span], gt: Sequence[Span], player_energy: Sequence,
                   min_overlap_s: float = 0.2, settle_kmh: float = SETTLE_KMH,
                   settle_window_s: float = SETTLE_WINDOW_S) -> dict:
    """Settledness split on its own, threshold-free: how many trainer-approved rallies vs
    phantoms start from a settled instant (`players_settled_before` at ``settle_kmh``). Judges
    the cue's separating power independently of any swing threshold; the CLI runs a small kmh
    grid so one Mac run also answers whether a looser/tighter settle bar separates better."""
    phantoms = phantom_spans(pred, gt, min_overlap_s)

    def settled(span: Span) -> bool:
        return players_settled_before(span[0], player_energy, settle_kmh, settle_window_s)

    return {"settle_kmh": settle_kmh,
            "gt_settled": sum(1 for g in gt if settled(g)), "gt_total": len(gt),
            "phantom_settled": sum(1 for p in phantoms if settled(p)),
            "phantom_total": len(phantoms)}


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
    output' claim silently breaks. Knobs (incl. RALLY_SWING_BRIDGE) come from env exactly like
    the pipeline, so this single path stays truthful once the swing-bridge flag flips on."""
    import json
    from pathlib import Path

    from .pipeline import rally_hit
    from .rally import segment as seg

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])

    dets = rally_hit.load_dets(str(cache / "shuttle.parquet"), frame_w, frame_h)

    out = seg.build_rallies(dets, load_player_energy(video_id, cfg), load_swings(video_id, cfg))
    return [(float(r["t_start_s"]), float(r["t_end_s"])) for r in out["rallies"]]


def load_player_energy(video_id: str, cfg) -> Optional[list]:
    """Player motion series ``[(t_s, kmh)]`` via ``rally_hit._player_energy`` from the cached
    tracks.parquet — the exact series the pipeline's serve cue sees (single definition; the
    diagnostics path re-reads the small parquet rather than threading dataframes around).
    None when the parquet is absent or empty."""
    from pathlib import Path

    from .pipeline import rally_hit

    tracks_path = Path(cfg.frame_cache_dir).expanduser() / str(video_id) / "tracks.parquet"
    if not tracks_path.exists():
        return None
    import polars as pl  # lazy

    tdf = pl.read_parquet(tracks_path)
    return rally_hit._player_energy(tdf) if not tdf.is_empty() else None


def load_swings(video_id: str, cfg) -> Optional[list]:
    """Production-shaped swing events (thresholded + merged, via rally_hit's own join) — the
    exact events the pipeline's swing-bridge rule sees. None when tracks/pose parquets are
    absent or empty (same guard as ``load_player_energy``)."""
    import json
    from pathlib import Path

    from .pipeline import rally_hit

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    tracks_path, pose_path = cache / "tracks.parquet", cache / "pose.parquet"
    if not tracks_path.exists() or not pose_path.exists():
        return None
    import polars as pl  # lazy

    tdf, pdf = pl.read_parquet(tracks_path), pl.read_parquet(pose_path)
    if tdf.is_empty() or pdf.is_empty():
        return None
    manifest = json.loads((cache / "frames.json").read_text())
    frame_t = {int(f["frame"]): float(f["t"]) for f in manifest["frames"]}
    return rally_hit._swings_from_frames(pdf, tdf, frame_t) or None


def load_swing_samples(video_id: str, cfg) -> Tuple[List[dict], Dict[int, str]]:
    """Per-sample swing strengths (emission floor 0, no coalescing — the raw distribution the
    thresholds are picked from) + track→court-half map. Pose rows join to tracks.parquet kept
    track ids; frame→t via frames.json; `source=\"interp\"` rows have no pose — harmless.
    Without tracks/pose parquets there is no swing signal at all — empty report, not a crash."""
    import json
    from pathlib import Path

    import polars as pl  # lazy

    from .rally.swing import swing_events

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    if not (cache / "tracks.parquet").exists() or not (cache / "pose.parquet").exists():
        return [], {}
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


# Low-end resolution matters: the naive missed==0 pin point is low, and the fused rule's whole
# point is raising thr past it — the decision happens between 0.25 and 2.5 tl/s.
SWEEP_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def _fmt_pct(st: dict) -> str:
    if not st["n"]:
        return "n=0"
    return (f"n={st['n']:>6}  p10={st['p10']:5.2f}  p25={st['p25']:5.2f}  p50={st['p50']:5.2f}"
            f"  p75={st['p75']:5.2f}  p90={st['p90']:5.2f}")


def _max_key(r: dict) -> tuple:
    """Sort key for per-rally records: NO-SWING (unsavable) first, then ascending strength."""
    return (r["max_strength"] is not None, r["max_strength"] or 0.0)


def _fmt_rec(r: dict) -> str:
    """One per-rally record line (serve-window records carry no span_s — omitted)."""
    span = f"span={r['span_s']:5.2f}s  " if r.get("span_s") is not None else ""
    ms = "NO-SWING " if r["max_strength"] is None else f"max={r['max_strength']:5.2f}"
    half = f"  {r['best_half'] or '?':4}" if r["max_strength"] is not None else ""
    rel = f"  t_rel={r['best_t_rel']:+5.2f}s" if r["best_t_rel"] is not None else ""
    return f"t_start={r['t_start']:8.2f}s  {span}{ms}{half}{rel}"


def _rally_max_summary(records: List[dict]) -> dict:
    """Percentiles of the non-None per-rally maxima + the NO-SWING count + raw records."""
    vals = [r["max_strength"] for r in records if r["max_strength"] is not None]
    return {"pctiles": _pctiles(vals),
            "n_no_swing": sum(1 for r in records if r["max_strength"] is None),
            "records": records}


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
                    help="swing-strength tuning report (in/between + serve-window near/far + "
                         "per-rally max gt/phantom + settled counts + swing/fused veto sweep)")
    ap.add_argument("--bridge-hist", action="store_true",
                    help="bridge-gap tuning report (T8): split-gap (should-bridge) vs real-gap "
                         "(must-not-bridge) swing-strength distributions + operating-point sweep")
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
        player_energy = load_player_energy(video["id"], cfg)
        phantoms = phantom_spans(pred, gt)
        split = swing_strength_split(samples, gt, half_by_track)
        serve = serve_window_report(samples, gt, half_by_track)
        gt_max = rally_max_strengths(samples, gt, half_by_track=half_by_track)
        ph_max = rally_max_strengths(samples, phantoms, half_by_track=half_by_track)
        settled = ([settled_counts(pred, gt, player_energy, settle_kmh=k)
                    for k in (3.0, 4.0, 6.0)] if player_energy else None)
        sweep = veto_operating_points(samples, pred, gt, SWEEP_THRESHOLDS,
                                      player_energy=player_energy)
        if args.json:
            print(json.dumps({"slug": args.slug, "split": split, "serve_window": serve,
                              "rally_max": {"gt": _rally_max_summary(gt_max),
                                            "phantom": _rally_max_summary(ph_max)},
                              "settled": settled,
                              "sweep": {str(k): v for k, v in sweep.items()}}, indent=2))
            return 0
        print(f"[{args.slug}] swing-strength tuning report (tl/s, per-sample, floor 0)")
        print("=" * 68)
        for key in ("in_rally", "in_rally_near", "in_rally_far", "between"):
            if key in split:
                print(f"{key:14}: {_fmt_pct(split[key])}")
        print("-" * 68)
        print(f"serve windows (t_start -{serve['pre_s']:.1f}/+{serve['post_s']:.1f} s, "
              "per-rally max, attribution = strongest swing; server UNLABELED — skew is the "
              "back-to-camera signal):")
        att = serve["attribution"]
        print(f"  attribution: near={att['near']}  far={att['far']}  none={att['none']}   "
              f"rallies with NO serve-window sample: {serve['no_sample_rallies']}")
        for h in ("near", "far"):
            print(f"  {h + '-attributed maxima':22}: {_fmt_pct(serve['max_by_half'][h])}")
        print("  weakest serve windows:")
        for r in sorted(serve["per_rally"], key=_max_key)[:5]:
            print(f"    {_fmt_rec(r)}")
        print("-" * 68)
        print("per-rally max strength over span ±0.5 s (THE veto quantity):")
        for name, recs in (("gt", gt_max), ("phantom", ph_max)):
            s = _rally_max_summary(recs)
            print(f"  {name:7}: NO-SWING={s['n_no_swing']:2}  {_fmt_pct(s['pctiles'])}")
        print("  weakest gt rallies (pin the veto threshold; NO-SWING = unsavable by swing "
              "alone):")
        for r in sorted(gt_max, key=_max_key)[:8]:
            print(f"    {_fmt_rec(r)}")
        print("-" * 68)
        if settled:
            print(f"settled-before-start (players_settled_before, window {SETTLE_WINDOW_S} s):")
            for s in settled:
                print(f"  kmh={s['settle_kmh']:3.1f}  gt {s['gt_settled']:3}/{s['gt_total']:<3}"
                      f"  phantoms {s['phantom_settled']:3}/{s['phantom_total']}")
            print(f"veto sweep (swing >= thr within ±0.5 s; fused = swing OR "
                  f"settled@{SETTLE_KMH:.1f} km/h):")
        else:
            print("settled-before-start: no tracks.parquet — settled/fused columns unavailable")
            print("veto sweep (swing >= thr within ±0.5 s saves a rally / kills a phantom):")
        for thr, pt in sweep.items():
            gate = "OK  " if pt["rallies_covered"] == pt["gt_total"] else "MISS"
            line = (f"  thr={thr:4.2f}  swing {pt['rallies_covered']:3}/{pt['gt_total']:<3} "
                    f"{gate} kills {pt['phantoms_killed']:3}/{pt['phantom_total']}")
            if "fused_rallies_covered" in pt:
                fgate = "OK  " if pt["fused_rallies_covered"] == pt["gt_total"] else "MISS"
                line += (f"  |  fused {pt['fused_rallies_covered']:3}/{pt['gt_total']:<3} "
                         f"{fgate} kills {pt['fused_phantoms_killed']:3}/{pt['phantom_total']}")
            print(line)
        return 0

    if args.bridge_hist:
        samples, _ = load_swing_samples(video["id"], cfg)
        splits = split_gap_spans(pred, gt)
        # candidate range only: gaps a swing-bridge could ever act on (longer real gaps are
        # never at risk — they sit well outside RALLY_SWING_BRIDGE_S regardless of threshold).
        cand_reals = [(gs, ge) for gs, ge in real_gap_spans(gt) if BRIDGE_S < ge - gs <= RALLY_SWING_BRIDGE_S]
        split_recs = bridge_gap_strengths(samples, splits, RALLY_SERVE_EXCLUSION_S)
        real_recs = bridge_gap_strengths(samples, cand_reals, RALLY_SERVE_EXCLUSION_S)
        sweep = bridge_operating_points(samples, splits, cand_reals, SWEEP_THRESHOLDS,
                                        RALLY_SERVE_EXCLUSION_S)
        if args.json:
            print(json.dumps({"slug": args.slug, "split_gaps": split_recs, "real_gaps": real_recs,
                              "sweep": {str(k): v for k, v in sweep.items()}}, indent=2))
            return 0
        print(f"[{args.slug}] bridge-gap tuning report (tl/s, T8; candidate range "
              f"{BRIDGE_S:.2f}–{RALLY_SWING_BRIDGE_S:.2f}s)")
        print("=" * 68)
        for name, recs in (("should-bridge (trainer-merged splits)", split_recs),
                          ("must-NOT-bridge (real gaps, same range)", real_recs)):
            s = _rally_max_summary(recs)
            print(f"{name}: n={len(recs):3}  NO-SWING={s['n_no_swing']:2}  {_fmt_pct(s['pctiles'])}")
        print("-" * 68)
        print("bridge sweep (swing >= thr in gap interior, excl. last "
              f"{RALLY_SERVE_EXCLUSION_S:.2f}s before the gap ends):")
        for thr, pt in sweep.items():
            gate = "OK  " if pt["false_merges"] == 0 else "MISS"
            print(f"  thr={thr:4.2f}  bridges {pt['splits_bridged']:3}/{pt['splits_total']:<3}"
                 f"  false-merges {pt['false_merges']:3}/{pt['real_gaps_total']:<3} {gate}")
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
