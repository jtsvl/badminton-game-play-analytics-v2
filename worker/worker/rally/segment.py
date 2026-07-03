"""Pure rally-segmentation logic for stage 6 (no torch/cv2 — tested under the dev extra).

Input is the shuttle spotter's detections as dicts ``{t, x, y, conf, frame, u, v}`` where ``x=px/W`` and
``y=py/H`` are NORMALIZED IMAGE coordinates (the trajectory lives here — the ground homography projects an
airborne shuttle past the horizon, so court ``(u, v)`` is unusable in flight and only surfaces at
find_rally_end for the floor-contact candidate). Thresholds are module constants, env-overridable like the
``SHUTTLE_*`` knobs, and tuned against real footage when the stage runs.

Honest limit: this *reduces* foreign-court intrusion (short discontinuous neighbour runs are dropped) but
cannot eliminate it — a foreign shuttle that lingers several in-mask frames with rally-like motion leaks.
Monocular, single-camera.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

Det = Dict[str, float]

# --- trajectory grouping ---------------------------------------------------------------------------
GAP_S = float(os.environ.get("RALLY_GAP_S", "0.5"))        # split a flight run on a temporal gap this big
MAX_JUMP = float(os.environ.get("RALLY_MAX_JUMP", "0.10")) # normalized step above which a jump needs to be
MIN_COS = float(os.environ.get("RALLY_MIN_COS", "0.3"))    # collinear (cos>=MIN_COS) to stay one run (smash)
# --- foreign-run rejection -------------------------------------------------------------------------
MIN_RUN_LEN = int(os.environ.get("RALLY_MIN_RUN_LEN", "4"))      # a run this long counts as sustained flight
BRIDGE_S = float(os.environ.get("RALLY_BRIDGE_S", "1.5"))        # max time gap to connect/bridge two runs
BRIDGE_DIST = float(os.environ.get("RALLY_BRIDGE_DIST", "0.15")) # max endpoint distance to connect two runs
# --- swing-conditional gap bridging (T8, specs/features/pose-swing-fusion.md) -----------------------
# Promoted 2026-07-03: --bridge-hist on all 5 reviewed games recovers ~80% of measured splits at
# thr=4.0 tl/s with zero measured false-merges (the real-gap population in the candidate range was
# empty on this corpus — an untestable-not-proven-safe gap, accepted deliberately; see spec).
RALLY_SWING_BRIDGE = os.environ.get("RALLY_SWING_BRIDGE", "1") == "1"  # master switch (default on)
RALLY_SWING_BRIDGE_S = float(os.environ.get("RALLY_SWING_BRIDGE_S", "2.5"))  # raised cap when swing-backed
RALLY_SWING_BRIDGE_MIN_STRENGTH = float(os.environ.get("RALLY_SWING_BRIDGE_MIN_STRENGTH", "4.0"))
RALLY_SERVE_EXCLUSION_S = float(os.environ.get("RALLY_SERVE_EXCLUSION_S", "0.75"))  # no bridge this close to gap end
# --- rally assembly --------------------------------------------------------------------------------
MIN_RALLY_S = float(os.environ.get("RALLY_MIN_RALLY_S", "0.5"))  # discard spans shorter than this
MIN_DETS = int(os.environ.get("RALLY_MIN_DETS", "5"))            # discard rallies with fewer detections
# --- serve / end / hits ----------------------------------------------------------------------------
SETTLE_KMH = float(os.environ.get("RALLY_SETTLE_KMH", "4.0"))         # players "settled" below this speed
SETTLE_WINDOW_S = float(os.environ.get("RALLY_SETTLE_WINDOW_S", "0.5"))  # over this window before the serve
SERVE_CONF_SETTLED = float(os.environ.get("RALLY_SERVE_CONF_SETTLED", "0.7"))   # serve cue confidence...
SERVE_CONF_FALLBACK = float(os.environ.get("RALLY_SERVE_CONF_FALLBACK", "0.4")) # ...settled vs first-det
END_MIN_CONF = float(os.environ.get("RALLY_END_MIN_CONF", "0.5"))     # ignore low-conf blips at rally end
HIT_DEADBAND = float(os.environ.get("RALLY_HIT_DEADBAND", "0.01"))    # ignore vertical jitter below this
HIT_CONF = float(os.environ.get("RALLY_HIT_CONF", "0.3"))            # provisional hit-event confidence


def _dist(a: Det, b: Det) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _unit(a: Det, b: Det) -> Optional[tuple]:
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n else None


def _cos(d1: Optional[tuple], d2: Optional[tuple]) -> float:
    return d1[0] * d2[0] + d1[1] * d2[1] if d1 and d2 else -1.0


def group_trajectories(dets: Sequence[Det], max_gap_s: float = GAP_S, max_jump: float = MAX_JUMP,
                       min_cos: float = MIN_COS) -> List[List[Det]]:
    """Split time-sorted detections into continuous flight runs.

    Break on a temporal gap (> ``max_gap_s``) or a large jump that is NOT a collinear continuation of the
    current motion. A genuine smash is consecutive large *collinear* steps — there the direction holds, so
    the run stays whole (same intent as ``geometry.reject_shuttle_outliers``, lifted to runs). A large jump
    that reverses or veers (a neighbour's shuttle teleporting in) starts a new run. A large jump at a run's
    very start is ambiguous (no prior direction) so it is kept — ``reject_foreign_runs`` cleans up isolated
    short runs afterward."""
    dets = list(dets)
    if not dets:
        return []
    runs: List[List[Det]] = [[dets[0]]]
    for prev, cur in zip(dets, dets[1:]):
        run = runs[-1]
        if cur["t"] - prev["t"] > max_gap_s:
            runs.append([cur])
            continue
        if _dist(prev, cur) <= max_jump:
            run.append(cur)
            continue
        prev_dir = _unit(run[-2], run[-1]) if len(run) >= 2 else None
        if prev_dir is None or _cos(prev_dir, _unit(prev, cur)) >= min_cos:
            run.append(cur)       # ambiguous start, or a collinear smash continuation
        else:
            runs.append([cur])    # non-collinear teleport → a different shuttle
    return runs


def _time_gap(a: List[Det], b: List[Det]) -> float:
    """Positive time gap between two time-ordered runs; 0 if they overlap in time."""
    if a[-1]["t"] < b[0]["t"]:
        return b[0]["t"] - a[-1]["t"]
    if b[-1]["t"] < a[0]["t"]:
        return a[0]["t"] - b[-1]["t"]
    return 0.0


def _endpoint_dist(a: List[Det], b: List[Det]) -> float:
    """Smallest image distance among the runs' endpoints (where a real continuation would connect)."""
    return min(_dist(pa, pb) for pa in (a[0], a[-1]) for pb in (b[0], b[-1]))


def reject_foreign_runs(runs: Sequence[List[Det]], min_run_len: int = MIN_RUN_LEN,
                        bridge_s: float = BRIDGE_S, bridge_dist: float = BRIDGE_DIST) -> List[List[Det]]:
    """Drop short runs that don't connect to any sustained run — a neighbour's shuttle crossing through.

    A run of >= ``min_run_len`` detections is sustained flight (kept). A short run is kept only if it abuts a
    sustained run in BOTH time (gap <= ``bridge_s``) and space (an endpoint within ``bridge_dist``) — i.e. it
    is a fragment of the same rally split off by a dropout or a smash, not an isolated foreign blip. Per-run
    association (not a global filter) so real smash fragments survive. If nothing is sustained, there is no
    rally and everything is dropped."""
    longs = [r for r in runs if len(r) >= min_run_len]
    if not longs:
        return []
    kept: List[List[Det]] = []
    for r in runs:
        if len(r) >= min_run_len:
            kept.append(r)
        elif any(_time_gap(r, lr) <= bridge_s and _endpoint_dist(r, lr) <= bridge_dist for lr in longs):
            kept.append(r)
    return kept


def _swing_in_gap(gap_start: float, gap_end: float, swings: Sequence[dict], min_strength: float,
                  serve_exclusion_s: float) -> bool:
    """True iff some swing >= ``min_strength`` falls in ``[gap_start, gap_end - serve_exclusion_s]``.

    The exclusion zone excludes swings close to the gap's end: a quick re-serve puts a genuine swing
    there, and bridging on it would glue two real rallies together (unrecoverable — no split op)."""
    hi = gap_end - serve_exclusion_s
    return hi > gap_start and any(gap_start <= s["t"] <= hi and s["strength"] >= min_strength
                                  for s in swings)


def merge_runs_into_rallies(runs: Sequence[List[Det]], max_bridge_s: float = BRIDGE_S,
                            min_rally_s: float = MIN_RALLY_S, min_dets: int = MIN_DETS,
                            swings: Optional[Sequence[dict]] = None,
                            swing_bridge: bool = RALLY_SWING_BRIDGE,
                            swing_bridge_s: float = RALLY_SWING_BRIDGE_S,
                            swing_bridge_min_strength: float = RALLY_SWING_BRIDGE_MIN_STRENGTH,
                            serve_exclusion_s: float = RALLY_SERVE_EXCLUSION_S) -> List[List[Det]]:
    """Group time-ordered runs into rallies: runs within ``max_bridge_s`` merge (a white-wall dropout stays
    one rally), a larger gap closes the rally. Each rally's detections are concatenated and time-sorted;
    spans shorter than ``min_rally_s`` or with fewer than ``min_dets`` detections are discarded as noise.

    When ``swing_bridge`` is on and ``swings`` is given, a gap in ``(max_bridge_s, swing_bridge_s]``
    also merges iff a swing carrying enough strength lies in its interior (players kept swinging through
    a longer shuttle-detection dropout) — see ``_swing_in_gap``. Default off until a measured tuning
    pass sets ``swing_bridge_min_strength`` (T8, specs/features/pose-swing-fusion.md)."""
    runs = [r for r in runs if r]
    if not runs:
        return []
    groups: List[List[List[Det]]] = [[runs[0]]]
    for prev, cur in zip(runs, runs[1:]):
        gap = _time_gap(prev, cur)
        if gap <= max_bridge_s:
            groups[-1].append(cur)
        elif (swing_bridge and swings and gap <= swing_bridge_s and
              _swing_in_gap(prev[-1]["t"], cur[0]["t"], swings, swing_bridge_min_strength,
                           serve_exclusion_s)):
            groups[-1].append(cur)
        else:
            groups.append([cur])
    rallies: List[List[Det]] = []
    for g in groups:
        dets = sorted((d for run in g for d in run), key=lambda d: d["t"])
        if dets[-1]["t"] - dets[0]["t"] >= min_rally_s and len(dets) >= min_dets:
            rallies.append(dets)
    return rallies


def players_settled_before(t: float, player_energy: Sequence, thresh: float = SETTLE_KMH,
                           window_s: float = SETTLE_WINDOW_S) -> bool:
    """True if player motion energy stayed below ``thresh`` (km/h) over ``window_s`` before time ``t``.

    ``player_energy`` is ``[(t_s, energy_kmh)]`` — the combined player step-speed series (built from
    ``insights.step_speeds_kmh``). The settled-players half of the serve cue is the reliable one in 2D
    ("shuttle low" is perspective-ambiguous monocular)."""
    window = [e for (tt, e) in player_energy if t - window_s <= tt < t]
    return bool(window) and max(window) < thresh


def find_serve_frame(rally_dets: Sequence[Det], player_energy: Optional[Sequence] = None,
                     settle_kmh: float = SETTLE_KMH, settle_window_s: float = SETTLE_WINDOW_S) -> Det:
    """Serve = the rally's first shuttle motion. Confidence is high when the players were settled just
    before it (the reliable cue); otherwise this is a lower-confidence fallback to the first detection."""
    first = rally_dets[0]
    settled = bool(player_energy) and players_settled_before(
        first["t"], player_energy, settle_kmh, settle_window_s)
    return {"t": first["t"], "frame": first["frame"], "settled": settled,
            "confidence": SERVE_CONF_SETTLED if settled else SERVE_CONF_FALLBACK}


def find_rally_end(rally_dets: Sequence[Det], min_conf: float = END_MIN_CONF) -> Det:
    """Last reliable in-flight point before the terminal gap. Exposes image ``(x, y)`` AND court
    ``(u, v)`` — the floor-contact candidate the future in/out feature will classify. We do NOT decide
    in/out here (that needs the dedicated landing-frame selection). Low-conf trailing blips are skipped."""
    reliable = [d for d in rally_dets if d.get("conf", 1.0) >= min_conf] or list(rally_dets)
    last = reliable[-1]
    return {"t": last["t"], "frame": last["frame"], "x": last["x"], "y": last["y"],
            "u": last.get("u"), "v": last.get("v")}


def hit_points(rally_dets: Sequence[Det], deadband: float = HIT_DEADBAND) -> List[Det]:
    """Provisional racket-contact points = significant reversals in the shuttle's image-space VERTICAL
    travel (each contact flips it between rising and falling). Image-space estimate, refined later by the
    stage-7 swing detector; jitter below ``deadband`` is ignored. Returns the detection at each reversal."""
    pts = list(rally_dets)
    if len(pts) < 3:
        return []
    hits: List[Det] = []
    prev_sign = 0
    for prev, cur in zip(pts, pts[1:]):
        dy = cur["y"] - prev["y"]
        if abs(dy) < deadband:
            continue
        sign = 1 if dy > 0 else -1
        if prev_sign and sign != prev_sign:
            hits.append(prev)   # the local extremum where travel reversed
        prev_sign = sign
    return hits


def count_hits(rally_dets: Sequence[Det], deadband: float = HIT_DEADBAND) -> Optional[int]:
    """Provisional shot count from vertical reversals; ``None`` when too few points to be meaningful."""
    if len(rally_dets) < 3:
        return None
    return len(hit_points(rally_dets, deadband))


def build_rallies(dets: Sequence[Det], player_energy: Optional[Sequence] = None,
                  swings: Optional[Sequence[dict]] = None,
                  swing_bridge: bool = RALLY_SWING_BRIDGE) -> Dict[str, list]:
    """Orchestrate stage 6: detections → rallies + events shaped for the DB (rallies/events columns).

    Dead time falls out for free — anything not inside a surviving rally span is simply never emitted.
    ``rallies`` rows carry no ``winner_side``/``end_shot_type`` (those are decoupled from the human-entered
    match score and refined by stage 7). ``events`` reference their rally by ``rally_idx`` (the store
    resolves it to ``rally_id`` on insert). Per rally: a ``serve`` event, provisional ``hit`` events, and a
    ``rally_end`` event carrying the floor-contact ``(x, y, u, v)`` for the future in/out feature.
    ``swings`` (player swing-activity events) only change output when ``swing_bridge`` is on — see
    ``merge_runs_into_rallies``."""
    dets = sorted(dets, key=lambda d: d["t"])
    rallies_dets = merge_runs_into_rallies(reject_foreign_runs(group_trajectories(dets)),
                                           swings=swings, swing_bridge=swing_bridge)

    rallies: List[dict] = []
    events: List[dict] = []
    for idx, rd in enumerate(rallies_dets):
        serve = find_serve_frame(rd, player_energy)
        end = find_rally_end(rd)
        hits = hit_points(rd)
        shots = count_hits(rd)
        frame_span = rd[-1]["frame"] - rd[0]["frame"]
        density = len(rd) / (frame_span + 1) if frame_span >= 0 else None
        rallies.append({
            "idx": idx,
            "t_start_s": serve["t"],
            "t_end_s": end["t"],
            "shot_count": (1 + len(hits)) if shots is not None else None,  # serve + contacts (estimate)
            "winner_side": None,
            "end_shot_type": None,
            "confidence": round(min(1.0, density), 3) if density is not None else None,
        })
        events.append({"rally_idx": idx, "t_s": serve["t"], "type": "serve",
                       "payload": {"settled": serve["settled"]}, "confidence": serve["confidence"]})
        for h in hits:
            events.append({"rally_idx": idx, "t_s": h["t"], "type": "hit",
                           "payload": {"provisional": True}, "confidence": HIT_CONF})
        events.append({"rally_idx": idx, "t_s": end["t"], "type": "rally_end",
                       "payload": {"x": end["x"], "y": end["y"], "u": end["u"], "v": end["v"]},
                       "confidence": None})
    return {"rallies": rallies, "events": events}
