"""Pure landing-frame selection + in/out call (no torch/cv2 — tested under the dev extra).

The one moment a monocular ground homography is valid for the shuttle is floor contact — so the whole
problem reduces to: which detection is the landing detection, and how sure are we? The search runs in
NORMALIZED IMAGE space (``x=px/W``, ``y=py/H``; y grows downward, so descent is positive vertical
velocity). Court ``(u, v)`` is only read off the chosen contact detection, where the per-video ground
homography already projected it validly (worker/shuttle/track.py).

Honest degradation is the contract (specs/features/shuttle-landing-inout.md): occlusion at contact,
airspace-mask exit, a 30 fps frame straddle, or a sparse tail all yield ``located=False`` — the arc is
NEVER extrapolated to the floor. All thresholds are untuned first guesses until real-footage
validation on the Mac Studio; env-overridable like the ``RALLY_*`` knobs in segment.py.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

from ..geometry import SINGLES_INSET_M
from ..insights import COURT_L_M, COURT_W_M
from ..manifest import REFERENCE_FPS, fps_from_frames_t, frames_for_seconds
from .segment import END_MIN_CONF

Det = Dict[str, float]

# Singles side lines, normalised by the doubles width: u ∈ [0.0754, 0.9246]. Baselines are the full
# length in singles and doubles alike (v ∈ [0, 1]).
SINGLES_U_MIN = SINGLES_INSET_M / COURT_W_M
SINGLES_U_MAX = 1.0 - SINGLES_U_MIN

# --- landing-frame selection (image space) ----------------------------------------------------------
LANDING_WINDOW_S = float(os.environ.get("LANDING_WINDOW_S", "0.5"))    # terminal window before the last det
# Detection counts / frame gaps are DURATIONS (specs/features/analysis-30fps.md): the tuned 8 / 4 / 2 were
# measured on the real ~10 fps cadence → 0.8 s / 0.4 s / 0.2 s. detect_landing derives the counts from the
# rally's cadence (`fps`, inferred from its (frame, t) span when not given); the *_DETS / *_FRAMES names
# keep the REFERENCE_FPS equivalents for direct callers.
LANDING_MAX_WINDOW_S = float(os.environ.get("LANDING_MAX_WINDOW_S", "0.8"))    # cap on dets in the window
LANDING_MIN_DETS_S = float(os.environ.get("LANDING_MIN_DETS_S", "0.4"))        # fewer → sparse
LANDING_MAX_END_GAP_S = float(os.environ.get("LANDING_MAX_END_GAP_S", "0.2"))  # larger → straddle
LANDING_MAX_WINDOW_DETS = frames_for_seconds(LANDING_MAX_WINDOW_S, REFERENCE_FPS)
LANDING_MIN_DETS = frames_for_seconds(LANDING_MIN_DETS_S, REFERENCE_FPS)
LANDING_MAX_END_GAP_FRAMES = frames_for_seconds(LANDING_MAX_END_GAP_S, REFERENCE_FPS)
LANDING_EDGE_MARGIN = float(os.environ.get("LANDING_EDGE_MARGIN", "0.04"))       # border zone → edge_cutoff
LANDING_UNCERTAIN_EDGE = float(os.environ.get("LANDING_UNCERTAIN_EDGE", "0.08")) # barrel-bow zone → uncertain
LANDING_FLAT_VY_ABS = float(os.environ.get("LANDING_FLAT_VY_ABS", "0.25"))   # |vy| (norm/s) below = at rest
LANDING_FLAT_VY_FRAC = float(os.environ.get("LANDING_FLAT_VY_FRAC", "0.35")) # …or below this × peak descent
# --- in/out call -------------------------------------------------------------------------------------
LANDING_BAND_CM = float(os.environ.get("LANDING_BAND_CM", "45"))       # flat ± uncertainty band
LANDING_MAX_OUT_CM = float(os.environ.get("LANDING_MAX_OUT_CM", "100"))  # farther out = elevated rest
LANDING_NET_MARGIN_M = float(os.environ.get("LANDING_NET_MARGIN_M", "1.0"))  # near-net = ambiguous half
LANDING_UNCERTAIN_CONF = 0.2


def _near_border(x: float, y: float, margin: float) -> bool:
    return min(x, 1.0 - x, y, 1.0 - y) < margin


def landing_counts(fps: float) -> Dict[str, int]:
    """The detection-count / frame-gap thresholds at a cache cadence of `fps` (REFERENCE_FPS → 8/4/2)."""
    return {
        "max_window_dets": frames_for_seconds(LANDING_MAX_WINDOW_S, fps),
        "min_dets": frames_for_seconds(LANDING_MIN_DETS_S, fps),
        "max_end_gap_frames": frames_for_seconds(LANDING_MAX_END_GAP_S, fps),
    }


def _cadence(rally_dets: Sequence[Det], fps: Optional[float]) -> float:
    return fps or fps_from_frames_t((d["frame"], d["t"]) for d in rally_dets if "frame" in d) or REFERENCE_FPS


def detect_landing(rally_dets: Sequence[Det], min_conf: float = END_MIN_CONF,
                   fps: Optional[float] = None) -> dict:
    """Find the floor-contact detection in a rally's terminal trajectory, or say honestly why not.

    Located ⇒ ``{located: True, t, frame, x, y, u, v, edge_uncertain, window_conf}`` where the landing
    detection is the FIRST of the flattened tail (descent stopped there). Degraded ⇒
    ``{located: False, reason}`` with reason ∈ sparse | straddle | edge_cutoff | no_descent | no_contact
    — and deliberately NO position: a fabricated landing pixel is worse than an absent one.
    ``fps`` is the cache cadence the count thresholds are derived from (``landing_counts``); omitted →
    inferred from the detections' (frame, t) span, else REFERENCE_FPS.
    """
    counts = landing_counts(_cadence(rally_dets, fps))
    reliable = [d for d in rally_dets if d.get("conf", 1.0) >= min_conf]
    if len(reliable) < counts["min_dets"]:
        return {"located": False, "reason": "sparse"}
    last, prev = reliable[-1], reliable[-2]
    # Checked BEFORE the window-density test: a terminal dropout longer than LANDING_MAX_END_GAP_S also
    # thins the 0.5 s window, and "contact fell into a gap" is the honest reason, at any cadence.
    if last["t"] - prev["t"] > LANDING_MAX_END_GAP_S + 1e-9:
        return {"located": False, "reason": "straddle"}
    t_last = last["t"]
    window = [d for d in reliable if t_last - d["t"] <= LANDING_WINDOW_S][-counts["max_window_dets"]:]
    if len(window) < counts["min_dets"]:
        return {"located": False, "reason": "sparse"}
    if _near_border(last["x"], last["y"], LANDING_EDGE_MARGIN):
        return {"located": False, "reason": "edge_cutoff"}  # descent left the frame / airspace mask

    vys = []
    for a, b in zip(window, window[1:]):
        dt = b["t"] - a["t"]
        vys.append(((b["y"] - a["y"]) / dt) if dt > 0 else 0.0)
    peak_descent = max(vys)
    if peak_descent < LANDING_FLAT_VY_ABS:
        return {"located": False, "reason": "no_descent"}   # nothing fell (caught / mid-air loss)
    flat_thr = max(LANDING_FLAT_VY_ABS, LANDING_FLAT_VY_FRAC * peak_descent)
    if abs(vys[-1]) > flat_thr:
        return {"located": False, "reason": "no_contact"}   # still falling when the track ends

    # first index whose step — and every later step — is flat: the start of the resting tail.
    start = len(vys) - 1
    while start > 0 and abs(vys[start - 1]) <= flat_thr:
        start -= 1
    d = window[start]
    return {"located": True, "t": d["t"], "frame": d["frame"], "x": d["x"], "y": d["y"],
            "u": d.get("u"), "v": d.get("v"),
            "edge_uncertain": _near_border(d["x"], d["y"], LANDING_UNCERTAIN_EDGE),
            "window_conf": sum(w.get("conf", 1.0) for w in window) / len(window)}


def classify_landing(u: float, v: float, band_cm: float = LANDING_BAND_CM,
                     discipline: str = "singles") -> dict:
    """In/out call for a floor-contact court position, with a flat ± uncertainty band.

    ``out``/``in`` only when the point clears the nearest boundary by MORE than the band; anything
    inside the band is ``uncertain`` — a monocular near-line call is not an umpire's call.
    ``margin_cm`` is signed (positive inside the court); ``line`` names the nearest boundary.
    """
    u_min, u_max = (0.0, 1.0) if discipline == "doubles" else (SINGLES_U_MIN, SINGLES_U_MAX)
    du_cm = min(u - u_min, u_max - u) * COURT_W_M * 100.0
    dv_cm = min(v, 1.0 - v) * COURT_L_M * 100.0
    margin_cm = min(du_cm, dv_cm)
    line = "side" if du_cm < dv_cm else "baseline"
    if margin_cm < -band_cm:
        call = "out"
    elif margin_cm > band_cm:
        call = "in"
    else:
        call = "uncertain"
    return {"call": call, "line": line, "margin_cm": round(margin_cm, 1)}


def landing_event(rally_dets: Sequence[Det], discipline: str = "singles",
                  min_conf: float = END_MIN_CONF, fps: Optional[float] = None) -> dict:
    """The ``events.type='landing'`` payload for one rally (spec shape).

    Located: ``{located, call, line, u, v, px, py, margin_cm, band_cm, confidence}`` — ``px, py`` are
    NORMALIZED image coordinates (same convention as the ``rally_end`` payload's x, y). Degraded:
    ``{located: False, call: 'uncertain', line: None, reason, band_cm, confidence: None}`` with
    reason ∈ sparse | straddle | edge_cutoff | no_descent | no_contact | implausible — the last one
    fires when the contact's court projection is > ``LANDING_MAX_OUT_CM`` outside every boundary
    (an above-floor rest seen through the ground homography, e.g. a racket catch).
    ``edge_uncertain`` landings keep their position but are forced ``uncertain`` (lens bow near the
    frame edges; undistortion is still a no-op).
    """
    d = detect_landing(rally_dets, min_conf, fps=fps)
    if not d["located"]:
        return {"located": False, "call": "uncertain", "line": None, "reason": d["reason"],
                "band_cm": LANDING_BAND_CM, "confidence": None}
    cls = classify_landing(d["u"], d["v"], discipline=discipline)
    if cls["margin_cm"] < -LANDING_MAX_OUT_CM:
        # A rally-ending landing this far outside every line is (almost) never a floor contact — it is
        # the ground projection of a shuttle that came to REST ABOVE the floor (racket catch, net cord,
        # between-points handover), which the image-space flatness test cannot tell from contact
        # (proven on real footage 2026-07-03). Degrade honestly instead of fabricating an out call.
        return {"located": False, "call": "uncertain", "line": None, "reason": "implausible",
                "band_cm": LANDING_BAND_CM, "confidence": None}
    call = "uncertain" if d["edge_uncertain"] else cls["call"]
    if call == "uncertain":
        confidence = LANDING_UNCERTAIN_CONF
    else:
        # full geometric confidence at 2× the band away from the line, scaled by tracker confidence.
        confidence = round(min(1.0, abs(cls["margin_cm"]) / (2.0 * LANDING_BAND_CM)) * d["window_conf"], 3)
    return {"located": True, "call": call, "line": cls["line"], "u": d["u"], "v": d["v"],
            "px": d["x"], "py": d["y"], "margin_cm": cls["margin_cm"],
            "band_cm": LANDING_BAND_CM, "confidence": confidence, "t": d["t"]}


def winner_half(v: float, net_margin_m: float = LANDING_NET_MARGIN_M) -> Optional[str]:
    """The court half an OUT landing awards the rally to (winner = the receiving side).

    The shuttle lands out on/beyond the receiver's half, so the landing half IS the winner's half; the
    hitter (the error) is the other half. Within ``net_margin_m`` of the net the shot may not have
    crossed at all (a wide net shot on the hitter's own half) → ``None``, no guess.
    """
    if abs(v - 0.5) * COURT_L_M < net_margin_m:
        return None
    return "near" if v >= 0.5 else "far"
