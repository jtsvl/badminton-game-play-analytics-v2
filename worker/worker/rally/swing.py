"""Pure swing-activity events from cached RTMPose keypoints (specs/features/pose-swing-fusion.md).

Input rows are pose.parquet rows already mapped to seconds: ``{t, track, kp, kp_score}`` (Halpe-26,
``kp`` = 26 [x, y] image points). No torch/cv2/polars — tested under the dev extra like segment.py.

Honest limit (12 fps): a swing's forward phase is 1–2 analysis frames, so this is an ACTIVITY cue
("racket-like arm motion in this window"), not a hit timer — ±83 ms. Consumers (the stage-6 veto and
bridge rules) only need presence + strength, never precise timing.

Signal per (frame, track): arm-tip speed of the most distal *confident* joint per arm — wrist at
``kp_score >= SWING_KP_CONF``, else elbow scaled by ``SWING_ELBOW_GAIN`` (the elbow moves less than
the racket hand; the gain makes the proxy comparable). Speeds are computed per joint series (never
wrist→elbow displacement, which measures arm extension, not motion). A middle frame lost to motion
blur bridges across the true 2·dt (a blurred peak still shows as one big step); anything beyond
``SWING_MAX_DT_S`` is a tracker dropout and yields no sample — a re-appearing player must not read
as a swing. Normalization: per-track ROLLING-MEDIAN torso length (head 17 → mid-hip 19) over
``SWING_TORSO_WINDOW_S`` — the instantaneous torso collapses in lunges and would spike exactly at
athletic moments. Unit: torso-lengths/second (tl/s), scale-free across near/far court.

A swing event = a maximal run of consecutive samples above ``SWING_MIN_TLPS``; runs closer than
``SWING_MERGE_S`` coalesce; no local-max logic (degenerate at 1–2 samples per swing).
"""
from __future__ import annotations

import math
import os
from bisect import bisect_left, bisect_right, insort
from typing import Dict, List, Optional, Sequence, Tuple

from ..geometry import ELBOW_KP, HEAD_KP, HIP_KP, WRIST_KP

SWING_KP_CONF = float(os.environ.get("SWING_KP_CONF", "0.3"))          # joint counts above this
SWING_MIN_TLPS = float(os.environ.get("SWING_MIN_TLPS", "3.0"))        # emission floor (T6 tunes)
SWING_MERGE_S = float(os.environ.get("SWING_MERGE_S", "0.3"))          # coalesce adjacent runs
SWING_MAX_DT_S = float(os.environ.get("SWING_MAX_DT_S", "0.25"))       # dropout guard
SWING_ELBOW_GAIN = float(os.environ.get("SWING_ELBOW_GAIN", "1.8"))    # elbow→wrist speed proxy
SWING_TORSO_WINDOW_S = float(os.environ.get("SWING_TORSO_WINDOW_S", "2.0"))

Event = Dict[str, float]


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def arm_tip(kp: Sequence, kp_score: Sequence[float], side: int,
            conf_thr: float = SWING_KP_CONF, elbow_gain: float = SWING_ELBOW_GAIN,
            ) -> Optional[Tuple[float, float, float]]:
    """Most distal confident joint of one arm as ``(x, y, gain)``: the wrist at gain 1.0, else the
    elbow at ``elbow_gain``, else None. ``side`` indexes (left, right) = (0, 1)."""
    for joint, gain in ((WRIST_KP[side], 1.0), (ELBOW_KP[side], elbow_gain)):
        if joint < len(kp_score) and kp_score[joint] >= conf_thr:
            return (float(kp[joint][0]), float(kp[joint][1]), gain)
    return None


def rolling_median_torso(samples: Sequence[Tuple[float, float]], window_s: float = SWING_TORSO_WINDOW_S,
                         ) -> List[float]:
    """Median torso length in a centered ±window_s/2 around each sample (samples time-sorted
    ``[(t, length)]``). Two-pointer sliding window with a sorted insertion list — O(n·w)."""
    times = [t for t, _ in samples]
    medians: List[float] = []
    window: List[float] = []  # sorted lengths currently inside the window
    lo = hi = 0
    for t, _ in samples:
        while hi < len(samples) and times[hi] <= t + window_s / 2:
            insort(window, samples[hi][1])
            hi += 1
        while lo < len(samples) and times[lo] < t - window_s / 2:
            window.pop(bisect_left(window, samples[lo][1]))
            lo += 1
        n = len(window)
        medians.append(window[n // 2] if n % 2 else (window[n // 2 - 1] + window[n // 2]) / 2.0)
    return medians


def _joint_speeds(rows: Sequence[dict], joint: int, gain: float, conf_thr: float,
                  max_dt_s: float) -> Dict[float, float]:
    """{t_of_later_frame: image-speed × gain} between consecutive confident sightings of one joint.
    Gaps wider than ``max_dt_s`` yield nothing (dropout guard); shorter gaps divide by the TRUE
    elapsed time, which is exactly the blurred-peak bridge."""
    pts = [(r["t"], r["kp"][joint]) for r in rows
           if joint < len(r["kp_score"]) and r["kp_score"][joint] >= conf_thr]
    out: Dict[float, float] = {}
    for (t1, p1), (t2, p2) in zip(pts, pts[1:]):
        dt = t2 - t1
        if 0 < dt <= max_dt_s:
            out[t2] = _dist(p1, p2) / dt * gain
    return out


def swing_events(pose_rows: Sequence[dict], *, kp_conf: float = SWING_KP_CONF,
                 min_tlps: float = SWING_MIN_TLPS, merge_s: float = SWING_MERGE_S,
                 max_dt_s: float = SWING_MAX_DT_S, elbow_gain: float = SWING_ELBOW_GAIN,
                 torso_window_s: float = SWING_TORSO_WINDOW_S) -> List[Event]:
    """Swing-activity events ``{"t", "track", "strength"}`` (time-sorted across tracks)."""
    by_track: Dict[int, List[dict]] = {}
    for r in pose_rows:
        by_track.setdefault(r["track"], []).append(r)

    events: List[Event] = []
    for track, rows in by_track.items():
        rows.sort(key=lambda r: r["t"])

        torso = [(r["t"], _dist(r["kp"][HEAD_KP], r["kp"][HIP_KP])) for r in rows
                 if r["kp_score"][HEAD_KP] >= kp_conf and r["kp_score"][HIP_KP] >= kp_conf]
        torso = [(t, ln) for t, ln in torso if ln > 0]
        if not torso:
            continue  # nothing to normalize against — no pose, no cue
        torso_t = [t for t, _ in torso]
        medians = rolling_median_torso(torso, torso_window_s)

        def torso_at(t: float) -> float:
            i = min(bisect_right(torso_t, t), len(medians) - 1)  # nearest-from-left is enough
            return medians[i - 1] if i > 0 and (i == len(medians) or t < torso_t[i]) else medians[i]

        # Per side: the wrist series wins where it exists; the elbow proxy fills the rest.
        samples: Dict[float, float] = {}
        for side in (0, 1):
            side_speeds = _joint_speeds(rows, ELBOW_KP[side], elbow_gain, kp_conf, max_dt_s)
            side_speeds.update(_joint_speeds(rows, WRIST_KP[side], 1.0, kp_conf, max_dt_s))
            for t, speed in side_speeds.items():
                tl = torso_at(t)
                if tl > 0:
                    samples[t] = max(samples.get(t, 0.0), speed / tl)

        # Threshold runs → events; runs closer than merge_s coalesce.
        run: List[Tuple[float, float]] = []
        runs: List[List[Tuple[float, float]]] = []
        for t in sorted(samples):
            s = samples[t]
            if s < min_tlps or (run and t - run[-1][0] > merge_s):
                if run:
                    runs.append(run)
                run = []
            if s >= min_tlps:
                run.append((t, s))
        if run:
            runs.append(run)
        merged: List[List[Tuple[float, float]]] = []
        for r in runs:
            if merged and r[0][0] - merged[-1][-1][0] < merge_s:
                merged[-1].extend(r)
            else:
                merged.append(r)
        for r in merged:
            t_peak, strength = max(r, key=lambda ts: ts[1])
            events.append({"t": t_peak, "track": track, "strength": strength})

    return sorted(events, key=lambda e: e["t"])
