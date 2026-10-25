"""Pure sample builders for stage-7 stroke classification (features/stroke-classification.md).

One shared `build_sample` feeds BOTH the dataset exporter and the pipeline stage so training and
inference see byte-identical features (the stage-5 dataset_export principle). Everything here is
pure math on rows/lists — parquet/DB I/O stays in stroke/dataset.py and pipeline/stroke.py.

REVISED 2026-07-05 after the Phase-1 gate measurements: 16-frame windows (the frame cache is the
12 fps analysis cadence, not 30 fps) and 4 fixed quadrant slots with pose recovery instead of the
rejected nearest-per-half hitter selector (hitter-track coverage 55 %, half signals ~48-49 % —
see the spec's design section for the numbers).
"""
from __future__ import annotations

from ..manifest import REFERENCE_FPS, frames_for_seconds, measured_fps

# Frames per window — a FIXED model dimension (BST4 has 16 time steps): ±0.8 s at the real ~10 fps
# cadence the stroke dataset was built on. On a 30 fps cache the same 16 frames span only ±0.27 s, so
# stage 7 needs a retrain before it may run on 30 fps caches (specs/features/analysis-30fps.md; stroke
# is not wired into the pipeline today — stroke/dataset._fps_guard warns).
SEQ_LEN = 16

SLOTS = ("near_left", "near_right", "far_left", "far_right")  # fixed stream order


def anchor_frame(frames: list[dict], t_s: float) -> int | None:
    """Nearest cached frame (by manifest time) to the hit anchor `t_s`. Empty manifest → None."""
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f["t"] - t_s))["frame"]


def window_frames(anchor: int, seq_len: int = SEQ_LEN) -> list[int]:
    """The stroke window: `seq_len` consecutive frames centered on the anchor (−8 … +7)."""
    half = seq_len // 2
    return list(range(anchor - half, anchor - half + seq_len))


# Interpolate shuttle gaps up to this DURATION within an arc (the stage 5/6 "within an arc only"
# principle): the tuned 6 frames at the real 10 fps cadence. build_sample derives the frame count
# from the manifest's cadence; SHUTTLE_MAX_GAP is the REFERENCE_FPS equivalent for direct callers.
SHUTTLE_MAX_GAP_S = 0.6
SHUTTLE_MAX_GAP = frames_for_seconds(SHUTTLE_MAX_GAP_S, REFERENCE_FPS)


def shuttle_stream(dets: dict[int, tuple[float, float]], window: list[int], anchor: int,
                   frame_w: float, frame_h: float,
                   max_gap: int = SHUTTLE_MAX_GAP) -> list[list[float]]:
    """Shuttle trajectory stream for one window: `(px/W, py/H)` per frame (BST convention).

    Gaps of ≤ `max_gap` missing frames between two detections are linearly interpolated —
    but never across the anchor: the racket contact breaks the arc, so bridging it would
    fabricate a trajectory through the hit. Undetected frames are honest `(0, 0)`.
    """
    known = sorted(f for f in dets)
    out = []
    for f in window:
        if f in dets:
            px, py = dets[f]
            out.append([px / frame_w, py / frame_h])
            continue
        prev = max((k for k in known if k < f), default=None)
        nxt = min((k for k in known if k > f), default=None)
        if (prev is None or nxt is None or nxt - prev - 1 > max_gap
                or prev < anchor < nxt):
            out.append([0.0, 0.0])
            continue
        w = (f - prev) / (nxt - prev)
        (px0, py0), (px1, py1) = dets[prev], dets[nxt]
        out.append([(px0 + (px1 - px0) * w) / frame_w, (py0 + (py1 - py0) * w) / frame_h])
    return out


N_JOINTS = 17     # body26 indices 0–16 are exactly COCO-17 — the 9 extras are dropped
KP_CONF_THR = 0.3  # house keypoint-confidence default (geometry.foot_keypoint, swing)
_MIN_CONF_JOINTS = 3


def normalize_joints(kp: list, kp_score: list, conf_thr: float = KP_CONF_THR) -> list[list[float]]:
    """COCO-17 joints, bbox-relative and center-aligned, scaled by the bbox diagonal (BST's
    documented convention). pose.parquet stores no bbox, so the envelope of the confident
    joints stands in for it; low-confidence joints don't shape the envelope but are still
    emitted. Fewer than 3 confident joints or a degenerate envelope → all zeros (honest miss).
    """
    pts = [kp[i] for i in range(N_JOINTS)]
    conf = [p for i, p in enumerate(pts) if kp_score[i] >= conf_thr]
    if len(conf) < _MIN_CONF_JOINTS:
        return [[0.0, 0.0] for _ in range(N_JOINTS)]
    x0, x1 = min(p[0] for p in conf), max(p[0] for p in conf)
    y0, y1 = min(p[1] for p in conf), max(p[1] for p in conf)
    diag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if diag == 0:
        return [[0.0, 0.0] for _ in range(N_JOINTS)]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [[(p[0] - cx) / diag, (p[1] - cy) / diag] for p in pts]


def half_of_role(role: str | None) -> str | None:
    """gate_track role → court half: near/near_left/near_right → near; far_* → far. The same
    prefix-collapse idiom as rally_hit._side_by_role; the hitter-label vocabulary matches 1:1."""
    if not role:
        return None
    if role.startswith("near"):
        return "near"
    if role.startswith("far"):
        return "far"
    return None


def quadrant(u: float, v: float) -> str:
    """Court (u,v) → quadrant slot. Absolute form of geometry.role_assign's relative sort:
    near = larger v, left = smaller u."""
    half = "near" if v > 0.5 else "far"
    side = "left" if u < 0.5 else "right"
    return f"{half}_{side}"


def slot_of_role(role: str | None) -> str | None:
    """gate_track role → stream slot. Doubles roles map 1:1; singles roles occupy the left
    slots (`near→near_left`, `far→far_left`) so one input layout serves both disciplines."""
    if role in SLOTS:
        return role
    if role == "near":
        return "near_left"
    if role == "far":
        return "far_left"
    return None


TRACK_ACTIVE_TOL = 0.5  # s — a role segment must have a row this close to the anchor to count


def assign_slots(track_rows: list[dict], t_s: float,
                 tol: float = TRACK_ACTIVE_TOL) -> dict[str, int | None]:
    """Time-aware slot assignment: per slot, the role track with a row nearest the anchor time
    (within `tol`). Track ids churn heavily (33 role segments for 4 players in game 1), so a
    whole-video role→id lookup would pick stale segments — measured at the Phase-1 gate."""
    best: dict[str, tuple[float, int]] = {}
    for r in track_rows:
        slot = slot_of_role(r.get("role"))
        if slot is None:
            continue
        dt = abs(r["t"] - t_s)
        if dt <= tol and (slot not in best or dt < best[slot][0]):
            best[slot] = (dt, r["track_id"])
    return {slot: best.get(slot, (None, None))[1] for slot in SLOTS}


IN_COURT_MARGIN = 0.06  # gate_track's court-normalized gating margin — same players, same rule
FOOT_CONF_THR = 0.3     # gate_track's POSE_CONF_THR default


def recover_slots(pose_rows: list[dict], taken: dict[str, int | None], homography,
                  empty_slots: list[str],
                  foot_conf_thr: float = FOOT_CONF_THR) -> dict[str, dict]:
    """Fill slots the court tracks lost from pose alone: pose.parquet sees everyone (the far
    pair included — only the track bookkeeping drops them, measured 40 % of hits), so any
    unassigned pose track whose precomputed foot projects into the court claims its quadrant.
    Per empty slot the candidate present in the most window frames wins (stability beats
    proximity for a fixed-slot input). Returns {slot: {"track": tid, "uv": {frame: (u,v)}}}.
    """
    from ..geometry import apply_homography

    taken_tids = {tid for tid in taken.values() if tid is not None}
    cands: dict[int, dict[int, tuple[float, float]]] = {}
    for r in pose_rows:
        tid = r["track"]
        if tid in taken_tids or r.get("foot_conf") is None or r["foot_conf"] < foot_conf_thr:
            continue
        u, v = apply_homography(homography, r["foot_x"], r["foot_y"])
        if not (-IN_COURT_MARGIN <= u <= 1 + IN_COURT_MARGIN
                and -IN_COURT_MARGIN <= v <= 1 + IN_COURT_MARGIN):
            continue
        cands.setdefault(tid, {})[r["frame"]] = (u, v)

    out: dict[str, dict] = {}
    for tid, uv in sorted(cands.items(), key=lambda kv: -len(kv[1])):
        us = [p[0] for p in uv.values()]
        vs = [p[1] for p in uv.values()]
        slot = quadrant(sorted(us)[len(us) // 2], sorted(vs)[len(vs) // 2])  # median foot
        if slot in empty_slots and slot not in out:
            out[slot] = {"track": tid, "uv": uv}
    return out


POSITION_MAX_DT = 0.1  # s — nearest-t join tolerance for the court-position stream


def position_stream(samples: list[tuple[float, float, float]], times: list[float],
                    max_dt: float = POSITION_MAX_DT) -> list[list[float]]:
    """Court-space (u, v) per window frame time via nearest-t join; no sample within
    `max_dt` → honest (0, 0). `samples` = [(t, u, v)] for ONE track, any order."""
    srt = sorted(samples)
    out = []
    for t in times:
        near = min(srt, key=lambda s: abs(s[0] - t), default=None)
        if near is None or abs(near[0] - t) > max_dt:
            out.append([0.0, 0.0])
        else:
            out.append([near[1], near[2]])
    return out


def build_sample(pose_rows: list[dict], track_rows: list[dict],
                 dets: dict[int, tuple[float, float]], frames: list[dict], t_s: float,
                 frame_w: float, frame_h: float, homography=None) -> dict | None:
    """The one sample builder both the dataset exporter and the pipeline stage call
    (train == inference by construction). Returns nested lists:
      joints    (4, 16, 17, 2)  per-slot skeletons (SLOTS order) per window frame
      shuttle   (16, 2)
      positions (4, 16, 2)      court-space (u, v) per slot
      meta      anchor / t_s / pad / shuttle_coverage / track_<slot> / source_<slot>
    Slots fill from time-aware role tracks, then pose recovery when a homography is given.
    Missing inputs degrade to zeros and None meta — never a crash. Empty frame manifest → None.
    """
    anchor = anchor_frame(frames, t_s)
    if anchor is None:
        return None
    window = window_frames(anchor)
    t_by_frame = {f["frame"]: f["t"] for f in frames}
    pad = sum(1 for f in window if f not in t_by_frame)

    fps = measured_fps(frames) or REFERENCE_FPS  # the gap is a duration; frames follow the cadence
    shuttle = shuttle_stream(dets, window, anchor, frame_w, frame_h,
                             max_gap=frames_for_seconds(SHUTTLE_MAX_GAP_S, fps))
    coverage = sum(1 for p in shuttle if p != [0.0, 0.0]) / len(window)

    window_pose = [r for r in pose_rows if window[0] <= r["frame"] <= window[-1]]
    pose_by_frame: dict[int, dict[int, dict]] = {}
    for r in window_pose:
        pose_by_frame.setdefault(r["frame"], {})[r["track"]] = r

    slots = assign_slots(track_rows, t_s)
    sources = {slot: ("track" if tid is not None else None) for slot, tid in slots.items()}
    recovered: dict[str, dict] = {}
    if homography is not None:
        empty = [s for s, tid in slots.items() if tid is None]
        recovered = recover_slots(window_pose, slots, homography, empty)
        for slot, rec in recovered.items():
            slots[slot] = rec["track"]
            sources[slot] = "recovery"

    joints, positions = [], []
    times = [t_by_frame.get(f) for f in window]
    for slot in SLOTS:
        tid = slots[slot]
        joints.append([
            normalize_joints(row["kp"], row["kp_score"])
            if tid is not None and (row := pose_by_frame.get(f, {}).get(tid)) is not None
            else [[0.0, 0.0] for _ in range(N_JOINTS)]
            for f in window
        ])
        if tid is None:
            positions.append([[0.0, 0.0]] * len(window))
        elif slot in recovered:
            uv = recovered[slot]["uv"]
            positions.append([[uv[f][0], uv[f][1]] if f in uv else [0.0, 0.0] for f in window])
        else:
            samples = [(r["t"], r["u"], r["v"]) for r in track_rows if r["track_id"] == tid]
            positions.append(position_stream(samples,
                                             [t if t is not None else -1e9 for t in times])
                             if samples else [[0.0, 0.0]] * len(window))

    meta = {"anchor": anchor, "t_s": t_s, "pad": pad, "shuttle_coverage": coverage}
    for slot in SLOTS:
        meta[f"track_{slot}"] = slots[slot]
        meta[f"source_{slot}"] = sources[slot]
    return {"joints": joints, "shuttle": shuttle, "positions": positions, "meta": meta}
