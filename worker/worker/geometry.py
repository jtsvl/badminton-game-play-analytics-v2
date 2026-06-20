"""Pure court geometry — homography, projection, gating, roles, resampling.

No numpy/cv2/torch: math only, so this installs and tests under the dev extra without the ml stack.
The homography solver is a direct port of the frontend `web/src/lib/homography.ts` (same DLT +
Gauss-Jordan), so the worker and the web derive the *same* projective map from the *same* corners.

Court coordinate convention (shared with the web and insights.py):
  u = left->right in [0,1], v = far baseline (0) -> near baseline (1).
Canonical confirmed-corner order is FL, FR, NR, NL, mapping to the unit square below.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Dict, List, Sequence, Tuple

from .insights import COURT_W_M, COURT_L_M

Pt = Tuple[float, float]
Sample = Tuple[float, float, float]  # (t, u, v)

# Unit court square in (u,v); rows match the FL, FR, NR, NL corner order.
UNIT_SQUARE: List[Pt] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Gauss-Jordan elimination with partial pivoting (mirrors homography.ts solveLinear)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(m[r][col]) > abs(m[piv][col]):
                piv = r
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("homography: singular system")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def solve_homography(src: Sequence[Pt], dst: Sequence[Pt]) -> List[float]:
    """3x3 homography mapping src[i] -> dst[i] for four point pairs. Returns [h0..h8] row-major, h8=1."""
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("homography: need exactly 4 point pairs")
    A: List[List[float]] = []
    b: List[float] = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * X, -y * X])
        b.append(X)
        A.append([0, 0, 0, x, y, 1, -x * Y, -y * Y])
        b.append(Y)
    h = _solve_linear(A, b)
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]


def homography_from_corners(corners_px: Sequence[Pt]) -> List[float]:
    """Image-pixel corners (FL,FR,NR,NL) -> unit court square (u,v). The worker's gating/projection H."""
    return solve_homography(corners_px, UNIT_SQUARE)


def apply_homography(H: Sequence[float], x: float, y: float) -> Pt:
    """Project (x,y) through H. Returns (u,v) = the dst-space coordinates."""
    denom = H[6] * x + H[7] * y + H[8]
    return (
        (H[0] * x + H[1] * y + H[2]) / denom,
        (H[3] * x + H[4] * y + H[5]) / denom,
    )


def reproject_error_m(
    H: Sequence[float],
    points_px: Sequence[Pt],
    points_uv: Sequence[Pt],
    w_m: float = COURT_W_M,
    l_m: float = COURT_L_M,
) -> float:
    """RMS metric residual (m) between H(points_px) and the expected (u,v). 0 for an exact 4-corner fit."""
    n = len(points_px)
    if n == 0:
        return 0.0
    sq = 0.0
    for (px, py), (eu, ev) in zip(points_px, points_uv):
        u, v = apply_homography(H, px, py)
        sq += ((u - eu) * w_m) ** 2 + ((v - ev) * l_m) ** 2
    return math.sqrt(sq / n)


def point_in_poly(pt: Pt, poly: Sequence[Pt]) -> bool:
    """Even-odd ray-cast point-in-polygon. poly is a list of (x,y) vertices."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def ankle_point(bbox: Tuple[float, float, float, float]) -> Pt:
    """Foot point = bottom-center of a (top-left x, top-left y, width, height) bbox."""
    x, y, w, h = bbox
    return (x + w / 2, y + h)


def foot_point(bbox: Tuple[float, float, float, float], k: float = 0.0) -> Pt:
    """Foot point with an optional downward bias of k*height, correcting the detector's systematic
    bbox-bottom truncation (the person box ends above the real feet → foot read too high → player
    appears too close to the net, amplified on the far half). k=0 is identical to ankle_point.

    Empirical fit k≈0.16 on the trainer's f35d0984 verify-tool corrections collapsed median court
    error ~112→27 cm (far ~158→31 cm); see worker/eval_corrections.py and specs/fix-later.md. This
    is the interim fix; the durable one is the pose stage reading the real ankle keypoint.
    """
    x, y, w, h = bbox
    return (x + w / 2, y + h * (1.0 + k))


# Halpe-26 (RTMPose body26) foot-defining keypoints. Ankles are the most reliably detected; heels/toes
# sit closer to the ground-contact point. FOOT_KP is the fitted default (see eval_corrections / fix-later).
ANKLE_KP: Tuple[int, int] = (15, 16)   # left_ankle, right_ankle
HEEL_KP: Tuple[int, int] = (24, 25)    # left_heel, right_heel
TOE_KP: Tuple[int, int] = (20, 21)     # left_big_toe, right_big_toe
FOOT_KP: Tuple[int, ...] = ANKLE_KP


def foot_keypoint(
    kp: Sequence[Pt], scores: Sequence[float],
    foot_indices: Sequence[int] = FOOT_KP, conf_thr: float = 0.3,
) -> Tuple[float, float, float] | None:
    """Pose-derived foot (x, y, confidence) = mean of the confident foot keypoints (ankles by default).

    Returns the mean of foot keypoints with score ≥ conf_thr (confidence = their mean score). If none
    clear the threshold it still returns the mean position but with confidence 0.0, signalling the
    caller to fall back. None only when no foot keypoint is present at all (empty pose)."""
    pts = [(kp[i], scores[i]) for i in foot_indices if i < len(kp) and i < len(scores)]
    if not pts:
        return None
    conf_pts = [(p, s) for p, s in pts if s >= conf_thr]
    use = conf_pts or pts
    x = sum(p[0] for p, _ in use) / len(use)
    y = sum(p[1] for p, _ in use) / len(use)
    conf = sum(s for _, s in conf_pts) / len(conf_pts) if conf_pts else 0.0
    return (x, y, conf)


# Halpe-26 upper-body keypoints that stay visible when the lower body (ankles) is occluded — the
# anchor for the Phase-4 head/height foot grounding. 17 = head top, 19 = mid-hip (pelvis centre).
HEAD_KP: int = 17
HIP_KP: int = 19


def fit_head_foot_model(samples: Sequence[Tuple[float, float, float]]) -> float | None:
    """Learn a per-player head→foot drop from confident, feet-visible frames (Phase-4 occlusion layer).

    Each sample is `(head_y, hip_y, foot_y)` in image pixels from a frame where the ankle *and* head/hip
    are confident. Returns the **median** `(foot_y - head_y) / (hip_y - head_y)` — the foot drop measured
    in torso lengths, so it transfers across distance using only the head+hip that remain visible during
    lower-body occlusion. Median makes it robust to the odd bad foot. None if no frame has a positive
    torso (degenerate / too little signal to ground from).

    Honest limits (surfaced as „Schätzung" downstream): real height varies ~±10 %, posture changes the
    torso↔leg ratio (a deep lunge compresses it), and the low standing camera foreshortens the vertical
    drop — a higher mount would sharpen it. Most valuable in doubles (P3) where players occlude each other.
    """
    ratios = [(fy - hy) / (py - hy) for hy, py, fy in samples if py - hy > 1e-6]
    if not ratios:
        return None
    return median(ratios)


def ground_foot_from_head(head_pt: Pt, hip_pt: Pt, ratio: float) -> Pt:
    """Predict the (occluded) foot from the still-visible head + hip and a fitted `ratio`
    (see fit_head_foot_model): foot sits `ratio` torso-lengths below the head, under the pelvis."""
    hx, hy = head_pt
    px, py = hip_pt
    return (px, hy + ratio * (py - hy))


def foot_from_keypoints(
    kp: Sequence[Pt], scores: Sequence[float], bbox: Tuple[float, float, float, float],
    k_fallback: float = 0.0, conf_thr: float = 0.3, foot_indices: Sequence[int] = FOOT_KP,
) -> Pt:
    """Durable foot point (Phase 3): the real ankle/foot keypoint when confident, else the Phase-2
    interim bbox nudge `foot_point(bbox, k_fallback)`. This is the three-layer foot resolution's first
    two layers; head/height grounding for full occlusion is Phase 4 (specs/fix-later.md)."""
    fk = foot_keypoint(kp, scores, foot_indices, conf_thr)
    if fk is not None and fk[2] >= conf_thr:
        return (fk[0], fk[1])
    return foot_point(bbox, k_fallback)


def net_depth(v: float, role: str) -> float:
    """Depth within the player's own half: 0 at the net, 1 at their baseline. Clamped [0,1].

    Fixes the insights.zones() orientation trap: canonical v=0 is the FAR baseline and the net is at
    v=0.5, so raw v is not 'distance from net'. `role` is 'near'/'far' (or 'near_left' etc.).
    """
    if role.startswith("near"):
        d = (v - 0.5) / 0.5
    else:  # far
        d = (0.5 - v) / 0.5
    return max(0.0, min(1.0, d))


def role_assign(medians: Dict[int, Pt], discipline: str) -> Dict[int, str]:
    """Assign a persistent role per track from its median (u,v).

    singles -> {'near','far'} by v (near = larger v). doubles -> the four quadrants by v then u.
    """
    items = sorted(medians.items(), key=lambda kv: kv[1][1], reverse=True)  # by v desc (near first)
    if discipline == "singles":
        roles: Dict[int, str] = {}
        if items:
            roles[items[0][0]] = "near"
            roles[items[-1][0]] = "far"
        return roles
    near = sorted(items[:2], key=lambda kv: kv[1][0])  # by u asc within the near half
    far = sorted(items[2:4], key=lambda kv: kv[1][0])
    roles = {}
    if len(near) == 2:
        roles[near[0][0]] = "near_left"
        roles[near[1][0]] = "near_right"
    if len(far) == 2:
        roles[far[0][0]] = "far_left"
        roles[far[1][0]] = "far_right"
    return roles


def enforce_n_tracks(counts: Dict[int, int], n: int) -> List[int]:
    """Keep the n track ids with the most in-court frames (drops adjacent-court / spurious tracks)."""
    return [tid for tid, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def stitch_role_fragments(
    primaries: Dict[str, dict], fragments: Dict[int, dict], max_court_dist: float = 0.5
) -> Dict[int, str]:
    """Re-associate dropped in-court track fragments with the role they continue (Phase-4 re-ID, RC3).

    BoT-SORT drops a small/low-contrast player (the far player on f35d0984) and re-acquires them under
    a *new* track id, which `enforce_n_tracks` then discards as a short fragment — leaving a hole. Each
    `fragments[tid]` is matched to a primary role iff it (a) is in the same court half, (b) does **not**
    temporally overlap that role's primary coverage (so it genuinely fills the role's hole, not a frame
    the role already has), and (c) is the nearest such role within `max_court_dist` (court-normalized) —
    a guard against cross-court blips. This recovers *real detected* positions; it does not fabricate.

    primaries: role -> {"median": (u,v), "span": (first_frame, last_frame)}.
    fragments: track_id -> {"median": (u,v), "span": (first_frame, last_frame)}.
    Returns {fragment_track_id: role}; fragments matching no role are omitted (stay dropped).
    """
    out: Dict[int, str] = {}
    for tid, frag in fragments.items():
        fu, fv = frag["median"]
        fa, fb = frag["span"]
        frag_far = fv < 0.5
        best_role = None
        best_dist = max_court_dist
        for role, prim in primaries.items():
            if role.startswith("far") != frag_far:
                continue  # different court half
            pa, pb = prim["span"]
            if pa <= fb and fa <= pb:
                continue  # overlaps the role's own coverage → fills no hole
            pu, pv = prim["median"]
            d = math.hypot(fu - pu, fv - pv)
            if d <= best_dist:
                best_dist = d
                best_role = role
        if best_role is not None:
            out[tid] = best_role
    return out


def smooth_track(track: List[Sample], window: int = 5) -> List[Sample]:
    """Centered moving-average of (u,v), keeping each sample's t. Damps ankle-point jitter so
    per-step speeds aren't inflated into implausible sprints (monocular foot tracking is noisy)."""
    n = len(track)
    if n <= 2 or window < 2:
        return list(track)
    half = window // 2
    out: List[Sample] = []
    for i in range(n):
        seg = track[max(0, i - half) : min(n, i + half + 1)]
        u = sum(s[1] for s in seg) / len(seg)
        v = sum(s[2] for s in seg) / len(seg)
        out.append((track[i][0], u, v))
    return out


FrameSample = Tuple[int, float, float]  # (frame, u, v)


def fill_track_gaps(
    samples: List[FrameSample], max_gap: int = 8
) -> List[Tuple[int, float, float, bool]]:
    """Fill short interior holes in a frame-indexed (frame, u, v) track by linear interpolation.

    Phase-4 short-gap repair (RC3): when a role's coverage has a hole of 2..`max_gap` *missing*
    frames between two confident detections, interpolate (u,v) at each missing frame — the same
    linear blend as `resample_10hz`, but keyed on integer frames. Holes wider than `max_gap` are
    left absent (a long dropout is recall, not a fillable gap), and nothing is produced before the
    first or after the last sample: **interpolate between two anchors only, never extrapolate**
    (a one-sided dropout — e.g. a player lost to the end of the clip — stays an honest hole).

    Returns the densified track as (frame, u, v, is_filled); is_filled marks interpolated frames so
    the caller can carry the higher uncertainty downstream (overlay „Schätzung")."""
    samples = sorted(samples)
    if len(samples) < 2:
        return [(f, u, v, False) for f, u, v in samples]
    out: List[Tuple[int, float, float, bool]] = []
    for (fa, ua, va), (fb, ub, vb) in zip(samples, samples[1:]):
        out.append((fa, ua, va, False))
        span = fb - fa
        if 1 < span <= max_gap:
            for f in range(fa + 1, fb):
                frac = (f - fa) / span
                out.append((f, ua + (ub - ua) * frac, va + (vb - va) * frac, True))
    out.append((samples[-1][0], samples[-1][1], samples[-1][2], False))
    return out


# --- Stage 5 (shuttle) crop geometry + light spike removal -----------------------------------------
# The shuttle is a handful of pixels in a wide 4K DJI shot, and the hall holds several courts at once.
# Cropping to the target court's calibration polygon (a) isolates that court (ignore other shuttles) and
# (b) restores effective shuttle resolution before a tracker downsizes to its ~512x288 input.


def court_crop_box(
    poly_px: Sequence[Pt], frame_w: int, frame_h: int,
    up: float = 0.5, side: float = 0.12, down: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Pixel crop box (x0,y0,x1,y1) around the target court's mask polygon, clamped to the frame.

    The polygon bounding box is expanded UPWARD by `up`*height (image up = smaller y) to keep the
    shuttle's arc over the net, sideways by `side`*width, and down by `down`*height. A tighter `up`
    isolates better (less white wall / fewer adjacent courts re-enter at the top) but clips high
    clears — the white-wall region where detection fails anyway (specs/30 stage 5)."""
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    bx0 = max(0.0, min(x0 - side * w, frame_w))
    bx1 = max(0.0, min(x1 + side * w, frame_w))
    by0 = max(0.0, min(y0 - up * h, frame_h))
    by1 = max(0.0, min(y1 + down * h, frame_h))
    return (int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1)))


def court_airspace_polygon(
    poly_px: Sequence[Pt], frame_w: int, frame_h: int, up: float = 0.7, side: float = 0.08,
) -> List[Pt]:
    """Target-court AIRSPACE polygon for the stage-5 isolation mask — the ground quad with its far (top)
    edge raised by `up`*court-height (image up = smaller y) and widened by `side`*court-width, so it
    covers where the shuttle actually flies while its sidelines still exclude the neighbouring courts.

    A strict ground-quad mask would black out the airborne shuttle (it sits well above the floor); this
    keeps the column of space above the target court. Robust to corner input order (far = the two with
    smaller y). Returns [far_left_up, far_right_up, near_right, near_left], clamped to the frame."""
    pts = sorted(poly_px, key=lambda p: p[1])
    far = sorted(pts[:2], key=lambda p: p[0])    # smaller y = far baseline (higher in image)
    near = sorted(pts[2:], key=lambda p: p[0])
    (fl, fr), (nl, nr) = far, near
    height = (nl[1] + nr[1]) / 2 - (fl[1] + fr[1]) / 2
    width = max(p[0] for p in poly_px) - min(p[0] for p in poly_px)

    def clamp(x, y):
        return (max(0.0, min(x, frame_w)), max(0.0, min(y, frame_h)))

    fl_up = clamp(fl[0] - side * width, fl[1] - up * height)
    fr_up = clamp(fr[0] + side * width, fr[1] - up * height)
    return [fl_up, fr_up, (nr[0], nr[1]), (nl[0], nl[1])]


def map_detection_to_frame(
    mx: float, my: float, crop_box: Tuple[int, int, int, int], model_w: int, model_h: int,
) -> Pt:
    """Map a tracker peak (mx,my) in model-input pixels back to full-frame pixels.

    The crop is resized straight to the model input (independent x/y scale), so invert that scale and
    add the crop offset. The result feeds `apply_homography` to reach court (u,v)."""
    x0, y0, x1, y1 = crop_box
    return (x0 + mx * (x1 - x0) / model_w, y0 + my * (y1 - y0) / model_h)


def frame_to_model_xy(
    px: float, py: float, crop_box: Tuple[int, int, int, int], model_w: int, model_h: int,
) -> Pt:
    """Inverse of `map_detection_to_frame`: full-frame pixels (px,py) -> model-input pixels (mx,my).

    Both the dataset exporter (label click -> 512x288 CSV X,Y) and the scoreboard (parquet's full-frame
    px,py and the normalized labels -> the single 512x288 space the metric is pinned to) need this. It is
    the exact algebraic inverse so a round-trip through `map_detection_to_frame` is the identity."""
    x0, y0, x1, y1 = crop_box
    return ((px - x0) * model_w / (x1 - x0), (py - y0) * model_h / (y1 - y0))


def reject_shuttle_outliers(track: List[Sample], max_step: float) -> List[Sample]:
    """Drop isolated court-space spikes: a point that juts > max_step from BOTH neighbours while those
    neighbours stay within max_step of each other (a false positive on another court / a misdetection).

    A genuine fast smash is consecutive large *collinear* steps — there the neighbours are also far
    apart, so nothing is dropped. This is deliberately light: it removes blips, it does NOT smooth the
    arc (averaging would flatten the very peaks we want). `track` items are (frame_or_t, u, v)."""
    if len(track) <= 2:
        return list(track)

    def d(a: Sample, b: Sample) -> float:
        return math.hypot(a[1] - b[1], a[2] - b[2])

    keep = [True] * len(track)
    for i in range(1, len(track) - 1):
        prev, cur, nxt = track[i - 1], track[i], track[i + 1]
        if d(cur, prev) > max_step and d(cur, nxt) > max_step and d(prev, nxt) <= max_step:
            keep[i] = False
    return [p for p, k in zip(track, keep) if k]


def split_on_gaps(track: List[Sample], max_dt: float) -> List[List[Sample]]:
    """Split a time-sorted (t,u,v) track wherever consecutive samples are more than `max_dt` apart.

    Used by the overlay for the shuttle: a real dropout (white-wall camouflage) must stay a gap, not be
    bridged by `resample_10hz`'s straight-line interpolation. Each returned segment is resampled on its
    own. An empty track yields no segments; a contiguous track yields one."""
    if not track:
        return []
    segments: List[List[Sample]] = [[track[0]]]
    for prev, cur in zip(track, track[1:]):
        if cur[0] - prev[0] > max_dt:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def resample_10hz(track: List[Sample], hz: int = 10) -> List[Sample]:
    """Resample a (t,u,v) track to a fixed hz with linear interpolation; endpoints preserved."""
    if not track:
        return []
    if len(track) == 1:
        return [track[0]]
    dt = 1.0 / hz
    t0 = track[0][0]
    tend = track[-1][0]
    steps = int(round((tend - t0) / dt))
    out: List[Sample] = []
    j = 0
    for i in range(steps + 1):
        t = t0 + i * dt
        if t > tend:
            t = tend
        while j < len(track) - 2 and track[j + 1][0] < t:
            j += 1
        ta, ua, va = track[j]
        tb, ub, vb = track[j + 1]
        if tb == ta:
            u, v = ua, va
        else:
            f = max(0.0, min(1.0, (t - ta) / (tb - ta)))
            u = ua + (ub - ua) * f
            v = va + (vb - va) * f
        out.append((t, u, v))
    return out
