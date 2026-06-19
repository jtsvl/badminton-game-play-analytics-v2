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
