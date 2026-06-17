"""Pure insight math — no third-party deps, dependency-free and unit-tested.

Inputs are tracks in normalized court coordinates: a sample is (t_seconds, u, v) with u,v in [0,1]
(u left->right, v far->near baseline). Court metric size defaults to BWF doubles. These functions
produce the exact fields the frontend consumes (specs/20-data-model.md). Stage 8 in the pipeline.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

Sample = Tuple[float, float, float]  # (t, u, v)

COURT_W_M = 6.10   # doubles width
COURT_L_M = 13.40  # full length


def _segments(track: List[Sample]):
    for a, b in zip(track, track[1:]):
        yield a, b


def total_distance_m(track: List[Sample], w_m: float = COURT_W_M, l_m: float = COURT_L_M) -> float:
    """Total path length in metres from court-space displacement."""
    d = 0.0
    for (_, u0, v0), (_, u1, v1) in _segments(track):
        d += math.hypot((u1 - u0) * w_m, (v1 - v0) * l_m)
    return d


def step_speeds_kmh(track: List[Sample], w_m: float = COURT_W_M, l_m: float = COURT_L_M) -> List[float]:
    out: List[float] = []
    for (t0, u0, v0), (t1, u1, v1) in _segments(track):
        dt = t1 - t0
        if dt <= 0:
            continue
        dist = math.hypot((u1 - u0) * w_m, (v1 - v0) * l_m)
        out.append(dist / dt * 3.6)
    return out


def avg_top_speed_kmh(speeds: List[float], moving_threshold: float = 0.5) -> Tuple[float, float]:
    """Average over moving samples, peak over all. Rounded to 1 decimal."""
    moving = [s for s in speeds if s >= moving_threshold]
    avg = sum(moving) / len(moving) if moving else 0.0
    top = max(speeds) if speeds else 0.0
    return round(avg, 1), round(top, 1)


def sprint_count(speeds: List[float], threshold_kmh: float = 15.0) -> int:
    """Count rising edges above the sprint threshold (one per burst)."""
    count = 0
    in_burst = False
    for s in speeds:
        if s >= threshold_kmh:
            if not in_burst:
                count += 1
                in_burst = True
        else:
            in_burst = False
    return count


def zones(depths: List[float], laterals: List[float]) -> Dict[str, int]:
    """Percent of samples in front/mid/rear (depth 0=net..1=baseline) and left/right (lateral 0..1)."""
    n = len(depths)
    if n == 0:
        return {"front": 0, "mid": 0, "rear": 0, "left": 0, "right": 0}
    front = sum(1 for d in depths if d < 1 / 3)
    mid = sum(1 for d in depths if 1 / 3 <= d < 2 / 3)
    rear = n - front - mid
    left = sum(1 for x in laterals if x < 0.5)
    right = n - left
    pct = lambda c: round(100 * c / n)
    return {"front": pct(front), "mid": pct(mid), "rear": pct(rear), "left": pct(left), "right": pct(right)}


def timeline_buckets(
    track: List[Sample], duration_min: float, bucket_min: int = 5,
    w_m: float = COURT_W_M, l_m: float = COURT_L_M,
) -> List[int]:
    """Distance (m) per `bucket_min`-minute bucket; length = ceil(duration_min/bucket_min)."""
    nb = max(1, math.ceil(duration_min / bucket_min))
    buckets = [0.0] * nb
    bsec = bucket_min * 60
    for (t0, u0, v0), (_, u1, v1) in _segments(track):
        idx = min(nb - 1, int(t0 // bsec))
        buckets[idx] += math.hypot((u1 - u0) * w_m, (v1 - v0) * l_m)
    return [round(b) for b in buckets]


def heatmap_kernels(
    points: List[Tuple[float, float]], k: int = 4, grid: int = 12, sigma: float = 0.13,
) -> List[Dict[str, float]]:
    """Reduce a position cloud to up to k gaussian kernels [{x,y,w,s}] (weights normalized to max=1).

    Matches the frontend `CourtHeatmap` input shape. Coarse grid-peak method (good enough for P0; a
    proper KDE/GMM fit can replace it later without changing the output contract).
    """
    if not points:
        return []
    cells: Dict[Tuple[int, int], int] = {}
    for x, y in points:
        cx = min(grid - 1, max(0, int(x * grid)))
        cy = min(grid - 1, max(0, int(y * grid)))
        cells[(cx, cy)] = cells.get((cx, cy), 0) + 1
    top = sorted(cells.items(), key=lambda kv: kv[1], reverse=True)[:k]
    max_count = top[0][1]
    return [
        {
            "x": round((cx + 0.5) / grid, 3),
            "y": round((cy + 0.5) / grid, 3),
            "w": round(c / max_count, 3),
            "s": sigma,
        }
        for (cx, cy), c in top
    ]
