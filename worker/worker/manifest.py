"""Frame-cache cadence helpers — the ONE place seconds↔frames conversions live (pure, stdlib only).

Why this exists (specs/features/analysis-30fps.md): `frames.json` records the fps that was *requested*
(`analysis_fps`), but ingest keeps a frame only when its PTS reaches `last_kept + 1/fps`, so the cache's
real cadence is `source_fps / ceil(source_fps / requested)` — on the 30 fps Session-2 sources a "12 fps"
cache is really 10 fps. Every threshold that used to be a frame count is therefore expressed in seconds
(the `frames × 0.1 s` convention: what the tuned values meant at the real 10 fps cadence) and converted
back to frames here, from the MEASURED spacing of the manifest's `t` values — never from the requested
number. Fallbacks keep hand-written test manifests (no `analysis_fps`) working.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence, Tuple

NOMINAL_FPS = 12.0    # last-resort fallback = the historical ingest default (ingest.DEFAULT_ANALYSIS_FPS)
REFERENCE_FPS = 10.0  # the REAL cadence of the production caches every frame-count threshold was tuned on


def _median(values: Sequence[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def measured_fps(frames: Sequence[Mapping]) -> Optional[float]:
    """Cadence actually present in a manifest's `frames` list: 1 / median(Δt) over consecutive `t`.

    The median makes a single dropped/duplicated frame irrelevant. None when fewer than two frames or
    the spacing is zero (nothing to measure)."""
    ts = [float(f["t"]) for f in frames]
    if len(ts) < 2:
        return None
    med = _median([b - a for a, b in zip(ts, ts[1:])])
    if med <= 0:
        return None
    return 1.0 / med


def fps_from_frames_t(pairs: Iterable[Tuple[int, float]]) -> Optional[float]:
    """Cadence from any (frame_index, t) samples that share a cache's frame numbering — e.g. shuttle
    detections with gaps: `(t_last − t_first) / (frame_last − frame_first)` is exact over the gaps.
    None when fewer than two distinct frames or a non-positive span."""
    pts = sorted((int(f), float(t)) for f, t in pairs)
    if len(pts) < 2:
        return None
    (f0, t0), (f1, t1) = pts[0], pts[-1]
    if f1 <= f0 or t1 <= t0:
        return None
    return (f1 - f0) / (t1 - t0)


def manifest_fps(manifest: Mapping) -> float:
    """The cadence a stage should convert seconds with: measured from `frames[*].t`, else the recorded
    `cadence_fps`/`analysis_fps`, else the nominal 12.0 (old hand-written manifests)."""
    frames = manifest.get("frames") or []
    fps = measured_fps(frames)
    if fps:
        return fps
    for key in ("cadence_fps", "analysis_fps"):
        value = manifest.get(key)
        if value:
            return float(value)
    return NOMINAL_FPS


def _check_fps(fps: float) -> None:
    if not fps or fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"fps must be a positive finite number (got {fps!r})")


def frames_for_seconds(seconds: float, fps: float, minimum: int = 1) -> int:
    """Frame count equivalent to `seconds` at `fps`, rounded half-up (not banker's), ≥ `minimum`."""
    _check_fps(fps)
    return max(int(minimum), int(math.floor(seconds * fps + 0.5)))


def seconds_for_frames(n_frames: int, fps: float) -> float:
    """Duration of `n_frames` consecutive frames at `fps`."""
    _check_fps(fps)
    return n_frames / fps


def per_step(rate_per_second: float, fps: float) -> float:
    """A per-second rate expressed per consecutive frame step (e.g. court-units/s → court-units/step)."""
    _check_fps(fps)
    return rate_per_second / fps
