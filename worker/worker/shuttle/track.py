"""Pure shuttle-track assembly: per-frame tracker peaks -> court-space shuttle rows.

No torch/cv2 — the heavy backends produce raw `(frame, mx, my, conf)` peaks; this turns them into the
`shuttle.parquet` rows (t,u,v,visibility + debug). Court coords come from the SAME homography the player
overlay uses, so the shuttle dot lives in the same top-down space (`apply_homography`). Honest limit: an
airborne shuttle projected through the GROUND homography lands "beyond" its true spot — a „Schätzung"
good for an overlay, wrong for landing/speed (specs/30 stage 5).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..geometry import apply_homography, map_detection_to_frame, reject_shuttle_outliers

RawPeak = Tuple[int, float, float, float]  # (frame, mx, my, conf) in model-input pixel space


def build_shuttle_track(
    raw: Sequence[RawPeak],
    H: Sequence[float],
    crop_box: Tuple[int, int, int, int],
    model_w: int,
    model_h: int,
    t_by_frame: Dict[int, float],
    conf_thr: float = 0.5,
    max_step: float = 0.5,
) -> List[dict]:
    """Turn raw per-frame peaks into time-sorted shuttle rows.

    Drops peaks below `conf_thr` (a real dropout — never a fabricated point), maps each surviving peak
    crop->frame->court, removes isolated court-space spikes (`max_step`, court units — e.g. a stray
    detection on an adjacent court), and returns row dicts
    `{t, u, v, visibility, frame, px, py, conf}` sorted by t."""
    pts = []  # (frame, u, v, px, py, conf)
    for frame, mx, my, conf in raw:
        if conf < conf_thr:
            continue
        px, py = map_detection_to_frame(mx, my, crop_box, model_w, model_h)
        u, v = apply_homography(H, px, py)
        pts.append((int(frame), u, v, px, py, float(conf)))
    pts.sort(key=lambda p: p[0])

    kept_frames = {f for f, _, _ in reject_shuttle_outliers([(p[0], p[1], p[2]) for p in pts], max_step)}

    rows: List[dict] = []
    for frame, u, v, px, py, conf in pts:
        if frame not in kept_frames:
            continue
        rows.append({
            "t": t_by_frame[frame], "u": u, "v": v, "visibility": conf,
            "frame": frame, "px": px, "py": py, "conf": conf,
        })
    rows.sort(key=lambda r: r["t"])
    return rows
