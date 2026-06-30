"""Diagnostic: draw raw shuttle detections in IMAGE space on the real frames.

The court-space (u,v) numbers can't tell "the tracker found the wrong thing" from "the tracker found the
right thing but the ground homography mislocates an airborne shuttle". Drawing the raw pixel peaks
(px,py from shuttle.parquet) straight on the frames removes the projection entirely, so we can SEE
whether the dots follow one shuttle on the target court or scatter/hop between courts.

Outputs (to <out_dir>): scatter.png (all detections on one frame, coloured early→late), key frames
(annotated stills), and overlay.mp4 (marker + fading trail, scrubable). Verification here is the eyeball,
not a unit test — this is a viz tool, not pipeline code. Run via `eval_shuttle --render <dir>`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("worker.shuttle.render")

GREEN = (0, 200, 0)
YELLOW = (0, 220, 220)


def _time_color(frac: float):
    """Blue (early) → red (late) BGR, so a detection's colour encodes when in the clip it fired."""
    return (int(255 * (1 - frac)), 0, int(255 * frac))


def _poly_int(poly):
    import numpy as np

    return np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype="int32")


def render_overlay(video_id: str, tracker_name: str, out_dir: str, video_max_w: int = 1920,
                   frame_range: tuple[int, int] | None = None) -> dict:
    import cv2
    import numpy as np
    import polars as pl

    from ..geometry import court_airspace_polygon
    from ..pipeline.shuttle import AIR_SIDE, AIR_UP
    from .. import eval_shuttle  # reuse run_stage + the calibration fetch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from ..config import load_config
    cfg = load_config()
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    frames_dir = cache / "frames"
    manifest = json.loads((cache / "frames.json").read_text())
    fw, fh = int(manifest["width"]), int(manifest["height"])
    frames = manifest["frames"]
    if frame_range:
        a, b = frame_range
        frames = [f for f in frames if a <= int(f["frame"]) <= b]
        if not frames:
            raise RuntimeError(f"render: no frames in range {a}:{b}")

    # Run the chosen tracker → shuttle.parquet, and fetch the court polygon for context.
    parquet = eval_shuttle.run_stage(video_id, tracker_name, frame_range=frame_range)
    det = {int(r["frame"]): r for r in pl.read_parquet(parquet).sort("frame").iter_rows(named=True)}
    from ..__main__ import _make_store
    cal = _make_store(cfg).get_calibration(video_id)
    poly = cal.get("mask_polygon_px") or cal.get("confirmed_points")
    air = court_airspace_polygon(poly, fw, fh, up=AIR_UP, side=AIR_SIDE)

    n = len(frames)
    rad = max(6, fw // 240)

    def draw_context(img):
        cv2.polylines(img, [_poly_int(poly)], True, GREEN, max(2, fw // 900))     # court (ground quad)
        cv2.polylines(img, [_poly_int(air)], True, YELLOW, max(2, fw // 1100))    # airspace mask boundary

    # 1) Scatter: every detection on one mid-clip frame, coloured early→late.
    mid = frames[len(frames) // 2]["frame"]
    base = cv2.imread(str(frames_dir / f"{int(mid):06d}.jpg"))
    draw_context(base)
    for i, f in enumerate(sorted(det)):
        r = det[f]
        cv2.circle(base, (int(r["px"]), int(r["py"])), rad, _time_color(i / max(len(det) - 1, 1)), -1)
    cv2.imwrite(str(out / "scatter.png"), base)

    # 2) Annotated key frames (spread across the clip, only frames that have a detection) — for inspection.
    have = sorted(det)
    picks = [have[int(k)] for k in np.linspace(0, len(have) - 1, min(8, len(have)))] if have else []
    for f in picks:
        img = cv2.imread(str(frames_dir / f"{int(f):06d}.jpg"))
        draw_context(img)
        r = det[f]
        p = (int(r["px"]), int(r["py"]))
        cv2.circle(img, p, rad + 4, (255, 255, 255), 2)
        cv2.circle(img, p, rad, (0, 0, 255), -1)
        cv2.putText(img, f"f{int(f)} vis={r['visibility']:.2f}", (p[0] + 12, p[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, fw / 1600, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"frame_{int(f):06d}.png"), img)

    # 3) overlay.mp4: current detection + a fading trail of the last 8, downscaled.
    scale = min(1.0, video_max_w / fw)
    ow, oh = int(fw * scale), int(fh * scale)
    fps = float(manifest.get("analysis_fps") or 12.0)
    vw = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    trail: list = []
    for f in frames:
        idx = int(f["frame"])
        img = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"))
        if img is None:
            continue
        draw_context(img)
        if idx in det:
            r = det[idx]
            trail.append((int(r["px"]), int(r["py"])))
            trail[:] = trail[-8:]
        for j, (tx, ty) in enumerate(trail):
            age = (len(trail) - j) / len(trail)
            cv2.circle(img, (tx, ty), max(3, int(rad * (1 - 0.6 * age))),
                       (0, int(140 + 115 * (1 - age)), 255), -1)
        if idx in det:
            cv2.circle(img, trail[-1], rad + 4, (255, 255, 255), 2)
        small = cv2.resize(img, (ow, oh))
        cv2.putText(small, f"{tracker_name}  f{idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
        vw.write(small)
    vw.release()

    summary = {"video_id": video_id, "tracker": tracker_name, "detections": len(det), "frames": n,
               "out_dir": str(out), "scatter": str(out / "scatter.png"),
               "video": str(out / "overlay.mp4"), "key_frames": [f"frame_{int(f):06d}.png" for f in picks]}
    log.info("render: %d detections over %d frames → %s", len(det), n, out)
    return summary
