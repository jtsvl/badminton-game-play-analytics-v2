"""Stage 5: shuttle tracking — detect the shuttle per frame on the TARGET court.

The hall holds several courts (several shuttles) and the shuttle is a few pixels in a wide 4K DJI shot,
so we crop to the target court's calibration polygon first (`geometry.court_crop_box`): that ignores the
other courts AND restores effective shuttle resolution before the tracker downsizes. A swappable
backend (WASB / TrackNetV3, behind `shuttle.get_tracker`) returns a per-frame peak; we map it through
the SAME homography the player overlay uses to court (u,v), drop low-confidence dropouts, reject isolated
spikes, and write `shuttle.parquet (t,u,v,visibility + debug)`. overlay then emits subject='shuttle'.

Honest limits (specs/30 stage 5, surfaced as „Schätzung"): an airborne shuttle projected through the
GROUND homography lands beyond its true spot; the 12 fps cache undersamples a fast smash; DJI ultrawide
undistortion is still a no-op. So this is a flight-path estimate — speed/landing wait for later stages.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..geometry import court_airspace_polygon, court_crop_box
from ..shuttle import get_tracker
from ..shuttle.track import build_shuttle_track

log = logging.getLogger("worker.pipeline.shuttle")

CONF_THR = float(os.environ.get("SHUTTLE_CONF_THR", "0.5"))   # peak below this = a real dropout
MAX_STEP = float(os.environ.get("SHUTTLE_MAX_STEP", "0.5"))   # court-unit isolated-spike guard
# Isolate the target court's AIRSPACE (ground quad raised up) so the side courts are masked out before
# the tracker sees the frame. up/side shape the far edge; down/out extend the near edge to keep high
# clears and just-out landings inside the mask; the bbox margin pads the crop around it.
# NOTE: defaults are deliberately conservative (down/out=0.0, up=0.7 = the legacy box) so production is
# unchanged. The wider box is exercised via env (SHUTTLE_AIR_UP/DOWN/OUT) for the A/B; flip the defaults
# only once the A/B with the fine-tuned model confirms precision holds (the model trained on the old crop).
MASK = os.environ.get("SHUTTLE_MASK", "1").lower() not in ("0", "", "false", "no")
AIR_UP = float(os.environ.get("SHUTTLE_AIR_UP", "0.7"))       # raise far edge by up*court-height
AIR_SIDE = float(os.environ.get("SHUTTLE_AIR_SIDE", "0.08"))  # widen far edge by side*court-width
AIR_DOWN = float(os.environ.get("SHUTTLE_AIR_DOWN", "0.0"))   # lower near edge by down*court-height
AIR_OUT = float(os.environ.get("SHUTTLE_AIR_OUT", "0.0"))     # spread near corners by out*court-width
BBOX_MARGIN = float(os.environ.get("SHUTTLE_BBOX_MARGIN", "0.02"))

SCHEMA = {
    "t": "Float64", "u": "Float64", "v": "Float64", "visibility": "Float64",
    "frame": "Int64", "px": "Float64", "py": "Float64", "conf": "Float64", "tracker": "Utf8",
}


def _calibration(ctx, video_id):
    """Homography + court polygon (pixels) from ctx (set by calibrate) or the DB. The polygon is the
    in-court mask (`mask_polygon_px`), falling back to the confirmed corners."""
    H = ctx.get("H")
    poly = ctx.get("mask_poly_px")
    if H and poly:
        return H, poly
    store = ctx.get("store")
    cal = store.get_calibration(video_id) if store else None
    H = H or (cal or {}).get("homography")
    poly = poly or (cal or {}).get("mask_polygon_px") or (cal or {}).get("confirmed_points")
    if not H or not poly:
        raise RuntimeError("shuttle: no homography/court polygon — calibrate must run first")
    return H, poly


def run(ctx) -> None:
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    frames_dir = cache / "frames"
    manifest = json.loads((cache / "frames.json").read_text())
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])
    frames = manifest["frames"]
    fr = ctx.get("frame_range")  # optional (bench/render only); production never sets it → full clip
    if fr:
        a, b = fr
        frames = [f for f in frames if a <= int(f["frame"]) <= b]
        if not frames:
            raise RuntimeError(f"shuttle: no frames in range {a}:{b}")
    t_by_frame = {int(f["frame"]): float(f["t"]) for f in frames}

    H, poly = _calibration(ctx, video_id)
    air = court_airspace_polygon(poly, frame_w, frame_h, up=AIR_UP, side=AIR_SIDE, down=AIR_DOWN, out=AIR_OUT)
    crop_box = court_crop_box(air, frame_w, frame_h, up=0.0, side=BBOX_MARGIN, down=0.0)
    x0, y0, x1, y1 = crop_box
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError(f"shuttle: degenerate crop {crop_box} — check the calibration polygon")

    import cv2  # lazy (ml extra)
    import numpy as np

    # Static airspace mask in crop-local coords (tripod): black out everything outside the target court.
    mask = None
    if MASK:
        mask = np.zeros((y1 - y0, x1 - x0), dtype="uint8")
        local = np.array([[int(round(px - x0)), int(round(py - y0))] for px, py in air], dtype="int32")
        cv2.fillPoly(mask, [local], 255)

    crops = []
    for f in frames:
        idx = int(f["frame"])
        img = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"))
        if img is None:
            continue
        c = img[y0:y1, x0:x1]
        if mask is not None:
            c = cv2.bitwise_and(c, c, mask=mask)
        crops.append((idx, c))
    if not crops:
        raise RuntimeError("shuttle: no frames in cache — ingest must run first")

    tracker = get_tracker()
    dets = tracker.predict(crops)
    raw = [(d.frame, d.x, d.y, d.visibility) for d in dets]
    mw, mh = tracker.model_size
    rows = build_shuttle_track(raw, H, crop_box, mw, mh, t_by_frame,
                               conf_thr=CONF_THR, max_step=MAX_STEP)
    for r in rows:
        r["tracker"] = tracker.name

    import polars as pl  # lazy

    out = cache / "shuttle.parquet"
    pl.DataFrame(rows, schema={k: getattr(pl, v) for k, v in SCHEMA.items()}).write_parquet(out)
    ctx["shuttle_path"] = str(out)

    store = ctx.get("store")
    if store is not None:
        store.upload(cfg.artifact_bucket, f"{video_id}/shuttle.parquet", out.read_bytes(),
                     "application/octet-stream")

    n_seen = sum(1 for d in dets if d.visibility >= CONF_THR)
    rate = 100.0 * len(rows) / max(len(crops), 1)
    log.info("shuttle: %s on %d frames → %d kept rows (%.0f%% coverage; %d raw peaks ≥thr, "
             "crop=%s mask=%s)", tracker.name, len(crops), len(rows), rate, n_seen, crop_box, MASK)
