"""Stage 5: shuttle tracking — detect the shuttle per frame on the TARGET court.

The hall holds several courts (several shuttles) and the shuttle is a few pixels in a wide 4K DJI shot,
so we crop to the target court's calibration polygon first (`geometry.court_crop_box`): that ignores the
other courts AND restores effective shuttle resolution before the tracker downsizes. A swappable
backend (WASB / TrackNetV3 / TrackNetV5, behind `shuttle.get_tracker`) returns a per-frame peak; we map it through
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
from ..manifest import REFERENCE_FPS, manifest_fps, per_step
from ..shuttle import get_tracker
from ..shuttle.base import chunk_spans
from ..shuttle.track import build_shuttle_track

log = logging.getLogger("worker.pipeline.shuttle")

# Peak-confidence gate — a peak below this is a real dropout. Unset → the backend's own published
# threshold (`tracker.conf_thr`, e.g. TrackNetV5's 0.35 Lite / 0.30 full), else 0.5; an explicit env
# always wins (resolve_conf_thr). CONF_THR keeps the module-level fallback for direct callers.
CONF_THR_ENV = os.environ.get("SHUTTLE_CONF_THR")
CONF_THR = float(CONF_THR_ENV) if CONF_THR_ENV else 0.5
# Court-space isolated-spike guard as a RATE (court units per SECOND); run() derives the per-step value
# from the manifest's measured cadence (specs/features/analysis-30fps.md). 5.0/s = the tuned 0.5 per
# step at the real 10 fps cadence; MAX_STEP is that per-step equivalent for direct callers.
MAX_STEP_PER_S = float(os.environ.get("SHUTTLE_MAX_STEP_PER_S", "5.0"))
MAX_STEP = per_step(MAX_STEP_PER_S, REFERENCE_FPS)
# Bounded-memory prediction: process the clip in chunks of this many crops instead of holding every
# full-res crop in RAM (a full match would OOM). Boundaries are aligned to the backend's temporal
# window (chunk_spans), and TrackNetV3's median background is computed per-clip (begin_clip), so
# chunked output is identical to the single-pass path. 240 is a multiple of both window sizes (8, 3)
# ≈ 2-3 GB peak crop RAM on a 4K court crop. 0 disables chunking (the legacy single-pass path).
CHUNK_FRAMES = int(os.environ.get("SHUTTLE_CHUNK_FRAMES", "240"))
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


def resolve_conf_thr(tracker, env_value: str | None = CONF_THR_ENV) -> float:
    """The peak-confidence gate for this run: `SHUTTLE_CONF_THR` when set, else the backend's declared
    `conf_thr` (its published operating point), else 0.5. Without this a backend whose threshold is
    below 0.5 would be silently re-thresholded by the stage."""
    if env_value not in (None, ""):
        return float(env_value)
    declared = getattr(tracker, "conf_thr", None)
    return float(declared) if declared else 0.5


class CropLoader:
    """Lazy (frame_idx, masked_crop) access over the cached JPEGs — only the chunk being predicted
    (plus the median subsample) is ever in memory. Unreadable frames are skipped with a warning
    (counted in `skipped`); note a skip shifts window grouping within its chunk, so bit-identity with
    the single-pass path is guaranteed for a complete frame cache (the normal case — ingest writes
    the cache atomically)."""

    def __init__(self, frames, frames_dir: Path, crop_box, mask) -> None:
        self.frame_ids = [int(f["frame"]) for f in frames]
        self._dir = frames_dir
        self._box = crop_box
        self._mask = mask
        self.skipped = 0

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, i: int):
        """(frame_idx, crop) — crop is None when the cached JPEG is unreadable."""
        import cv2  # lazy (ml extra)

        idx = self.frame_ids[i]
        img = cv2.imread(str(self._dir / f"{idx:06d}.jpg"))
        if img is None:
            log.warning("shuttle: unreadable cached frame %06d.jpg — skipped", idx)
            self.skipped += 1
            return idx, None
        x0, y0, x1, y1 = self._box
        c = img[y0:y1, x0:x1]
        if self._mask is not None:
            c = cv2.bitwise_and(c, c, mask=self._mask)
        return idx, c

    def load_span(self, a: int, b: int) -> list:
        return [(idx, img) for idx, img in (self[i] for i in range(a, b)) if img is not None]


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
    fps = manifest_fps(manifest)  # measured cadence — the spike guard is a rate, applied per frame step
    max_step = per_step(MAX_STEP_PER_S, fps)
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

    loader = CropLoader(frames, frames_dir, crop_box, mask)

    tracker = get_tracker()
    conf_thr = resolve_conf_thr(tracker)
    begin_clip = getattr(tracker, "begin_clip", None)
    if callable(begin_clip):
        begin_clip(loader)  # per-clip state (TrackNetV3 median, Fake phase) shared across chunks
    align = int(getattr(tracker, "window", 1) or 1)
    dets, n_crops = [], 0
    for a, b in chunk_spans(len(loader), CHUNK_FRAMES, align):
        span = loader.load_span(a, b)  # only this chunk's crops are in memory
        n_crops += len(span)
        if span:
            dets.extend(tracker.predict(span))
    if not n_crops:
        raise RuntimeError("shuttle: no frames in cache — ingest must run first")

    raw = [(d.frame, d.x, d.y, d.visibility) for d in dets]
    mw, mh = tracker.model_size
    rows = build_shuttle_track(raw, H, crop_box, mw, mh, t_by_frame,
                               conf_thr=conf_thr, max_step=max_step)
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

    n_seen = sum(1 for d in dets if d.visibility >= conf_thr)
    rate = 100.0 * len(rows) / max(n_crops, 1)
    log.info("shuttle: %s on %d frames → %d kept rows (%.0f%% coverage; %d raw peaks ≥%.2f, "
             "max_step=%.3f/frame @%.2f fps, crop=%s mask=%s chunk=%d)%s", tracker.name, n_crops,
             len(rows), rate, n_seen, conf_thr, max_step, fps, crop_box, MASK, CHUNK_FRAMES,
             f", {loader.skipped} unreadable frames skipped" if loader.skipped else "")
