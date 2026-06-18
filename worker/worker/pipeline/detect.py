"""Stage: detect — YOLO11 person detection + BoT-SORT tracking over the cached frames.

PyTorch MPS primary (fp32, PYTORCH_ENABLE_MPS_FALLBACK=1); one-time NaN check falls back to CPU on
silent MPS failure. Writes detections.parquet (frame, track, x, y, w, h, score) in PIXEL coords with
the top-left convention geometry.ankle_point expects. specs/30-ml-pipeline.md stage 2/3.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("worker.pipeline.detect")

DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolo11m.pt")
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))  # 4K → 1280 keeps far/small players detectable


def _pick_device(model, sample_path: str):
    import torch

    if not torch.backends.mps.is_available():
        return "cpu"
    try:
        res = model.predict(sample_path, device="mps", imgsz=IMGSZ, classes=[0], verbose=False)[0]
        xy = res.boxes.xyxy
        if xy is not None and bool(torch.isnan(xy).any()):
            log.warning("detect: NaN in MPS output — falling back to CPU")
            return "cpu"
        return "mps"
    except Exception as e:  # noqa: BLE001
        log.warning("detect: MPS probe failed (%s) — using CPU", e)
        return "cpu"


def run(ctx) -> None:
    store = ctx["store"]
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    manifest = json.loads((cache / "frames.json").read_text())
    frames = manifest["frames"]
    frames_dir = cache / "frames"
    if not frames:
        raise RuntimeError("detect: empty frame cache (ingest must run first)")

    from ultralytics import YOLO  # lazy

    model = YOLO(DEFAULT_MODEL)
    first = str(frames_dir / f"{frames[0]['frame']:06d}.jpg")
    device = _pick_device(model, first)
    log.info("detect: %s on %s, imgsz=%d, %d frames", DEFAULT_MODEL, device, IMGSZ, len(frames))

    rows = []
    for f in frames:
        p = str(frames_dir / f"{f['frame']:06d}.jpg")
        res = model.track(
            p, persist=True, classes=[0], conf=0.25, iou=0.7, imgsz=IMGSZ, device=device,
            tracker="botsort.yaml", verbose=False,
        )[0]
        b = res.boxes
        if b is None or b.xyxy is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy() if b.id is not None else [None] * len(xyxy)
        for (x1, y1, x2, y2), tid, sc in zip(xyxy, ids, conf):
            rows.append(
                {
                    "frame": int(f["frame"]),
                    "track": int(tid) if tid is not None else None,
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "score": float(sc),
                }
            )

    import polars as pl  # lazy

    out = cache / "detections.parquet"
    pl.DataFrame(rows, schema={"frame": pl.Int64, "track": pl.Int64, "x": pl.Float64,
                              "y": pl.Float64, "w": pl.Float64, "h": pl.Float64, "score": pl.Float64}).write_parquet(out)
    ctx["detections_path"] = str(out)
    store.upload(cfg.artifact_bucket, f"{video_id}/detections.parquet", out.read_bytes(), "application/octet-stream")
    n_tracks = len({r["track"] for r in rows if r["track"] is not None})
    log.info("detect: %d boxes, %d raw tracks", len(rows), n_tracks)
