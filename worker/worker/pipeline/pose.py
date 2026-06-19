"""Stage: pose — RTMPose top-down keypoints, the durable fix for the foot point (RC1).

The detector's bbox bottom sits above the real feet (~21px@1080 / ~42px@4K), and the homography
amplifies that ~8x on the far half (specs/fix-later.md). This stage feeds each detector crop to
RTMPose-m body26 (Halpe-26, Apache) and derives the foot from the real ankle keypoint; gate_track then
projects that instead of the bbox bottom. Low-confidence ankles fall back to the Phase-2 bbox nudge,
and the full keypoint array is kept for the Phase-4 head/height occlusion fallback + stroke stage.

Runs *before* gate_track (detect -> pose -> gate_track) so the gating/projection uses the pose foot.
RTMPose runs on ONNXRuntime; device='mps' selects the CoreML EP (CLAUDE.md allows CoreML for RTMPose),
with a one-time NaN probe falling back to CPU. specs/30-ml-pipeline.md stage 4.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..geometry import (
    FOOT_KP, HEAD_KP, HIP_KP, fit_head_foot_model, foot_keypoint, ground_foot_from_head,
)

log = logging.getLogger("worker.pipeline.pose")

# RTMPose-m body26 (Halpe-26), 256x192 top-down ONNX (OpenMMLab model zoo, Apache-2.0). Override with a
# local /models/*.onnx path via RTMPOSE_MODEL once the weights are cached on the Mac Studio.
DEFAULT_RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
)
RTMPOSE_MODEL = os.environ.get("RTMPOSE_MODEL", DEFAULT_RTMPOSE_URL)
RTMPOSE_INPUT = (192, 256)  # (w, h) for the 256x192 model
POSE_CONF_THR = float(os.environ.get("POSE_CONF_THR", "0.3"))


def pose_row(frame: int, track: int, bbox, kp, scores,
             conf_thr: float = POSE_CONF_THR, foot_indices=FOOT_KP) -> dict:
    """Assemble one pose.parquet row: full keypoint array + the derived foot (mean confident ankle) and
    its confidence. foot_conf < conf_thr signals gate_track to fall back to the bbox nudge.

    `foot_source` records *where* the foot came from for downstream uncertainty: "ankle" when the real
    ankle keypoint is confident, else "none" — a placeholder the second pass (`ground_occluded_feet`)
    may upgrade to "head" when it can predict the foot from the still-visible head + hip."""
    fk = foot_keypoint(kp, scores, foot_indices, conf_thr)
    fx, fy, fc = fk if fk is not None else (None, None, 0.0)
    return {
        "frame": int(frame),
        "track": int(track),
        "kp": [[float(p[0]), float(p[1])] for p in kp],
        "kp_score": [float(s) for s in scores],
        "foot_x": fx,
        "foot_y": fy,
        "foot_conf": float(fc),
        "foot_source": "ankle" if fc >= conf_thr else "none",
    }


def ground_occluded_feet(rows: list, conf_thr: float = POSE_CONF_THR) -> list:
    """Second pass (Phase-4 occlusion layer): recover the foot of ankle-occluded frames from the head.

    Per track, fit a head→foot drop (`fit_head_foot_model`) from that player's *own* confident,
    feet-visible frames (ankle, head and hip all ≥ conf_thr), then for each frame whose ankle is hidden
    (foot_conf < conf_thr) but whose head + hip are still confident, predict the foot
    (`ground_foot_from_head`) and mark `foot_source="head"`. The grounded foot carries a real but lower
    confidence (min of head/hip), so it stays an estimate („Schätzung"). Rows are mutated in place and
    returned. No fitted model for a track (no feet-visible frame) → that track is left untouched.

    Honest limits: height varies ~±10 %, posture shifts the torso↔leg ratio, and the low standing camera
    foreshortens the drop — see `geometry.fit_head_foot_model`. Most valuable in doubles (P3)."""
    by_track: dict[int, list] = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)
    for trows in by_track.values():
        samples = [
            (r["kp"][HEAD_KP][1], r["kp"][HIP_KP][1], r["foot_y"])
            for r in trows
            if r["foot_conf"] >= conf_thr
            and r["kp_score"][HEAD_KP] >= conf_thr and r["kp_score"][HIP_KP] >= conf_thr
        ]
        ratio = fit_head_foot_model(samples)
        if ratio is None:
            continue
        for r in trows:
            if r["foot_conf"] >= conf_thr:
                continue  # ankle already confident — leave it
            hc, pc = r["kp_score"][HEAD_KP], r["kp_score"][HIP_KP]
            if hc < conf_thr or pc < conf_thr:
                continue  # head/hip also hidden — nothing to ground from
            fx, fy = ground_foot_from_head(r["kp"][HEAD_KP], r["kp"][HIP_KP], ratio)
            r["foot_x"], r["foot_y"], r["foot_conf"], r["foot_source"] = fx, fy, min(hc, pc), "head"
    return rows


def _load_model(frames_dir: Path, first_frame: int, first_bbox):
    """Build RTMPose on CoreML (device='mps'); on a NaN probe (CLAUDE.md) rebuild on CPU."""
    import cv2  # lazy
    import numpy as np
    from rtmlib import RTMPose

    img = cv2.imread(str(frames_dir / f"{first_frame:06d}.jpg"))
    for device in ("mps", "cpu"):
        try:
            model = RTMPose(onnx_model=RTMPOSE_MODEL, model_input_size=RTMPOSE_INPUT,
                            backend="onnxruntime", device=device)
            kps, _ = model(img, [first_bbox])
            if not bool(np.isnan(kps).any()):
                log.info("pose: RTMPose body26 on %s", device)
                return model
            log.warning("pose: NaN keypoints on %s — trying next device", device)
        except Exception as e:  # noqa: BLE001
            log.warning("pose: RTMPose init/probe failed on %s (%s)", device, e)
    raise RuntimeError("pose: RTMPose failed on both mps and cpu")


def run(ctx) -> None:
    cfg = ctx["cfg"]
    job = ctx["job"]
    video_id = job["video_id"]

    cache = Path(cfg.frame_cache_dir).expanduser() / str(video_id)
    frames_dir = cache / "frames"
    det_path = ctx.get("detections_path") or str(cache / "detections.parquet")

    import cv2  # lazy
    import polars as pl

    det = pl.read_parquet(det_path)
    by_frame: dict[int, list] = {}
    for r in det.iter_rows(named=True):
        if r["track"] is None:
            continue  # untracked boxes never reach gate_track, so no pose needed
        by_frame.setdefault(int(r["frame"]), []).append(r)
    if not by_frame:
        raise RuntimeError("pose: no tracked detections (detect must run first)")

    frame0 = sorted(by_frame)[0]
    d0 = by_frame[frame0][0]
    model = _load_model(frames_dir, frame0, [d0["x"], d0["y"], d0["x"] + d0["w"], d0["y"] + d0["h"]])

    rows = []
    for frame_idx in sorted(by_frame):
        dets = by_frame[frame_idx]
        img = cv2.imread(str(frames_dir / f"{frame_idx:06d}.jpg"))
        if img is None:
            continue
        bboxes = [[r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]] for r in dets]
        keypoints, scores = model(img, bboxes)  # (N, K, 2), (N, K)
        for r, kp, sc in zip(dets, keypoints, scores):
            rows.append(pose_row(frame_idx, r["track"], (r["x"], r["y"], r["w"], r["h"]), kp, sc))

    ground_occluded_feet(rows)  # Phase-4: head/height-ground the ankle-occluded frames

    out = cache / "pose.parquet"
    pl.DataFrame(rows, schema={
        "frame": pl.Int64, "track": pl.Int64,
        "kp": pl.List(pl.List(pl.Float64)), "kp_score": pl.List(pl.Float64),
        "foot_x": pl.Float64, "foot_y": pl.Float64, "foot_conf": pl.Float64,
        "foot_source": pl.Utf8,
    }).write_parquet(out)
    ctx["pose_path"] = str(out)

    store = ctx.get("store")
    if store is not None:
        store.upload(cfg.artifact_bucket, f"{video_id}/pose.parquet", out.read_bytes(),
                     "application/octet-stream")
    n_conf = sum(1 for r in rows if r["foot_source"] == "ankle")
    n_head = sum(1 for r in rows if r["foot_source"] == "head")
    log.info("pose: %d keypoint sets, %d ankle-foot / %d head-grounded / %d bbox-fallback (conf>=%.2f)",
             len(rows), n_conf, n_head, len(rows) - n_conf - n_head, POSE_CONF_THR)
