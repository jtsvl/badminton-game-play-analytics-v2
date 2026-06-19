"""Shuttle tracker adapter — the interface every backend (WASB, TrackNetV3, …) implements.

Separation of concerns: the pipeline stage crops each frame to the target court and hands the tracker a
list of `(frame_idx, crop_bgr)` images; the tracker handles its own sequence windowing + resize to its
fixed model input and returns one `ShuttleDetection` per frame in MODEL-input pixel space. The stage
then maps those back to court coords (`shuttle.track.build_shuttle_track`). Keeping the court geometry
out of the backends makes them swappable and keeps the math pure + unit-tested.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

log = logging.getLogger("worker.shuttle")

Crop = Tuple[int, "object"]  # (frame_idx, HxWx3 BGR ndarray)


@dataclass
class ShuttleDetection:
    """One per processed frame. (x,y) are the heatmap peak in the model's input pixels; `visibility`
    is the peak confidence in [0,1] (0 ⇒ the tracker saw no shuttle this frame)."""
    frame: int
    x: float
    y: float
    visibility: float


class ShuttleTracker(Protocol):
    name: str
    model_size: Tuple[int, int]  # (model_w, model_h) — the fixed input the peaks live in

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        ...


def heatmap_peak(heatmap) -> Tuple[float, float, float]:
    """Argmax of a 2D confidence heatmap → (x, y, conf). conf is the peak value clamped to [0,1]
    (heatmaps from these trackers are already in [0,1]); used as the shuttle visibility."""
    import numpy as np

    h = np.asarray(heatmap, dtype="float32")
    idx = int(h.argmax())
    y, x = divmod(idx, h.shape[1])
    return float(x), float(y), float(min(max(h.flat[idx], 0.0), 1.0))


def probe_device(run_once, log_prefix: str = "shuttle") -> str:
    """Mirror pose._load_model: try MPS, run a one-shot probe, fall back to CPU on NaN/error (CLAUDE.md
    fp32 + PYTORCH_ENABLE_MPS_FALLBACK). `run_once(device)` must build+run the model once and return its
    output tensor/array (or raise). Returns the first device that yields a finite result."""
    import numpy as np

    for device in ("mps", "cpu"):
        try:
            out = run_once(device)
            if out is None or not bool(np.isnan(np.asarray(out, dtype="float32")).any()):
                log.info("%s: model on %s", log_prefix, device)
                return device
            log.warning("%s: NaN output on %s — trying next device", log_prefix, device)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: init/probe failed on %s (%s)", log_prefix, device, e)
    raise RuntimeError(f"{log_prefix}: model failed on both mps and cpu")


class FakeTracker:
    """Weightless synthetic backend — a deterministic parabolic 'shuttle' with two scripted dropouts.

    Lets the whole stage (crop → tracker → court mapping → parquet → overlay → eval stats) run and be
    verified end-to-end WITHOUT any model weights, and gives the eval harness a known-good signal to
    sanity-check its own statistics against. Selected via SHUTTLE_TRACKER=fake."""

    name = "fake"
    model_size = (512, 288)

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        out: List[ShuttleDetection] = []
        mw, mh = self.model_size
        n = max(len(crops), 1)
        for i, (frame_idx, _img) in enumerate(crops):
            phase = i / n
            # a left→right sweep with a parabolic arc in y; two short camouflage dropouts.
            x = mw * (0.1 + 0.8 * phase)
            y = mh * (0.85 - 1.2 * phase * (1 - phase))  # up at mid-arc
            vis = 0.0 if (10 <= i < 14 or 30 <= i < 33) else 0.85
            out.append(ShuttleDetection(frame=frame_idx, x=x, y=y, visibility=vis))
        return out


def assert_finite(value: float) -> float:
    """Guard against NaN/inf leaking into parquet/court coords."""
    if not math.isfinite(value):
        raise ValueError("shuttle: non-finite coordinate")
    return value
