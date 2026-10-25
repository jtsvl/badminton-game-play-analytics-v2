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

    # Backends may additionally expose (the stage discovers both via getattr, so they stay optional):
    #   window: int             — temporal window size; chunk boundaries must be multiples of it
    #   begin_clip(loader)      — compute per-clip state once (e.g. TrackNetV3's median background)
    #                             so chunked predict() calls share it


def chunk_spans(n_frames: int, chunk_size: int, align: int) -> List[Tuple[int, int]]:
    """Contiguous (start, end) spans covering 0..n_frames for bounded-memory prediction.

    Every span boundary is a multiple of `align` (the backend's temporal window), so the backend's
    non-overlapping windowing groups frames exactly as it would over the whole clip — chunked
    predictions are identical to full-clip ones, just without holding every crop in RAM.
    chunk_size <= 0 or >= n_frames means a single span (the legacy full-clip path)."""
    if n_frames <= 0:
        return []
    align = max(int(align), 1)
    if chunk_size <= 0 or chunk_size >= n_frames:
        return [(0, n_frames)]
    eff = max(align, chunk_size - chunk_size % align)
    return [(a, min(a + eff, n_frames)) for a in range(0, n_frames, eff)]


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
    window = 1  # no temporal windowing — any chunk boundary is fine

    def __init__(self) -> None:
        self._clip_len: int | None = None
        self._pos: dict | None = None

    def begin_clip(self, loader) -> None:
        """Pin the arc's phase to the WHOLE clip so chunked predict() calls trace the same parabola
        a single full-clip call would (phase is clip position, not position within the call)."""
        ids = list(loader.frame_ids)
        self._clip_len = max(len(ids), 1)
        self._pos = {fid: i for i, fid in enumerate(ids)}

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        out: List[ShuttleDetection] = []
        mw, mh = self.model_size
        n = self._clip_len or max(len(crops), 1)
        for local_i, (frame_idx, _img) in enumerate(crops):
            i = self._pos.get(frame_idx, local_i) if self._pos is not None else local_i
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


# ── shared backend helpers (TrackNetV3 + TrackNetV5; lazy numpy/cv2 so this module stays dev-extra light) ──

def resize_rgb(img_bgr, model_size: Tuple[int, int]):
    """A BGR crop → RGB uint8 at the model's (W, H) — the one resize both TrackNet backends feed."""
    import cv2

    mw, mh = model_size
    return cv2.resize(img_bgr, (mw, mh))[:, :, ::-1]


def median_of_frames(rgb_u8, samples: int = 700):
    """Background median (float64 HWC) of an RGB uint8 stack from a stride subsample of ≤ `samples`
    frames (≤ samples → every frame, byte-identical to a full median). The subsample caps the np.median
    transient that once ~doubled peak memory on a whole clip and SIGKILLed the process."""
    import numpy as np

    stride = max(1, len(rgb_u8) // samples)
    return np.median(rgb_u8[::stride], axis=0)


def clip_median(loader, model_size: Tuple[int, int], samples: int = 700):
    """Per-CLIP background median computed ONCE from a stride subsample of the whole clip (same resize /
    BGR→RGB / dtype as predict) so every chunked predict() call shares the clip-wide background — a
    per-chunk median would differ and change detections. None when the clip is empty."""
    import numpy as np

    n = len(loader)
    if n == 0:
        return None
    stride = max(1, n // samples)
    sample = [loader[i] for i in range(0, n, stride)]
    rgb = np.stack([resize_rgb(img, model_size) for _, img in sample if img is not None]).astype("uint8")
    return np.median(rgb, axis=0)


def largest_blob_bbox(binary) -> tuple:
    """Largest-area blob bbox (x, y, w, h) of a binary heatmap — a faithful reimplementation of the
    TrackNet repos' `test.predict_location` (not imported: their test.py pulls pycocotools and the
    vendored module names collide across backends). (0, 0, 0, 0) when the map is empty."""
    import cv2
    import numpy as np

    if np.amax(binary) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    return max(rects, key=lambda r: r[2] * r[3])


def decode_heatmap(hm, thr: float):
    """TrackNet heatmap → (x, y, conf) of the largest blob above `thr`, conf = the map's (finite) peak;
    None when nothing is above the threshold. `thr` is the backend's operating point (V3 0.5; V5 0.35
    Lite / 0.30 full per the published sweep) — the stage gates `visibility` with the same value."""
    import numpy as np

    x0, y0, w, h = largest_blob_bbox((hm > thr).astype("uint8"))
    if w == 0 and h == 0:
        return None
    finite = hm[np.isfinite(hm)]
    conf = float(finite.max()) if finite.size else 0.0
    return x0 + w / 2, y0 + h / 2, conf


def checkpoint_tag(path, n: int = 8) -> str:
    """First `n` hex chars of a file's SHA-256 — the provenance suffix in `shuttle.parquet.tracker`
    (`tracknetv5-lite@<sha8>`), so a baseline and a fine-tune of the same backend stay distinguishable."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def vendor_commit(repo) -> str | None:
    """The checked-out commit of a vendored git repo (reads .git/HEAD, following one symbolic ref),
    or None when it is not a git checkout."""
    from pathlib import Path

    git = Path(repo) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:].strip()
            direct = git / ref
            if direct.exists():
                return direct.read_text().strip() or None
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
            return None
        return head or None
    except OSError:
        return None
