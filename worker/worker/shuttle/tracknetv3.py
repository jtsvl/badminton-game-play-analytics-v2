"""TrackNetV3 backend (qaz812345/TrackNetV3, MIT) — the recall-champion shuttlecock tracker.

A conv U-Net that takes an 8-frame window + a per-clip median background (bg_mode='concat', 27 input
channels) and emits one [0,1] heatmap per frame; the median-subtraction is its built-in white-wall
mitigation (specs/30 stage 5, layer 1). We reuse the upstream `Shuttlecock_Trajectory_Dataset` to build
the exact 27-channel input the published weights expect (avoids a hand-rolled channel-layout bug), feed
our court crops pre-resized to the model's 512x288, and read the heatmap peak per frame.

Setup (one-time, on the Mac Studio):
  git clone https://github.com/qaz812345/TrackNetV3 worker/.cache/_vendor/TrackNetV3
  # weights: TrackNetV3_ckpts.zip -> models/weights/shuttle/ckpts/{TrackNet,InpaintNet}_best.pt
Override paths with TRACKNETV3_REPO / TRACKNETV3_MODEL. Needs `pandas` + `parse` (the repo's deps).
Slice 1 uses TrackNet only (no InpaintNet rectification) to see the RAW detector — recall recovery is a
later lever once the baseline failures are measured.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Sequence

from .base import Crop, ShuttleDetection, clip_median, decode_heatmap, median_of_frames, resize_rgb

log = logging.getLogger("worker.shuttle.tracknetv3")

_ROOT = Path(__file__).resolve().parents[3]
REPO = os.environ.get("TRACKNETV3_REPO", str(_ROOT / "worker/.cache/_vendor/TrackNetV3"))
MODEL = os.environ.get("TRACKNETV3_MODEL", str(_ROOT / "models/weights/shuttle/ckpts/TrackNet_best.pt"))
BATCH = int(os.environ.get("TRACKNETV3_BATCH", "4"))
# Background median is computed from a SUBSAMPLE of frames (not the whole stack). The vendored dataset
# otherwise runs np.median over every frame at once — a transient that ~doubles peak memory on a full clip
# and SIGKILLs the process. For a static tripod the subsample median ≈ the full median. ≤MEDIAN_SAMPLES
# frames → stride 1 → byte-identical to the old path.
MEDIAN_SAMPLES = int(os.environ.get("TRACKNETV3_MEDIAN_SAMPLES", "700"))
THR = 0.5  # heatmap binarisation = the published V3 operating point (shared decode in shuttle/base.py)


def _in_dims(seq_len: int, bg_mode: str) -> tuple:
    """Mirror utils.general.get_model's channel arithmetic for the TrackNet variants."""
    if bg_mode == "subtract":
        return seq_len, seq_len
    if bg_mode == "subtract_concat":
        return seq_len * 4, seq_len
    if bg_mode == "concat":
        return (seq_len + 1) * 3, seq_len
    return seq_len * 3, seq_len


class TrackNetV3Tracker:
    name = "tracknetv3"
    model_size = (512, 288)  # (W, H) — the peaks live here; matches HEIGHT=288, WIDTH=512

    def __init__(self, repo: str = REPO, model_path: str = MODEL):
        import torch

        if not Path(model_path).exists():
            raise RuntimeError(f"tracknetv3: weights not found at {model_path} (see module docstring)")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from model import TrackNet  # vendored repo (torch-only)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.seq_len = int(ckpt["param_dict"]["seq_len"])
        self.bg_mode = ckpt["param_dict"]["bg_mode"]
        self.window = self.seq_len  # chunk boundaries must align to the 8-frame windows
        self._median = None
        in_dim, out_dim = _in_dims(self.seq_len, self.bg_mode)
        model = TrackNet(in_dim, out_dim)
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.device = self._pick_device(model)
        self.model = model.to(self.device)
        log.info("tracknetv3: seq_len=%d bg_mode=%s on %s", self.seq_len, self.bg_mode, self.device)

    def _pick_device(self, model) -> str:
        """MPS with a one-shot NaN probe → CPU fallback (CLAUDE.md fp32 + MPS-first)."""
        import torch

        if not torch.backends.mps.is_available():
            return "cpu"
        in_dim, _ = _in_dims(self.seq_len, self.bg_mode)
        probe = torch.zeros((1, in_dim, self.model_size[1], self.model_size[0]), dtype=torch.float32)
        try:
            with torch.no_grad():
                out = model.to("mps")(probe.to("mps"))
            if not bool(torch.isnan(out).any()):
                return "mps"
            log.warning("tracknetv3: NaN on MPS probe — using CPU")
        except Exception as e:  # noqa: BLE001
            log.warning("tracknetv3: MPS probe failed (%s) — using CPU", e)
        model.to("cpu")
        return "cpu"

    def begin_clip(self, loader) -> None:
        """Compute the per-clip median background ONCE from a stride subsample of the whole clip, so
        chunked predict() calls share it (shared `base.clip_median`: same frames, same resize/BGR→RGB,
        same dtype as the legacy in-predict computation → byte-identical median, and each chunk sees the
        clip-wide background — a per-chunk median would differ and change detections)."""
        self._median = None
        if not self.bg_mode:
            return
        self._median = clip_median(loader, self.model_size, MEDIAN_SAMPLES)

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from dataset import Shuttlecock_Trajectory_Dataset  # vendored repo

        frame_ids = [idx for idx, _ in crops]
        # Pre-resize each crop to model size (keeps the median-background cheap) and go BGR→RGB.
        rgb = np.stack([resize_rgb(img, self.model_size) for _, img in crops]).astype("uint8")

        # Per-clip median from begin_clip when the stage runs chunked; else (direct predict() callers)
        # compute it from THESE crops exactly as before. Handing the median to the dataset (which only
        # computes np.median itself when median=None) caps the full-stack memory spike (the crash).
        # Channels-last float64, same layout a computed median would have.
        median = self._median
        if median is None and self.bg_mode:  # '' / None bg_modes never use a median
            median = median_of_frames(rgb, MEDIAN_SAMPLES)

        ds = Shuttlecock_Trajectory_Dataset(
            seq_len=self.seq_len, sliding_step=self.seq_len, data_mode="heatmap",
            bg_mode=self.bg_mode, frame_arr=rgb, padding=True, median=median,
        )
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)

        best: dict[int, ShuttleDetection] = {}
        for data_idx, x in loader:
            with torch.no_grad():
                y = self.model(x.float().to(self.device)).cpu().numpy()  # (B, L, H, W) in [0,1]
            di = data_idx.numpy()
            for b in range(y.shape[0]):
                for f in range(y.shape[1]):
                    pos = int(di[b, f, 1])  # index into the crops list
                    if pos in best:
                        continue  # padding repeats the last frame — keep the first prediction
                    hit = decode_heatmap(y[b, f], THR)
                    best[pos] = (ShuttleDetection(frame_ids[pos], *hit) if hit
                                 else ShuttleDetection(frame_ids[pos], 0.0, 0.0, 0.0))
        return [best[i] for i in sorted(best)]
