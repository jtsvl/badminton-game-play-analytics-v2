"""WASB backend (nttcom/WASB-SBDT, BMVC2023, MIT) — the SOTA badminton ball detector/tracker.

An HRNet that takes a 3-frame window (9 RGB channels, ImageNet-normalized, 512x288) and emits a [0,1]
heatmap per frame; high-resolution features + temporal consistency are what put it top of the
Papers-with-Code badminton benchmark. We build the HRNet straight from the repo's `wasb.yaml` (via
OmegaConf, which the model's mixed dict/attr config access needs) + the published weights, run our court
crops through it, and read the heatmap argmax per frame. We deliberately bypass the repo's
postprocessor/tracker (coupled to its dataloader's affine transforms) — argmax + sigmoid(peak) is enough
for the slice-1 baseline (specs/30 stage 5).

Setup (one-time, on the Mac Studio):
  git clone https://github.com/nttcom/WASB-SBDT worker/.cache/_vendor/WASB-SBDT
  # weights: MODEL_ZOO WASB/Badminton -> models/weights/shuttle/wasb_badminton_best.pth.tar
Override with WASB_REPO / WASB_MODEL. Needs `omegaconf`.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Sequence

from .base import Crop, ShuttleDetection, heatmap_peak

log = logging.getLogger("worker.shuttle.wasb")

_ROOT = Path(__file__).resolve().parents[3]
REPO = os.environ.get("WASB_REPO", str(_ROOT / "worker/.cache/_vendor/WASB-SBDT"))
MODEL = os.environ.get("WASB_MODEL", str(_ROOT / "models/weights/shuttle/wasb_badminton_best.pth.tar"))
BATCH = int(os.environ.get("WASB_BATCH", "4"))

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class WASBTracker:
    name = "wasb"
    model_size = (512, 288)  # (W, H)

    def __init__(self, repo: str = REPO, model_path: str = MODEL):
        import torch
        from omegaconf import OmegaConf

        if not Path(model_path).exists():
            raise RuntimeError(f"wasb: weights not found at {model_path} (see module docstring)")
        src = str(Path(repo) / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from models.hrnet import HRNet  # vendored repo (torch-only)

        cfg = OmegaConf.load(str(Path(src) / "configs/model/wasb.yaml"))
        self.frames_in = int(cfg["frames_in"])
        self.window = self.frames_in  # chunk boundaries must align to the 3-frame windows
        model = HRNet(cfg)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            log.warning("wasb: load_state_dict missing=%d unexpected=%d", len(missing), len(unexpected))
        model.eval()
        self.device = self._pick_device(model)
        self.model = model.to(self.device)
        log.info("wasb: HRNet frames_in=%d on %s", self.frames_in, self.device)

    def _pick_device(self, model) -> str:
        import torch

        if not torch.backends.mps.is_available():
            return "cpu"
        probe = torch.zeros((1, self.frames_in * 3, self.model_size[1], self.model_size[0]))
        try:
            with torch.no_grad():
                out = model.to("mps")(probe.to("mps"))[0]
            if not bool(torch.isnan(out).any()):
                return "mps"
            log.warning("wasb: NaN on MPS probe — using CPU")
        except Exception as e:  # noqa: BLE001
            log.warning("wasb: MPS probe failed (%s) — using CPU", e)
        model.to("cpu")
        return "cpu"

    def _prep(self, img):
        """BGR crop -> normalized (3,288,512) float32 (RGB/255, ImageNet mean/std)."""
        import cv2
        import numpy as np

        mw, mh = self.model_size
        rgb = cv2.resize(img, (mw, mh))[:, :, ::-1].astype("float32") / 255.0
        rgb = (rgb - np.array(_MEAN, dtype="float32")) / np.array(_STD, dtype="float32")
        return np.moveaxis(rgb, -1, 0)  # (3, H, W)

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        import numpy as np
        import torch

        frame_ids = [idx for idx, _ in crops]
        prepped = [self._prep(img) for _, img in crops]  # each (3,H,W)
        n = len(prepped)
        fi = self.frames_in

        # Non-overlapping windows of `frames_in`, padding the last with its final frame.
        windows, win_pos = [], []
        for start in range(0, n, fi):
            idxs = [min(start + k, n - 1) for k in range(fi)]
            windows.append(np.concatenate([prepped[k] for k in idxs], axis=0))  # (9,H,W)
            win_pos.append(idxs)

        best: dict[int, ShuttleDetection] = {}
        for b0 in range(0, len(windows), BATCH):
            batch = windows[b0:b0 + BATCH]
            x = torch.from_numpy(np.stack(batch)).float().to(self.device)
            with torch.no_grad():
                y = torch.sigmoid(self.model(x)[0]).cpu().numpy()  # (B, frames_in, H, W) in [0,1]
            for bi, idxs in enumerate(win_pos[b0:b0 + BATCH]):
                for f, pos in enumerate(idxs):
                    if pos in best:
                        continue  # padding repeat — keep first
                    px, py, conf = heatmap_peak(y[bi, f])
                    best[pos] = ShuttleDetection(frame_ids[pos], px, py, conf)
        return [best[i] for i in sorted(best)]
