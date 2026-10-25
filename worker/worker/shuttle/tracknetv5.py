"""TrackNetV5 backend (gino6178/TrackNetV5, ICIP 2026 — Chang, Wang, Huang) — third shuttle tracker.

R(2+1)D stem + gated residual motion prompt + spatial-channel-gate decoder; input = 5 frames × (RGB + a
background-difference channel) = 20 channels at 512×288 (`bg_mode=subtract_concat`), output = one sigmoid
heatmap per frame. Pinned to commit 9116161 (TRACKNETV5_SHA). Specs: specs/features/shuttle-tracknetv5.md.

Why the input is hand-rolled (unlike the V3 backend, which reuses its vendored dataset): the V5 repo's
module names (`dataset`, `test`, `utils`, `models`) collide with TrackNetV3's (`model`, `dataset`, `utils`,
`test`) and WASB's (`models`) inside the one process `eval_shuttle --compare` runs them in — the first
import would silently win. So only `models/tracknetv5.py` (torch-only) is loaded, BY FILE PATH under a
private module name, and the 20-channel tensor is built here from the exact upstream formula
(`background_diff` / `build_input`, TDD'd in tests/test_tracknetv5.py).

Setup (one-time, on the Mac Studio):
  git clone https://github.com/gino6178/TrackNetV5 worker/.cache/_vendor/TrackNetV5
  git -C worker/.cache/_vendor/TrackNetV5 checkout 9116161772183e841bd1b51d6747c70f269b8b1c
  uv run python -m worker convert-tracknetv5 --variant lite --probe   # releases/*.pt → models/weights/shuttle/
  uv run python -m worker convert-tracknetv5 --variant full --probe
Env: TRACKNETV5_REPO, TRACKNETV5_VARIANT (lite|full), TRACKNETV5_MODEL, TRACKNETV5_THR, TRACKNETV5_BATCH,
TRACKNETV5_DIFF (wrap|clip), TRACKNETV5_MEDIAN_SAMPLES. Select with SHUTTLE_TRACKER=tracknetv5.
Licence: the upstream repo has no LICENSE file; the user confirmed a licence grant (2026-10-25).
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .base import (Crop, ShuttleDetection, checkpoint_tag, clip_median, decode_heatmap, median_of_frames,
                   probe_device, resize_rgb, vendor_commit)

log = logging.getLogger("worker.shuttle.tracknetv5")

_ROOT = Path(__file__).resolve().parents[3]
TRACKNETV5_SHA = "9116161772183e841bd1b51d6747c70f269b8b1c"
UPSTREAM_URL = "https://github.com/gino6178/TrackNetV5"
SEQ_LEN = 5                       # frames per window (the released weights' --seq_len)
CHANNELS_PER_FRAME = 4            # R, G, B, background-difference (bg_mode=subtract_concat)
IN_CHANNELS = SEQ_LEN * CHANNELS_PER_FRAME
MODEL_W, MODEL_H = 512, 288       # utils/general.py WIDTH, HEIGHT

# The two released checkpoints (upstream `releases/`), their width multiplier and the detection threshold at
# which the README's F1 sweep peaks (98.363 Lite @0.35, 98.331 full @0.30; 0.5 is the protocol default).
VARIANTS: Dict[str, dict] = {
    "lite": {"alpha": 0.5, "thr": 0.35, "weights": "tracknetv5_lite.pt",
             "upstream": "TrackNetV5_Lite_best.pt",
             "upstream_sha256": "c7ca025e5117d292f9efa60b3132d401434ca1afe26aba9353133ad2823e7a07"},
    "full": {"alpha": 1.0, "thr": 0.30, "weights": "tracknetv5.pt",
             "upstream": "TrackNetV5_best.pt",
             "upstream_sha256": "cf9025285cdcd3089229a45a19ed8ed4075f6a4f459cbf99e450b1f968f7198d"},
}


def default_model_path(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"tracknetv5: unknown variant {variant!r} (expected one of {sorted(VARIANTS)})")
    return str(_ROOT / "models/weights/shuttle" / VARIANTS[variant]["weights"])


REPO = os.environ.get("TRACKNETV5_REPO", str(_ROOT / "worker/.cache/_vendor/TrackNetV5"))
VARIANT = os.environ.get("TRACKNETV5_VARIANT", "lite").lower()
MODEL = os.environ.get("TRACKNETV5_MODEL", default_model_path(VARIANT if VARIANT in VARIANTS else "lite"))
THR_ENV = os.environ.get("TRACKNETV5_THR")
BATCH = int(os.environ.get("TRACKNETV5_BATCH", "4"))
DIFF_MODE = os.environ.get("TRACKNETV5_DIFF", "wrap").lower()   # wrap = upstream's uint8 cast; clip = experiment
MEDIAN_SAMPLES = int(os.environ.get("TRACKNETV5_MEDIAN_SAMPLES", "700"))


def resolve_thr(variant: str, env_value: Optional[str], config: Optional[dict]) -> float:
    """Detection threshold: explicit env > the converted checkpoint's config > the variant's sweep value."""
    if env_value not in (None, ""):
        return float(env_value)
    if config and config.get("thr"):
        return float(config["thr"])
    return float(VARIANTS[variant]["thr"])


# ── pure input construction (numpy only; mirrors dataset.py `bg_mode == 'subtract_concat'` at 9116161) ──

def background_diff(rgb_u8, median, mode: str = "wrap"):
    """Upstream's 4th channel: `np.sum(np.absolute(frame − median), 2).astype('uint8')`.

    The sum over RGB reaches 765 while uint8 stops at 255; the cast truncates then WRAPS modulo 256
    (both x86 and arm64 convert through a 32-bit integer before the byte store), so a strongly changed
    pixel can read as a small difference. `wrap` replicates that bit-exactly — the released weights were
    trained with it, so it is NOT "fixed" silently; `clip` (saturate at 255) is the measurable experiment
    (TRACKNETV5_DIFF=clip)."""
    import numpy as np

    s = np.abs(rgb_u8.astype(np.float64) - np.asarray(median, dtype=np.float64)).sum(axis=2)
    if mode == "wrap":
        return (np.floor(s).astype(np.int64) & 0xFF).astype(np.uint8)
    if mode == "clip":
        return np.minimum(np.floor(s), 255).astype(np.uint8)
    raise ValueError(f"tracknetv5: TRACKNETV5_DIFF must be wrap|clip (got {mode!r})")


def build_input(frames_rgb_u8, median, mode: str = "wrap"):
    """(SEQ_LEN, H, W, 3) RGB uint8 + the clip median → the model's (SEQ_LEN×4, H, W) float32 in [0, 1]:
    per frame [R, G, B, D] (channels-first), frames concatenated in order, everything / 255."""
    import numpy as np

    frames = np.asarray(frames_rgb_u8)
    if frames.ndim != 4 or frames.shape[0] != SEQ_LEN or frames.shape[-1] != 3:
        raise ValueError(f"tracknetv5: expected ({SEQ_LEN}, H, W, 3) frames, got {frames.shape}")
    chans = []
    for f in frames:
        chans.append(np.moveaxis(f, -1, 0).astype(np.float32))
        chans.append(background_diff(f, median, mode)[None].astype(np.float32))
    return np.concatenate(chans, axis=0) / np.float32(255.0)


def windows(n: int, seq_len: int = SEQ_LEN) -> List[List[int]]:
    """Non-overlapping windows over n frames, the last padded by repeating the final frame — the same
    grouping as upstream's `sliding_step = seq_len, padding=True` (and WASB/V3 here), so chunk boundaries
    aligned to `window` give identical results to a single pass."""
    return [[min(start + k, n - 1) for k in range(seq_len)] for start in range(0, n, seq_len)] if n > 0 else []


# ── model + checkpoint plumbing (torch; lazy) ──────────────────────────────────────────────────────────

def load_vendored_model_class(repo: str):
    """`TrackNetV5` from <repo>/models/tracknetv5.py loaded BY FILE PATH under a private module name: no
    sys.path insert, no `models` package in sys.modules (WASB vendors one). The file imports torch only."""
    path = Path(repo) / "models" / "tracknetv5.py"
    if not path.exists():
        raise RuntimeError(f"tracknetv5: vendored repo not found at {repo} (see module docstring; "
                           f"pin {TRACKNETV5_SHA})")
    spec = importlib.util.spec_from_file_location("_vendor_tracknetv5_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.TrackNetV5


def _numpy_safe_globals():
    """The only non-torch globals in the upstream checkpoints (`val_f1` is a numpy scalar): allow-list
    exactly those for weights_only=True — `numpy.dtype`, `numpy.core.multiarray.scalar` and the numpy
    dtype classes. The pickle names the scalar `numpy.core.multiarray.scalar`, but under numpy ≥ 2 the live
    function lives in `numpy._core`, so it is registered with its pickled name via torch's `(obj, name)`
    form (torch ≥ 2.6). Never falls back to weights_only=False."""
    import numpy as np

    allowed: list = [np.dtype]
    scalar = None
    for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            scalar = importlib.import_module(modname).scalar
            break
        except (ImportError, AttributeError):
            continue
    if scalar is not None:
        allowed.append(scalar)
        for legacy in ("numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar"):
            if legacy != f"{scalar.__module__}.{scalar.__qualname__}":
                allowed.append((scalar, legacy))
    try:  # dtype classes pickled by numpy ≥ 1.25 (e.g. numpy.dtypes.Float64DType)
        import numpy.dtypes as npd

        allowed.extend(getattr(npd, n) for n in dir(npd) if n.endswith("DType"))
    except ImportError:
        pass
    return allowed


def load_checkpoint(path):
    """`(state_dict, config)` from either our converted file (`{model_state_dict, config, source}`) or an
    upstream release (`{epoch, model_state_dict, val_f1, args}`), always with `weights_only=True`."""
    import torch

    with torch.serialization.safe_globals(_numpy_safe_globals()):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], dict(ckpt.get("config") or {}), ckpt
    return ckpt, {}, {"model_state_dict": ckpt}


def tracker_name(variant: str, model_path) -> str:
    """`shuttle.parquet.tracker` provenance: backend-variant@<sha8 of the checkpoint file>."""
    return f"tracknetv5-{variant}@{checkpoint_tag(model_path)}"


def only_primitives(obj) -> bool:
    if isinstance(obj, dict):
        return all(isinstance(k, str) and only_primitives(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return all(only_primitives(v) for v in obj)
    return obj is None or isinstance(obj, (str, int, float, bool))


def conversion_payload(state_dict, upstream_args: Optional[dict], variant: str, src_name: str,
                       src_sha256: str, vendor_sha: Optional[str], val_f1: Optional[float]) -> dict:
    """The re-saved checkpoint: plain state_dict + a primitives-only config/source block that
    `weights_only=True` loads without any allow-list (stroke/model.py convention)."""
    args = {k: v for k, v in (upstream_args or {}).items() if only_primitives(v) and not isinstance(v, (dict, list, tuple))}
    return {
        "model_state_dict": state_dict,
        "config": {
            "backend": "tracknetv5", "variant": variant, "alpha": float(VARIANTS[variant]["alpha"]),
            "seq_len": SEQ_LEN, "bg_mode": "subtract_concat", "in_dim": IN_CHANNELS, "out_dim": SEQ_LEN,
            "input_wh": [MODEL_W, MODEL_H], "thr": float(VARIANTS[variant]["thr"]),
        },
        "source": {
            "repo": UPSTREAM_URL, "commit": TRACKNETV5_SHA, "vendor_commit": vendor_sha,
            "file": src_name, "sha256": src_sha256,
            "sha256_expected": VARIANTS[variant]["upstream_sha256"],
            "val_f1": float(val_f1) if val_f1 is not None else None,
            "upstream_args": args,
        },
    }


class TrackNetV5Tracker:
    model_size = (MODEL_W, MODEL_H)  # (W, H)
    window = SEQ_LEN                 # chunk boundaries must align to the 5-frame windows

    def __init__(self, repo: str = REPO, model_path: str = MODEL, variant: str = VARIANT,
                 thr: Optional[float] = None, diff_mode: str = DIFF_MODE):
        import torch

        if variant not in VARIANTS:
            raise ValueError(f"tracknetv5: unknown variant {variant!r} (expected one of {sorted(VARIANTS)})")
        if not Path(model_path).exists():
            raise RuntimeError(f"tracknetv5: weights not found at {model_path} — run "
                               f"`python -m worker convert-tracknetv5 --variant {variant}` (module docstring)")
        sd, config, _ = load_checkpoint(model_path)
        if config.get("variant") and config["variant"] != variant:
            log.warning("tracknetv5: checkpoint says variant=%s but TRACKNETV5_VARIANT=%s — using the "
                        "checkpoint's alpha", config["variant"], variant)
        alpha = float(config.get("alpha", VARIANTS[variant]["alpha"]))
        self.variant = variant
        self.diff_mode = diff_mode
        self.conf_thr = float(thr) if thr is not None else resolve_thr(variant, THR_ENV, config)
        self.name = tracker_name(variant, model_path)
        self._median = None

        TrackNetV5 = load_vendored_model_class(repo)
        model = TrackNetV5(in_dim=IN_CHANNELS, out_dim=SEQ_LEN, alpha=alpha)
        model.load_state_dict(sd, strict=True)
        model.eval()  # train mode returns (out, aux) tuples — eval mode the sigmoid heatmaps only

        def run_once(device: str):
            probe = torch.zeros((1, IN_CHANNELS, MODEL_H, MODEL_W), dtype=torch.float32)
            with torch.no_grad():
                return model.to(device)(probe.to(device)).float().cpu().numpy()

        self.device = probe_device(run_once, "tracknetv5")  # MPS NaN/failure probe → CPU (fp32)
        self.model = model.to(self.device)
        log.info("tracknetv5: %s alpha=%.1f thr=%.2f diff=%s on %s (vendor %s)", self.name, alpha,
                 self.conf_thr, diff_mode, self.device, (vendor_commit(repo) or "?")[:8])

    def begin_clip(self, loader) -> None:
        """One median per VIDEO (upstream inference: `Video_IterableDataset`; training used per-rally
        medians — a later experiment), shared by every chunked predict() call."""
        self._median = clip_median(loader, self.model_size, MEDIAN_SAMPLES)

    def predict(self, crops: Sequence[Crop]) -> List[ShuttleDetection]:
        import numpy as np
        import torch

        frame_ids = [idx for idx, _ in crops]
        rgb = np.stack([resize_rgb(img, self.model_size) for _, img in crops]).astype("uint8")
        median = self._median if self._median is not None else median_of_frames(rgb, MEDIAN_SAMPLES)

        best: dict[int, ShuttleDetection] = {}
        wins = windows(len(rgb), SEQ_LEN)
        for b0 in range(0, len(wins), BATCH):
            batch = wins[b0:b0 + BATCH]
            x = np.stack([build_input(rgb[idxs], median, self.diff_mode) for idxs in batch])
            with torch.no_grad():
                y = self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()  # (B, 5, H, W)
            for bi, idxs in enumerate(batch):
                for f, pos in enumerate(idxs):
                    if pos in best:
                        continue  # padding repeats the last frame — keep the first prediction
                    hit = decode_heatmap(y[bi, f], self.conf_thr)
                    best[pos] = (ShuttleDetection(frame_ids[pos], *hit) if hit
                                 else ShuttleDetection(frame_ids[pos], 0.0, 0.0, 0.0))
        return [best[i] for i in sorted(best)]


# ── convert-tracknetv5: upstream release → plain weights_only-loadable checkpoint (+ sidecar, probe) ───

def _sha256(path) -> str:
    return checkpoint_tag(path, 64)


def parity_check(model_factory, seed: int = 0) -> dict:
    """CPU-vs-MPS check on one seeded random input: finite outputs on each device and the max |Δ| between
    them (the CLAUDE.md 'validate new models once against CPU for silent NaNs' rule)."""
    import numpy as np
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.rand((1, IN_CHANNELS, MODEL_H, MODEL_W), generator=g)
    out: dict = {"devices": {}}
    ref = None
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            out["devices"]["mps"] = "unavailable"
            continue
        try:
            model = model_factory().to(device).eval()
            with torch.no_grad():
                y = model(x.to(device)).float().cpu().numpy()
            finite = bool(np.isfinite(y).all())
            out["devices"][device] = {"shape": list(y.shape), "finite": finite,
                                      "mean": float(np.nanmean(y)), "max": float(np.nanmax(y))}
            if device == "cpu":
                ref = y
            elif ref is not None:
                out["max_abs_diff_cpu_mps"] = float(np.nanmax(np.abs(y - ref)))
        except Exception as e:  # noqa: BLE001
            out["devices"][device] = f"failed: {e}"
    return out


def convert(src: Path, out: Path, variant: str, repo: str, probe: bool = False) -> dict:
    import json

    import torch

    sd, _config, raw = load_checkpoint(src)
    src_sha = _sha256(src)
    expected = VARIANTS[variant]["upstream_sha256"]
    if src_sha != expected:
        log.warning("tracknetv5: %s has SHA-256 %s, the spec records %s for the %s release — converting "
                    "anyway (a fine-tune or a newer upstream file?)", src.name, src_sha, expected, variant)
    val_f1 = raw.get("val_f1") if isinstance(raw, dict) else None
    payload = conversion_payload(sd, raw.get("args") if isinstance(raw, dict) else None, variant, src.name,
                                 src_sha, vendor_commit(repo), float(val_f1) if val_f1 is not None else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out))
    sidecar = {
        "file": out.name, "sha256": _sha256(out), "torch": torch.__version__,
        "config": payload["config"], "source": payload["source"],
    }
    if probe:
        TrackNetV5 = load_vendored_model_class(repo)

        def factory():
            m = TrackNetV5(in_dim=IN_CHANNELS, out_dim=SEQ_LEN, alpha=payload["config"]["alpha"])
            m.load_state_dict(sd, strict=True)
            return m

        sidecar["probe"] = parity_check(factory)
    # Re-load what we wrote exactly the way the backend will (strict weights_only, no allow-list needed).
    check = torch.load(str(out), map_location="cpu", weights_only=True)
    assert set(check) == {"model_state_dict", "config", "source"}, "converted checkpoint shape drifted"
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def convert_cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="worker convert-tracknetv5",
                                 description="Re-save a TrackNetV5 release/fine-tune as a plain weights_only "
                                             "checkpoint + JSON sidecar (SHA-256, provenance, optional probe).")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="lite")
    ap.add_argument("--repo", default=REPO, help="vendored TrackNetV5 checkout (TRACKNETV5_REPO)")
    ap.add_argument("--src", help="checkpoint to convert (default: <repo>/releases/<variant release>)")
    ap.add_argument("--out", help="output .pt (default: models/weights/shuttle/<variant weights>)")
    ap.add_argument("--probe", action="store_true", help="also run the CPU-vs-MPS NaN/parity check")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    src = Path(args.src) if args.src else Path(args.repo) / "releases" / VARIANTS[args.variant]["upstream"]
    out = Path(args.out) if args.out else Path(default_model_path(args.variant))
    if not src.exists():
        raise SystemExit(f"source checkpoint not found: {src}")
    sidecar = convert(src, out, args.variant, args.repo, probe=args.probe)
    print(json.dumps(sidecar, indent=2))
    print(f"wrote {out} (+ {out.with_suffix('.json').name}); tracker name will be "
          f"{tracker_name(args.variant, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(convert_cli())
