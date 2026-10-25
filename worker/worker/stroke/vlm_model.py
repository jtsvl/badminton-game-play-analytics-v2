"""VideoJev model side of the stroke pilot — the only module that touches torch/transformers,
and only inside functions, so the suite and the light worker install never load them.

Upstream API (MCG-NJU/VideoJev, read 2026-10-18; the inference code ships INSIDE the HF snapshot
as remote code, so pinning the revision is the vendoring): `AutoModel/AutoProcessor
(trust_remote_code=True)` → `model.prepare_for_inference()` (casts the scorer head + vision
tower to fp32, language weights keep their loaded dtype) → `prepare_video` + `encode_video`
(one visual encoding per clip) → `predict_features(processor, features, record, questions)`
returning `{id, task, probabilities, answer_index, answer}` with a softmax over the options.
Encoding once and scoring every preset against the same features is what makes the dev sweep
affordable. One question per pass: several questions in one call share a causal sequence.

Apple Silicon (CLAUDE.md): MPS primary, fp32 default, bf16 only as the recorded memory fallback;
no flash-attn on Mac → sdpa (eager retry). Upstream autocasts only under CUDA, so on MPS/CPU the
tensors run in their native dtype — which is why the CPU-fp32-vs-device probe is the gate.
"""
from __future__ import annotations

import gc
import logging
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .vlm_eval import Preset, build_question, probe_summary

log = logging.getLogger("worker.stroke.vlm")

UPSTREAM_MAX_FRAMES = 2048   # predict_video's default cap; irrelevant for 1.4 s clips, passed for parity


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def torch_dtype(name: str):
    import torch

    try:
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]
    except KeyError:
        raise ValueError(f"dtype must be fp32 or bf16, got {name!r}") from None


def load_videojev(path_or_id: str, *, device: str, dtype: str = "fp32", attn: str = "sdpa",
                  revision: str | None = None):
    """-> (model, processor, provenance). sdpa failing to load retries eager, loudly."""
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor

    kwargs: dict = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(path_or_id, **kwargs)
    try:
        model = AutoModel.from_pretrained(path_or_id, dtype=torch_dtype(dtype),
                                          attn_implementation=attn, **kwargs)
    except Exception as e:  # noqa: BLE001 — attention backend availability is environment-specific
        if attn != "sdpa":
            raise
        log.warning("vlm: loading with sdpa attention failed (%s) — retrying with eager", e)
        attn = "eager"
        model = AutoModel.from_pretrained(path_or_id, dtype=torch_dtype(dtype),
                                          attn_implementation="eager", **kwargs)
    model = model.to(device)
    model.prepare_for_inference()
    cfg = model.config
    provenance = {
        "id": str(path_or_id), "revision": revision,
        "model_type": getattr(cfg, "model_type", None),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "attn": attn, "dtype": dtype, "device": device,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    return model, processor, provenance


def _remote_module(model):
    """The snapshot's modeling module (where prepare_video lives) — found via the model class."""
    return sys.modules[type(model).__module__]


def encode_clip(model, processor, clip_path: Path, *, fps: float, max_frames: int | None,
                context: int):
    """-> (features, record, wall_s). record['frames'] is the frame count upstream sampled."""
    import torch

    prepare_video = getattr(_remote_module(model), "prepare_video")
    t0 = time.perf_counter()
    pixels, record = prepare_video(processor, str(clip_path), fps=float(fps),
                                   max_frames=max_frames if max_frames is not None
                                   else UPSTREAM_MAX_FRAMES, context=int(context))
    features = model.encode_video(pixels, torch.tensor([record["grid_thw"]]))
    return features, record, time.perf_counter() - t0


def predict_from_features(model, processor, features, record, question: dict, *,
                          context: int) -> list[float]:
    """One choice question against pre-encoded features -> probabilities in option order."""
    results = model.predict_features(processor, features, record, [question], context=int(context))
    probs = [float(x) for x in results[0]["probabilities"]]
    if len(probs) != len(question["options"]):
        raise ValueError(f"model returned {len(probs)} probabilities for "
                         f"{len(question['options'])} options")
    if not all(math.isfinite(x) for x in probs):
        raise ValueError(f"non-finite probabilities from the model: {probs}")
    return probs


def score_clip(model, processor, clip_path: Path, presets: Sequence[Preset], *, fps: float,
               max_frames: int | None, context: int):
    """Encode the clip once at `fps`, score every preset against the same features.
    -> [(preset, probs_option_order, n_frames, wall_s)]; the encode time is shared out evenly."""
    features, record, wall_enc = encode_clip(model, processor, clip_path, fps=fps,
                                             max_frames=max_frames, context=context)
    share = wall_enc / max(1, len(presets))
    out = []
    for p in presets:
        t0 = time.perf_counter()
        probs = predict_from_features(model, processor, features, record, build_question(p),
                                      context=context)
        out.append((p, probs, int(record["frames"]), share + time.perf_counter() - t0))
    return out


def _probe_probs(model, processor, clips: Sequence[Path], preset: Preset, *, context: int):
    return [predict_from_features(model, processor, *encode_clip(model, processor, c, fps=preset.fps,
                                                                  max_frames=preset.max_frames,
                                                                  context=context)[:2],
                                  build_question(preset), context=context) for c in clips]


def run_probe(path_or_id: str, clips: Sequence[Path], preset: Preset, *, device: str, dtype: str,
              attn: str, revision: str | None, context: int, tol: float = 0.05):
    """CPU-fp32 reference vs the requested device/dtype on the same clips (CLAUDE.md: validate a
    new model once against CPU for silent NaNs). Returns (probe_summary, (model, processor,
    provenance)) where the model is the one to keep scoring with.

    fp32 on a device: the very same weights are moved, so only the backend differs. bf16: the
    model is reloaded in bf16 — a blanket .to(bfloat16) would undo prepare_for_inference's fp32
    casts of the head and vision tower — after the reference model is freed (64 GB box)."""
    ref_model, processor, ref_prov = load_videojev(path_or_id, device="cpu", dtype="fp32",
                                                   attn=attn, revision=revision)
    reference = _probe_probs(ref_model, processor, clips, preset, context=context)
    if device == "cpu" and dtype == "fp32":
        model, prov = ref_model, ref_prov
        device_probs = reference
    elif dtype == "fp32":
        model = ref_model.to(device)
        prov = dict(ref_prov, device=device)
        device_probs = _probe_probs(model, processor, clips, preset, context=context)
    else:
        del ref_model
        gc.collect()
        model, processor, prov = load_videojev(path_or_id, device=device, dtype=dtype, attn=attn,
                                               revision=revision)
        device_probs = _probe_probs(model, processor, clips, preset, context=context)
    summary = probe_summary(reference, device_probs, tol=tol)
    summary.update(device=device, dtype=dtype, reference="cpu/fp32", preset=preset.name,
                   clips=[Path(c).name for c in clips])
    return summary, (model, processor, prov)
