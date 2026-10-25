"""VideoJev-4B zero-shot stroke pilot — the deterministic half (specs/features/stroke-vlm-pilot.md).

Why this exists: the stage-7 BST4 classifier failed its 0.70 gate at 0.400 LOGO-CV with a
train/held-out gap of 0.945/0.402 — it memorizes 385 rows and does not transfer across games. A
model with a pretrained *video* representation might; this pilot measures that zero-shot on the
695 labeled 1.4 s clips before anyone spends a fine-tune on it. VideoJev (MCG-NJU, Apache-2.0)
scores text options against a clip and returns a softmax over them — the same shape FR-22's
τ machinery already consumes. It is NOT TypeSafe's Jev (text-only, hosted; see the spec's
"Alternatives evaluated" note in stroke-classification.md).

Honesty rules, enforced here rather than by discipline:
- presets (prompt wording, option wording/order, fps) are the ONLY knobs and are frozen in code;
- a variant may be chosen on ONE dev game (G5); the eval run reads its settings from that dev
  report (`frozen_settings`) and refuses override flags; dev accuracy never counts;
- `unknown`/Unsicher rows are never scored for accuracy — they are the confidence-honesty set;
- the model sees pixels only: no serve hint, rally index or label metadata reaches the prompt.

Torch/transformers never enter this module; the model side is vlm_model.py (lazy imports), so
the suite and the light worker install collect without the vlm extra.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .train import (CLASSES, GATE_MEAN_ACCURACY, HARD_PAIRS, _CLASS_INDEX, _none_if_nan,
                    accuracy, confusion, hard_pair_block, per_class_recall, sweep_tau,
                    unknown_rate_per_match, unsicher_abstention)

log = logging.getLogger("worker.stroke.vlm")

MODEL_ID = "ZhuYuhan/VideoJev-4B"
_ROOT = Path(__file__).resolve().parents[3]
STROKE_VLM_MODEL = os.environ.get("STROKE_VLM_MODEL",
                                  str(_ROOT / "models/weights/stroke_vlm/VideoJev-4B"))
VIDEOJEV_REVISION: str | None = None   # pin the HF snapshot SHA after the first download (runbook §0)
CLIP_DURATION_S = 1.4                  # stroke_clips DEFAULT_PRE_S + DEFAULT_POST_S
DEFAULT_CONTEXT = 32768                # ≥ 2× the ~8.2k visual tokens a 30 fps clip produces
FPS_GRID = (2.0, 10.0, 30.0)           # upstream default 2 fps ≈ 3 frames of a clip; 30 fps = every proxy frame
BASELINES = {"chance": 1 / len(CLASSES), "majority": 165 / 554,     # drop share of the 554 confident rows
             "bst_logo_mean": 0.400, "gate": GATE_MEAN_ACCURACY}

# decision tiers (spec §Decision rule) — pooled eval accuracy on confident rows
TIER_CLOSED_BELOW = 0.35               # indistinguishable from majority-class 0.30, below BST 0.40
TIER_FINETUNE_AT = 0.55                # clearly above BST with margin
TOP2_ASSIST_AT = 0.75                  # top-2 alone can justify labeling assist
FINETUNE_MIN_CLASSES_WITH_RECALL = 6   # a fine-tune candidate must not collapse onto a few classes


# --- presets: the only knobs that exist ------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    query: str
    options: tuple[str, ...]          # option texts, in the order the model sees them
    option_classes: tuple[str, ...]   # the CLASSES token behind each option position
    fps: float
    max_frames: int | None = None
    probe: bool = False               # honesty probe (dev only): recorded, never selectable


QUERY_PLAIN = "Which badminton stroke is hit at the center of this clip?"
QUERY_CONTEXT = ("Doubles badminton, elevated end-on camera. Exactly one stroke is hit in the "
                 "middle of the clip. Which stroke type is it?")

OPTION_TEXT = {
    "plain": {"clear": "clear", "smash": "smash", "drop": "drop", "net": "net shot",
              "drive": "drive", "lift": "lift", "serve": "serve"},
    "described": {
        "clear": "clear – a high overhead shot sent deep to the opponent's back court",
        "smash": "smash – a steep, powerful downward overhead hit",
        "drop": "drop – a soft overhead shot that falls just over the net",
        "net": "net shot – a gentle shot from near the net that barely crosses it",
        "drive": "drive – a flat, fast shot hit roughly parallel to the floor",
        "lift": "lift – an underhand shot lifted high to the opponent's back court",
        "serve": "serve – the stroke that starts the rally",
    },
}
_PROMPTS = {"plain": (QUERY_PLAIN, OPTION_TEXT["plain"]),
            "described": (QUERY_PLAIN, OPTION_TEXT["described"]),
            "context": (QUERY_CONTEXT, OPTION_TEXT["described"])}


def validate_preset(p: Preset) -> None:
    """7 options whose classes are a permutation of CLASSES; positive fps; sane cap."""
    if len(p.options) != len(CLASSES) or len(p.option_classes) != len(CLASSES):
        raise ValueError(f"preset {p.name}: needs exactly {len(CLASSES)} options")
    if sorted(p.option_classes) != sorted(CLASSES):
        raise ValueError(f"preset {p.name}: option_classes must be a permutation of {CLASSES}")
    if len(set(p.options)) != len(p.options):
        raise ValueError(f"preset {p.name}: duplicate option texts")
    if not (p.fps > 0):
        raise ValueError(f"preset {p.name}: fps must be positive")
    if p.max_frames is not None and (not isinstance(p.max_frames, int) or p.max_frames < 1):
        raise ValueError(f"preset {p.name}: max_frames must be a positive int or None")


def _preset(name: str, query: str, texts: dict, fps: float, *, reverse: bool = False,
            probe: bool = False) -> Preset:
    classes = tuple(reversed(CLASSES)) if reverse else tuple(CLASSES)
    return Preset(name=name, query=query, options=tuple(texts[c] for c in classes),
                  option_classes=classes, fps=float(fps), max_frames=None, probe=probe)


def _build_presets() -> dict[str, Preset]:
    table: dict[str, Preset] = {}
    for prompt, (query, texts) in _PROMPTS.items():
        for fps in FPS_GRID:
            name = f"{prompt}_fps{int(fps)}"
            table[name] = _preset(name, query, texts, fps)
    # position-bias probe: same words as plain_fps10, options reversed
    table["plain_rev_fps10"] = _preset("plain_rev_fps10", *_PROMPTS["plain"], 10.0,
                                       reverse=True, probe=True)
    for p in table.values():
        validate_preset(p)
    return table


PRESETS: dict[str, Preset] = _build_presets()


def build_question(p: Preset, qid: str = "stroke") -> dict:
    """One VideoJev multiple-choice question; options are raw text (no letters needed)."""
    return {"id": qid, "task": "choice", "query": p.query, "options": list(p.options)}


def probs_in_class_order(probs: Sequence[float], p: Preset):
    """Model probabilities (option order) -> numpy (7,) in CLASSES order; finite or raise."""
    import numpy as np

    arr = np.asarray(list(probs), dtype=np.float64)
    if arr.shape != (len(CLASSES),):
        raise ValueError(f"expected {len(CLASSES)} probabilities, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite probability from the model")
    out = np.zeros(len(CLASSES), dtype=np.float64)
    for value, cls in zip(arr, p.option_classes):
        out[_CLASS_INDEX[cls]] = value
    return out


def expected_frames(duration_s: float, fps: float, max_frames: int | None = None) -> int:
    """Frames upstream will sample: mirrors videojev_temporal.time_grid (ceil(duration*fps) on a
    k/fps grid) + videojev_video.sampling_grid's cap (drop grid points at/after the end, then cap)."""
    if duration_s <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    n = int(math.ceil(duration_s * fps))
    if max_frames is not None:
        if not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        n = sum(1 for k in range(n) if k / fps < float(duration_s))
        n = min(n, max_frames)
    return n


# --- labels <-> clips ---------------------------------------------------------------------------

def load_tasks(labeling_root: Path) -> dict[str, dict]:
    """Every tasks_<game>.json the clip exporter wrote -> {event_id: {clip, video_id, match_id,
    hit_t_s, game}}; `clip` is the path relative to the labeling root (after Label Studio's `?d=`)."""
    root = Path(labeling_root)
    tasks: dict[str, dict] = {}
    for path in sorted(root.glob("tasks_*.json")):
        for task in json.loads(path.read_text(encoding="utf-8")):
            data = task.get("data") or {}
            video = str(data.get("video", ""))
            rel = video.split("?d=", 1)[1] if "?d=" in video else video
            eid = data.get("event_id")
            if not eid or not rel or "hit_t_s" not in data:
                log.warning("vlm: skipping malformed task in %s: %s", path.name, data)
                continue
            tasks[str(eid)] = {"clip": rel, "video_id": str(data.get("video_id")),
                               "match_id": str(data.get("match_id")),
                               "hit_t_s": float(data["hit_t_s"]),
                               "game": str(data.get("game") or path.stem.removeprefix("tasks_"))}
    return tasks


def clip_index_from_files(labeling_root: Path) -> dict[str, list[str]]:
    """stroke/<game>/*.mp4 grouped by the 8-char event id the exporter puts last in the stem."""
    root = Path(labeling_root)
    index: dict[str, list[str]] = {}
    for path in sorted((root / "stroke").glob("*/*.mp4")):
        event8 = path.stem.rsplit("_", 1)[-1]
        index.setdefault(event8, []).append(path.relative_to(root).as_posix())
    return index


def join_labels_to_clips(labels: Sequence[dict], tasks: dict, clip_index: dict, *,
                         tol_s: float = 0.05) -> tuple[list[dict], list[dict]]:
    """Label rows -> rows with `clip_path` + `join_method`, plus the rows that found no clip.

    event_id is exact when labels and clips come from the same export. After a stage-6 re-run
    the ids regenerate (replace_rallies is delete-then-insert), so two fallbacks exist: the same
    video's task within ±tol_s (`t_s`), then a unique clip filename carrying event_id[:8]
    (`filename`). Anything ambiguous is reported missing, never guessed."""
    by_video: dict[str, list[dict]] = {}
    for t in tasks.values():
        by_video.setdefault(str(t["video_id"]), []).append(t)
    rows, missing = [], []
    for lab in labels:
        shot = lab["shot_type"]
        if shot != "unknown" and shot not in _CLASS_INDEX:
            raise ValueError(f"shot_type {shot!r} is not a stroke label (classes {CLASSES} + unknown)")
        eid, vid, t_s = str(lab["event_id"]), str(lab["video_id"]), float(lab["t_s"])
        clip = method = None
        task = tasks.get(eid)
        if task is not None:
            clip, method = task["clip"], "event_id"
        else:
            near = [t for t in by_video.get(vid, []) if abs(t["hit_t_s"] - t_s) <= tol_s]
            cands = clip_index.get(eid[:8], [])
            if len(near) == 1:
                clip, method = near[0]["clip"], "t_s"
            elif len(cands) == 1:
                clip, method = cands[0], "filename"
            else:
                reason = ("ambiguous" if len(near) > 1 or len(cands) > 1
                          else "no task or clip matched")
                missing.append({"event_id": eid, "video_id": vid, "t_s": t_s, "reason": reason})
                continue
        rows.append({**lab, "event_id": eid, "video_id": vid, "t_s": t_s,
                     "is_calibration": shot == "unknown", "clip_path": clip, "join_method": method})
    return rows, missing


def split_rows(rows: Sequence[dict], dev_video_id: str, split: str) -> list[dict]:
    """dev = the one development game, eval = every other game. Nothing else exists."""
    if split not in ("dev", "eval"):
        raise ValueError(f"split must be 'dev' or 'eval', got {split!r}")
    vids = {str(r["video_id"]) for r in rows}
    if str(dev_video_id) not in vids:
        raise ValueError(f"dev video_id {dev_video_id} is not among the labeled games {sorted(vids)}")
    want_dev = split == "dev"
    return [r for r in rows if (str(r["video_id"]) == str(dev_video_id)) == want_dev]


# --- protocol ----------------------------------------------------------------------------------

def select_preset(dev_accuracies: dict[str, float | None]) -> str:
    """Argmax dev accuracy over selectable presets; ties -> table order; probes never win."""
    unknown = set(dev_accuracies) - set(PRESETS)
    if unknown:
        raise ValueError(f"unknown presets in dev results: {sorted(unknown)}")
    order = list(PRESETS)
    cands = [n for n in order if n in dev_accuracies and dev_accuracies[n] is not None
             and not PRESETS[n].probe]
    if not cands:
        raise ValueError("no selectable preset with a dev accuracy")
    return max(cands, key=lambda n: (dev_accuracies[n], -order.index(n)))


def frozen_settings(dev_report: dict, dev_video_id: str) -> dict:
    """The eval run's settings, read from a dev report — the only place they may come from."""
    if dev_report.get("split") != "dev":
        raise ValueError("--frozen-from must point at a dev run (report.split == 'dev')")
    if str(dev_report.get("dev_video_id")) != str(dev_video_id):
        raise ValueError(f"dev report was selected on game {dev_report.get('dev_video_id')}, "
                         f"not {dev_video_id}")
    sel = dev_report.get("selected") or {}
    name = sel.get("preset")
    if name not in PRESETS or PRESETS[name].probe:
        raise ValueError(f"dev report selected preset {name!r}, which is not a selectable preset")
    p = PRESETS[name]
    if sel.get("fps") is None or float(sel["fps"]) != p.fps or sel.get("max_frames") != p.max_frames:
        raise ValueError(f"dev report fps/max_frames disagree with preset {name}")
    for key in ("dtype", "context", "attn"):
        if key not in sel:
            raise ValueError(f"dev report 'selected' lacks {key}")
    return {"preset": name, "fps": p.fps, "max_frames": p.max_frames, "dtype": str(sel["dtype"]),
            "context": int(sel["context"]), "attn": str(sel["attn"])}


def decision_tier(pooled_accuracy, per_class_recall_by_class: dict, pmax_median: dict,
                  top2_accuracy) -> str | None:
    """Spec §Decision rule. None recall (no support) never counts toward recall breadth."""
    if pooled_accuracy is None:
        return None
    if pooled_accuracy >= GATE_MEAN_ACCURACY:
        return "gate"
    n_recall = sum(1 for v in per_class_recall_by_class.values() if v is not None and v > 0)
    unsicher, confident = pmax_median.get("unsicher"), pmax_median.get("confident")
    honest = unsicher is not None and confident is not None and unsicher < confident
    if (pooled_accuracy >= TIER_FINETUNE_AT and n_recall >= FINETUNE_MIN_CLASSES_WITH_RECALL
            and honest):
        return "finetune_candidate"
    if pooled_accuracy >= TIER_CLOSED_BELOW or (top2_accuracy is not None
                                                 and top2_accuracy >= TOP2_ASSIST_AT):
        return "labeling_assist"
    return "closed"


# --- rows and reports --------------------------------------------------------------------------

def prediction_row(base: dict, p: Preset, probs_option_order: Sequence[float], *,
                   n_frames_expected: int, wall_s: float) -> dict:
    """One scored (clip, preset): class-order probs, argmax prediction, confidence."""
    probs = probs_in_class_order(probs_option_order, p)
    i = int(probs.argmax())
    return {**base, "preset": p.name, "n_frames_expected": int(n_frames_expected),
            "probs": [float(x) for x in probs], "pred": CLASSES[i], "p_max": float(probs[i]),
            "wall_s": float(wall_s)}


def aggregate_report(rows: Sequence[dict], *, split: str) -> dict:
    """run_logo_cv-shaped report over one preset's rows of one split. Accuracy uses confident
    rows only; Unsicher rows feed abstention/confidence honesty; per-game spread is reported,
    never averaged away."""
    import numpy as np

    conf = [r for r in rows if not r["is_calibration"]]
    unk = [r for r in rows if r["is_calibration"]]
    rep: dict = {"split": split, "n": len(rows), "n_confident": len(conf), "n_unknown": len(unk)}
    unk_pmax = np.array([r["p_max"] for r in unk], dtype=np.float64)
    if not conf:
        rep.update(per_game=[], mean_accuracy=None, std_accuracy=None, pooled_accuracy=None,
                   top2_accuracy=None, per_class_recall={c: None for c in CLASSES},
                   confusion=None, hard_pairs=None, tau=None, unsicher_abstention=None,
                   unknown_rate_per_match=None,
                   pmax_median={"confident": None,
                                "unsicher": float(np.median(unk_pmax)) if unk else None},
                   pred_share=None, pred_share_max=None, gate_passed=False, decision_tier=None)
        return rep

    y = np.array([_CLASS_INDEX[r["shot_type"]] for r in conf], dtype=np.int64)
    probs = np.array([r["probs"] for r in conf], dtype=np.float64)
    pred = probs.argmax(axis=1)
    p_max = probs.max(axis=1)
    vids = np.array([str(r["video_id"]) for r in conf])
    per_game = []
    for v in sorted(set(vids.tolist())):
        m = vids == v
        per_game.append({"video_id": v, "n": int(m.sum()), "accuracy": accuracy(y[m], pred[m]),
                         "per_class_recall": [_none_if_nan(r) for r in
                                              per_class_recall(y[m], pred[m])]})
    accs = [g["accuracy"] for g in per_game]
    top2_hits = [int(y[i] in np.argsort(-probs[i])[:2]) for i in range(len(y))]
    cm = confusion(y, pred)
    tau = sweep_tau(p_max, pred == y)
    pcr = {c: _none_if_nan(r) for c, r in zip(CLASSES, per_class_recall(y, pred))}
    pred_share = {c: float(np.mean(pred == i)) for i, c in enumerate(CLASSES)}
    pmax_median = {"confident": float(np.median(p_max)),
                   "unsicher": float(np.median(unk_pmax)) if unk else None}
    pooled = accuracy(y, pred)
    top2 = float(np.mean(top2_hits))
    all_pmax = np.array([r["p_max"] for r in rows], dtype=np.float64)
    all_vids = [str(r["video_id"]) for r in rows]
    rep.update(
        per_game=per_game,
        mean_accuracy=float(np.mean(accs)), std_accuracy=float(np.std(accs)),
        pooled_accuracy=pooled, top2_accuracy=top2,
        per_class_recall=pcr, confusion=cm.tolist(),
        hard_pairs={f"{a}/{b}": hard_pair_block(cm, a, b) for a, b in HARD_PAIRS},
        tau=tau,
        unsicher_abstention=unsicher_abstention(unk_pmax, tau["tau"]) if unk else None,
        unknown_rate_per_match=unknown_rate_per_match(all_pmax, all_vids, tau["tau"]),
        pmax_median=pmax_median, pred_share=pred_share, pred_share_max=max(pred_share.values()),
        gate_passed=bool(pooled >= GATE_MEAN_ACCURACY),
        decision_tier=decision_tier(pooled, pcr, pmax_median, top2),
    )
    return rep


_PRED_COLUMNS = ("event_id", "video_id", "match_id", "t_s", "shot_type", "is_calibration", "split",
                 "preset", "clip_path", "join_method", "n_frames_expected", "pred", "p_max")


def predictions_frame(rows: Sequence[dict]):
    """Flat polars frame: one row per (event, preset); probs fan out to prob_<class> columns."""
    import polars as pl

    schema = {"event_id": pl.Utf8, "video_id": pl.Utf8, "match_id": pl.Utf8, "t_s": pl.Float64,
              "shot_type": pl.Utf8, "is_calibration": pl.Boolean, "split": pl.Utf8,
              "preset": pl.Utf8, "clip_path": pl.Utf8, "join_method": pl.Utf8,
              "n_frames_expected": pl.Int64, "pred": pl.Utf8, "p_max": pl.Float64,
              **{f"prob_{c}": pl.Float64 for c in CLASSES}, "wall_s": pl.Float64}
    data: dict[str, list] = {k: [] for k in schema}
    for r in rows:
        for k in _PRED_COLUMNS:
            v = r.get(k)
            if k == "t_s" and v is not None:
                v = float(v)
            elif k == "n_frames_expected" and v is not None:
                v = int(v)
            elif k in ("event_id", "video_id", "match_id") and v is not None:
                v = str(v)
            data[k].append(v)
        for c in CLASSES:
            data[f"prob_{c}"].append(float(r["probs"][_CLASS_INDEX[c]]))
        data["wall_s"].append(float(r["wall_s"]) if r.get("wall_s") is not None else None)
    return pl.DataFrame(data, schema=schema, strict=False)


# --- run planning + probe comparison (pure; the CLI turns ValueError into argparse errors) --------

def plan_run(split: str, *, preset: str | None, frozen_report: dict | None, dev_video_id: str,
             dtype: str | None, attn: str | None, context: int | None) -> dict:
    """What a run may do, from its flags. dev: any preset(s), defaults fp32/sdpa/DEFAULT_CONTEXT.
    eval: every setting comes from the dev report; override flags are refused outright."""
    if split == "eval":
        if frozen_report is None:
            raise ValueError("--split eval requires --frozen-from <dev run dir>")
        given = [flag for flag, value in (("--preset", preset), ("--dtype", dtype),
                                          ("--attn", attn), ("--context", context))
                 if value is not None]
        if given:
            raise ValueError("--split eval takes its settings from --frozen-from; not allowed: "
                             + ", ".join(given))
        s = frozen_settings(frozen_report, dev_video_id)
        return {"split": "eval", "presets": [s["preset"]], "dtype": s["dtype"], "attn": s["attn"],
                "context": s["context"], "frozen": True}
    if split == "dev":
        if frozen_report is not None:
            raise ValueError("--frozen-from is only for --split eval")
        name = preset or "all"
        if name == "all":
            names = list(PRESETS)
        elif name in PRESETS:
            names = [name]
        else:
            raise ValueError(f"unknown preset {name!r}; known: {list(PRESETS)}")
        return {"split": "dev", "presets": names, "dtype": dtype or "fp32", "attn": attn or "sdpa",
                "context": int(context or DEFAULT_CONTEXT), "frozen": False}
    raise ValueError(f"split must be 'dev' or 'eval', got {split!r}")


def probe_summary(reference: Sequence[Sequence[float]], device: Sequence[Sequence[float]], *,
                  tol: float = 0.05) -> dict:
    """CPU-fp32 reference vs device probabilities on the same clips: passes only when every
    value is finite, the largest probability difference is ≤ tol and every argmax agrees."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    dev = np.asarray(device, dtype=np.float64)
    if ref.ndim != 2 or ref.shape != dev.shape or ref.shape[0] == 0:
        raise ValueError(f"probe arrays must be equal non-empty (n, classes); got {ref.shape} vs {dev.shape}")
    nonfinite = int((~np.isfinite(ref)).sum() + (~np.isfinite(dev)).sum())
    diff = float(np.max(np.abs(ref - dev))) if nonfinite == 0 else None
    agree = int(np.sum(ref.argmax(axis=1) == dev.argmax(axis=1)))
    passed = nonfinite == 0 and diff <= tol and agree == ref.shape[0]
    return {"n": int(ref.shape[0]), "max_abs_prob_diff": diff, "argmax_agreement": agree,
            "nonfinite": nonfinite, "tol": float(tol), "passed": bool(passed)}


# --- CLI shell (I/O + model calls; the pure pieces above are what the tests pin) --------------------

def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


CAVEATS = [
    "one session / one lighting / all doubles: this measures cross-game, not cross-session generalization",
    "zero-shot: the model never saw a label; the preset was chosen on the dev game only and the dev number never counts",
    "the model sees pixels only — the human labelers also used the audio (smash crack vs net tap)",
    "MPS is not bit-deterministic; the CPU-fp32 probe is the reproducibility reference",
    "seconds per clip is far from pipeline budgets — runtime feasibility is not what this pilot measures",
]


def eval_cli(argv: Sequence[str] | None = None) -> int:
    """`worker eval-stroke-vlm` — VideoJev zero-shot over the labeled stroke clips, dev or eval
    split, with the FR-22 τ sweep and the spec's decision tier. Every number ships with the
    command, the model revision and the preset that produced it."""
    import argparse
    import hashlib
    import shlex
    import time
    from datetime import datetime, timezone

    import polars as pl

    from ..config import load_config
    from ..stroke_clips import sample_evenly

    ap = argparse.ArgumentParser(
        prog="worker eval-stroke-vlm",
        description="Score the labeled 1.4 s stroke clips zero-shot with VideoJev-4B "
                    "(specs/features/stroke-vlm-pilot.md). dev = choose the preset on one game; "
                    "eval = the number of record on the other games, settings frozen from the dev run.")
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--dev-video-id", required=True, help="match_videos.id of the development game (G5)")
    ap.add_argument("--preset", help="dev only: a preset name or 'all' (default all)")
    ap.add_argument("--frozen-from", help="eval only (required): dev run dir whose report.json fixes "
                                          "preset/fps/dtype/context/attn")
    ap.add_argument("--labels", help="stroke_labels.parquet (default: <labeling>/stroke_labels.parquet)")
    ap.add_argument("--labeling-root", help="dir with stroke/<game>/*.mp4 + tasks_<game>.json "
                                            "(default: .labeling next to the frame cache)")
    ap.add_argument("--model", default=STROKE_VLM_MODEL,
                    help="local snapshot dir or HF id (default: $STROKE_VLM_MODEL)")
    ap.add_argument("--revision", default=VIDEOJEV_REVISION, help="HF snapshot SHA to pin")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], help="dev only (default fp32; bf16 = memory fallback)")
    ap.add_argument("--attn", choices=["sdpa", "eager"], help="dev only (default sdpa)")
    ap.add_argument("--context", type=int, help=f"dev only (default {DEFAULT_CONTEXT})")
    ap.add_argument("--probe", type=int, default=5,
                    help="clips for the CPU-fp32-vs-device agreement probe (0 skips, loudly)")
    ap.add_argument("--allow-probe-fail", action="store_true")
    ap.add_argument("--allow-missing-clips", action="store_true")
    ap.add_argument("--limit", type=int, help="score only the first N clips (plumbing checks)")
    ap.add_argument("--resume", help="existing run dir: continue its predictions.jsonl")
    ap.add_argument("--out", help="run dir (default: <labeling>/stroke_runs/<UTC>_videojev_zeroshot_<split>)")
    args = ap.parse_args(argv)

    frozen_report = None
    if args.frozen_from:
        report_path = Path(args.frozen_from).expanduser() / "report.json"
        if not report_path.exists():
            ap.error(f"no report.json in {args.frozen_from}")
        frozen_report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        plan = plan_run(args.split, preset=args.preset, frozen_report=frozen_report,
                        dev_video_id=args.dev_video_id, dtype=args.dtype, attn=args.attn,
                        context=args.context)
    except ValueError as e:
        ap.error(str(e))
    presets = [PRESETS[n] for n in plan["presets"]]
    if not Path(args.model).expanduser().exists() and not args.revision:
        ap.error(f"--model {args.model} is not a local snapshot dir, so it is a Hugging Face id: "
                 "pass --revision <SHA> to pin the snapshot (remote code runs from it)")

    cfg = load_config()
    labeling = (Path(args.labeling_root).expanduser() if args.labeling_root
                else Path(cfg.frame_cache_dir).expanduser().parent / ".labeling")
    labels_path = Path(args.labels).expanduser() if args.labels else labeling / "stroke_labels.parquet"
    if not labels_path.exists():
        print(f"no labels at {labels_path} — run import-stroke-labels first")
        return 1
    labels = list(pl.read_parquet(labels_path).iter_rows(named=True))
    rows, missing = join_labels_to_clips(labels, load_tasks(labeling), clip_index_from_files(labeling))
    present = []
    for r in rows:
        if (labeling / r["clip_path"]).exists():
            present.append(r)
        else:
            missing.append({"event_id": r["event_id"], "video_id": r["video_id"], "t_s": r["t_s"],
                            "reason": f"clip file missing: {r['clip_path']}"})
    try:
        rows = split_rows(present, args.dev_video_id, args.split)
    except ValueError as e:
        ap.error(str(e))
    rows.sort(key=lambda r: (str(r["video_id"]), float(r["t_s"])))
    if args.limit:
        rows = rows[:args.limit]
    is_dev = args.split == "dev"
    split_missing = [m for m in missing if (str(m["video_id"]) == str(args.dev_video_id)) == is_dev]
    if split_missing and not args.allow_missing_clips:
        print(f"{len(split_missing)} labeled hits in the {args.split} split have no clip "
              f"(first: {split_missing[0]}) — re-run export-stroke-clips, or pass "
              f"--allow-missing-clips to score the rest")
        return 1
    if not rows:
        print("no rows to score in this split")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (Path(args.resume).expanduser() if args.resume
               else Path(args.out).expanduser() if args.out
               else labeling / "stroke_runs" / f"{stamp}_videojev_zeroshot_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "predictions.jsonl"
    scored = _read_jsonl(jsonl) if args.resume else []
    done = {(r["event_id"], r["preset"]) for r in scored}
    print(f"{len(rows)} clips x {len(presets)} presets in split={args.split} "
          f"({len(done)} already scored) -> {out_dir}")

    from . import vlm_model as vm

    device = vm.resolve_device(args.device)
    probe_preset = next((p for p in presets if not p.probe), presets[0])
    t_load = time.perf_counter()
    if args.probe > 0:
        probe_clips = [labeling / r["clip_path"] for r in sample_evenly(rows, args.probe)]
        probe, (model, processor, prov) = vm.run_probe(
            args.model, probe_clips, probe_preset, device=device, dtype=plan["dtype"],
            attn=plan["attn"], revision=args.revision, context=plan["context"])
        print(f"probe cpu/fp32 vs {device}/{plan['dtype']} on {probe['n']} clips: "
              f"max|dp|={probe['max_abs_prob_diff']}, argmax agreement "
              f"{probe['argmax_agreement']}/{probe['n']}, nonfinite {probe['nonfinite']} -> "
              f"{'PASSED' if probe['passed'] else 'FAILED'}")
        if not probe["passed"] and not args.allow_probe_fail:
            (out_dir / "probe_failed.json").write_text(json.dumps(probe, indent=2))
            print("probe FAILED — aborting (--allow-probe-fail records the run anyway, flagged)")
            return 1
    else:
        model, processor, prov = vm.load_videojev(args.model, device=device, dtype=plan["dtype"],
                                                  attn=plan["attn"], revision=args.revision)
        probe = None
        log.warning("vlm: --probe 0 — no cpu-vs-%s agreement check; the report says so", device)
    load_s = time.perf_counter() - t_load

    by_fps: dict[float, list[Preset]] = {}
    for p in presets:
        by_fps.setdefault(p.fps, []).append(p)
    t_run = time.perf_counter()
    n_clips = 0
    with jsonl.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            todo = {fps: [p for p in ps if (r["event_id"], p.name) not in done]
                    for fps, ps in by_fps.items()}
            if not any(todo.values()):
                continue
            base = {k: r.get(k) for k in ("event_id", "match_id", "video_id", "t_s", "shot_type",
                                          "is_calibration", "clip_path", "join_method")}
            base["split"] = args.split
            for fps, ps in todo.items():
                if not ps:
                    continue
                for p, probs, n_frames, wall in vm.score_clip(
                        model, processor, labeling / r["clip_path"], ps, fps=fps,
                        max_frames=ps[0].max_frames, context=plan["context"]):
                    row = prediction_row(base, p, probs, n_frames_expected=n_frames, wall_s=wall)
                    scored.append(row)
                    done.add((row["event_id"], p.name))
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
            n_clips += 1
            if n_clips % 10 == 0 or i + 1 == len(rows):
                elapsed = time.perf_counter() - t_run
                per = elapsed / n_clips
                print(f"  {i + 1}/{len(rows)} clips, {per:.1f} s/clip, "
                      f"ETA {(len(rows) - i - 1) * per / 60:.0f} min", flush=True)
    total_s = time.perf_counter() - t_run

    per_preset = {p.name: aggregate_report([x for x in scored if x["preset"] == p.name],
                                           split=args.split) for p in presets}
    walls = sorted(float(x["wall_s"]) for x in scored) or [0.0]
    n_conf = sum(1 for r in rows if not r["is_calibration"])
    report: dict = {
        "split": args.split, "dev_video_id": args.dev_video_id, "frozen_from": args.frozen_from,
        "model": prov, "presets_run": [p.name for p in presets], "per_preset": per_preset,
        "baselines": BASELINES,
        "labels": {"path": str(labels_path),
                   "sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                   "n_rows": len(labels), "n_joined": len(present), "n_split": len(rows),
                   "n_confident": n_conf, "n_unknown": len(rows) - n_conf,
                   "n_missing_clips": len(split_missing), "missing": split_missing[:20]},
        "probe": probe,
        "timing": {"clips": len(rows), "passes": len(scored), "load_s": load_s, "total_s": total_s,
                   "wall_s_median": walls[len(walls) // 2], "wall_s_mean": sum(walls) / len(walls)},
        "command": shlex.join(["python", "-m", "worker", "eval-stroke-vlm", *(argv or [])]),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": _git_sha(),
        "caveats": list(CAVEATS) + ([] if probe is not None else
                                    ["NO cpu-vs-device probe was run (--probe 0)"]),
    }
    if is_dev:
        dev_acc = {n: rep["pooled_accuracy"] for n, rep in per_preset.items()}
        try:
            best = select_preset(dev_acc)
        except ValueError:
            best = None
        report["selected"] = None if best is None else {
            "preset": best, "fps": PRESETS[best].fps, "max_frames": PRESETS[best].max_frames,
            "dtype": plan["dtype"], "context": plan["context"], "attn": plan["attn"],
            "dev_accuracy": dev_acc[best]}
        if {"plain_fps10", "plain_rev_fps10"} <= set(per_preset):
            a, b = per_preset["plain_fps10"], per_preset["plain_rev_fps10"]
            report["position_bias"] = {
                "plain_fps10": a["pooled_accuracy"], "plain_rev_fps10": b["pooled_accuracy"],
                "accuracy_delta": (None if a["pooled_accuracy"] is None or b["pooled_accuracy"] is None
                                   else a["pooled_accuracy"] - b["pooled_accuracy"]),
                "pred_share_plain": a["pred_share"], "pred_share_reversed": b["pred_share"]}
    else:
        p = presets[0]
        rep = per_preset[p.name]
        report["settings"] = {"preset": p.name, "query": p.query, "options": list(p.options),
                              "option_classes": list(p.option_classes), "fps": p.fps,
                              "max_frames": p.max_frames, "context": plan["context"],
                              "dtype": plan["dtype"], "attn": plan["attn"]}
        report["headline"] = {k: rep[k] for k in ("pooled_accuracy", "mean_accuracy", "std_accuracy",
                                                   "top2_accuracy", "decision_tier", "gate_passed")}
    predictions_frame(scored).write_parquet(out_dir / "predictions.parquet")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    for name, rep in per_preset.items():
        if rep["pooled_accuracy"] is None:
            print(f"{name}: n_confident=0 (nothing to score)")
            continue
        print(f"{name}: n={rep['n_confident']} pooled acc {rep['pooled_accuracy']:.3f} "
              f"(games {rep['mean_accuracy']:.3f} ± {rep['std_accuracy']:.3f}), top-2 "
              f"{rep['top2_accuracy']:.3f}, tau* {rep['tau']['tau']:.3f}"
              f"{' [knee]' if rep['tau']['knee_fallback'] else ''}, tier {rep['decision_tier']}")
    if is_dev:
        sel = report["selected"]
        print("selected for eval: " + (f"{sel['preset']} (dev acc {sel['dev_accuracy']:.3f}) — dev "
                                       f"accuracy never counts; run --split eval --frozen-from {out_dir}"
                                       if sel else "none"))
    else:
        h = report["headline"]
        print(f"EVAL ({len(rows)} clips, {n_conf} confident): pooled {h['pooled_accuracy']:.3f}, "
              f"tier {h['decision_tier']}, gate {'PASSED' if h['gate_passed'] else 'FAILED'} "
              f"(baselines: majority {BASELINES['majority']:.2f}, BST {BASELINES['bst_logo_mean']:.2f}, "
              f"gate {BASELINES['gate']:.2f})")
    print(f"-> {out_dir}")
    return 0
