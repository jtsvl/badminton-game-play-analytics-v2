"""GBDT pre-annotations for Label Studio stroke-task exports (Session 3+).

Turns the labeling-assist artifact (gbdt.py --fit-full, spec carve-out 2026-07-12) into
per-task suggestions: above `min_score` the answer is PRE-SELECTED via a Label Studio
`predictions` entry (verify/correct beats choose-from-9); below it the model's guess appears
only as sortable display columns (`data.model_pred` / `model_score` / `rare_score`) — the
same bias stance as the serve hint. `rare_score` = max(prob_net, prob_drive) drives the
mine-rare workflow: sorting by it surfaces the starved classes at the measured ~5–10×
enrichment. All German label text derives from the import map's inverse
(import_stroke_labels.GERMAN_BY_CANONICAL) — one source of truth for both directions.

Kept separate from stroke_clips.py so the exporter stays import-light: xgboost/polars enter
only here, and only when --prelabel is actually used. Feature rows come from the SAME
builder as training (traj.build_video_rows) — train == inference by construction. Note the
cache parquets are the feature source: if a rally-review edit has not been rally_sync'ed
into the cache yet, affected hits degrade to all-NaN rows and get low-information
predictions — degraded, never wrong-by-crash. (specs/features/stroke-labeling.md)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Sequence

from ..import_stroke_labels import GERMAN_BY_CANONICAL
from .train import CLASSES

log = logging.getLogger("worker.stroke.prelabel")

# must match the labeling config (stroke-labeling.md): <Choices name= toName=>
LS_FROM_NAME = "shot_type"
LS_TO_NAME = "video"
RARE_CLASSES = ("net", "drive")


@dataclass
class AssistModel:
    """A loaded labeling-assist model: predict_fn maps (n, len(feature_cols)) float64 with
    NaN missing values -> (n, 7) class probabilities in CLASSES order."""
    predict_fn: Callable
    feature_cols: List[str]
    classes: tuple
    model_version: str


def load_assist(path: Path) -> AssistModel:
    """Load assist_model.ubj + assist_meta.json from a run dir (or the .ubj path itself).
    Hard-refuses anything not stamped `labeling_assist_only` — the spec carve-out's
    guardrail — and any class-order drift. Validation happens before xgboost is imported."""
    path = Path(path)
    run_dir = path.parent if path.suffix else path
    meta_path = run_dir / "assist_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no assist_meta.json next to {path} — "
                                "run `worker train-stroke-gbdt --fit-full` first")
    meta = json.loads(meta_path.read_text())
    if meta.get("purpose") != "labeling_assist_only":
        raise ValueError(f"refusing {meta_path}: purpose={meta.get('purpose')!r} is not "
                         "'labeling_assist_only' (spec carve-out guardrail)")
    if tuple(meta.get("classes", ())) != CLASSES:
        raise ValueError(f"refusing {meta_path}: classes {meta.get('classes')} != {CLASSES}")

    import xgboost as xgb

    booster = xgb.Booster(model_file=str(run_dir / "assist_model.ubj"))
    feature_cols = list(meta["feature_columns"])

    def predict_fn(X):
        return booster.predict(xgb.DMatrix(X, feature_names=feature_cols))

    return AssistModel(predict_fn=predict_fn, feature_cols=feature_cols,
                       classes=CLASSES, model_version=str(meta["model_version"]))


def pseudo_labels(hits: Sequence[Mapping], *, video_id: str, match_id: str) -> List[dict]:
    """hits_in_rallies output -> label dicts for traj.build_feature_row (shot_type=None:
    the human hasn't answered yet — that is the point). t_s comes from the detected event
    itself, so the builder's LABEL_MATCH_DT join is exact by construction."""
    return [{
        "event_id": h["event"]["id"], "match_id": match_id, "video_id": video_id,
        "t_s": float(h["event"]["t_s"]), "shot_type": None, "hitter": None,
        "rally_idx": h["rally_idx"], "hit_idx": h["hit_idx"],
    } for h in hits]


def predict_rows(assist: AssistModel, rows: Sequence[Mapping]) -> dict:
    """Feature rows -> {event_id: {pred, score, prob_net, prob_drive}}. X is assembled in
    assist.feature_cols order (the artifact's inference contract); None -> NaN rides
    XGBoost's native missing routing, same convention as training."""
    import numpy as np

    if not rows:
        return {}
    X = np.array([[np.nan if r.get(c) is None else float(r[c])
                   for c in assist.feature_cols] for r in rows], dtype=np.float64)
    probs = assist.predict_fn(X)
    out = {}
    for r, p in zip(rows, probs):
        best = int(np.argmax(p))
        out[str(r["event_id"])] = {
            "pred": CLASSES[best],
            "score": float(p[best]),
            **{f"prob_{c}": float(p[CLASSES.index(c)]) for c in RARE_CLASSES},
        }
    return out


def prediction_entry(canonical: str, score: float, *, model_version: str,
                     result_id: str) -> dict:
    """One Label Studio pre-annotation (LS 1.23 choices schema). The UI pre-selects
    value.choices when the project setting 'Show predictions to annotators' is on."""
    return {
        "model_version": model_version,
        "score": round(float(score), 4),
        "result": [{
            "id": result_id,
            "type": "choices",
            "from_name": LS_FROM_NAME,
            "to_name": LS_TO_NAME,
            "value": {"choices": [GERMAN_BY_CANONICAL[canonical]]},
        }],
    }


def attach_predictions(tasks: List[dict], preds: Mapping[str, Mapping], *,
                       model_version: str, min_score: float = 0.8,
                       skip_first_of_rally: bool = False) -> dict:
    """Mutate tasks in place: always add the sortable display columns (model_pred German,
    model_score, rare_score); add a pre-selecting `predictions` entry only at
    score >= min_score (user decision 2026-07-12: pre-fill when confident, hint otherwise).
    Returns counts for the export summary."""
    counts = {"prelabeled": 0, "skipped_low_score": 0, "skipped_first": 0, "no_features": 0}
    for task in tasks:
        data = task["data"]
        p = preds.get(str(data["event_id"]))
        if p is None:
            data["model_pred"] = ""
            data["model_score"] = None
            data["rare_score"] = None
            counts["no_features"] += 1
            continue
        data["model_pred"] = GERMAN_BY_CANONICAL[p["pred"]]
        data["model_score"] = round(float(p["score"]), 4)
        data["rare_score"] = round(max(float(p[f"prob_{c}"]) for c in RARE_CLASSES), 3)
        if skip_first_of_rally and data.get("first_of_rally"):
            counts["skipped_first"] += 1
        elif float(p["score"]) >= min_score:
            task["predictions"] = [prediction_entry(
                p["pred"], p["score"], model_version=model_version,
                result_id=f"pred_{data['event_id'][:8]}")]
            counts["prelabeled"] += 1
        else:
            counts["skipped_low_score"] += 1
    return counts
