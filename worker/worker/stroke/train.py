"""Stage-7 stroke-classifier training: LOGO-CV over the exported dataset + FR-22 τ sweep.

Pure logic first (folds, encoding, bones, jitter, metrics, τ) — torch enters only in the
training-loop section below and is imported lazily, so the module collects light.
Locked protocol (specs/features/stroke-classification.md): train the 7-class softmax on the
confident rows only (`unknown` = abstention, never a class); eval = leave-one-game-out CV
keyed by video_id; τ = smallest grid value with OOF covered-accuracy ≥ 0.75 subject to
coverage ≥ 0.70, else the accuracy-coverage knee, flagged. Labels join the NPZ by row order
(no key in the NPZ) — load_dataset hard-fails when meta.parquet is misaligned.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

CLASSES = ("clear", "smash", "drop", "net", "drive", "lift", "serve")  # frozen; index = class id
_CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# vendor preparing_data/shuttleset_dataset.py get_bone_pairs('coco') — copied verbatim
BONE_PAIRS = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),   # head
    (3, 5), (4, 6),                           # ears to shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
)

GATE_MEAN_ACCURACY = 0.70  # spec acceptance: LOGO-CV mean blocks pipeline wiring below this


def encode_labels(shot_types: Sequence[str]):
    """shot_type tokens -> int64 class ids. `unknown` is abstention, never a trained class."""
    import numpy as np

    bad = sorted({s for s in shot_types if s not in _CLASS_INDEX})
    if bad:
        raise ValueError(f"not trainable shot_type tokens: {bad} (classes: {CLASSES})")
    return np.array([_CLASS_INDEX[s] for s in shot_types], dtype=np.int64)


def fold_split(video_ids: Sequence[str]) -> list[str]:
    """LOGO fold keys: sorted unique video_ids (5 games -> 5 folds)."""
    return sorted(set(video_ids))


def class_weights(y, n_class: int = len(CLASSES)):
    """Balanced inverse-frequency CE weights: n/(k*count) over present classes, 0 if absent."""
    import numpy as np

    counts = np.bincount(y, minlength=n_class).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_class, dtype=np.float64)
    w[present] = len(y) / (present.sum() * counts[present])
    return w


def make_bones(joints):
    """(..., 17, 2) joints -> (..., 19, 2) bones, vendor create_bones semantics:
    end - start where both coordinates are nonzero (elementwise), else 0."""
    import numpy as np

    bones = []
    for start, end in BONE_PAIRS:
        s, e = joints[..., start, :], joints[..., end, :]
        bones.append(np.where((s != 0.0) & (e != 0.0), e - s, 0.0))
    return np.stack(bones, axis=-2).astype(joints.dtype)


def to_model_arrays(joints, positions, pose_style: str = "J_only"):
    """Exporter layout -> model layout.

    joints (N, slot, t, J, 2) -> J (N, t, slot, J*2) flattened per-slot features
    (JnB_bone appends the 19 bones after the 17 joints -> in_dim 72);
    positions (N, slot, t, 2) -> P (N, t, slot, 2).
    """
    import numpy as np

    if pose_style == "JnB_bone":
        joints = np.concatenate((joints, make_bones(joints)), axis=-2)
    elif pose_style != "J_only":
        raise ValueError(f"unknown pose_style {pose_style!r} (J_only | JnB_bone)")
    n, n_slots, t = joints.shape[:3]
    j = joints.transpose(0, 2, 1, 3, 4).reshape(n, t, n_slots, -1)
    p = positions.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(j), np.ascontiguousarray(p)


def jitter_shift(arrays: tuple, shifts):
    """Per-sample temporal shift on axis 1 (the window), zero-filling the vacated frame.

    One shift per sample, applied to every array (joints/shuttle/positions must move
    together). ±1 frame at the 12 fps cadence absorbs the observed ±0.2 s stage-6 anchor
    drift; the shuttle's within-arc interpolation break moves with the content, which is
    exactly the drift being simulated. Train-time only — never jitter eval/OOF.
    """
    out = []
    for a in arrays:
        b = a.copy()
        for i, shift in enumerate(shifts):
            shift = int(shift)
            if shift > 0:
                b[i, shift:] = a[i, :-shift]
                b[i, :shift] = 0
            elif shift < 0:
                b[i, :shift] = a[i, -shift:]
                b[i, shift:] = 0
        out.append(b)
    return tuple(out)


def accuracy(y_true, y_pred) -> float:
    import numpy as np

    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_recall(y_true, y_pred, n_class: int = len(CLASSES)):
    """Recall per class; NaN where the class has zero support (never averaged silently —
    game 9aaa599b has no net rows, its fold must show NaN, not 0)."""
    import numpy as np

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recall = np.full(n_class, np.nan)
    for c in range(n_class):
        support = y_true == c
        if support.any():
            recall[c] = float(np.mean(y_pred[support] == c))
    return recall


def confusion(y_true, y_pred, n_class: int = len(CLASSES)):
    """Confusion matrix, rows = true class, cols = predicted class."""
    import numpy as np

    c = np.zeros((n_class, n_class), dtype=np.int64)
    np.add.at(c, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return c


def sweep_tau(p_max, correct, acc_target: float = 0.75, cov_floor: float = 0.70) -> dict:
    """FR-22 threshold rule on OOF confident rows: smallest τ with covered-accuracy ≥
    acc_target subject to coverage ≥ cov_floor; unsatisfiable -> accuracy+coverage knee,
    flagged. Grid = {0} ∪ observed softmax maxima; full curve returned for the report."""
    import numpy as np

    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    curve = []
    for tau in np.unique(np.concatenate(([0.0], p_max))):
        covered = p_max >= tau
        acc = float(correct[covered].mean()) if covered.any() else 0.0
        curve.append({"tau": float(tau), "covered_accuracy": acc,
                      "coverage": float(covered.mean())})
    feasible = [pt for pt in curve
                if pt["covered_accuracy"] >= acc_target and pt["coverage"] >= cov_floor]
    if feasible:
        pick, knee = min(feasible, key=lambda pt: pt["tau"]), False
    else:
        pick, knee = max(curve, key=lambda pt: pt["covered_accuracy"] + pt["coverage"]), True
    return {"tau": pick["tau"], "covered_accuracy": pick["covered_accuracy"],
            "coverage": pick["coverage"], "knee_fallback": knee, "curve": curve}


def unknown_rate_per_match(p_max, video_ids, tau: float) -> dict[str, float]:
    """Share of hits per game that would ship as `unknown` at τ (FR-22: surfaced, not hidden)."""
    import numpy as np

    p_max = np.asarray(p_max, dtype=np.float64)
    ids = np.asarray(video_ids)
    return {str(v): float(np.mean(p_max[ids == v] < tau)) for v in sorted(set(video_ids))}


def unsicher_abstention(p_max_calibration, tau: float) -> float:
    """Share of the held-out Unsicher rows the model abstains on at τ — the honesty
    differential. Reported, never tuned on."""
    import numpy as np

    return float(np.mean(np.asarray(p_max_calibration, dtype=np.float64) < tau))


def load_dataset(dataset_dir: Path) -> tuple[dict, "pl.DataFrame"]:  # noqa: F821
    """Load stroke_dataset.npz + row-aligned meta.parquet with a hard alignment guard:
    the NPZ has no key column, so row counts must match and the per-row shuttle
    nonzero-rate must reproduce meta.shuttle_coverage exactly (same 16-frame arithmetic)."""
    import numpy as np
    import polars as pl

    dataset_dir = Path(dataset_dir)
    npz = np.load(dataset_dir / "stroke_dataset.npz")
    data = {k: npz[k] for k in ("joints", "shuttle", "positions")}
    meta = pl.read_parquet(dataset_dir / "meta.parquet")
    n = data["joints"].shape[0]
    if not (data["shuttle"].shape[0] == n == data["positions"].shape[0] == meta.height):
        raise ValueError(f"dataset misaligned: joints {n}, shuttle {data['shuttle'].shape[0]}, "
                         f"positions {data['positions'].shape[0]}, meta {meta.height} rows")
    recomputed = (data["shuttle"] != 0.0).any(axis=2).mean(axis=1)
    stored = meta["shuttle_coverage"].to_numpy()
    if not np.allclose(recomputed, stored, atol=1e-6):
        off = int(np.sum(~np.isclose(recomputed, stored, atol=1e-6)))
        raise ValueError(f"meta.parquet is not row-aligned with the NPZ: shuttle_coverage "
                         f"mismatch on {off}/{n} rows — re-export the dataset")
    return data, meta
