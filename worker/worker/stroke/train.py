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

import logging
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("worker.stroke")

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


# --- torch section (lazy imports; nothing above needs the ml stack) ---------------------

HARD_PAIRS = (("drop", "net"), ("clear", "lift"))  # spec: confusion called out for these


def hard_pair_block(cm, a: str, b: str) -> dict:
    """2x2 slice of a confusion matrix (rows true, cols pred) for one hard pair — shared report
    format between the BST LOGO-CV and the VideoJev zero-shot pilot (stroke/vlm_eval.py)."""
    ia, ib = _CLASS_INDEX[a], _CLASS_INDEX[b]
    return {"labels": [a, b],
            "matrix": [[int(cm[ia, ia]), int(cm[ia, ib])],
                       [int(cm[ib, ia]), int(cm[ib, ib])]]}


def _none_if_nan(x) -> float | None:
    import math

    v = float(x)
    return None if math.isnan(v) else v


def resolve_device(requested: str = "auto") -> str:
    """MPS-first fp32 (CLAUDE.md): one real forward+backward probe — loss AND every grad
    finite — else CPU with a loud log. Explicit cpu/mps requests are honored unprobed."""
    import torch

    if requested != "auto":
        return requested
    if not torch.backends.mps.is_available():
        return "cpu"
    from .model import BST4

    try:
        torch.manual_seed(0)
        model = BST4(drop_p=0.0).to("mps")
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(4, 16, 4, 34, device="mps"),
                  torch.randn(4, 16, 2, device="mps"),
                  torch.rand(4, 16, 4, 2, device="mps")),
            torch.zeros(4, dtype=torch.long, device="mps"))
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all().item()
                           for p in model.parameters() if p.grad is not None)
        if torch.isfinite(loss).item() and grads_finite:
            return "mps"
        log.warning("stroke: MPS fwd+bwd probe non-finite — falling back to cpu")
    except Exception as e:  # noqa: BLE001 — any MPS failure means: train on CPU
        log.warning("stroke: MPS probe failed (%s) — falling back to cpu", e)
    return "cpu"


def train_fold(J, S, P, y, cfg: dict, device: str, rng) -> "BST4":  # noqa: F821
    """Fixed-budget training (no early stopping — there is no honest val set inside a
    fold, and a fixed budget keeps the final all-data train well-defined): AdamW, 5-epoch
    linear warmup then cosine to 0, class-weighted CE from THIS split's rows, fresh ±1
    temporal jitter each epoch, drop_last batches (keeps BatchNorm batches full)."""
    import math

    import numpy as np
    import torch

    from .model import BST4

    torch.manual_seed(int(cfg.get("seed", 42)))
    model = BST4(in_dim=int(J.shape[-1]), seq_len=int(J.shape[1]), n_class=len(CLASSES),
                 variant=cfg.get("variant", "cg_ap")).to(device)
    weights = torch.tensor(class_weights(y), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                            weight_decay=1e-4)
    epochs = int(cfg.get("epochs", 150))
    warmup = min(5, epochs)

    def lam(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    n = len(y)
    batch = min(int(cfg.get("batch_size", 32)), n)
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(device)
    model.train()
    for _ in range(epochs):
        jj, sj, pj = jitter_shift((J, S, P), rng.integers(-1, 2, n))
        perm = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            jb, sb, pb = (torch.from_numpy(np.ascontiguousarray(a[idx])).float().to(device)
                          for a in (jj, sj, pj))
            opt.zero_grad()
            loss = loss_fn(model(jb, sb, pb), y_t[idx])
            loss.backward()
            opt.step()
        sched.step()
    return model


def predict_probs(model, J, S, P, device: str, batch: int = 256):
    """(n, 7) softmax probabilities. eval mode (BatchNorm running stats — batch-independent
    by test), no jitter, fp32."""
    import numpy as np
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(J), batch):
            jb, sb, pb = (torch.from_numpy(np.ascontiguousarray(a[i:i + batch]))
                          .float().to(device) for a in (J, S, P))
            # .cpu() before .double(): MPS has no float64
            out.append(torch.softmax(model(jb, sb, pb), dim=1).cpu().double().numpy())
    return np.concatenate(out, axis=0)


def run_logo_cv(data: dict, meta, cfg: dict):
    """Leave-one-game-out CV (locked eval): per fold, train on the OTHER games' confident
    rows, predict every row of the held-out game — confident and Unsicher alike — so all
    rows get exactly one out-of-fold prediction. Returns (oof_df in meta row order, report).
    Gate metrics use confident rows only; τ is swept on the OOF confident rows (FR-22)."""
    import numpy as np
    import polars as pl

    J, P = to_model_arrays(data["joints"], data["positions"],
                           cfg.get("pose_style", "J_only"))
    S = data["shuttle"]
    video_ids = np.asarray([str(v) for v in meta["video_id"].to_list()])
    confident = ~meta["is_calibration"].to_numpy()
    shot_types = meta["shot_type"].to_list()
    y = np.full(meta.height, -1, dtype=np.int64)
    y[confident] = encode_labels([s for s, keep in zip(shot_types, confident) if keep])

    device = resolve_device(cfg.get("device", "auto"))
    folds = fold_split(video_ids.tolist())
    probs = np.zeros((meta.height, len(CLASSES)), dtype=np.float64)
    fold_of_row = np.zeros(meta.height, dtype=np.int64)
    per_fold = []
    for k, held in enumerate(folds):
        train_mask = (video_ids != held) & confident
        test_mask = video_ids == held
        fold_of_row[test_mask] = k
        rng = np.random.default_rng([int(cfg.get("seed", 42)), k])
        model = train_fold(J[train_mask], S[train_mask], P[train_mask], y[train_mask],
                           cfg, device, rng)
        probs[test_mask] = predict_probs(model, J[test_mask], S[test_mask], P[test_mask],
                                         device)
        hm = test_mask & confident
        fold_pred = probs[hm].argmax(axis=1)
        fold_acc = accuracy(y[hm], fold_pred)
        per_fold.append({
            "fold": k, "video_id": held, "n": int(hm.sum()), "accuracy": fold_acc,
            "per_class_recall": [_none_if_nan(r) for r in per_class_recall(y[hm], fold_pred)],
        })
        log.info("stroke: fold %d/%d (%s) n=%d acc=%.3f",
                 k + 1, len(folds), held[:8], int(hm.sum()), fold_acc)

    pred = probs.argmax(axis=1)
    p_max = probs.max(axis=1)
    tau_res = sweep_tau(p_max[confident], pred[confident] == y[confident])
    cm = confusion(y[confident], pred[confident])
    fold_accs = [f["accuracy"] for f in per_fold]
    mean_acc = float(np.mean(fold_accs))

    report = {
        "classes": list(CLASSES),
        "folds": per_fold,
        "mean_accuracy": mean_acc,
        "std_accuracy": float(np.std(fold_accs)),
        "pooled_accuracy": accuracy(y[confident], pred[confident]),
        "per_class_recall": {c: _none_if_nan(r) for c, r in
                             zip(CLASSES, per_class_recall(y[confident], pred[confident]))},
        "confusion": cm.tolist(),
        "hard_pairs": {f"{a}/{b}": hard_pair_block(cm, a, b) for a, b in HARD_PAIRS},
        "tau": tau_res,
        "unsicher_abstention": (unsicher_abstention(p_max[~confident], tau_res["tau"])
                                if (~confident).any() else None),
        "unknown_rate_per_match": unknown_rate_per_match(p_max, video_ids.tolist(),
                                                         tau_res["tau"]),
        "gate_passed": bool(mean_acc >= GATE_MEAN_ACCURACY),
        "device": device,
        "config": dict(cfg),
    }
    oof_cols: dict = {c: meta[c] for c in
                      ("event_id", "video_id", "t_s", "shot_type", "is_calibration")
                      if c in meta.columns}
    oof_cols["fold"] = fold_of_row
    for i, c in enumerate(CLASSES):
        oof_cols[f"prob_{c}"] = probs[:, i]
    oof_cols["pred"] = [CLASSES[i] for i in pred]
    oof_cols["p_max"] = p_max
    return pl.DataFrame(oof_cols), report


def train_cli(argv: Sequence[str] | None = None) -> int:
    """`worker train-stroke` — LOGO-CV eval + FR-22 τ sweep; final all-data train and
    (gated) promotion to models/weights/stroke/. Every metric ships with command + seed."""
    import argparse
    import hashlib
    import json
    import shlex
    import shutil
    import subprocess
    from datetime import datetime, timezone

    import numpy as np

    from ..config import load_config

    ap = argparse.ArgumentParser(
        prog="worker train-stroke",
        description="Train + LOGO-CV-evaluate the BST4 stroke classifier on the exported "
                    "stroke dataset (specs/features/stroke-classification.md).")
    ap.add_argument("--dataset", help="stroke_dataset dir (default: worker/.labeling/stroke_dataset)")
    ap.add_argument("--variant", choices=["cg_ap", "bst0"], default="cg_ap")
    ap.add_argument("--pose-style", choices=["J_only", "JnB_bone"], default="J_only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--out", help="run dir (default: <labeling>/stroke_runs/<UTC>_<variant>_<pose-style>)")
    ap.add_argument("--promote", action="store_true",
                    help="copy the final checkpoint to models/weights/stroke/ + sidecar "
                         "(refused unless the LOGO gate passed)")
    args = ap.parse_args(argv)

    cfg_env = load_config()
    labeling = Path(cfg_env.frame_cache_dir).expanduser().parent / ".labeling"
    dataset_dir = Path(args.dataset).expanduser() if args.dataset else labeling / "stroke_dataset"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (Path(args.out).expanduser() if args.out
               else labeling / "stroke_runs" / f"{stamp}_{args.variant}_{args.pose_style}")

    data, meta = load_dataset(dataset_dir)
    cfg = {"variant": args.variant, "pose_style": args.pose_style, "lr": args.lr,
           "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
           "device": args.device}
    oof, report = run_logo_cv(data, meta, cfg)

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        git_sha = None
    import torch

    report.update({
        "command": shlex.join(["python", "-m", "worker", "train-stroke", *(argv or [])]),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "dataset": {
            "path": str(dataset_dir),
            "sha256": {f: hashlib.sha256((dataset_dir / f).read_bytes()).hexdigest()
                       for f in ("stroke_dataset.npz", "meta.parquet")},
        },
        "caveats": [
            "one session / one lighting: LOGO-CV measures cross-game, not cross-session generalization",
            "MPS is not bit-deterministic — the CPU command in the runbook is the reproducibility reference",
            "tau is calibrated on the fold models' OOF confidences but shipped with the all-data model",
        ],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.write_parquet(out_dir / "oof_predictions.parquet")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    for f in report["folds"]:
        print(f"fold {f['fold']} ({f['video_id'][:8]}): n={f['n']} acc={f['accuracy']:.3f}")
    print(f"LOGO mean accuracy: {report['mean_accuracy']:.3f} ± {report['std_accuracy']:.3f} "
          f"(pooled {report['pooled_accuracy']:.3f}, device {report['device']}, seed {args.seed})")
    tau = report["tau"]
    print(f"tau* = {tau['tau']:.3f}: covered-acc {tau['covered_accuracy']:.3f}, "
          f"coverage {tau['coverage']:.3f}"
          + (" [KNEE FALLBACK — FR-22 targets unsatisfiable]" if tau["knee_fallback"] else ""))
    print(f"Unsicher abstention at tau*: {report['unsicher_abstention']:.3f}" if
          report["unsicher_abstention"] is not None else "no calibration rows")
    gate = report["gate_passed"]
    print(f"GATE (mean >= {GATE_MEAN_ACCURACY:.2f}): {'PASSED' if gate else 'FAILED'} -> {out_dir}")

    if not gate:
        if args.promote:
            print("PROMOTE REFUSED: gate not passed — no checkpoint written")
            return 1
        return 0

    # final all-confident-rows train, same fixed budget (well-defined: no early stopping)
    J, P = to_model_arrays(data["joints"], data["positions"], args.pose_style)
    confident = ~meta["is_calibration"].to_numpy()
    y = encode_labels([s for s, keep in zip(meta["shot_type"].to_list(), confident) if keep])
    device = resolve_device(args.device)
    rng = np.random.default_rng([args.seed, len(report["folds"])])
    final = train_fold(J[confident], data["shuttle"][confident], P[confident], y,
                       cfg, device, rng)
    from .model import BST_SHA, STROKE_MODEL, save_checkpoint

    model_cfg = {"in_dim": int(J.shape[-1]), "seq_len": int(J.shape[1]),
                 "n_class": len(CLASSES), "variant": args.variant}
    save_checkpoint(out_dir / "bst_final.pt", final, model_cfg)
    print(f"final model (n={len(y)}) -> {out_dir / 'bst_final.pt'}")

    if args.promote:
        target = Path(STROKE_MODEL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "bst_final.pt", target)
        sidecar = {
            "model_version": f"bst4-{args.variant}-{args.pose_style}-v1",
            "classes": list(CLASSES),
            "variant": args.variant,
            "pose_style": args.pose_style,
            "in_dim": model_cfg["in_dim"],
            "seq_len": model_cfg["seq_len"],
            "tau": tau["tau"],
            "mean_logo_accuracy": report["mean_accuracy"],
            "git_sha": git_sha,
            "bst_vendor_sha": BST_SHA,
            "date": report["date"],
            "train_run": str(out_dir),
            "caveats": report["caveats"],
        }
        target.with_name("bst_best.json").write_text(json.dumps(sidecar, indent=2))
        print(f"promoted -> {target} (+ sidecar bst_best.json, tau {tau['tau']:.3f})")
    return 0
