"""M3 — GBDT learning curve at 140/280/554 confident labels.

Throwaway measurement (spec `stroke-classification-v2.md` §10, memo §11). Reuses (does not
modify) `worker.stroke.gbdt`/`worker.stroke.train` — only new code here is `run_logo_capped`,
which mirrors `gbdt.run_logo`'s fold loop but subsamples each fold's training pool down to a
fixed cap (stratified by class label, fixed per-fold seed) before calling `fit_predict`.

Per-video confident-label counts (121/51/88/125/169, sum 554) mean every LOGO fold's
other-4-videos pool is >=385 rows, so a cap of 554 never binds — that data point is already
measured (2026-07-12 run, `stroke_runs/20260712T205001Z_gbdt_traj/report.json`, mean acc
0.536 +- 0.023). This script re-measures it once at cap=554 as a correctness check on the
capping wrapper (should reproduce ~0.536), then measures the two new points (140, 280).

Usage: uv run python ../scratch/stroke_v2/m3_learning_curve.py
(run from worker/worker/ so the relative .labeling path resolves)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from worker.stroke import gbdt as gbdt_mod
from worker.stroke.gbdt import CLASSES, xgb_fit_predict
from worker.stroke.train import accuracy, class_weights, fold_split, per_class_recall

FEATURES_PATH = Path(".labeling/stroke_traj/features.parquet")
OUT_DIR = Path(".labeling/phase1_measurements/m3_learning_curve")
BASELINE_REPORT = Path(".labeling/stroke_runs/20260712T205001Z_gbdt_traj/report.json")

PARAMS = {"max_depth": 4, "eta": 0.05, "n_estimators": 400,
          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}
SEED = 42
CAPS = (140, 280, 554)


def stratified_cap(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Indices into y (already the fold's train-eligible rows), subsampled to <= cap total,
    proportionally per class (remainder distributed to the largest classes), fixed seed."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n <= cap:
        return np.arange(n)
    classes, counts = np.unique(y, return_counts=True)
    target = np.floor(counts * (cap / n)).astype(int)
    target = np.maximum(target, np.minimum(counts, 1))  # keep >=1 of every present class
    while target.sum() > cap:
        i = np.argmax(target)
        target[i] -= 1
    while target.sum() < cap:
        room = counts - target
        i = np.argmax(room)
        if room[i] <= 0:
            break
        target[i] += 1
    picked = []
    for c, k in zip(classes, target):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        picked.append(idx_c[:k])
    return np.concatenate(picked)


def run_logo_capped(df, feature_cols, cfg: dict, cap: int, seed: int) -> dict:
    X = df.select(list(feature_cols)).to_numpy().astype(np.float64)
    video_ids = np.asarray([str(v) for v in df["video_id"].to_list()])
    confident = ~df["is_calibration"].to_numpy()
    from worker.stroke.train import encode_labels

    shot_types = df["shot_type"].to_list()
    y = np.full(df.height, -1, dtype=np.int64)
    y[confident] = encode_labels([s for s, keep in zip(shot_types, confident) if keep])

    folds = fold_split(video_ids.tolist())
    per_fold = []
    oof_true, oof_pred = [], []  # pooled OOF, confident rows only -> macro-recall (matches
                                  # gbdt.run_logo's own headline metric, not just mean accuracy)
    train_true_all, train_pred_all = [], []  # in-sample, for the G2 train-eval memorization gap
    for k, held in enumerate(folds):
        train_mask = (video_ids != held) & confident
        test_mask = (video_ids == held) & confident
        train_idx_full = np.where(train_mask)[0]
        y_train_full = y[train_idx_full]

        keep_local = stratified_cap(y_train_full, cap, seed=hash((seed, k, cap)) & 0x7FFFFFFF)
        train_idx = train_idx_full[keep_local]

        fold_seed = int(np.random.default_rng([seed, k, cap]).integers(2**31 - 1))
        sw = class_weights(y[train_idx])[y[train_idx]]
        probs = xgb_fit_predict(X[train_idx], y[train_idx], sw, X[test_mask], fold_seed,
                                params=PARAMS, feature_names=list(feature_cols))
        pred = probs.argmax(axis=1)
        fold_acc = accuracy(y[test_mask], pred)
        oof_true.append(y[test_mask]); oof_pred.append(pred)

        train_probs = xgb_fit_predict(X[train_idx], y[train_idx], sw, X[train_idx], fold_seed,
                                      params=PARAMS, feature_names=list(feature_cols))
        train_pred_all.append(train_probs.argmax(axis=1)); train_true_all.append(y[train_idx])

        per_fold.append({"fold": k, "video_id": held, "n_train": int(len(train_idx)),
                         "n_train_available": int(len(train_idx_full)),
                         "n_test": int(test_mask.sum()), "accuracy": fold_acc})
        print(f"    fold {k} ({held[:8]}): n_train={len(train_idx)}/{len(train_idx_full)} "
              f"n_test={int(test_mask.sum())} acc={fold_acc:.3f}")

    accs = [f["accuracy"] for f in per_fold]
    oof_true, oof_pred = np.concatenate(oof_true), np.concatenate(oof_pred)
    recalls = per_class_recall(oof_true, oof_pred)  # NaN where a class has zero OOF support
    macro_recall = float(np.nanmean(recalls))
    per_class = {c: (None if np.isnan(r) else float(r)) for c, r in zip(CLASSES, recalls)}
    support = {c: int((oof_true == i).sum()) for i, c in enumerate(CLASSES)}

    train_true_all = np.concatenate(train_true_all)
    train_pred_all = np.concatenate(train_pred_all)
    train_acc = accuracy(train_true_all, train_pred_all)  # in-sample pooled, memorization check

    return {"cap": cap, "folds": per_fold, "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)), "pooled_oof_accuracy": accuracy(oof_true, oof_pred),
            "macro_recall": macro_recall, "per_class_recall": per_class, "class_support_oof": support,
            "train_accuracy_in_sample": train_acc,
            "train_eval_gap_pts": (train_acc - accuracy(oof_true, oof_pred)) * 100}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = gbdt_mod.load_features(FEATURES_PATH, include_ppos=True)
    cfg = {**PARAMS, "seed": SEED, "include_ppos": True}

    baseline = json.loads(BASELINE_REPORT.read_text())
    print(f"existing uncapped (554) baseline: mean_acc={baseline['mean_accuracy']:.4f} "
          f"+- {baseline['std_accuracy']:.4f} (2026-07-12 run, reused as-is)")

    points = {554: {"cap": 554, "mean_accuracy": baseline["mean_accuracy"],
                    "std_accuracy": baseline["std_accuracy"], "source": "reused_2026-07-12_run"}}

    for cap in (554, 280, 140):
        print(f"\n--- cap={cap} ---")
        res = run_logo_capped(df, feature_cols, cfg, cap, SEED)
        res["source"] = "measured_this_run"
        print(f"  cap={cap}: mean_acc={res['mean_accuracy']:.4f} +- {res['std_accuracy']:.4f}  "
              f"macro_recall={res['macro_recall']:.4f}  train_eval_gap={res['train_eval_gap_pts']:+.1f}pts")
        if cap == 554:
            print(f"  [correctness check] vs 2026-07-12 baseline mean_acc 0.536: "
                  f"delta={res['mean_accuracy'] - baseline['mean_accuracy']:+.4f} "
                  "(small delta expected: this run uses its own fold-seed derivation, "
                  "not a byte-identical rerun)")
            print(f"  per_class_recall (n=554, G2 needs every n>=30 class >=0.40): "
                  f"{ {c: (None if r is None else round(r,3)) for c, r in res['per_class_recall'].items()} }")
            print(f"  class_support_oof: {res['class_support_oof']}")
        points[cap] = res

    ordered = [points[c] for c in CAPS]
    deltas = []
    for i in range(1, len(ordered)):
        deltas.append({"from_cap": ordered[i - 1]["cap"], "to_cap": ordered[i]["cap"],
                       "delta_mean_accuracy": ordered[i]["mean_accuracy"] - ordered[i - 1]["mean_accuracy"],
                       "delta_macro_recall": ordered[i]["macro_recall"] - ordered[i - 1]["macro_recall"]})

    # log2(labels) vs metric linear fit -> extrapolate to session2/session3 label counts.
    # Two curves: mean_accuracy (the harness's headline number, comparable to the historical
    # 0.536 baseline) and macro_recall (G2's ACTUAL gate metric — pooled-accuracy figures can
    # look fine while thin classes stay at recall 0, which is exactly what broke v1's gate).
    caps_arr = np.array([p["cap"] for p in ordered], dtype=float)

    def fit_and_predict(metric_key):
        vals = np.array([p[metric_key] for p in ordered], dtype=float)
        slope, intercept = np.polyfit(np.log2(caps_arr), vals, 1)
        return {"slope_per_log2_label": float(slope),
                "predicted_at_1400_session2": float(slope * np.log2(1400) + intercept),
                "predicted_at_2100_session3": float(slope * np.log2(2100) + intercept)}

    extrapolation = {"mean_accuracy": fit_and_predict("mean_accuracy"),
                     "macro_recall": fit_and_predict("macro_recall")}

    report = {"caps": list(CAPS), "points": {str(c): points[c] for c in CAPS},
              "deltas": deltas, "extrapolation": extrapolation,
              "g2_gate": {"macro_recall_target": 0.55, "per_class_n30_recall_target": 0.40,
                         "train_eval_gap_max_pts": 15},
              "spec_estimate_band_mean_accuracy": {
                  "session2_~1400_labels": [0.60, 0.70],
                  "session3_~2100_labels": [0.65, 0.75],
              }}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== LEARNING CURVE ===")
    for p in ordered:
        print(f"  n={p['cap']:>4}: mean_acc={p['mean_accuracy']:.4f} +- {p['std_accuracy']:.4f}  "
              f"macro_recall={p['macro_recall']:.4f}  train_eval_gap={p['train_eval_gap_pts']:+.1f}pts  ({p['source']})")
    for d in deltas:
        print(f"  {d['from_cap']} -> {d['to_cap']}: d_mean_acc={d['delta_mean_accuracy']:+.4f}  "
              f"d_macro_recall={d['delta_macro_recall']:+.4f}")
    ea, er = extrapolation["mean_accuracy"], extrapolation["macro_recall"]
    print(f"  mean_accuracy extrapolation: @1400 -> {ea['predicted_at_1400_session2']:.3f} "
          f"(spec band 0.60-0.70) | @2100 -> {ea['predicted_at_2100_session3']:.3f} (spec band 0.65-0.75)")
    print(f"  macro_recall extrapolation:  @1400 -> {er['predicted_at_1400_session2']:.3f} "
          f"| @2100 -> {er['predicted_at_2100_session3']:.3f}  (G2 gate: macro_recall >= 0.55)")
    print(f"\nWrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
