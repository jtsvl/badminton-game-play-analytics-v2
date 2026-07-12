"""Trajectory-geometry GBDT stroke spike: LOGO-CV eval of XGBoost on stroke/traj.py features.

Spec: features/stroke-classification.md, "Alternative candidate: trajectory-geometry GBDT".
Protocol identical to BST4 by construction — same fold_split/sweep_tau/metric functions from
stroke/train.py, train on confident rows only, one OOF prediction per held-out row, gate =
mean LOGO accuracy >= 0.70 — but parameterized by a `fit_predict` callable so the whole
protocol tests with a stub on light installs; xgboost is imported only inside
`xgb_fit_predict`. Eval only: no promotion path, no pipeline wiring, whatever the result
(locked). One user-approved carve-out (2026-07-12): `--fit-full` additionally saves a
LABELING-ASSIST artifact (assist_model.ubj + assist_meta.json, `purpose:
labeling_assist_only`) trained on all confident rows — it pre-fills Label Studio tasks and
nothing else; the ≥0.70 gate for wiring stage 7 into the pipeline is untouched.
Deterministic by construction: `hist` on CPU, nthread=1, per-fold seeds.

    python -m worker train-stroke-gbdt [--features P] [--seed 42] [--out DIR] [--ppos]
                                       [--max-depth 4] [--eta 0.1] [--n-estimators 200]
                                       [--subsample 0.8] [--colsample-bytree 0.8]
                                       [--fit-full]
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .traj import FEATURE_COLS, PPOS_COLS
from .train import (
    CLASSES,
    GATE_MEAN_ACCURACY,
    HARD_PAIRS,
    _CLASS_INDEX,
    _none_if_nan,
    accuracy,
    class_weights,
    confusion,
    encode_labels,
    fold_split,
    per_class_recall,
    sweep_tau,
    unknown_rate_per_match,
    unsicher_abstention,
)

log = logging.getLogger("worker.stroke.gbdt")

REQUIRED_META = ("video_id", "t_s", "shot_type", "is_calibration")

DEFAULT_PARAMS = {"max_depth": 4, "eta": 0.1, "n_estimators": 200,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}


def load_features(path: Path, include_ppos: bool = False):
    """features.parquet -> (df, ordered feature columns). Hard guards: required meta columns,
    at least one known feature column, and >= 2 distinct video_ids (1 video = empty train
    masks under LOGO)."""
    import polars as pl

    df = pl.read_parquet(path)
    missing = [c for c in REQUIRED_META if c not in df.columns]
    if missing:
        raise ValueError(f"features parquet is missing required columns {missing} ({path})")
    wanted = FEATURE_COLS + (PPOS_COLS if include_ppos else ())
    cols = [c for c in wanted if c in df.columns]
    if not cols:
        raise ValueError(f"no known feature columns in {path} (expected from FEATURE_COLS)")
    if df["video_id"].n_unique() < 2:
        raise ValueError("LOGO-CV needs >= 2 distinct video_ids — got 1")
    return df, cols


def run_logo(df, feature_cols: Sequence[str], fit_predict, cfg: dict):
    """The locked LOGO/OOF/τ protocol on a features frame. `fit_predict(X_tr, y_tr,
    sample_weight, X_te, seed) -> (n_te, 7) probs` is the only model-specific piece —
    everything here runs with a stub. NaNs pass through unimputed (XGBoost missing routing).
    Returns (oof_df in df row order, report) with the BST4 report shape."""
    import numpy as np
    import polars as pl

    X = df.select(list(feature_cols)).to_numpy().astype(np.float64)  # nulls -> NaN
    video_ids = np.asarray([str(v) for v in df["video_id"].to_list()])
    confident = ~df["is_calibration"].to_numpy()
    shot_types = df["shot_type"].to_list()
    y = np.full(df.height, -1, dtype=np.int64)
    y[confident] = encode_labels([s for s, keep in zip(shot_types, confident) if keep])

    folds = fold_split(video_ids.tolist())
    probs = np.zeros((df.height, len(CLASSES)), dtype=np.float64)
    fold_of_row = np.zeros(df.height, dtype=np.int64)
    seed = int(cfg.get("seed", 42))
    per_fold = []
    for k, held in enumerate(folds):
        train_mask = (video_ids != held) & confident
        test_mask = video_ids == held
        fold_of_row[test_mask] = k
        fold_seed = int(np.random.default_rng([seed, k]).integers(2 ** 31 - 1))
        sample_weight = class_weights(y[train_mask])[y[train_mask]]
        probs[test_mask] = fit_predict(X[train_mask], y[train_mask], sample_weight,
                                       X[test_mask], fold_seed)
        hm = test_mask & confident
        fold_pred = probs[hm].argmax(axis=1)
        fold_acc = accuracy(y[hm], fold_pred)
        per_fold.append({
            "fold": k, "video_id": held, "n": int(hm.sum()), "accuracy": fold_acc,
            "per_class_recall": [_none_if_nan(r) for r in per_class_recall(y[hm], fold_pred)],
        })
        log.info("stroke gbdt: fold %d/%d (%s) n=%d acc=%.3f",
                 k + 1, len(folds), held[:8], int(hm.sum()), fold_acc)

    pred = probs.argmax(axis=1)
    p_max = probs.max(axis=1)
    tau_res = sweep_tau(p_max[confident], pred[confident] == y[confident])
    cm = confusion(y[confident], pred[confident])
    fold_accs = [f["accuracy"] for f in per_fold]
    mean_acc = float(np.mean(fold_accs))

    def _pair_block(a: str, b: str) -> dict:
        ia, ib = _CLASS_INDEX[a], _CLASS_INDEX[b]
        return {"labels": [a, b],
                "matrix": [[int(cm[ia, ia]), int(cm[ia, ib])],
                           [int(cm[ib, ia]), int(cm[ib, ib])]]}

    report = {
        "classes": list(CLASSES),
        "folds": per_fold,
        "mean_accuracy": mean_acc,
        "std_accuracy": float(np.std(fold_accs)),
        "pooled_accuracy": accuracy(y[confident], pred[confident]),
        "per_class_recall": {c: _none_if_nan(r) for c, r in
                             zip(CLASSES, per_class_recall(y[confident], pred[confident]))},
        "confusion": cm.tolist(),
        "hard_pairs": {f"{a}/{b}": _pair_block(a, b) for a, b in HARD_PAIRS},
        "tau": tau_res,
        "unsicher_abstention": (unsicher_abstention(p_max[~confident], tau_res["tau"])
                                if (~confident).any() else None),
        "unknown_rate_per_match": unknown_rate_per_match(p_max, video_ids.tolist(),
                                                         tau_res["tau"]),
        "gate_passed": bool(mean_acc >= GATE_MEAN_ACCURACY),
        "device": "cpu",
        "config": dict(cfg),
        "feature_importance_gain": None,  # filled by train_cli from the fold boosters
    }
    oof_cols: dict = {c: df[c] for c in
                      ("event_id", "video_id", "t_s", "shot_type", "is_calibration")
                      if c in df.columns}
    oof_cols["fold"] = fold_of_row
    for i, c in enumerate(CLASSES):
        oof_cols[f"prob_{c}"] = probs[:, i]
    oof_cols["pred"] = [CLASSES[i] for i in pred]
    oof_cols["p_max"] = p_max
    return pl.DataFrame(oof_cols), report


def fit_full(df, feature_cols: Sequence[str], fit_predict, cfg: dict) -> int:
    """Fit ONE model on ALL confident rows — the labeling-assist artifact, never the
    pipeline. Masks/encoding/weights identical to run_logo minus the fold holdout; the
    test matrix is empty (nothing honest to predict — this model has seen every label).
    The caller captures the booster out-of-band (xgb_fit_predict's `boosters` param).
    Returns the number of training rows for the artifact metadata."""
    import numpy as np

    X = df.select(list(feature_cols)).to_numpy().astype(np.float64)
    confident = ~df["is_calibration"].to_numpy()
    y = encode_labels([s for s, keep in zip(df["shot_type"].to_list(), confident) if keep])
    sample_weight = class_weights(y)[y]
    # seed slot after the folds, mirroring the BST final all-data train (train.py)
    n_folds = len(fold_split([str(v) for v in df["video_id"].to_list()]))
    full_seed = int(np.random.default_rng([int(cfg.get("seed", 42)), n_folds])
                    .integers(2 ** 31 - 1))
    fit_predict(X[confident], y, sample_weight, X[:0], full_seed)
    return int(confident.sum())


def save_assist(booster, out_dir: Path, *, feature_cols: Sequence[str], params: dict,
                seed: int, n_train_rows: int, dataset_sha256: str,
                git_sha: str | None, xgboost_version: str) -> Path:
    """Write assist_model.ubj + assist_meta.json. `feature_columns` ORDER is the inference
    contract — prelabel must assemble X exactly in this order. `purpose`/`never_promote`
    are the guardrail: loaders must refuse anything not marked labeling_assist_only."""
    import json

    model_path = out_dir / "assist_model.ubj"
    booster.save_model(str(model_path))
    meta = {
        "purpose": "labeling_assist_only",
        "never_promote": True,
        "model_version": out_dir.name,
        "classes": list(CLASSES),
        "feature_columns": list(feature_cols),
        "params": dict(params),
        "seed": int(seed),
        "n_train_rows": int(n_train_rows),
        "dataset_sha256": dataset_sha256,
        "git_sha": git_sha,
        "xgboost_version": xgboost_version,
    }
    (out_dir / "assist_meta.json").write_text(json.dumps(meta, indent=2))
    return model_path


def mean_gain(boosters: Sequence, feature_cols: Sequence[str]) -> dict:
    """Mean gain importance across the fold boosters, 0 where a feature went unused,
    sorted descending — the spike's main diagnostic beyond the headline accuracy."""
    totals = {c: 0.0 for c in feature_cols}
    for b in boosters:
        for name, gain in b.get_score(importance_type="gain").items():
            totals[name] = totals.get(name, 0.0) + float(gain)
    n = max(len(boosters), 1)
    return dict(sorted(((c, t / n) for c, t in totals.items()), key=lambda kv: -kv[1]))


def xgb_fit_predict(X_tr, y_tr, sample_weight, X_te, seed: int, *, params: dict | None = None,
                    feature_names: Sequence[str] | None = None,
                    boosters: list | None = None):
    """The only xgboost-touching function. Native Booster API (no scikit-learn), NaN routed
    natively (missing=NaN default), fixed tree budget (no early stopping — no honest
    validation set inside a LOGO fold), nthread=1 + seed -> bit-deterministic on CPU."""
    import numpy as np
    import xgboost as xgb

    p = dict(DEFAULT_PARAMS, **(params or {}))
    names = list(feature_names) if feature_names is not None else None
    dtrain = xgb.DMatrix(np.asarray(X_tr, dtype=np.float64), label=y_tr,
                         weight=sample_weight, feature_names=names)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": len(CLASSES),
         "tree_method": "hist", "device": "cpu", "nthread": 1, "seed": int(seed),
         "max_depth": int(p["max_depth"]), "eta": float(p["eta"]),
         "subsample": float(p["subsample"]), "colsample_bytree": float(p["colsample_bytree"]),
         "min_child_weight": float(p["min_child_weight"])},
        dtrain, num_boost_round=int(p["n_estimators"]))
    if boosters is not None:
        boosters.append(booster)
    if np.asarray(X_te).shape[0] == 0:  # fit_full's empty test matrix — nothing to predict
        return np.zeros((0, len(CLASSES)))
    return booster.predict(xgb.DMatrix(np.asarray(X_te, dtype=np.float64),
                                       feature_names=names))


def train_cli(argv: Sequence[str] | None = None) -> int:
    """`worker train-stroke-gbdt` — LOGO-CV eval + FR-22 τ sweep on trajectory features.
    Eval only: report + OOF parquet, no model artifact, no promotion (spec-locked)."""
    import argparse
    import hashlib
    import json
    import shlex
    import subprocess
    from datetime import datetime, timezone

    from ..config import load_config

    ap = argparse.ArgumentParser(
        prog="worker train-stroke-gbdt",
        description="LOGO-CV-evaluate an XGBoost stroke classifier on hit-to-hit trajectory "
                    "features (specs/features/stroke-classification.md, GBDT spike).")
    ap.add_argument("--features", help="features.parquet (default: worker/.labeling/stroke_traj/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="run dir (default: <labeling>/stroke_runs/<UTC>_gbdt_traj)")
    ap.add_argument("--ppos", action="store_true",
                    help="include the ppos_ player-position ablation features")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    ap.add_argument("--eta", type=float, default=DEFAULT_PARAMS["eta"])
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    ap.add_argument("--subsample", type=float, default=DEFAULT_PARAMS["subsample"])
    ap.add_argument("--colsample-bytree", type=float,
                    default=DEFAULT_PARAMS["colsample_bytree"])
    ap.add_argument("--fit-full", action="store_true",
                    help="additionally fit on ALL confident rows and save the "
                         "labeling-assist artifact (assist_model.ubj — pre-labels Label "
                         "Studio tasks only, never the pipeline)")
    args = ap.parse_args(argv)

    cfg_env = load_config()
    labeling = Path(cfg_env.frame_cache_dir).expanduser().parent / ".labeling"
    features_path = (Path(args.features).expanduser() if args.features
                     else labeling / "stroke_traj" / "features.parquet")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (Path(args.out).expanduser() if args.out
               else labeling / "stroke_runs" / f"{stamp}_gbdt_traj")

    df, feature_cols = load_features(features_path, include_ppos=args.ppos)
    params = {"max_depth": args.max_depth, "eta": args.eta,
              "n_estimators": args.n_estimators, "subsample": args.subsample,
              "colsample_bytree": args.colsample_bytree,
              "min_child_weight": DEFAULT_PARAMS["min_child_weight"]}
    cfg = {**params, "nthread": 1, "seed": args.seed, "include_ppos": args.ppos,
           "feature_columns": feature_cols}

    boosters: list = []

    def fit(X_tr, y_tr, sample_weight, X_te, fold_seed):
        return xgb_fit_predict(X_tr, y_tr, sample_weight, X_te, fold_seed, params=params,
                               feature_names=feature_cols, boosters=boosters)

    oof, report = run_logo(df, feature_cols, fit, cfg)
    report["feature_importance_gain"] = mean_gain(boosters, feature_cols)

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        git_sha = None
    import xgboost

    report.update({
        "command": shlex.join(["python", "-m", "worker", "train-stroke-gbdt", *(argv or [])]),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "xgboost_version": xgboost.__version__,
        "dataset": {
            "path": str(features_path),
            "sha256": {"features.parquet": hashlib.sha256(
                features_path.read_bytes()).hexdigest()},
            "n_rows": df.height,
            "n_features": len(feature_cols),
        },
        "caveats": [
            "one session / one lighting: LOGO-CV measures cross-game, not cross-session generalization",
            "spike: eval only — nothing is promoted; tau reported from fold-model OOF confidences for comparability",
            "court-projection features are sparse (~37% valid): elevated contact points break the floor homography",
        ],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.write_parquet(out_dir / "oof_predictions.parquet")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    for f in report["folds"]:
        print(f"fold {f['fold']} ({f['video_id'][:8]}): n={f['n']} acc={f['accuracy']:.3f}")
    print(f"LOGO mean accuracy: {report['mean_accuracy']:.3f} ± {report['std_accuracy']:.3f} "
          f"(pooled {report['pooled_accuracy']:.3f}, {len(feature_cols)} features, "
          f"seed {args.seed})")
    tau = report["tau"]
    print(f"tau* = {tau['tau']:.3f}: covered-acc {tau['covered_accuracy']:.3f}, "
          f"coverage {tau['coverage']:.3f}"
          + (" [KNEE FALLBACK — FR-22 targets unsatisfiable]" if tau["knee_fallback"] else ""))
    print(f"Unsicher abstention at tau*: {report['unsicher_abstention']:.3f}" if
          report["unsicher_abstention"] is not None else "no calibration rows")
    top = list(report["feature_importance_gain"].items())[:5]
    print("top gain: " + ", ".join(f"{c} {g:.1f}" for c, g in top))
    gate = report["gate_passed"]
    print(f"GATE (mean >= {GATE_MEAN_ACCURACY:.2f}): {'PASSED' if gate else 'FAILED'} "
          f"-> {out_dir}")

    if args.fit_full:
        full_boosters: list = []

        def fit_all(X_tr, y_tr, sample_weight, X_te, full_seed):
            return xgb_fit_predict(X_tr, y_tr, sample_weight, X_te, full_seed,
                                   params=params, feature_names=feature_cols,
                                   boosters=full_boosters)

        n_train = fit_full(df, feature_cols, fit_all, cfg)
        model_path = save_assist(
            full_boosters[0], out_dir, feature_cols=feature_cols, params=params,
            seed=args.seed, n_train_rows=n_train,
            dataset_sha256=report["dataset"]["sha256"]["features.parquet"],
            git_sha=git_sha, xgboost_version=xgboost.__version__)
        print(f"labeling-assist model (n={n_train}) -> {model_path} "
              "[labeling_assist_only — pre-fills Label Studio, never the pipeline]")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_cli())
