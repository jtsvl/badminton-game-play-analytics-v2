# BST stroke classifier — local training runbook (Mac Studio M4 Max)

Stage-7 stroke classification (`specs/features/stroke-classification.md`, FR-21/22/23).
All commands run from `worker/`. Unlike the TrackNetV3 fine-tune, the training loop is
**in-tree and TDD'd** (`worker/stroke/train.py` + `worker/stroke/model.py`); the vendored repo
contributes only the model building blocks (TCN, transformer encoders, cross-attention, MLP
heads) — none of upstream's training scripts or preprocessing run here, so no MPS patch is
needed (`worker/training/bst_mps.patch` intentionally does not exist).

## 0. Prerequisites

- Vendored model source (one-time, git-ignored; SHA is pinned and recorded in the promoted
  sidecar):

  ```bash
  git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer .cache/_vendor/BST
  git -C .cache/_vendor/BST checkout fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df
  ```

  MIT license (non-distributed use is unconstrained either way, CLAUDE.md). Env override:
  `BST_REPO=<path>` (defaults to `worker/.cache/_vendor/BST/stroke_classification`).
- Deps: `uv sync --extra ml --extra dev` (the ml extra carries `positional-encodings` +
  `torchinfo`, imported at the vendored files' module top).
- Tests green first: `uv run pytest` (the guarded `tests/test_stroke_model.py` needs the
  vendor checkout + ml extra; it self-skips otherwise).

## 1. Export the dataset

```bash
uv run python -m worker export-stroke-dataset
```

Writes `worker/.labeling/stroke_dataset/{stroke_dataset.npz,meta.parquet}` from
`stroke_labels.parquet` + the per-video frame caches (details + gate numbers in the spec).
Current export (2026-07-05): 695 rows — 554 confident across 7 classes, 141 Unsicher
calibration rows. `train-stroke` hard-fails on a misaligned NPZ/meta pair; re-export if so.

## 2. CPU NaN-probe (run before any MPS run)

```bash
uv run python -m worker train-stroke --device cpu --epochs 1 --seed 42
```

Finite losses + a written `report.json` prove the plumbing; the numbers are meaningless at
1 epoch. `--device auto` additionally runs a one-shot MPS forward+backward finite-probe and
falls back to CPU loudly (fp32 everywhere, CLAUDE.md).

## 3. LOGO-CV baseline (the eval of record)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker train-stroke --seed 42
```

Defaults: `--variant cg_ap --pose-style J_only --lr 1e-3 --epochs 150 --batch-size 32`.
Protocol (locked): 5 leave-one-game-out folds keyed by `video_id`; train on the other games'
554-row confident subset; every row of the held-out game (confident + Unsicher) gets exactly
one out-of-fold prediction. τ per FR-22: smallest grid value with OOF covered-accuracy ≥ 0.75
subject to coverage ≥ 0.70, else the accuracy+coverage knee, flagged. Run artifacts
(`oof_predictions.parquet`, `report.json`, checkpoints) land in
`worker/.labeling/stroke_runs/<UTC>_<variant>_<pose-style>/`.

## 4. Sanctioned sweep (only these knobs — locked)

`{cg_ap vs bst0, J_only vs JnB_bone, --lr, --epochs, class-weighted CE (always on)}`.
Every run gets a row here — command + seed + result, no exceptions.

**Sweep of 2026-07-05** (dataset export of 2026-07-05, 554 confident rows; all commands
prefixed `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker train-stroke --seed 42`,
device mps, worker git `4aa65f3`+f64-fix; ~4.5 min/run):

| run | extra args | mean LOGO acc | pooled | note |
|---|---|---|---|---|
| baseline | — | 0.376 ± 0.039 | 0.388 | folds 0.31–0.43 |
| lr | `--lr 3e-4` | **0.400 ± 0.056** | 0.403 | best |
| epochs+ | `--epochs 300` | 0.390 ± 0.049 | 0.397 | |
| epochs− | `--epochs 75` | 0.377 ± 0.024 | 0.384 | |
| bones | `--pose-style JnB_bone` | 0.349 ± 0.023 | 0.356 | bones hurt |
| ablation | `--variant bst0` | 0.353 ± 0.026 | 0.354 | CG+AP worth ≈ +0.02 |

**Result: GATE FAILED on every run** (best 0.400 vs the 0.70 bar); the FR-22 τ targets were
unsatisfiable everywhere (knee fallback at τ = 0, i.e. no useful confidence separation).
Best-run per-class recall (pooled OOF): clear .53, lift .54, serve .40, drop .36, smash .30,
**net .00, drive .00**; predictions collapse toward the majority classes (drop/lift).

**Diagnosis — generalization gap, not underfit** (fold ff0d8099, lr 3e-4 config, seed 42, via
the snippet in the session log / re-runnable with `train_fold` + `predict_probs` on the train
split): train accuracy **0.945** vs held-out **0.402**. The model memorizes 385 training rows
and does not transfer across games — more epochs/LR cannot close this; the levers are data
volume (more sessions), input signal (near-half recovery v2 — the spec's named first lever —
and shuttle coverage), not hyperparameters. No promotion, no pipeline wiring (spec gate).

## 5. Gate + promote (only on a measured win)

Gate: **mean LOGO-CV accuracy ≥ 0.70** — below it, stop; no pipeline wiring, no promotion
(`--promote` refuses on a failed gate). On a pass the CLI trains the final model on all 554
confident rows (same fixed budget — no early stopping, so the all-data train is well-defined)
and `--promote` copies it to `models/weights/stroke/bst_best.pt` + sidecar `bst_best.json`
(model_version, classes, variant, seq_len, τ, git SHA, vendor SHA, date). Inference selects
weights via `STROKE_MODEL` (defaults to that path), mirroring the stage-5 shuttle convention.

## 6. Caveats (report these with every number)

- **One session, one lighting, 5 games (all doubles):** LOGO-CV measures cross-game, not
  cross-session generalization. The Session-3 re-eval is the real test.
- **MPS is not bit-deterministic** — seeds are recorded, but the §2 CPU command is the
  reproducibility reference.
- **τ is calibrated on the fold models' OOF confidences** and shipped with the all-data
  model; the confidence distributions differ. Documented in the sidecar, not re-tuned
  (the spec locks OOF-only τ).
- Unsicher abstention is a reported honesty differential, never a tuning target.

## 7. Trajectory-geometry XGBoost spike (2026-07-12)

Alternative candidate after the §4 gate failure (spec section "Alternative candidate:
trajectory-geometry GBDT" — eval only, no wiring/promotion either way). Torch-free path:
hit-to-hit geometry features + XGBoost, same LOGO protocol and report shape.

```bash
uv sync --extra ml            # xgboost 3.2.0; macOS needs: brew install libomp
uv run python -m worker export-stroke-traj      # -> worker/.labeling/stroke_traj/features.parquet
uv run python -m worker train-stroke-gbdt --seed 42 [--ppos] [--max-depth 4 --eta 0.1 ...]
```

**Sweep of 2026-07-12** (features export 2026-07-12: 695 rows / 554 confident, 28 core
features + 3 `ppos_` ablation; all commands prefixed `uv run python -m worker
train-stroke-gbdt --seed 42`, device cpu `nthread=1` — **bit-deterministic, replicate
verified identical** — worker git `8410897`, xgboost 3.2.0, ~3 s/run):

| run | extra args | mean LOGO acc | pooled | note |
|---|---|---|---|---|
| baseline | — | 0.523 ± 0.042 | 0.525 | |
| shallow | `--max-depth 3` | 0.519 ± 0.038 | 0.522 | |
| trees+ | `--n-estimators 400` | 0.519 ± 0.037 | 0.518 | |
| slow | `--eta 0.05 --n-estimators 400` | 0.524 ± 0.027 | 0.529 | best core |
| shallow+slow | `--max-depth 3 --eta 0.05 --n-estimators 400` | 0.519 ± 0.025 | 0.523 | |
| no-bagging | `--subsample 1.0 --colsample-bytree 1.0` | 0.504 ± 0.037 | 0.504 | bagging helps |
| +player | `--eta 0.05 --n-estimators 400 --ppos` | **0.536 ± 0.023** | 0.534 | best; seeds 43/44 → 0.526 / 0.523 |

**Result: GATE FAILED (best 0.536 vs the 0.70 bar) — but decisively above BST4** (best
0.400 ± 0.056): +12–14 pp, well beyond the ~5–8 pp fold-noise band, with tighter fold spread
(folds 0.50–0.56) and a fraction of the training cost. Per-class recall (best run, pooled
OOF): serve .65, lift .61, clear .60, drop .59, **net .30, drive .14** (recovered from BST4's
0.00 / 0.00), smash .21 (lags — 12 fps sampling + rally_end flight-time inflation on
rally-ending smashes). Hard pairs: net→drop confusion persists (7/13 pooled net rows → drop);
clear/lift largely separated (10/103 lift → clear). τ per FR-22 stays **unsatisfiable** (knee
fallback at τ = 0, Unsicher abstention 0.00) — like BST4, no usable confidence separation.
Top gain features: `first_of_rally`, `out_du_abs`, `out_end_is_rally_end`, `out_end_dist_net`,
`out_apex_rise`; the sparse court-projection features still earn mid-table gain (the ~37 %
where the floor projection is valid — see the spec's D′/E′ amendment — carry real signal).

**Decision (user rules, spec section):** 0.5–0.7 → stopped for discussion; no pipeline
wiring, no promotion. The measured levers on the table: more labeled sessions (every
comparable system trains on ≥ 10× our rows), net/drive mining from unlabeled hits, near-half
recovery v2, and a 30 fps re-ingest for the smash/drive time resolution.

## 8. Pretrained-weight transfer experiment (2026-07-12)

The spec's design-time rejection of pretraining ("the 4-slot revision breaks the pretrained
2-player input shape") was **audited and found wrong in the load-bearing part**: all
slot-processing weights are shared across slots and BST4 kept upstream's attribute names, so
the released 2-player `BST_CG_AP` state_dict maps onto the 4-slot model key-for-key. With
`--pose-style JnB_bone` (the checkpoint's in_dim=72), **86/94 tensors ≈ 89.1 % of parameters
transfer**; re-initialized by design: the 3 positional embeddings (LEARNED upstream, rows
encode front-anchored hit-to-hit windows at 30 fps — not our centered ±8 @ 12 fps) and the
25-class/300-wide head (ours: 7-class/500-wide). Known transfer-quality risks, accepted and
measured rather than argued: TCN receptive field 0.57 s @ 30 fps vs 1.42 s @ 12 fps (motion
filters run 2.5× too slow), bbox top-left vs center-aligned joint anchoring, broadcast pro
singles vs steep-camera amateur doubles.

### Checkpoint
- Source: the official repo's Google Drive (README "Weights trained on ShuttleSet").
  Serial choice per `results_and_images/result_table.xlsx`, sheet "ShuttleSet_merged (30)":
  **`bst_CG_AP_JnB_bone_merged.pt` = serial_1** (accuracy 0.826, macro-F1 0.803 — best
  macro-F1 of the ten 2D serials; Drive file id `1PhnbUAyqq-sfsj0cI6sSIYHkQcRxV8qb`).
- Store at `worker/.cache/_vendor/BST_weights/shuttleset/` (git-ignored). sha256:
  `10fb0d6c049602e29c690a9eaf5babda606a92fe5e4f74d84a4c251546f4f48e` (downloaded 2026-07-12;
  gdown by file id — the folder download rate-limits after ~47 files).
- Verify before first use (shapes prove it is the 25-class seq-30 JnB_bone variant):
  `tcn_pose.net.0.weight (100,72,5)` · `embedding_tem (1,31,100)` · `mlp_head.mlp.mlp.3.weight (25,400)`.
  The rung-1 audit must report exactly 86 matched / 8 shape-skipped / 0 missing / 0 unexpected.

### Commands (from the worktree `worker/`; data + vendor live in the main checkout)
```bash
export BST_REPO=<main>/worker/.cache/_vendor/BST/stroke_classification
export PYTORCH_ENABLE_MPS_FALLBACK=1
DS=<main>/worker/worker/.labeling/stroke_dataset
CKPT=<main>/worker/.cache/_vendor/BST_weights/shuttleset/bst_CG_AP_JnB_bone_merged.pt
RUN="uv run python -m worker train-stroke --seed 42 --dataset $DS --pose-style JnB_bone"
$RUN --device cpu --epochs 1 --init-weights $CKPT      # rung 1: plumbing + audit
$RUN --lr 3e-4                                         # rung 2: lr-matched scratch control
$RUN --lr 3e-4 --init-weights $CKPT                    # rung 3a: fine-tune all
$RUN --lr 1e-4 --init-weights $CKPT                    # rung 3b
$RUN --lr 3e-4 --init-weights $CKPT --freeze-epochs 150  # rung 4a: frozen-trunk probe
$RUN --lr 3e-4 --init-weights $CKPT --freeze-epochs 30   # rung 4b: warmup-unfreeze
# rung 5: best of 3-4 with --seed 43 / --seed 44
```

### Decision rules (locked before running; primary comparison = best of rung 3/4 vs rung 2)
Δ < +5 pp → pretraining does not help at 12 fps; close the experiment. +5–8 pp but < 0.536 →
real but the GBDT stays the candidate. ≥ 0.536 but < 0.70 → stop and discuss (0.5–0.7 rule).
≥ 0.70 → gate passes, normal promote path; sidecar additionally records the upstream weight
file + sha256 (MIT). Honest prior: realistic band 0.35–0.48, P(beats GBDT) ~10–15 %,
P(≥ 0.70) < 5 % — the durable payoff is converting the spec's reasoned rejection into a
measured fact, and the frozen-probe number prices every future pretraining path (e.g.
30 fps re-ingest + init).

### Results
| rung | config | mean LOGO acc | pooled | notes |
|---|---|---|---|---|
| 2 | JnB_bone scratch lr 3e-4 | 0.374 ± 0.052 | 0.384 | lr-matched control, run `20260712T204812Z` (MPS). Confirms bones hurt from scratch (J_only lr 3e-4 = 0.400) — the checkpoint must first buy back that deficit |
