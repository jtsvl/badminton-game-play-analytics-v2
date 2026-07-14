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
