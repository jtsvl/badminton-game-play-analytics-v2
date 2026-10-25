# TrackNetV5 — vendoring, conversion, fine-tune and A/B runbook (Mac Studio M4 Max, MPS)

Adds the ICIP-2026 TrackNetV5 (gino6178/TrackNetV5, commit `9116161`) as the third shuttle backend and
A/Bs it against the promoted TrackNetV3 fine-tune. Spec: `specs/features/shuttle-tracknetv5.md` (the
decision rule and the result tables live there — **no metric enters the spec without the command that
produced it**). Mirrors `tracknetv3_finetune.md`; everything runs from `worker/`.

> **Honest expectation first.** V5's published gain over V3 is ~1 F1 point on broadcast footage,
> self-reported; an independent re-implementation found none. 30 fps (`analysis-30fps.md`) is the bigger
> lever. Report what you measure. The default stays `tracknetv3` until the spec's rule says otherwise.

---

## 0. Prerequisites
- `uv sync --extra ml --extra dev` (torch/numpy/opencv/pillow + the vendored repo's `parse`, `pandas`,
  `pycocotools`, `tqdm` are all in the `ml` extra; otherwise pass them per run with `uv run --with …`).
- A 30 fps frame cache for every clip you score (`WORKER_FRAME_CACHE_DIR=$PWD/../worker/.cache-30fps`):
  game1 (`7be72b61…`, labelled) and the Session-2 clip you label for the cross-session gate.
- The fine-tuned V3 (`models/weights/shuttle/ckpts/TrackNet_best.pt`) and WASB weights present (baselines).

## 1. Vendor the repo at the pin (read the code before importing it — it is an external download)
```bash
git clone https://github.com/gino6178/TrackNetV5 .cache/_vendor/TrackNetV5
git -C .cache/_vendor/TrackNetV5 checkout 9116161772183e841bd1b51d6747c70f269b8b1c
git -C .cache/_vendor/TrackNetV5 rev-parse HEAD      # must print 9116161772183e841bd1b51d6747c70f269b8b1c
```
Override the location with `TRACKNETV5_REPO`. The repo has **no LICENSE file**; the user confirmed a
licence grant (2026-10-25). Only `models/tracknetv5.py` is imported by the worker (by file path — the
repo's `dataset`/`test`/`utils`/`models` names collide with the V3 and WASB vendored modules).

## 2. Convert the released checkpoints (never `weights_only=False`)
Upstream loads its `releases/*.pt` with `weights_only=False` (arbitrary pickled code). The converter loads
them with `weights_only=True` (allow-listing only the numpy scalar/dtype that `val_f1` pickles), verifies
the SHA-256 against the spec, re-saves a plain `{model_state_dict, config, source}` file plus a JSON
sidecar, and with `--probe` runs the CPU-vs-MPS NaN/parity check:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker convert-tracknetv5 --variant lite --probe
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker convert-tracknetv5 --variant full --probe
```
Outputs: `models/weights/shuttle/tracknetv5_lite.pt` (+ `.json`) and `tracknetv5.pt` (+ `.json`). Paste
the sidecars' `sha256`, `probe.devices` and `probe.max_abs_diff_cpu_mps` into the spec's Weights table.
Expected upstream SHA-256 (recorded in the spec): Lite `c7ca025e…7a07`, full `cf902528…198d` — a mismatch
only warns (a newer upstream file or a fine-tune) but must be explained in the spec.

## 3. Smoke the backend on a labelled clip (zero-shot, both variants)
```bash
export WORKER_FRAME_CACHE_DIR=$PWD/../worker/.cache-30fps PYTORCH_ENABLE_MPS_FALLBACK=1
V=7be72b61-…   # game1
uv run python -m worker.eval_shuttle --video-id $V --tracker tracknetv5 --labels --frames 173:4091
TRACKNETV5_VARIANT=full uv run python -m worker.eval_shuttle --video-id $V --tracker tracknetv5 --labels --frames 173:4091
```
The parquet's `tracker` column reads `tracknetv5-lite@<sha8>` / `tracknetv5-full@<sha8>`. The stage gates
`visibility` at the backend's published threshold (0.35 Lite / 0.30 full) unless `SHUTTLE_CONF_THR` is
set; `TRACKNETV5_THR` overrides the backend value. `TRACKNETV5_DIFF=clip` is the one input experiment
(saturate instead of the upstream uint8 wrap) — measure before believing it helps.

## 4. Patch CUDA → MPS (prerequisite for training, not optional)
`train.py` hard-codes `.cuda()`, bf16 `autocast(device_type='cuda')` and a `GradScaler`; `test.py`'s
`eval_tracknet` has `.cuda()` too. Apply the shipped patches (idempotent — `--forward` skips an applied one):
```bash
cd .cache/_vendor/TrackNetV5
patch -p0 --forward < ../../../training/tracknetv5_mps_train.patch
patch -p0 --forward < ../../../training/tracknetv5_mps_test.patch
cd ../../..
```
What the patch does (scope = `train.py` + the in-training `eval_tracknet`):
- global `DEVICE` via `pick_device()` (mps→cuda→cpu) + `--device {auto,mps,cpu,cuda}`; `.cuda()`→`.to(DEVICE)`
  for the model, the train batches, the teacher and the eval loop (`param_dict['device']`);
- autocast **enabled only on CUDA** (fp32 on MPS/CPU), `GradScaler` off CUDA, `pin_memory` only on CUDA,
  CUDA seed/cudnn guarded;
- `--init_from` / `--teacher_weights` loaded with `weights_only=True` (same numpy allow-list as the worker);
- a one-shot **MPS NaN probe → CPU fallback** on the first batch (train-mode output is a tuple; the probe
  checks the main head).
- **Not patched:** `test.py`'s standalone CLI (full-rally/COCO/InpaintNet paths) and `evaluate.py` (which
  also needs `drop_frame.json`, a TrackNetV3-dataset artefact we do not have). We score with
  `eval_shuttle`, never with the vendored CLIs.
Verify: `grep -n "\.cuda(" .cache/_vendor/TrackNetV5/train.py | grep -v PATCH` prints nothing.

## 5. Export the dataset (V5 window = 5)
`train.py` defaults to `--data_dir data_288x512` relative to its cwd; our export already writes 512×288
PNGs, so upstream's `preprocess_images.py` is **not** run (the loader generates `img_config_*.npz` itself).
Use `--frames A:B` for an explicit within-clip rally split and record the spans in the spec:
```bash
D=.cache/_vendor/TrackNetV5/data_288x512
uv run python -m worker export-shuttle-dataset --video-id $V --split train --match 1 --seq-len 5 --frames 173:2600 --out $D
uv run python -m worker export-shuttle-dataset --video-id $V --split val   --match 1 --seq-len 5 --frames 2601:3300 --out $D
uv run python -m worker export-shuttle-dataset --video-id $V --split test  --match 1 --seq-len 5 --frames 3301:4091 --out $D
```
(Spans are placeholders — cut at rally boundaries from `shuttle_labels.json`, never mid-rally; the test
span is in-sample for V3-ft anyway. The cross-session Session-2 clip is scored separately, §7.) The
exporter writes `median.npz` as float64 (V5's `subtract_concat` subtracts it; a uint8 median would wrap).
**Re-export after any relabel.**

## 6. Fine-tune (same recipe shape as the V3 fine-tune)
2-batch CPU NaN probe first:
```bash
cd .cache/_vendor/TrackNetV5
uv run python train.py --model_name TrackNetV5 --model_alpha 0.5 --seq_len 5 --bg_mode subtract_concat \
  --data_dir data_288x512 --train_splits train --eval_split val --epochs 1 --batch_size 2 --device cpu \
  --init_from releases/TrackNetV5_Lite_best.pt --optim Adam --learning_rate 1e-4 --lr_scheduler '' \
  --mixup_alpha 0.5 --ds_weight 0.25 --seed 26 --save_dir _probe
```
then the MPS run (fp32; the probe falls back to CPU on NaNs):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py --model_name TrackNetV5 --model_alpha 0.5 --seq_len 5 \
  --bg_mode subtract_concat --data_dir data_288x512 --train_splits train --eval_split val \
  --init_from releases/TrackNetV5_Lite_best.pt --optim Adam --learning_rate 1e-4 --lr_scheduler '' \
  --epochs 10 --batch_size 4 --mixup_alpha 0.5 --ds_weight 0.25 --seed 26 --device mps --save_dir runs/ft_game1
cd ../../..
```
- Do **not** pass `--trim_train/--trim_eval`: they rescale labels assuming 1280×720 sources; ours are
  already 512×288. `--amp_dtype` is ignored off CUDA (fp32).
- Mirror the V3 recipe's regime: LR 1e-4 (try 1e-5), 5–15 epochs, batch 2–4, mixup 0.5. Record every run
  (command + seed + `metrics.csv` best epoch) in the spec — no exceptions.
- Convert the winner so the backend loads it strictly and the parquet names it:
  `uv run python -m worker convert-tracknetv5 --variant lite --src .cache/_vendor/TrackNetV5/runs/ft_game1/TrackNetV5_best.pt --out models/weights/shuttle/tracknetv5_lite_ft_game1.pt --probe`
  (the SHA-256 "mismatch" warning is expected for a fine-tune — note it in the spec).

## 7. A/B — all at 30 fps, both label sets
```bash
export WORKER_FRAME_CACHE_DIR=$PWD/../worker/.cache-30fps PYTORCH_ENABLE_MPS_FALLBACK=1
for VID in $V $S2; do   # S2 = the Session-2 clip labelled with `verify --shuttle` in the 30 fps cache
  uv run python -m worker.eval_shuttle --video-id $VID --compare --labels                 # wasb · v3-ft · v5-lite · fake
  TRACKNETV5_VARIANT=full uv run python -m worker.eval_shuttle --video-id $VID --tracker tracknetv5 --labels
  TRACKNETV5_MODEL=$PWD/../models/weights/shuttle/tracknetv5_lite_ft_game1.pt \
    uv run python -m worker.eval_shuttle --video-id $VID --tracker tracknetv5 --labels
done
# downstream, per tracker (the --compare run keeps shuttle.<tracker>.parquet copies next to shuttle.parquet):
uv run python -m worker.eval_rally --slug <S2 slug> --landings --hits \
  --shuttle-parquet $WORKER_FRAME_CACHE_DIR/$S2/shuttle.tracknetv5-lite@<sha8>.parquet
```
Label the Session-2 clip first (~400+ visible frames, white-wall + over-exposure rallies first):
`WORKER_FRAME_CACHE_DIR=$PWD/../worker/.cache-30fps uv run python -m worker verify --shuttle <S2 video id>`.
Put every row in the spec's tables, labelled **in-sample (game1)** vs **cross-session (Session-2)**.

## 8. Promote only on the spec's pre-registered rule
Promotion = the converted weights at the default path (`models/weights/shuttle/tracknetv5_lite.pt`, named
copy kept) **and** `SHUTTLE_TRACKER` default flipped to `tracknetv5` (`shuttle/__init__.py`,
`eval_shuttle.py`, the `test_shuttle.py` default lock) — the `40-worker-contract.md` convention. The user
decides from the numbers; a non-win is recorded as such.
