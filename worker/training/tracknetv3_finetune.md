# TrackNetV3 fine-tune — local MPS runbook (Mac Studio M4 Max)

Domain-fine-tune TrackNetV3 on our own white-wall/overhead footage and **measure** whether it beats the
frozen baseline at raw-pixel detection. This runs **locally on MPS**, intentionally overriding
`specs/30-ml-pipeline.md`'s "cloud NVIDIA only for fine-tuning" line (the fine-tune is small; the whole
label→train→benchmark loop stays on the Mac against the real frame cache + weights).

> **Data gate (Slice 2b — do this first or nothing here runs).** Need **≥3–4 clips across ≥2 sessions /
> lighting** (over-exposure included): ≥2 train, 1 val, 1 held-out test, all distinct sessions, so the
> metric measures generalization not memorization. Only ONE ~30s clip exists today. `train.py` *requires*
> a populated `val` split. Adjacent 12 fps frames are near-duplicates → every split is a whole clip/rally,
> never a random-frame split. 12 fps undersamples a smash (1–3 blurred frames): either re-ingest label
> clips at a higher `analysis_fps` or scope the fine-tune to non-smash rallies and mark smashes
> `visibility=0` — record whichever as an honest limit.

All commands run from `worker/`. The vendored repo lives at `worker/.cache/_vendor/TrackNetV3` (git-ignored)
and weights at `models/weights/shuttle/ckpts/` (git-ignored). The patches that make MPS work live at
`worker/training/tracknetv3_mps_train.patch` and `tracknetv3_mps_test.patch` (tracked).

---

## 0. Prerequisites
- Clone the vendored repo if absent:
  `git clone https://github.com/qaz812345/TrackNetV3 .cache/_vendor/TrackNetV3`
- The baseline weights `models/weights/shuttle/ckpts/TrackNet_best.pt` (the one
  `worker/shuttle/tracknetv3.py:30` loads for inference) must be present.
- Vendored-repo Python deps (until `uv sync` picks up the ml-extra on this machine, pass per-run):
  `uv run --with pandas --with parse --with omegaconf --with tensorboard --with pycocotools python …`

## 1. Patch CUDA → MPS (prerequisite, not optional)
Upstream `train.py`/`test.py` hard-code `.cuda()` with no `--device` flag; `PYTORCH_ENABLE_MPS_FALLBACK=1`
only redirects unimplemented ops, **not** explicit `.cuda()`, which hard-fails on a CUDA-less Mac. Apply the
shipped patches (idempotent — skip if already applied):

```bash
cd .cache/_vendor/TrackNetV3
patch -p0 --forward < ../../../training/tracknetv3_mps_train.patch
patch -p0 --forward < ../../../training/tracknetv3_mps_test.patch
cd ../../..
```

What the patch does (scope = **TrackNet train + the in-training eval only**):
- Routes the TrackNet path through a global `DEVICE` (`pick_device()` → mps→cpu, mirroring
  `worker/shuttle/tracknetv3.py:83-100`): `.cuda()`→`.to(DEVICE)` at the model, the train batch
  (`train_tracknet`), `get_random_mask`, and `eval_tracknet` (`test.py`).
- Guards `torch.cuda.manual_seed` / `cudnn.deterministic` behind `torch.cuda.is_available()`.
- `pin_memory=True → False` on both loaders (MPS).
- Adds `--device {auto,mps,cpu,cuda}` and a one-shot **MPS NaN probe → CPU fallback** before training.
- Adds `--init_ckpt` (load weights + **fresh optimizer**) — see §2.
- **Not patched (out of scope):** the InpaintNet path (`train_inpaintnet`) and `test.py`'s standalone
  full-rally / COCO CLI still use `.cuda()`. We don't train InpaintNet, and we re-benchmark via our own
  `eval_shuttle.py` (already MPS via `tracknetv3.py`), **not** the vendored `test.py` CLI.

Verify: `grep -n "\.cuda()" .cache/_vendor/TrackNetV3/train.py` shows only the InpaintNet sites (none on
the TrackNet path).

## 2. Init from baseline weights — `--init_ckpt`, NOT `--resume_training`
`--resume_training` loads `{save_dir}/{model}_cur.pt` and its `param_dict` **overrides your LR/epochs** — it
is a resume-an-interrupted-run flag, not an init-from-checkpoint flag. Use the patched `--init_ckpt` instead:
it loads the state_dict after model creation and keeps a **fresh optimizer**, so your CLI LR/epochs stand.

## 3. Export the dataset (§4 of the spec)
Export each labeled clip into its split (whole clip → one match → one rally). `train.py` reads `./data`
relative to the cwd, so export to `.cache/_vendor/TrackNetV3/data`:

```bash
# ≥2 train clips, 1 val clip, 1 held-out test clip (all distinct sessions):
uv run python -m worker export-shuttle-dataset --video-id <CLIP_A> --split train --match 1 --out .cache/_vendor/TrackNetV3/data
uv run python -m worker export-shuttle-dataset --video-id <CLIP_B> --split train --match 2 --out .cache/_vendor/TrackNetV3/data
uv run python -m worker export-shuttle-dataset --video-id <CLIP_C> --split val   --match 1 --out .cache/_vendor/TrackNetV3/data
uv run python -m worker export-shuttle-dataset --video-id <CLIP_D> --split test  --match 1 --out .cache/_vendor/TrackNetV3/data
```

The exporter emits the SAME airspace-masked 512×288 crops inference sees, writes `Frame,Visibility,X,Y`
CSVs (`corrected_csv/` for test), generates `median.npz` per match (the `bg_mode=concat` white-wall lever),
and deletes the dataset's stale `img_config_*.npz` / `data_l*.npz` caches first. It refuses a rally shorter
than 8 frames (`seq_len`) — a too-short rally builds zero training windows. **Re-export after any relabel.**

## 4. 2-batch CPU NaN-probe (run before any MPS run)
```bash
cd .cache/_vendor/TrackNetV3
uv run --with pandas --with parse --with omegaconf --with tensorboard --with pycocotools python train.py \
  --model_name TrackNet --bg_mode concat --seq_len 8 --epochs 1 --batch_size 2 --device cpu \
  --save_dir _probe --verbose
cd ../../..
```
Expect it to reach `Done......` with finite (non-NaN) loss. (Validated on this machine 2026-06-20:
finite `max_val_acc`, no NaN weights.) Delete `_probe/` after.

## 5. Fine-tune on MPS (low-data regime)
```bash
cd .cache/_vendor/TrackNetV3
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with pandas --with parse --with omegaconf --with tensorboard --with pycocotools python train.py \
  --model_name TrackNet --bg_mode concat --seq_len 8 \
  --init_ckpt ../../../models/weights/shuttle/ckpts/TrackNet_baseline.pt \
  --learning_rate 1e-4 --epochs 10 --batch_size 2 --alpha 0.5 \
  --device mps --save_dir exp_ft --verbose
cd ../../..
```
- **LR `1e-4`** (try `1e-5`) vs the `1e-3` default; **5–15 epochs**; **`--alpha 0.5`** (mixup — the paper's
  headline lever, 87.72→97.51% acc); small **`--batch_size` 2–4**; keep `WBCELoss`. fp32.
- `--device mps` (or `auto`); the patched NaN probe falls back to CPU automatically if MPS produces NaNs.

## 6. Preserve the baseline; promote only on a measured win
`train.py` saves its winner to `{save_dir}/TrackNet_best.pt`. Using a **fresh `--save_dir exp_ft` keeps the
inference baseline untouched**. Before anything:

```bash
cp models/weights/shuttle/ckpts/TrackNet_best.pt models/weights/shuttle/ckpts/TrackNet_baseline.pt
```
After training, point inference at the fine-tune via env (don't overwrite the baseline yet):
```bash
cp .cache/_vendor/TrackNetV3/exp_ft/TrackNet_best.pt models/weights/shuttle/ckpts/TrackNet_ft_<clip>.pt
export TRACKNETV3_MODEL=$PWD/models/weights/shuttle/ckpts/TrackNet_ft_<clip>.pt
```
**Promote** (`cp TrackNet_ft_<clip>.pt TrackNet_best.pt`) **only after** §7 proves it beats baseline.
(`TrackNet_best.pt` is git-ignored — no VCS recovery, hence the explicit baseline copy.)

## 7. Re-benchmark — the deliverable
On the **held-out test clip**, score raw-pixel precision/recall in the pinned 512×288 space (tol=4px):
```bash
# baseline (uses the default TrackNet_best.pt):
uv run python -m worker.eval_shuttle --video-id <CLIP_D> --tracker tracknetv3 --labels
# fine-tune (TRACKNETV3_MODEL pointed at the ft weights):
TRACKNETV3_MODEL=$PWD/models/weights/shuttle/ckpts/TrackNet_ft_<clip>.pt \
  uv run python -m worker.eval_shuttle --video-id <CLIP_D> --tracker tracknetv3 --labels
# WASB stays frozen as the before/after reference:
uv run python -m worker.eval_shuttle --video-id <CLIP_D> --compare --labels
```
Record P/R/F1 + median px error **with `n` (visible frames)** in `specs/features/shuttle-tracking.md`. A
meaningful win is **recall +~10pp at tol=4px**; a few-point delta on a small `n` is noise. **The metric is
void / downgraded-to-eyeball if no second independent clip exists.** Report a non-improvement honestly — a
plausible-but-wrong metric is worse than an absent one.
```
