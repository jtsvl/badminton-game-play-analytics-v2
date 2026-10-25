# Feature — TrackNetV5 shuttle backend (ICIP 2026) and the V3-vs-V5 A/B

**Status: BUILT, NOT MEASURED (2026-10-25).** Third `SHUTTLE_TRACKER` backend (`tracknetv5`), checkpoint
converter, fine-tune patches/runbook and the A/B tooling exist and are unit-tested; every number below is
*not run yet* until the Mac run. **The production default stays `tracknetv3`** (the own-footage fine-tune,
F1 0.953 in-sample) — promotion only by the pre-registered rule in § A/B. Branch `claude/new-session-kuz5k5`.
Runbook: `worker/training/tracknetv5_finetune.md`.

## Goal
Add the ICIP-2026 TrackNetV5 (Chang, Wang, Huang; `github.com/gino6178/TrackNetV5`, commit
`9116161772183e841bd1b51d6747c70f269b8b1c`) as a swappable backend, A/B it against the fine-tuned
TrackNetV3 (and frozen WASB) **at 30 fps** on an in-sample clip and a cross-session clip, and promote it only
if it wins where it matters (cross-session F1) without a downstream regression.

**Honest prior.** Upstream reports F1 98.363 (Lite, thr 0.35) / 98.331 (full, thr 0.30) vs 97.59 for the
released V3 weights on the TrackNetV3 broadcast test split (10,836 frames, 4 px) — ~1 F1 point,
self-reported; an independent re-implementation (AnInsomniacy/tracknet-series-pytorch, 30 epochs) scored
V5 *worst* of V2–V5 (0.773 vs V3+rectifier 0.809) and warns evaluation choices move F1 by >15 points. Our
white-wall / overhead footage is nothing like broadcast. 30 fps (`analysis-30fps.md`) is the bigger lever.

## Locked decisions (user, 2026-10-25)
1. **The ICIP V5 only.** The other "TrackNetV5" (codelancera SDK, arXiv 2512.02789) is proprietary with
   private weights, a different model — excluded.
2. **Licence.** The repo has **no LICENSE file**; the user confirmed a licence grant (2026-10-25). Recorded
   here so locked decision #6 (no deps without a licence grant) stays coherent; no further documentation.
3. **Pin** `9116161772183e841bd1b51d6747c70f269b8b1c` (`TRACKNETV5_SHA`); vendored at
   `worker/.cache/_vendor/TrackNetV5` (`TRACKNETV5_REPO`), git-ignored.
4. **Default unchanged** until the A/B passes the rule below; A/B at 30 fps only.
5. **Replicate the pretrained input exactly** (incl. the uint8 wrap below); the "fixed" variant is an
   experiment flag, never a silent change.

## Model contract (read from the pinned source, not the paper)
- `models/tracknetv5.py`: `TrackNetV5(in_dim, out_dim, alpha)`; `in_channels = in_dim // out_dim`;
  R(2+1)D stem (`Conv3d` + `BatchNorm3d`), gated residual motion prompt, 3-level MBConv U-Net with
  spatial-channel gates. **Eval mode returns the sigmoid heatmaps `(B, 5, 288, 512)`; train mode returns
  `(out, [aux1, aux2])`** → the backend calls `.eval()`.
- Variants: **Lite** `alpha=0.5` (`releases/TrackNetV5_Lite_best.pt`, 2.58 M params per the project page),
  **full** `alpha=1.0` (`releases/TrackNetV5_best.pt`, 10.09 M). Both trained from scratch on train+val of
  the TrackNetV3 dataset, seq_len 5, Adam 1e-3, 30 epochs, mixup 0.5, bf16 AMP, deep-supervision 0.25.
- **Input** (`dataset.py`, `bg_mode='subtract_concat'`): 5 frames × `[R, G, B, D]` = 20 channels at 512×288,
  all `/255`. `D = np.sum(np.absolute(frame − median), 2).astype('uint8')` — the RGB sum reaches 765 while
  uint8 stops at 255, and the cast **wraps modulo 256** (both x86 and arm64 convert through a 32-bit int
  before the byte store), so a strongly changed pixel can read as a small difference. The released weights
  were trained with that; `TRACKNETV5_DIFF=wrap` (default) replicates it bit-exactly, `clip` saturates
  (the measurable experiment).
- **Median**: inference = one float64 median per video (`Video_IterableDataset.__gen_median__`); training =
  per rally/match `median.npz` (float64). We reuse the V3 backend's per-clip stride-subsampled median
  (shared `shuttle/base.clip_median`, ≤700 frames). A per-rally median is a later experiment.
- **Windows**: non-overlapping 5-frame windows, last padded with the final frame (`padding=True`) — the
  same grouping WASB/V3 use here, so `SHUTTLE_CHUNK_FRAMES=240` (a multiple of 5) chunks identically.
- **Decode** (`test.py`): binarise at a threshold, largest-area blob's bbox centre; published sweep peaks
  at **0.35 (Lite) / 0.30 (full)**, 0.5 is the protocol default. Shared `shuttle/base.decode_heatmap` (V3
  keeps 0.5). The stage's `visibility` gate follows the backend's threshold unless `SHUTTLE_CONF_THR` is set
  (`pipeline/shuttle.resolve_conf_thr`) — otherwise the 0.5 gate would silently undo the sweep.
- **Checkpoints**: `{epoch, model_state_dict, val_f1 (numpy scalar), args (dict)}`; the only non-torch pickle
  globals are `numpy.dtype` + `numpy.core.multiarray.scalar` → loaded with `weights_only=True` inside
  `torch.serialization.safe_globals([...])`; **never `weights_only=False`** (upstream does, running arbitrary
  pickled code). Our re-saved files need no allow-list at all.
- **CUDA-only upstream code**: `train.py`/`test.py` hard-code `.cuda()` + bf16 autocast; there is no
  `predict.py`. Inference is our own (`shuttle/tracknetv5.py`), training uses the shipped MPS patches.
- **Module-name hazard**: the repo's `dataset`/`test`/`utils`/`models` collide with V3's `model`/`dataset`/
  `utils`/`test` and WASB's `models` — `eval_shuttle --compare` runs all backends in one process. Only
  `models/tracknetv5.py` (torch-only) is loaded, **by file path under a private module name**; the input is
  hand-rolled and pinned by tests (`tests/test_tracknetv5.py`) plus an upstream-loader parity check.

## Weights (git-ignored; convert-tracknetv5 writes file + JSON sidecar)
| file | role | SHA-256 |
|---|---|---|
| upstream `releases/TrackNetV5_Lite_best.pt` | source, 10,221,090 B | `c7ca025e5117d292f9efa60b3132d401434ca1afe26aba9353133ad2823e7a07` |
| upstream `releases/TrackNetV5_best.pt` | source, 40,317,929 B | `cf9025285cdcd3089229a45a19ed8ed4075f6a4f459cbf99e450b1f968f7198d` |
| `models/weights/shuttle/tracknetv5_lite.pt` (+`.json`) | converted Lite, loaded by default | *filled by the Mac run (sidecar `sha256`)* |
| `models/weights/shuttle/tracknetv5.pt` (+`.json`) | converted full | *filled by the Mac run* |
| `models/weights/shuttle/tracknetv5_lite_ft_<clip>.pt` | fine-tune(s), one per run | *filled by the Mac run* |
Converted file = `{model_state_dict, config{backend, variant, alpha, seq_len 5, bg_mode, in_dim 20, out_dim 5,
input_wh, thr}, source{repo, commit, vendor_commit, file, sha256, sha256_expected, val_f1, upstream_args}}`
(primitives only, `stroke/model.py` convention). `--probe` records `probe.devices.{cpu,mps}` (finite, mean,
max) and `max_abs_diff_cpu_mps` in the sidecar — the CLAUDE.md "validate new models once against CPU" rule;
paste it here: *not run yet*.

## Integration — as built
- `worker/worker/shuttle/tracknetv5.py`: `TrackNetV5Tracker` (`name = tracknetv5-<variant>@<sha8>`,
  `model_size (512, 288)`, `window 5`, `conf_thr`, `begin_clip` → per-clip median, `predict`), pure
  `background_diff` / `build_input` / `windows`, `load_vendored_model_class`, `load_checkpoint`,
  `convert`/`convert_cli`, `parity_check`. Device via `base.probe_device` (MPS NaN probe → CPU, fp32,
  `PYTORCH_ENABLE_MPS_FALLBACK=1`).
- `shuttle/base.py` shared helpers (factored out of V3, byte-identical for it): `resize_rgb`,
  `median_of_frames`, `clip_median`, `largest_blob_bbox`, `decode_heatmap`, `checkpoint_tag`, `vendor_commit`.
- Registry `shuttle/__init__.py` + `eval_shuttle.TRACKERS` (`--compare` runs wasb · tracknetv3 · tracknetv5
  · fake, keeps `shuttle.<tracker>.parquet` per backend and restores the default's `shuttle.parquet` — the
  Fake tracker's output used to be left on disk). CLI `python -m worker convert-tracknetv5`.
- **Provenance**: `shuttle.parquet.tracker` = `tracknetv5-lite@<sha8>` (the checkpoint file's SHA-256 prefix),
  so Lite/full/fine-tunes stay distinguishable (`40-worker-contract.md`). V3 keeps the bare `tracknetv3`.
- Env: `TRACKNETV5_REPO`, `TRACKNETV5_VARIANT` (lite|full, default lite), `TRACKNETV5_MODEL` (default
  `models/weights/shuttle/tracknetv5_lite.pt` / `tracknetv5.pt`), `TRACKNETV5_THR` (default 0.35 / 0.30),
  `TRACKNETV5_BATCH` (4), `TRACKNETV5_DIFF` (wrap|clip), `TRACKNETV5_MEDIAN_SAMPLES` (700).
- Fine-tune path: `export-shuttle-dataset --seq-len 5 --frames A:B` (float64 `median.npz`),
  `worker/training/tracknetv5_mps_train.patch` + `tracknetv5_mps_test.patch` (DEVICE routing, fp32 off
  CUDA, `weights_only=True` loads, MPS NaN probe; `--trim_*` must stay off — they assume 1280×720 labels),
  recipe = the V3 fine-tune's regime (`--init_from` the Lite release, Adam 1e-4, 5–15 epochs, batch 2–4,
  mixup 0.5). The V3 fine-tune's rally split was never recorded; **this time the spans are recorded here**.

## A/B protocol (pre-registered; all at 30 fps, `WORKER_FRAME_CACHE_DIR=worker/.cache-30fps`)
Contenders: WASB (frozen), **TrackNetV3 fine-tuned (production)**, V5-Lite zero-shot, V5-full zero-shot,
V5-Lite fine-tuned on game1 (train/val spans recorded below). Tool: `eval_shuttle --compare --labels`
(4 px tolerance in the pinned 512×288 space, `n` = visible frames) per clip; downstream per tracker with
`eval_rally --shuttle-parquet … --landings --hits`. Two label sets:
- **in-sample (game1, `7be72b61…`, 30 fps, 424 visible frames, span 173:4091)** — V3-ft trained on rallies
  of this clip, so its number is in-sample; V5 zero-shot is clean here; V5-ft's held-out span is recorded.
- **cross-session (one Session-2 clip, to be labelled by the user with `verify --shuttle` in the 30 fps
  cache, ≥ 400 visible frames)** — the gate. No Session-2 shuttle labels exist today.

**Decision rule (set before the numbers).** Promote V5 (file at the default path + `SHUTTLE_TRACKER`
default flip, named copies kept — `40-worker-contract.md` convention) **only if**, on the cross-session clip,
its F1 ≥ 0.80 (the existing gate, `30-ml-pipeline.md` stage 5) **and** exceeds V3-ft's F1 by ≥ 0.02 with
n ≥ 400 **and** neither rally boundary F1 (`eval_rally`, ±0.5 s) nor landings located drop against V3-ft's
detections. In-sample wins alone promote nothing. The user decides from the table; a non-win is recorded.

### Results — in-sample, game1 (*not run yet*)
Run: `eval_shuttle --compare --labels --frames 173:4091` on `7be72b61…` (30 fps cache) + the two extra
`--tracker tracknetv5` runs (full; fine-tuned).

| backend | precision | recall | F1 | median px err | n (visible) | tracker tag |
|---|---|---|---|---|---|---|
| WASB (frozen) | *not run yet* | | | | | `wasb` |
| TrackNetV3 fine-tuned (production, in-sample) | *not run yet* (0.926 / 0.981 / 0.953 / 0.49 / 424 on 2026-07-02) | | | | | `tracknetv3` |
| TrackNetV5-Lite zero-shot | *not run yet* | | | | | `tracknetv5-lite@…` |
| TrackNetV5-full zero-shot | *not run yet* | | | | | `tracknetv5-full@…` |
| TrackNetV5-Lite fine-tuned (train span …, val …, test …) | *not run yet* | | | | | `tracknetv5-lite@…` |

### Results — cross-session, Session-2 clip `<video id>` (*not run yet; clip not yet labelled*)
Same table; plus downstream per tracker:

| backend | rally boundary F1 (±0.5 s) | false rallies / splits / missed / merges | landings located N/M | hit recall @±0.25 s |
|---|---|---|---|---|
| TrackNetV3 fine-tuned | *not run yet* | | | |
| TrackNetV5-Lite (best of zero-shot / fine-tuned) | *not run yet* | | | |

### Fine-tune runs (every run gets a row — command + seed + result)
| date | init | split (frames) | epochs / lr / batch / mixup | best val F1 (epoch) | converted file | command |
|---|---|---|---|---|---|---|
| *not run yet* | | | | | | |

## Honest limits
- Cross-session evidence requires a label set that does not exist yet; until then every V5 number is
  in-sample or zero-shot on a clip V3 was fine-tuned on — favourable to V3-ft, not a fair promotion basis.
- The uint8 wrap in the difference channel is a quirk the weights were trained with; "fixing" it may help or
  hurt — only `TRACKNETV5_DIFF=clip` measured on labels can say.
- `Conv3d`/`BatchNorm3d` on MPS: the probe falls back to CPU on NaNs; `max_abs_diff_cpu_mps` in the sidecar
  is the parity evidence (not run yet).
- Per-rally medians (training) vs per-clip median (our inference) is a domain shift we accept for now.
- Runtime/memory of the full model on 4K court crops is unmeasured; the chunked stage bounds crop RAM, not
  model RAM.

## Non-goals
Per-rally medians, InpaintNet-style rectification, CoreML export, retraining from scratch, the codelancera
SDK, and the report's other spec corrections (raised separately).

## Verification
- `cd worker && uv run pytest`: `tests/test_tracknetv5.py` (variants/thresholds, uint8 wrap + clip, channel
  layout, windows, decode incl. NaN safety, provenance helpers, file-path import leaves `sys.path`/`models`
  untouched, conversion payload primitives, registry), `test_shuttle.py` (default lock, `resolve_conf_thr`),
  `test_dataset_export.py` (seq_len/frames/float64 median), `test_eval_shuttle_range.py` (compare copies).
- **Container check against the real upstream files — RUN 2026-10-25** (Linux, torch 2.14.1 CPU, numpy 2.5.4;
  the two `releases/*.pt` downloaded at the pin, SHA-256 as in the Weights table):
  - `load_checkpoint` with `weights_only=True` + the numpy allow-list succeeds for both files (keys `args,
    epoch, model_state_dict, val_f1`; epoch 10; `val_f1` 0.98094 Lite / 0.98107 full; `args.seq_len` 5,
    `bg_mode` subtract_concat, `model_alpha` 0.5 / 1.0). Without the `(obj, "numpy.core.multiarray.scalar")`
    tuple form the load fails under numpy ≥ 2 — that is why the helper registers the pickled name.
  - `TrackNetV5(20, 5, α)` built from `models/tracknetv5.py` by file path, `load_state_dict(strict=True)` OK;
    params **2,526,228** (Lite) / **10,041,108** (full); eval forward on a random `(1, 20, 288, 512)` →
    `(1, 5, 288, 512)`, finite, in [0, 1].
  - `convert` → file with keys `{config, model_state_dict, source}` re-loads with a plain
    `weights_only=True` (no allow-list); `source.sha256 == sha256_expected`; CPU probe finite (MPS n/a here).
    Container sidecar SHA-256 `854eb080…` — the Mac's torch 2.12 serialisation will differ; record the Mac's.
  - **Input parity:** `build_input` vs upstream `Shuttlecock_Trajectory_Dataset(frame_arr, 'subtract_concat',
    median)` over 3 windows of random 512×288 frames: identical window indices, max |Δ| **2.97e-08** (float32
    rounding); the 300-sum pixel reads **44** in both (the uint8 wrap is real upstream).
- Mac: §2–§7 of the runbook; paste commands + numbers into the tables above.
