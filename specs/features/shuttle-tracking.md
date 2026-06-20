# Feature — Shuttle tracking (pipeline stage 5), slice 1

Status: **slice 1 built + benchmarked** (feat/p1-shuttle). FR-15/16/17. See `specs/30-ml-pipeline.md`
stage 5 and `specs/40-worker-contract.md`.

## Goal
Detect the shuttle per frame **on the target court only** → `shuttle.parquet (t,u,v,visibility)` in court
coords → flight-path overlay (`track_keyframes` subject=`shuttle`). Later stages (rally_hit / stroke /
speed) build on this. Slice 1 is the **thin baseline**: run a real tracker, see how bad the white-wall /
multi-court / tiny-shuttle problem is **before** building mitigations (decision: option A, run-both).

## Why a baseline first
The hall holds 3–4 simultaneous courts (several shuttles), the upper third of the frame is bright white
wall + skylights (locked decision #5), and the shuttle is a few pixels in a wide 4K DJI shot. Rather than
guess at mitigations, slice 1 measures the failure on real footage (`f35d0984`) and lets the numbers
direct the next slice.

## Architecture
```
frames (12fps cache) ─┐
calibration H + mask ──┤→ court crop (geometry.court_crop_box) → ShuttleTracker.predict(crops)
                       │        ↑ isolate target court + restore shuttle resolution   ↓ peaks (model px)
                       │   shuttle.track.build_shuttle_track: map crop→frame→court (apply_homography),
                       │        drop <conf_thr (real dropout), reject isolated spikes
                       └→ shuttle.parquet (t,u,v,visibility + frame,px,py,conf,tracker)
                                              │
              overlay: split_on_gaps → resample_10hz per segment → subject='shuttle' [{t,u,v}]
```

### Modules
- `worker/worker/geometry.py` (pure, TDD'd): `court_airspace_polygon` (target-court airspace = ground quad
  raised up for the shuttle's flight zone, bounded by its own sidelines → excludes the side courts),
  `court_crop_box`, `map_detection_to_frame`, `reject_shuttle_outliers`, `split_on_gaps`
  (+ existing `apply_homography`, `point_in_poly`, `resample_10hz`).
- `worker/worker/shuttle/track.py` (pure, TDD'd): `build_shuttle_track` — peaks → court rows.
- `worker/worker/shuttle/base.py`: `ShuttleTracker` protocol, `ShuttleDetection`, `heatmap_peak`,
  `FakeTracker` (weightless synthetic backend — runs/verifies the whole stage with no weights).
- `worker/worker/shuttle/{wasb,tracknetv3}.py`: the two real backends (below). `__init__.get_tracker`
  selects via `SHUTTLE_TRACKER` (default `wasb`), lazy-imported so the pure path stays torch-free.
- `worker/worker/pipeline/shuttle.py`: the stage (`run(ctx)`), registered in `_REAL`. Masks each frame to
  the target-court airspace polygon (`SHUTTLE_MASK`, default on) before the tracker sees it.
- `worker/worker/eval_shuttle.py`: dev runner + auto-stats bench (`--video-id [--tracker] [--compare]`),
  and `--render DIR` → `shuttle/render.py` draws the raw detections in **image space** (scatter +
  annotated key frames + `overlay.mp4`) — the diagnostic that disambiguates detection vs projection.

### Tracker backends (best-model-per-stage, both MIT)
Both reuse the upstream repos (cloned to `worker/.cache/_vendor/`, git-ignored) + published weights
(`models/weights/shuttle/`, git-ignored). Apple-Silicon: MPS-first with a one-shot NaN probe → CPU
(`PYTORCH_ENABLE_MPS_FALLBACK=1`, fp32); no CoreML in slice 1 (defer).
- **WASB** (`nttcom/WASB-SBDT`, BMVC2023) — SOTA on the PwC badminton benchmark. HRNet, 3-frame/9ch
  input, ImageNet-normalized, built from `wasb.yaml` via OmegaConf. Weights: MODEL_ZOO WASB/Badminton.
  Needs `omegaconf`.
- **TrackNetV3** (`qaz812345/TrackNetV3`) — recall champion. Conv U-Net, 8-frame window + **median
  background** (`bg_mode=concat`, 27ch) — the built-in white-wall mitigation. We reuse its
  `Shuttlecock_Trajectory_Dataset` to build the exact input (avoids a channel-layout bug) and run
  TrackNet only (no InpaintNet rectification yet). Needs `pandas` + `parse`.

## Honest limits (surfaced; not bugs to hide)
- **Airborne shuttle ≠ ground point.** `apply_homography` maps the **ground** plane; a shuttle in the air
  projects "beyond" its true spot, and a pixel above the court's ground-plane horizon projects toward
  infinity. So court-space (u,v) is a „Schätzung" — and for high shots it is **unusable**, not just noisy.
- **12 fps cache** undersamples a smash; some dropouts are undersampling, not camouflage.
- **DJI ultrawide undistortion is a no-op** → edge positions carry lens error.
- No shuttle ground truth yet → judged by **eye + auto-stats** (coverage %, gap structure, inter-frame
  step plausibility, % on-court for multi-court leakage), not a measured F1.

## Slice-1 result (f35d0984, 356 frames @12fps; bench `eval_shuttle --compare`)
Crop = `(57,483,3720,2160)` — the target court fills most of the frame width, so the crop helps mainly by
removing the white-wall ceiling + far-end court **vertically**, not by zooming.

| tracker | coverage | on-court | step m (med/p90/max) | longest gap |
|---|---|---|---|---|
| WASB | 60% (213/356) | **8%** | 6.3 / 73 / 12785 | 31f / 2.6s |
| TrackNetV3 | 81% (290/356) | **7%** | 6.7 / 124 / 2129 | 12f / 1.0s |
| fake (sanity) | 98% | 100% | 0.03 / 0.04 / 0.11 | 4f |

**Finding (unmasked):** the bbox crop did **not** isolate one court — the image-space scatter showed
detections spread across the **left/right neighbouring courts**, clustered at net/racket height. Only ~7–8%
landed on the target court; the rest were other courts + airborne mis-projection (kilometre steps =
horizon singularity).

### Airspace-mask result (`SHUTTLE_MASK=1`, crop=`(352,268,3424,2153)`)
Blacking out everything outside the target-court **airspace** before the tracker triples on-court and
removes the cross-court scatter:

| tracker | coverage | on-court | step m (med/p90/max) |
|---|---|---|---|
| TrackNetV3 (mask) | 46% (164/356) | **26%** | 3.8 / 67 / 2683 |
| WASB (mask) | 33% (117/356) | **20%** | 5.3 / 95 / 477 |

**Conclusion (the two blockers, separated by elimination):**
1. **Multi-court isolation — SOLVED** by the airspace mask (kept in the pipeline). Cross-court scatter
   gone; on-court ~tripled (7→26 %).
2. **Out-of-domain detection — the real residual.** Within the target column, coverage *drops* to 46 %
   (the tracker finds nothing on >half the frames) and what it does fire on clusters at the far
   net/scoreboard, on players/rackets — not the tiny low-contrast shuttle. Masking can't fix this; the
   broadcast-trained nets need **domain fine-tuning** on our white-wall/overhead footage (the spec's #1 ML
   ROI). Projection (airborne → off-court) is the secondary cleanup once detection is real.

## Next slice (directed by the finding): label → fine-tune
Plan/runbook: `specs/features/shuttle-finetune-plan.md`. Locked decisions: **fine-tune TrackNetV3 only**
(WASB stays frozen as the before/after benchmark — its public repo is eval-only); **runs locally on the
Mac Studio M4 Max (MPS)** (overrides `30-ml-pipeline.md`'s cloud-NVIDIA line; see there); **measure on a
held-out clip** (whole-clip train/val/test splits, never random frames).

**Slice 2a — answer-key + scoreboard infrastructure: BUILT** (this change).
- **Shuttle labeler** `verify --shuttle` — native-4K frame + hover-loupe for the few-pixel shuttle; click =
  anchor, Shift/right-click = visibility 0, `a` = segment boundary; server-rendered live interpolation
  preview. Persists sparse anchors+boundaries to `shuttle_labels.json`.
- Pure `shuttle/label.py::interpolate_anchors` (dense fill, never crosses a boundary, gaps→vis 0) →
  `import-shuttle-labels` → `shuttle_labels` table (migration `0005`).
- `shuttle/dataset_export.py` — emits the SAME airspace-masked 512×288 crops inference sees (train ≈
  inference, byte-identity verified), strict TrackNetV3 layout + per-match `median.npz` (concat lever).
- `eval_shuttle.py::score_against_labels` — precision/recall/F1 + median px error, pinned to 512×288 /
  tol=4px, via `--labels`.
- TrackNetV3 CUDA→MPS training patch + runbook under `worker/training/` (the vendored repo hard-codes
  `.cuda()`; verified: 2-batch CPU step runs without NaNs).

**Slice 2b — the actual run (blocked on data):** annotate ~15–30 rallies (white-wall + over-exposure first)
across **≥3–4 clips / ≥2 sessions** → fine-tune on MPS (LR 1e-4, mixup α 0.5) → re-benchmark raw **pixel**
detection on the held-out clip. Only ONE ~30s clip exists today, so 2b waits on more recordings.

### Before/after raw-pixel detection (held-out clip, 512×288, tol=4px) — *filled in 2b*
| backend | precision | recall | F1 | median px err | n (visible) |
|---|---|---|---|---|---|
| WASB (frozen) | — | — | — | — | — |
| TrackNetV3 baseline | — | — | — | — | — |
| TrackNetV3 fine-tuned | — | — | — | — | — |

**Projection honesty** (after detection works): image-space overlay, or horizon-guard `apply_homography` and
only court-project near-ground events. Then the spec's layered mitigations (motion cues, InpaintNet
rectification) + 2D speed. Speed/rally/hit remain deferred.

## Config / env
`SHUTTLE_TRACKER` (wasb|tracknetv3|fake), `SHUTTLE_CONF_THR` (0.5), `SHUTTLE_MAX_STEP` (0.5 court units),
`SHUTTLE_MASK` (1), `SHUTTLE_AIR_UP` (0.7), `SHUTTLE_AIR_SIDE` (0.08), `SHUTTLE_BBOX_MARGIN` (0.02),
`SHUTTLE_OVERLAY_GAP_S` (0.3), `WASB_REPO`/`WASB_MODEL`, `TRACKNETV3_REPO`/`TRACKNETV3_MODEL`.

## Wiring note
`shuttle` is in `_REAL` (runs when invoked) but **not yet in `RESUME_STAGES`** (`__main__.py`) — it joins
the post-calibration resume only once it produces a usable court-space track, so a normal job doesn't
depend on weights / emit garbage coords. Until then run it standalone via `eval_shuttle`.
