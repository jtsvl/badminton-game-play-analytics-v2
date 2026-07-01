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
- **Frame rate (shuttle = 30 fps; movement = 12 fps).** Shuttle fine-tune/eval clips now ingest at **30 fps**
  (`enqueue --analysis-fps 30`) so smashes are sampled; the player-movement pipeline stays at the default
  12 fps. (At 12 fps some dropouts were undersampling, not camouflage.) 60 fps remains a re-ingestable
  comparison option — see `shuttle-finetune-plan.md` §2b "Framerate decision".
- **Guard-rail: `gate_track` is fps-coupled.** Its `SMOOTH_WINDOW=9` / `MAX_GAP=8` are 12fps frame-counts
  (≈0.75 s / ≈0.67 s). The 30fps shuttle path runs only ingest→label→eval (not the movement stages), so this
  never bites today — but if a 30fps clip is ever pushed through the full movement pipeline, scale them ×2.5
  via env (`SMOOTH_WINDOW=23`, `GAP_FILL_MAX_FRAMES=20`) or make them duration-based.
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
  preview. Persists sparse anchors+boundaries to `shuttle_labels.json`. **Contrast/visibility views** (`c`
  cycles; CSS presets + server-side **background-difference** "Bewegung" and **CLAHE** "Lokalkontrast" views,
  `shuttle/enhance.py`) make the white-on-white shuttle easier to spot for accurate clicks — CLAHE is the
  local-contrast cue the v1 app identified, and complements the motion-difference view at the apex.
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
`SHUTTLE_MASK` (1), `SHUTTLE_AIR_UP` (0.7), `SHUTTLE_AIR_SIDE` (0.08), **`SHUTTLE_AIR_DOWN` (0.0)**,
**`SHUTTLE_AIR_OUT` (0.0)**, `SHUTTLE_BBOX_MARGIN` (0.02), `SHUTTLE_OVERLAY_GAP_S` (0.3),
`SHUTTLE_CHUNK_FRAMES` (240, see below), `WASB_REPO`/`WASB_MODEL`, `TRACKNETV3_REPO`/`TRACKNETV3_MODEL`.

### Bounded-memory chunked prediction (`SHUTTLE_CHUNK_FRAMES`, default 240)
The stage used to materialize **every** full-res court crop before calling the tracker (~130 GB for a
20-min match — fine on the 30 s eval clips, OOM on a real one). Now a lazy `CropLoader` reads crops
per chunk and the clip is predicted in spans of `SHUTTLE_CHUNK_FRAMES` crops (0 = legacy single pass).
**Identity guarantee:** both backends window their input *without overlap* (TrackNetV3
`sliding_step=seq_len=8`, WASB `frames_in=3`), so `chunk_spans` aligns every chunk boundary to a
multiple of the backend's `window` and the window grouping — hence the predictions — is exactly the
full-clip grouping. The only per-clip state, TrackNetV3's median background, is computed once per clip
(`begin_clip`, same stride subsample as before → byte-identical) instead of per predict() call.
240 is a multiple of both window sizes ≈ 2–3 GB peak crop RAM on a 4K court crop. Identity holds for a
complete frame cache (an unreadable JPEG shifts windows only within its chunk — warned and counted).
Verified: stage-level test (chunked vs single-pass parquet equality, FakeTracker); re-verify with real
weights on the Mac Studio via `eval_shuttle --compare` before trusting new benchmark numbers.

### Wider-box knobs (off by default — `AIR_DOWN`/`AIR_OUT`, raise-`AIR_UP`)
`court_airspace_polygon` gained near-edge margins (`down` lowers the near baseline, `out` spreads the near
corners) so high clears + just-out landings stay in the mask. **Defaults are 0.0 (= the legacy box) on
purpose.** A/B on game1 with the **fine-tuned** model (`SHUTTLE_AIR_UP=1.0 DOWN=0.10 OUT=0.06`) **regressed**
it: F1 0.953→0.854, recall 0.981→0.849. Cause = **train/inference crop mismatch** — the fine-tune was trained
on 512×288 crops from the *tight* box; a wider crop downscales the shuttle ~23% smaller than the model learned
(misses more) and re-admits more neighbouring-court noise. The box maths is correct (the wider yellow boundary
visibly reaches the high clears / out-landings); the model isn't. **Don't flip the defaults until a re-export +
re-fine-tune on the wider crop** — blocked on 2nd-session data anyway. The knobs ship reachable for that day.

## Honest limits (cont.)
- **Foreign-court intrusion is reduced, not solved.** Stage 6 rally continuity (`reject_foreign_runs`) drops
  short discontinuous neighbour runs, and the airspace mask isolates the target court — but a foreign shuttle
  that lingers several in-mask frames with rally-like motion still leaks. Monocular, single-camera: this is a
  *mitigation*, not a fix (a second camera or per-court depth would be the real fix). Also logged in `fix-later.md`.

## Wiring note
`shuttle` **and** `rally_hit` are now in `_REAL` **and** `RESUME_STAGES` (`__main__.py`) — shuttle is verified
on real footage (full-clip F1 0.958, fine-tuned) and rally_hit segments its output into rallies. `stroke`
(stage 7) stays stubbed. Run shuttle standalone via `eval_shuttle`; run `rally_hit` standalone with a
`stage="rally_hit"` job (shuttle/tracks cached).
