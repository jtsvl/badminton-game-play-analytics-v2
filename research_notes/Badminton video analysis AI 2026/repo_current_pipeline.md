# Repo inventory — current ML/CV pipeline (for matching external research)

Snapshot: `HEAD detached at refs/heads/main` = `fe1e949` (merge PR #21 `feat/stroke-vlm-pilot`, 2026-10-18).
CLAUDE.md still names `feat/p0-foundations` as the working branch; that branch no longer exists locally
(only `main` / `origin/main`). Read-only analysis; no tests were executed (test counts below come from
`grep 'def test_'`). Metrics quoted are only those recorded in the repo, with their source.

Status legend: **BUILT+VERIFIED** (code + real-footage measurement recorded) · **BUILT (pure-tested)**
(logic unit-tested; model inference only validated manually on the Mac) · **BUILT, NOT RUN** ·
**SPEC ONLY** · **DEFERRED** (consciously parked, usually in `specs/fix-later.md`).

---

## 0. Git context (`git log --oneline -30`, newest first, dates from `git log --date=short`)
- 2026-10-18 `ac6fb91` feat(stroke): evaluate TypeSafe Jev, add VideoJev-4B zero-shot pilot (PR #21)
- 2026-07-14 `b9df7e8` fix(worker): pose/overlay/stats coverage — identity join, re-ID v2, thinning; `0a9c807` ignore alt-fps caches (PR #20)
- 2026-07-13 `d5afc2d` feat(overlay): pose skeleton overlay on the match report video
- 2026-07-05 `4ea39a2` docs: stroke Phase-2 LOGO-CV results — **gate FAILED at 0.400, no wiring**; `b4fe037` OOF softmax to CPU (MPS has no f64); `38d75b2` BST runbook; `4aa65f3` train-stroke CLI; `e77bfaf` BST4 LOGO-CV loop; `32b626a` BST4 4-slot model; `da7a1f1` training pure logic; `f28e588` BST deps; `88bf40a` stage-7 feature builders + dataset exporter; `9d0b394` lock stage-7 design; `6cd86cd` stroke-labeling tooling + import
- 2026-07-03 `ca85888`/`f84cc63` promote `RALLY_SWING_BRIDGE` on (T8, thr 4.0 tl/s); `7a3f4f8` swing-conditional gap bridging; `ca57638` T6 veto dropped; `50805b0` degrade implausibly-far landings; `f2a34d0` shuttle landing + in/out (v1); PRs #13–#17 merged
- Earlier (not in the 30): 2026-07-02 default shuttle tracker → fine-tuned TrackNetV3 (`22952a5`), rally review, 720p proxy, eval_rally; 2026-07-01 chunked shuttle prediction, heartbeat reclaim; 2026-06-30 Stage 6 rally segmentation (`b2744fe`); 2026-06-20 shuttle labeling + fine-tune infra; 2026-06-19 shuttle slice 1 WASB/TrackNetV3 + airspace mask (`9bec142`), far-player re-ID; 2026-06-17 P0 foundations (`f8496ee`).

## 1. Locked decisions & hard constraints (CLAUDE.md:29-41, plus conventions)
1. Greenfield v2; `handovers/` + v1 repo refs = intent only (CLAUDE.md:30).
2. Supabase self-hosted on Coolify is authoritative; DDL via the `.mcp.json` `supabase` MCP (CLAUDE.md:31-35).
3. Full-breadth pipeline in phases P0→P4; **singles first, doubles P3+** (CLAUDE.md:36). *Reality:* all
   labeled/processed Session-2 footage is **doubles** (`specs/features/stroke-labeling.md:31-33`,
   `stroke-classification.md:197-199`), so doubles is de-facto being processed now; per-player shot/error
   stats stay singles-gated.
4. Court calibration = **auto-detect + manual correction** (drag 4 corners, pick court) (CLAUDE.md:37).
   *Reality:* manual only (see Stage 1).
5. **White shuttle vs white wall** is first-class (CLAUDE.md:38; FR-16 `specs/10-functional-requirements.md:36-38`).
6. Licensing: non-distributed/non-commercial → **best model per stage regardless of license** (AGPL,
   CC-BY-NC, Adobe-NC OK); only hard-avoid deps with **no license grant** (e.g. Deep-EIoU); if it ever
   becomes commercial, revisit with RF-DETR/RTMDet + own-footage training (CLAUDE.md:39-41).
- Apple Silicon: **PyTorch MPS primary**, `PYTORCH_ENABLE_MPS_FALLBACK=1`, fp32, one-time NaN check vs CPU;
  **no MLX** (slow Conv2d); CoreML only for hot fixed-shape models (detector, RTMPose), benchmark vs MPS;
  decode once → cache frames; undistort DJI ultrawide before geometry; Parquet intermediates, Postgres
  aggregates (CLAUDE.md "Apple Silicon conventions"; `specs/30-ml-pipeline.md:206-215`).
- Honesty persona: "a plausible-but-wrong metric is worse than an absent one"; monocular speed/3D are
  estimates surfaced with uncertainty („Schätzung") (CLAUDE.md persona; `specs/30-ml-pipeline.md:217-219`).
- Data-shape contract: court coords u,v∈[0,1] (u left→right, v far→near); heatmap = gaussian kernels
  `[{x,y,w,s}]`; never pixel coords on web rows except the normalized-image precedents (`rally_end{x,y}`,
  `landing{px,py}`, `pose_keyframes`) (`specs/20-data-model.md:8-18,88-99`).
- Hardware: Mac Studio **M4 Max, 64 GB** (`specs/30-ml-pipeline.md:3`). Camera: iPhone or **DJI Osmo Action 6**
  (Wide/155° used so far), tripod ~3–4 m behind baseline, multiple adjacent courts in frame, multi-sport lines
  (`specs/00-product-spec.md:25-30`; `specs/recording-guide.md`).

## 2. Pipeline topology (what actually runs)
- Stage registry `worker/worker/pipeline/__init__.py:12-16`:
  `STAGE_ORDER = ingest, calibrate, detect, pose, gate_track, shuttle, rally_hit, stroke, aggregate, overlay`
  (pose runs **before** gate_track). `_REAL` (`__init__.py:50-51`) = all of those **except `stroke`** + `remap`,
  `rally_sync`, `proxy`. `stroke` resolves to the `_todo` no-op stub (`__init__.py:23-28`).
- Job slices `worker/worker/__main__.py:24,40-63`: `ingest`→[ingest, proxy]→`calibrating`; `calibrate`→
  `RESUME_STAGES=[calibrate, detect, pose, gate_track, shuttle, rally_hit, aggregate, overlay]`→`review`;
  `remap`→[remap, rally_sync, aggregate, overlay]; `rally_review`→[remap, rally_sync, aggregate];
  `retrack`→[gate_track, rally_sync, aggregate, overlay]; `proxy` standalone.
- Not implemented though specified: side-angle reduced pipeline (`SIDE_ANGLE_STAGES` defined at
  `__init__.py:16` but referenced nowhere); `params.from_stage/to_stage` re-processing ranges
  (`specs/40-worker-contract.md:74`; no code hits).
- Human gates: calibration corners (web `/match/:slug/calibrate`), player identity
  (`/match/:slug/review`, `features/player-identity-review.md`), rally drop/merge (`/match/:slug/rallies`,
  `features/rally-shot-review.md`). Target end-state "graduated autonomy" is **SPEC ONLY**
  (`specs/features/graduated-autonomy.md:3-6`): calibration → automate (P4), rally gate → confidence
  auto-accept after rule tuning, identity → stays human (GDPR/minors, `graduated-autonomy.md:59-69`).
- Frame rates: default analysis cache **12 fps** (`pipeline/ingest.py:19`); shuttle fine-tune/eval clips
  were ingested at **30 fps** (`enqueue --analysis-fps 30`, `features/shuttle-finetune-plan.md:174-182`);
  source footage ~60 fps 4K (`recording-guide.md:12,29-30`). **Production Session-2 matches run every stage,
  incl. the shuttle tracker, on the 12 fps cache** (`stroke-classification.md:33-36`;
  `shuttle-landing-inout.md:9-11` "21 % shuttle coverage at ~12 fps") — i.e. the promoted shuttle model
  (trained on 30 fps game1) runs at 12 fps in production. `gate_track` smoothing/gap params are 12 fps frame
  counts (`features/shuttle-tracking.md:67-70`).
- Tests: 39 test files / 599 `def test_` functions under `worker/tests/`; all pure or fake-backed (no real
  model weights in CI). Real-model checks = manual Mac Studio runs recorded in specs
  (`specs/40-worker-contract.md:162-169`). `tests/test_stroke_model.py` imports torch and self-skips without
  the vendored BST checkout; `tests/test_stroke_vlm_smoke.py` is opt-in (`STROKE_VLM_SMOKE=1`).
- Locked dependency versions (`worker/uv.lock`): ultralytics 8.4.70, torch 2.12.0, torchvision 0.27.0,
  rtmlib 0.0.15, onnxruntime 1.27.0, av (PyAV) 17.1.0, opencv-python 4.13.0.92, polars 1.41.2, numpy 2.4.6.
  The `vlm` extra (`transformers==5.14.1`, …, `worker/pyproject.toml:40-47`) is **not in uv.lock** yet.
- `models/` is empty in git (weights git-ignored via `.gitignore` `models/weights/`, `*.pt`, `*.onnx`, …); no
  model configs are committed despite CLAUDE.md's "weights+configs". All weights/vendored repos live only on
  the Mac Studio (`worker/.cache/_vendor/…`, `models/weights/…`).

---

## Stage 0 — Decode & preprocess (FR-8)
- **Choice:** PyAV (`av.open`, `thread_type="AUTO"`) decode → resample to analysis fps → **full-resolution
  JPEG q88** per kept frame + `frames.json` manifest (PTS seconds) (`pipeline/ingest.py:57-114`). Calibration
  still + `CALIB_FRAME_COUNT`=12 snapshots uploaded (`ingest.py:117-124`). 720p playback proxy via ffmpeg
  CLI with `-hwaccel videotoolbox` + `h264_videotoolbox` (`pipeline/proxy.py:32-34`; `features/video-proxy.md`).
- **Status:** BUILT (decode-once cache works; disk ~0.8 MB/4K JPEG, ~4 GB per 7-min game @12 fps, ~10 GB @30 fps,
  **no eviction** — `specs/40-worker-contract.md:148-154`).
- **Gaps vs spec:** spec says "ffmpeg + VideoToolbox HW decode via PyAV" (`30-ml-pipeline.md:9`) — ingest passes
  **no hwaccel option** (software decode). **Lens undistortion is a no-op**: `frames.json.undistorted=False`
  (`ingest.py:84,110`); no DJI intrinsics. Spec acceptance "undistortion validated on a known straight line"
  (`30-ml-pipeline.md:12`) unmet.
- **Known issues:** trainer visually confirmed court lines **bow** under the straight overlay on Osmo Action 6
  Wide/155° footage (2026-06-24) → 4-corner homography can't fit curved lines; biases positions, heatmaps,
  shuttle court path, speed (`fix-later.md:94-105`). Mitigations: (a) shoot "Standard (Dewarp)" FOV
  (`recording-guide.md:8,20-28`) — caveat: shuttle model was fine-tuned on distorted Wide frames, so Dewarp
  changes its pixel domain; (b) lens-intrinsics calibration + undistort in ingest (deferred, RC2).
  If undistortion lands, the proxy must be undistorted too or image-space overlays drift
  (`features/pose-skeleton-overlay.md:42-44`).

## Stage 1 — Court calibration (FR-5/6/7/8) — locked decision #4
- **Choice (built):** fully **manual 4-corner** placement in the web (`pages/Calibrate.tsx`; generic starting
  trapezoid `defaultCorners()` at `Calibrate.tsx:27,110`; drag magnifier; multi-frame snapshot switcher) →
  worker `calibrate` stage computes a pure **DLT homography** from exactly 4 corners (FL, FR, NR, NL → unit
  square), `mask_polygon_px` = the 4 corners, status `finalized` (`pipeline/calibrate.py:33-58`;
  `geometry.homography_from_corners` `geometry.py:53-72`; same DLT in the web `solveHomography`,
  `features/p1-cv-vertical.md:8-15`). `apply_homography` raises on horizon points (`geometry.py:73-89`).
- **Status:** BUILT (manual) + tested (`tests/test_geometry.py`, 83 tests incl. homography). **Auto-detect: DEFERRED
  to P4** (`fix-later.md:82-89`) — the worker never writes `proposed_points` or `auto_accepted`; FR-6
  "system auto-proposes the 4 corners" is unmet.
- **Accuracy evidence:** with 4 points the reprojection residual is ~0 by construction — a degeneracy check,
  not accuracy (`calibrate.py:3-6`). Spec target "< ~1 % of court width" (`30-ml-pipeline.md:19`) is therefore
  not really measured. One clip's homography refit to 0.04 cm in the corrections scorecard (`fix-later.md:9-10`).
- **Named candidates/approach (spec):** primary semi-automatic 4–6-point homography (OpenCV
  `findHomography`/`getPerspectiveTransform`); **auto-assist CourtKeyNet (MIT)** proposes corners (single-court
  trained → suggests, never decides; fine-tune for clutter); avoid **PnLCalib** (GPL-2.0, soccer) and
  **TennisCourtDetector** (no license) (`30-ml-pipeline.md:15-18`). fix-later adds: reuse the player foot-point
  cloud to pick the right court in a multi-court hall, then snap corners to nearby lines (`fix-later.md:86-88`).
  Graduated-autonomy gate 1: auto-accept when detection confidence ≥ threshold AND homography sanity
  (dimensions plausible, corners in frame); reuse previous game's calibration as prior (`graduated-autonomy.md:34-39`).
- **Open:** camera motion — single static homography assumes fixed tripod; per-segment re-calibration or
  auto-court-detect suggested (`fix-later.md:109-110`). Multi-point (>4) fitting/lens model not built.

## Stage 2 — Player detection (FR-9)
- **Choice (built):** Ultralytics **YOLO11m** (`yolo11m.pt`, stock COCO-pretrained, env `YOLO_MODEL`),
  person class only, `imgsz=1280` (`YOLO_IMGSZ`), `conf=0.25`, `iou=0.7`, per-frame `model.track(...)` with
  `tracker="botsort.yaml"` (ultralytics default config, not overridden) (`pipeline/detect.py:16-17,58-63`).
  PyTorch **MPS** with a one-shot NaN probe → CPU (`detect.py:20-34`). License AGPL (accepted under decision #6;
  `pyproject.toml:21`).
- **Status:** BUILT, **not fine-tuned**; no unit test for the model path (no test imports `pipeline.detect`).
  Not exported to CoreML (spec allows/asks to benchmark CoreML vs MPS for the detector — not done).
- **Measured:** first clip `doubles_match.MOV` (4K/59.94 fps/29.7 s): 18 raw tracks → gate_track exactly 4 in-court
  (`features/p1-cv-vertical.md:47-50`). Detector box bottom sits ~1–3 % frame height **above** real feet —
  112/112 trainer box fixes pulled the foot down on f35d0984 (`fix-later.md:5-11`). Far-court detection-recall
  ceiling ≈ **90 % of in-rally frames per far player** (both far players detected 82 %); near ≈ 97 %
  (`fix-later.md:56-59`; `pose-skeleton-overlay.md:146-149`).
- **Spec acceptance:** ≥99 % recall on the 2/4 in-court players; `eval_corrections` misses ≈0 after re-ID
  (measured 69→0 with k=0.16 / →4 with k=0 on f35d0984) (`30-ml-pipeline.md:25-27`).
- **Known issues (fix-later):** far-half **recall** is the deferred upstream cause of identity holes — "wants
  tiling / higher-res inference for the far half and/or stronger BoT-SORT ReID persistence" (`fix-later.md:56-59`).
- **Named candidates:** benchmark **RF-DETR** (Apache; "best occlusion/domain-shift per research") vs YOLO11 on own
  footage, keep winner; **RTMDet** (Apache) = OpenMMLab option (`30-ml-pipeline.md:22-23`). Fine-tune flagged
  "high ROI": ~500–1.5k own-footage bboxes (overhead+side, small adjacent-court players, blur, edges)
  (`30-ml-pipeline.md:24`). None started.

## Stage 3 — Court gating + tracking + identity (FR-10, FR-35)
- **Choice (built):** tracking = **ultralytics built-in BoT-SORT** inside `detect` (no BoxMOT, no Deep-OC-SORT,
  no explicit ReID model, no gta-link). Gating/identity in `pipeline/gate_track.py` (pure core
  `build_role_tracks` `:57-123`):
  1. Foot point → H → keep inside court ±`MARGIN=0.06` court units (~37 cm wide / 80 cm long) (`:24,126-154`).
     Foot = pose ankle if `USE_POSE_FOOT=1` and conf ≥ `POSE_CONF_THR=0.3`, else bbox-bottom nudged by
     `FOOT_BIAS_K` — **both default OFF/0** (`:33,38-39`), i.e. **production default = raw bbox bottom**.
  2. **Half-aware primaries** `geometry.select_role_primaries` (`geometry.py:265`): per court half the
     most-present coexisting pair; n = 2 singles / 4 doubles (`gate_track.py:185-186`).
  3. **Framewise re-ID v2** `geometry.stitch_fragments_framewise` (`geometry.py:318`): attach fragments that fill
     per-frame holes, velocity-continuous boundary medians (`STITCH_BASE_GATE_M=1.0` + `STITCH_SPEED_CAP_M_S=6.0`·gap),
     ambiguity margin `STITCH_AMBIGUITY_M=0.5` (honest hole over partner swap), mutual-exclusion by elimination,
     fixpoint iteration (`gate_track.py:43-45`; `30-ml-pipeline.md:40-60`).
  4. Bounded interp ≤ `GAP_FILL_MAX_FRAMES=8` (`geometry.fill_track_gaps`), never extrapolated; then
     centered moving-average `smooth_track` window `SMOOTH_WINDOW=9` (~0.75 s @12 fps) (`gate_track.py:25,29,118-119`).
  5. `tracks.parquet (t,track_id,u,v,role,source∈detected/reid/interp)`; auto-seed `match_players.track_label`
     by role near=A / far=B (`gate_track.py:47-54,225-239`). Consumers bind label→role→all rows of that role
     (`pipeline/identity.py`; `40-worker-contract.md:96-102`).
- **Status:** BUILT+VERIFIED (pure logic: `test_geometry.py` 83, `test_gate_track.py` 7, `test_identity.py` 13;
  real-footage coverage harness `worker/eval_pose_coverage.py`).
- **Measured:** f35d0984 far player B1 lost frames 284–355 (69 manual re-adds): re-ID → scorecard misses
  **69→0** (k=0.16) / **69→4** (k=0) (`30-ml-pipeline.md:61-67`). Court-position error medians near/far: raw bbox
  **0.7/1.6 m**, pose ankle un-fitted **0.4/0.5 m** (far p90 2.5→1.4 m), clip-fitted nudge k=0.16 **0.2/0.3 m**
  (`fix-later.md:21-25`); recovered (re-ID) far positions ~1.3 m depth error (separate bucket, `30-ml-pipeline.md:64-66`).
  Session-2 after the 2026-07-13 identity rebuild: skeleton/identity coverage **82–100 % wall-clock per player, far
  court 85–96 %** (was 5–16 %); before the fix 18/20 role slots were near-half tracks — far players absent from all
  shipped stats (`pose-skeleton-overlay.md:113-149`; `fix-later.md:48-55`).
- **Spec acceptance:** median court error ≤0.5 m near / ≤1.0 m far on the scorecard (`30-ml-pipeline.md:36-38`).
- **Known issues (fix-later):** far recall (above); **doubles near-pair ambiguity** — a lone centre fragment while
  both near roles are untracked is dropped (main residual coverage cost, e.g. A1 game2 81.7 %)
  (`fix-later.md:59-60`; `pose-skeleton-overlay.md:156-159`); RC2: court-space smoothing can't undo bias and
  over-smooths fast far movement → want image-space One-Euro/Kalman + distance-weighted position uncertainty
  (`fix-later.md:106-108`); retrack re-seeded Session-2 `track_label`s → trainer must re-confirm in PlayerReview
  (`fix-later.md:158-163`). Pose foot vs nudge verdict needs a multi-clip / direct-foot-drag answer key (`fix-later.md:26-32`).
- **Named candidates (spec):** BoT-SORT-ReID / **Deep-OC-SORT** via **BoxMOT** with camera-motion compensation;
  offline global tracklet association **gta-link (MIT)**; avoid **Deep-EIoU** (no license); don't rely on appearance
  ReID alone (same-club kit) (`30-ml-pipeline.md:31-35`). Appearance/jersey re-ID suggested for the near-pair
  ambiguity (`fix-later.md:59-60`). Cross-session person re-ID deliberately NOT pursued (GDPR/minors,
  `graduated-autonomy.md:60-61`).
- **Honesty/UI:** `source` reid/interp and `foot_source` head are meant to surface as „Schätzung"
  (`40-worker-contract.md:92-95`) — **but `overlay` writes `track_keyframes` as bare `{t,u,v}` and drops `source`**
  (`pipeline/overlay.py:156-158`), so the web cannot distinguish re-ID/interp positions today (spec↔code drift).
  PlayerReviewBoard labels the mapping „Schätzung – bitte prüfen" (`web/src/components/review/PlayerReviewBoard.tsx:83`).

## Stage 4 — Pose (FR-24/25)
- **Choice (built):** **RTMPose-m body26 (Halpe-26)**, 256×192 top-down, ONNX from the OpenMMLab model zoo
  (`rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip`, Apache-2.0) via **rtmlib on
  ONNXRuntime**; `device='mps'` → CoreML EP, NaN probe → CPU (`pipeline/pose.py:25-33,94-112`). Fed by **all**
  tracked detector boxes per frame (not only gated players) (`pose.py:127-152`). Foot = mean of confident ankles
  (`geometry.foot_keypoint`); **head/height grounding** second pass: per-track head→foot drop in torso lengths
  (head kp17, mid-hip kp19) predicts hidden feet, `foot_source="head"` (`pose.py:58-91`).
  Output `pose.parquet (frame,track,kp[26],kp_score[26],foot_x,foot_y,foot_conf,foot_source)`.
- **Consumers:** gate_track foot (opt-in), stage-6 swing cue, stage-7 features (kp[:17] = COCO-17), skeleton overlay.
- **Status:** BUILT. Pure row assembly tested (`tests/test_pose.py` 7); RTMPose inference validated only on the Mac.
  Head grounding: **built + synthetic-unit-tested, unverified on real data** (moves nothing on f35d0984; needs a
  partial-occlusion answer key, mostly doubles) (`30-ml-pipeline.md:87-98`; `fix-later.md:60-62`).
- **Measured:** f35d0984 (CoreML, 30 s) **99.8 % confident feet**; far-court median error 1.6→0.5 m vs raw bbox
  (`30-ml-pipeline.md:81-84`). Skeleton overlay payload 0.8–2.6 MB/player, 3.7–9.5 MB/doubles match
  (`pose-skeleton-overlay.md:52-56`).
- **Known limits:** 2D image keypoints only (no 3D, any future angle is a 2D estimate); racket and fingers not
  tracked; far-court/low-contrast frames degrade; foot keypoints on the tiny far player are noise-amplified by
  the far homography (`pose-skeleton-overlay.md:151-168`; `30-ml-pipeline.md:84-85`). Kong REST without gzip →
  pose fetch ~8.5 MB on the wire (`fix-later.md:150-156`).
- **Named candidates (spec):** RTMPose-m/-l; RTMPose-26 for footwork; **wholebody-133** for racket hand when
  needed; fallbacks **RTMO** (wide frames), **YOLO11-pose** (single-model path); auto-label hard frames offline
  with **ViTPose++ / Sapiens** (label-only); optional 3D later **MonoTrack / MotionBERT** (approx)
  (`30-ml-pipeline.md:70-73`). None beyond RTMPose-m built. Footwork metrics (lunge/recovery, FR-25), joint
  angles, swing quality: out of scope so far (`pose-skeleton-overlay.md:170-175`).
- **Honesty/UI:** web `lib/pose.ts poseAt` nearest-frame only (no interpolation), hides >0.3 s gaps; bones with
  endpoint score <0.3 hidden; skeleton exported only for non-interp rows (`pose-skeleton-overlay.md:97-104,64-70`).

## Stage 5 — Shuttle tracking incl. white-wall (FR-15/16/17) — locked decision #5
- **Choice (built):** swappable `ShuttleTracker` backends (`worker/worker/shuttle/`), selected by
  `SHUTTLE_TRACKER` (default **`tracknetv3`**, `shuttle/__init__.py:18`):
  - **TrackNetV3** (`qaz812345/TrackNetV3`, MIT): conv U-Net, 8-frame window, **median background
    `bg_mode=concat` (27 ch)**, 512×288 input, heatmap >0.5 → largest-blob centre; **TrackNet only — no InpaintNet
    rectification** (`shuttle/tracknetv3.py:1-15,153-176`). Median from a ≤700-frame stride subsample
    (`TRACKNETV3_MEDIAN_SAMPLES`, `:36`). **Production weights = own-footage fine-tune** at
    `models/weights/shuttle/ckpts/TrackNet_best.pt` (named copy `TrackNet_ft_session1.pt`, upstream kept as
    `TrackNet_baseline.pt`) (`40-worker-contract.md:138-141`; `shuttle-tracking.md:130-137`); unit test locks the
    no-env default (`tests/test_shuttle.py:58-68`).
  - **WASB** (`nttcom/WASB-SBDT`, BMVC 2023, MIT): HRNet, 3-frame/9-ch, ImageNet-norm, 512×288, sigmoid argmax;
    bypasses the repo's postprocessor/tracker; loads `state_dict(strict=False)` (`shuttle/wasb.py:1-15,57,113`).
    **Frozen pre-fine-tune baseline** (public repo eval-only).
  - `FakeTracker` (weightless parabola) for end-to-end runs without weights (`shuttle/base.py:89-123`).
- **Stage glue** `pipeline/shuttle.py`: target-court **airspace polygon** (ground quad raised `SHUTTLE_AIR_UP=0.7`
  court-heights, widened `AIR_SIDE=0.08`, near edge `AIR_DOWN=AIR_OUT=0` = legacy tight box) → crop box
  (`BBOX_MARGIN=0.02`) → black out outside the mask (`SHUTTLE_MASK=1`) → tracker → `build_shuttle_track`
  (drop conf < `SHUTTLE_CONF_THR=0.5` as honest dropout, map crop→frame→court via ground H, reject isolated
  spikes > `SHUTTLE_MAX_STEP=0.5` court units) (`pipeline/shuttle.py:28-47,126-164`; `shuttle/track.py`).
  Bounded-memory chunking `SHUTTLE_CHUNK_FRAMES=240`, window-aligned, identical to single pass
  (`shuttle.py:35,143-157`; `tests/test_shuttle_chunk.py` 11). Output `shuttle.parquet (t,u,v,visibility,frame,px,py,conf,tracker)`.
  Device: MPS with NaN probe → CPU in both backends; **no CoreML** (`shuttle-tracking.md:49-50`).
- **Status:** BUILT+VERIFIED (in `_REAL` and `RESUME_STAGES`). Pure math tested (`test_shuttle.py`, `test_geometry.py`,
  `test_eval_shuttle_score.py` 7, `test_shuttle_label.py` 8, `test_dataset_export.py` 7, `test_enhance.py` 7).
- **Measured (all from specs):**
  - Slice 1, f35d0984 356 frames @12 fps, unmasked crop: coverage WASB 60 % / TrackNetV3 81 %, but only **8 % / 7 %
    on-court**; court steps up to km (horizon singularity) (`shuttle-tracking.md:75-88`).
  - Airspace mask: on-court TrackNetV3 7→**26 %**, WASB 8→**20 %**; coverage dropped to 46 % / 33 %; survivors cluster
    on far net/players/rackets → "isolation solved, out-of-domain detection is the residual" (`shuttle-tracking.md:90-106`).
  - Fine-tune (game1 `2026-06-24-game1`, 30 fps, 3919-frame span, **424 visible label frames**, 512×288, tol 4 px,
    measured 2026-07-02) (`shuttle-tracking.md:139-147`):
    WASB frozen P 0.526 / R 0.601 / F1 0.561 / med err 1.26 px; TrackNetV3 baseline 0.352 / 0.734 / 0.476 / 0.69 px;
    **TrackNetV3 fine-tuned 0.926 / 0.981 / 0.953 / 0.49 px**. Headline also quoted as F1 0.483→0.958
    (`30-ml-pipeline.md:136-138`). **Largely in-sample** — rally split within the single labeled clip
    (`shuttle-tracking.md:149-154`).
  - Session-2 (5 doubles games, processed 2026-07-01 at 12 fps): **21–27 % frame coverage** — coverage-only
    generalisation signal, no precision/recall (`fix-later.md:142-148`).
  - Wider box A/B with the fine-tuned model (`AIR_UP=1.0, DOWN=0.10, OUT=0.06`): F1 0.953→**0.854**, recall
    0.981→0.849 — train/inference crop mismatch (shuttle ~23 % smaller) (`shuttle-tracking.md:180-188`).
- **Fine-tune recipe actually used / planned** (`worker/training/tracknetv3_finetune.md:90-102`): init from baseline
  via patched `--init_ckpt` (fresh optimizer), LR 1e-4, ~10 epochs (5–15), batch 2–4, `--alpha 0.5` mixup, keep WBCELoss,
  fp32 on MPS; vendored `train.py`/`test.py` `.cuda()`→`.to(DEVICE)` patches tracked at
  `worker/training/tracknetv3_mps_{train,test}.patch` (InpaintNet path left unpatched). Exact hyperparameters of the
  promoted "ft_smoke" run are **not recorded** in the repo (only the runbook defaults).
- **White-wall mitigation status** (layered plan `30-ml-pipeline.md:109-115`):
  1. Background-median subtraction — **BUILT** (TrackNetV3 `bg_mode=concat`).
  2. Motion cues / frame-difference channel (MOG2; "avoid RAFT on MPS") — **SPEC ONLY** (labeler has a
     background-difference "Bewegung" view, human aid only: `shuttle/enhance.py:11-19`).
  3. **TrackNetV4 motion-attention** (stated "now MIT") — **SPEC ONLY**; named as the next lever if fine-tuned V3
     recall stays low on white-wall clips (`shuttle-finetune-plan.md:19-21`).
  4. Trajectory recovery (TrackNetV3 InpaintNet + MonoTrack-style per-shot parabola + Kalman/RTS smoother, within an
     arc only) — **SPEC ONLY** (InpaintNet not run).
  5. Targeted fine-tuning + over-exposure/brightness + mixup — **PARTLY BUILT** (one clip, mixup per runbook; v1
     prior-art augmentation recipe — brightness/contrast/gamma jitter, CLAHE-as-augmentation, motion blur, background
     swap, hard negatives, horizontal flip only — documented but not implemented, `tracknetv3_finetune.md:104-117`).
  6. Optional CLAHE-on-L front end (train+infer) — **SPEC ONLY** (labeler-only CLAHE view, `enhance.py:22-32`).
  Capture-side mitigation: under-expose −0.3…−0.7 EV so the wall doesn't clip (`recording-guide.md:13-14,36-39`).
- **Open items / fix-later:** cross-session scoring — label ≥1 Session-2 clip (`verify --shuttle`) and run
  `eval_shuttle --labels` out-of-session, target **F1 ≥ 0.80** (`fix-later.md:142-148`); wider airspace box needs
  re-export + re-fine-tune (gate = labeling effort; needed for landing/out-calls) (`fix-later.md:120-128`); the plan's
  data gate (≥3–4 clips across ≥2 sessions/lighting) was **never met** — fine-tune ran on one clip
  (`shuttle-finetune-plan.md:41-44,170-173`); f35d0984 re-ingested at 30 fps (890 frames) "being re-labeled" — status
  not recorded; Dewarp FOV domain shift expected (`recording-guide.md:26-28`); foreign-court intrusion reduced not solved
  (`fix-later.md:112-118`); 12 fps production cadence vs 30 fps training cadence (see §2).
- **Projection limit (load-bearing):** ground homography cannot place an airborne shuttle; pixels above the court
  horizon project toward infinity → court (u,v) unusable for high shots; downstream logic therefore works in
  **normalized image space** (`shuttle-tracking.md:59-62`). Planned: image-space overlay or horizon-guarded projection
  (`shuttle-tracking.md:156-158`). **2D only**; no 3D trajectory (`30-ml-pipeline.md:116`; product out-of-scope
  measurement-grade 3D, `00-product-spec.md:39`).
- **Spec acceptance:** same-session held-out F1 ≥ 0.90 @ tol 4 px (512×288) (measured 0.958 on game1); cross-session F1
  ≥ 0.80 before trusting a new session unlabeled (`30-ml-pipeline.md:117-119`).
- **Named alternatives:** WASB vs TrackNetV3 ("benchmark both"); "SOTA as of 2026 (web-checked): WASB tops the PwC
  badminton benchmark; TrackNetV3 leads recall (inpainting); **TrackNetV4/V5 are incremental** — keep WASB/TrackNetV3
  for now" (`30-ml-pipeline.md:101-108`).
- **Honesty/UI:** `visibility` = heatmap peak conf; sub-threshold frames absent (honest gaps); overlay keyframes split on
  gaps > `SHUTTLE_OVERLAY_GAP_S=0.3` (`overlay.py:29-45`). **Web gap:** the "Shuttle" chip exists
  (`OverlayChips.tsx:44`) but `VideoOverlay.tsx` never draws the `'shuttle'` track (comment at `:129`; only the unused
  demo helper `lib/overlay.ts:35 shuttleAt`) — the real shuttle path written by the worker is **not rendered**.

## Stage 6 — Rally segmentation + hit detection (FR-18/19/20) + landing/in-out
- **Choice (built):** rule-based heuristics, all in **normalized image space** (`worker/rally/segment.py`,
  `pipeline/rally_hit.py`): `group_trajectories` (break on gap > `RALLY_GAP_S=0.5` or non-collinear jump >
  `RALLY_MAX_JUMP=0.10`, cos < 0.3) → `reject_foreign_runs` (short runs < 4 dets kept only if within 1.5 s / 0.15 of a
  sustained run) → `merge_runs_into_rallies` (bridge gaps ≤ `RALLY_BRIDGE_S=1.5`; discard < 0.5 s or < 5 dets) →
  serve = first detection, confidence 0.7 if players "settled" (< 4 km/h over 0.5 s) else 0.4 → rally end = last
  conf ≥ 0.5 point → **provisional hits = vertical-direction reversals** (deadband 0.01, confidence 0.3)
  (`segment.py:21-47,64-233`). **"Not a learned segmenter"** — settled 2026-07-02 (`pose-swing-fusion.md:7-8`).
- **Pose/swing fusion** (`worker/rally/swing.py`; `features/pose-swing-fusion.md`): arm-tip (wrist, else elbow×1.8)
  speed in **torso-lengths/s** (rolling-median torso, kp17→kp19), swing events above `SWING_MIN_TLPS=3.0`
  (`swing.py:32-37`). **Veto (phantom-rally kill) DROPPED** 2026-07-03 (fused swing-OR-settled kills only 2/76 at
  `missed==0`). **Bridge PROMOTED ON**: gaps in (1.5, 2.5] s merge iff a swing ≥ 4.0 tl/s lies in the gap interior,
  excluding 0.75 s before gap end (`segment.py:33-36,129-174`). Serve corroboration (confidence only).
- **Landing + in/out** (sub-step, `worker/rally/landing.py`, `features/shuttle-landing-inout.md`): terminal window 0.5 s
  / ≤8 dets; flattening vertical velocity → first resting detection = contact; degrade to `located=false` with reason
  `sparse|straddle|edge_cutoff|no_descent|no_contact|implausible` (never extrapolate) (`landing.py:31-43,50-90`); in/out vs
  singles (`u∈[0.0754,0.9246]`) or doubles width, flat **±45 cm** band, forced `uncertain` near frame edges (lens bow),
  `LANDING_MAX_OUT_CM=100` plausibility guard for above-floor rests (`landing.py:93-148`); out → `rallies.winner_side`
  (receiver's half; ±1 m net margin = no attribution) → `player_shot_stats.errors` (singles only, `shot_type='unknown'`)
  (`pipeline/rally_hit.py:113-130`; `pipeline/aggregate.py:74-95,172-185`).
- **Status:** segmentation BUILT+VERIFIED (`test_rally.py` 32, `test_rally_hit.py` 13, `test_swing.py` 9,
  `test_eval_rally.py` 37, `test_landing.py` 23). Landing BUILT, real-footage-smoke-tested; **all `LANDING_*` and
  most `RALLY_*` thresholds untuned / fit to one game** (`30-ml-pipeline.md:164-167,174-176`).
- **Measured:** game1 labeled span 6/6 rallies, ~11 % in-play (`30-ml-pipeline.md:163-165`). After trainer review of
  all 5 Session-2 games (flags off, ±0.5 s): machine 246 → kept 149 rallies; boundary-F1 **0.742–0.775**, precision
  0.59–0.63, recall 1.000; **76 false rallies (~40 % noise)**, 20 splits (`pose-swing-fusion.md:157-172`). T8 bridge at
  4.0 tl/s recovers **17/21 (81 %)** splits; false-merge safety **untestable** (0 real gaps in the 1.5–2.5 s band —
  vacuous, not proven safe) (`pose-swing-fusion.md:268-301`). 90 `rally_corrections` rows captured (60 drops, 15 merge
  pairs) (`pose-swing-fusion.md:22-25`). Hit-detection precision from stroke labels: **43 `no_hit` / 738 ≈ 5.8 % FP →
  ≈ 94.2 % precision**; hit recall unmeasured; provisional hit timing drifts up to ~0.2 s
  (`stroke-classification.md:62-63`; `stroke-labeling.md:16-20`). Landing smoke run (Session2_Game_5, 23 rallies):
  located **1/23, and that one was a false positive** (racket catch projected 2.85 m out) → guard added → **0 located**;
  ceiling = 21 % shuttle coverage at ~12 fps (`shuttle-landing-inout.md:3-12,132-146`).
- **GT limitation:** rally-review only drops/merges, so kept boundaries = machine boundaries; boundary-moving
  improvements (serve-snap, end-trim) unmeasurable → deferred until a hand-labeled boundary set (~20–30 rallies)
  (`pose-swing-fusion.md:37-48`). No rally labeler exists; spec acceptance ±0.5 s boundaries is provisional.
- **Known issues:** phantom short rallies (pickup tosses, walk-backs, neighbour-court leaks, between-points
  push-backs) remain a manual one-click drop (`pose-swing-fusion.md:27-35`; `graduated-autonomy.md:43-45`); real short
  rallies (0.5–3 s) overlap the phantom band so no span guard separates them (`pose-swing-fusion.md:182-190`); 12 fps
  aliasing (swing = 1–2 frames) (`swing.py:6-8`); "which half hit" signals fail on this footage (trajectory
  vertical-flip 48 %, shuttle-y vs midline 49 %) (`stroke-classification.md:74-76`); above-floor rests that project
  *inside* the court are undetectable in landing v1 (`shuttle-landing-inout.md:110-115`); trainer edits are clobbered by a
  full re-run (loud warning only) (`rally-shot-review.md:49-55`).
- **Named/out-of-scope candidates:** Sensors-2024 "Shot Refinement" swing-detector fusion (~90 % F1 clean footage; its
  +0.18 F1 "not comparable" to this GT regime) (`30-ml-pipeline.md:150-152`; `pose-swing-fusion.md:47-48`); **audio
  hit-detection spike** ("agreed next step"), tiny GRU/BiLSTM sub-classifiers — out of scope of the fusion spec
  (`pose-swing-fusion.md:346-349`); "do not reuse broadcast rally/replay classifiers" (`30-ml-pipeline.md:150`).
  Stage-7 v2 line mentions **learned hit detection (HitNet-pattern GRU + audio-onset timing refinement)**
  (`30-ml-pipeline.md:181`) — see Stage 7 caveat.
- **Honesty/UI:** `winner_side`/`end_shot_type`/`shot_count`/landing are estimates decoupled from the human-entered
  `matches.score` (`40-worker-contract.md:89-91`); RallyReviewBoard says „Die Ballwechsel sind eine Schätzung"
  (`RallyReviewBoard.tsx:43`); „Aus-Schläge: N (Schätzung, Untergrenze – nur sicher erkannte Landungen)"
  (`ShotMixCard.tsx:28`), hidden for doubles / when never run; `outLine` rally sentence deferred
  (`20-data-model.md:120-123`).

## Stage 7 — Stroke classification (FR-21/22/23) incl. VLM pilot
- **Taxonomy:** coarse 7 `clear, smash, drop, net, drive, lift, serve` + `unknown`/„Sonstige" abstention
  (`stroke/train.py:19`; `web/src/lib/labels.ts:18`); FR-22 shows „Sonstige" only when count > 0.
- **v1 model (built, failed gate):** **BST** (Chang et al., "BST: Badminton Stroke-type Transformer…", CVPRW 2026,
  arXiv:2502.21085, `Va6lue/BST-Badminton-Stroke-type-Transformer`, MIT, vendored at SHA
  `fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df`) generalized to **BST4**: 4 fixed court-quadrant player streams
  (near_left/near_right/far_left/far_right) with presence mask, shared TCNs, CleanGate + AimPlayer (`BST_CG_AP`), **trained
  from scratch** (`stroke/model.py:1-35,68`; `stroke-classification.md:106-118`). Inputs per hit window
  **16 frames @12 fps (≈±0.67 s)**: COCO-17 joints (bbox-relative, diag-normalized), shuttle `(px/W,py/H)` with ≤6-frame
  in-arc interpolation, court positions; pose-recovery fills slots role tracks miss (`stroke/features.py:14-16,32-63,142-193`).
  LOGO-CV over 5 games, τ sweep (covered-acc ≥ 0.75 at coverage ≥ 0.70), gated promote (`stroke/train.py:31,143-176`;
  `worker/training/bst_stroke_train.md`).
- **Status:** BUILT (features, dataset export, model, training CLI; `test_stroke_features.py` 34, `test_stroke_train.py` 28,
  `test_stroke_model.py` 14 guarded, `test_stroke_dataset.py` 4) — **not wired**: no `pipeline/stroke.py`, not in `_REAL` or
  `RESUME_STAGES`, no `Store.upsert_player_shot_stats` (`stroke-classification.md:25-29,169-175`). `player_shot_stats` counts
  come only from `seed.sql`; `events.hit` stays `{provisional:true}`.
- **Measured (2026-07-05, all with `--seed 42`, MPS, ~4.5 min/run)** (`bst_stroke_train.md:67-90`): baseline 0.376 ± 0.039;
  **best `--lr 3e-4` 0.400 ± 0.056 (pooled 0.403)**; `--epochs 300` 0.390; `--epochs 75` 0.377; `JnB_bone` 0.349; `bst0`
  ablation 0.353 → **gate ≥ 0.70 FAILED** on every run; τ unsatisfiable (knee at 0); per-class recall clear .53, lift .54,
  serve .40, drop .36, smash .30, **net .00, drive .00**; collapse toward majority (drop/lift). Diagnosis: train acc
  **0.945 vs held-out 0.402** → cross-game generalization gap, not underfit; sanctioned hyperparameter space exhausted.
  Phase-1 slot coverage: hitter's slot filled **83.6 %** (308 track / 155 pose-recovery); hitting half ≥1 player near 94.0 %
  / far 100 %; both players near 52.3 % / far 82.9 %; shuttle window coverage median 1.00 (`stroke-classification.md:90-104`).
- **Labels:** 738 Session-2 hit events labeled in self-hosted **Label Studio 1.23.0** (German, 720p clips ±0.7 s with audio)
  → 695 training rows (drop 165, lift 153, unknown 141, clear 78, smash 61, serve 55, drive 22, net 20) + 43 `no_hit`; 4-way
  hitter label banked but unused; all doubles, one session/lighting (`stroke-labeling.md:165-168`).
- **Named first levers (user decision pending):** near-half recovery v2 (missing partner), more labeled sessions, shuttle
  coverage; 30 fps re-ingest as a separately measured experiment (smash/drive recall) (`stroke-classification.md:21-24,115-118`).
- **v2 plan (SPEC REFERENCED BUT MISSING):** `30-ml-pipeline.md:178-191` says stage 7 is "governed by
  `features/stroke-classification-v2.md` (signed off 2026-07-13)": learned hit detection (**HitNet-pattern GRU + audio-onset
  timing refinement**) → **5-way hitter attribution** → **BST-4-slot from the ShuttleSet-25 `_merged` checkpoint**, trained on
  **coarse-8** (17 fine labels for labeling); acceptance = **G0–G5 gate stack** (hit timing, attribution, per-class recall,
  calibrated abstention, outcomes, trainer edit rate); below-gate models = labeling-assist only. **The file
  `specs/features/stroke-classification-v2.md` does not exist in the repo or git history** (reference added in `b9df7e8`,
  2026-07-14). Also contradicts v1 on taxonomy facts: `30-ml-pipeline.md:184-185` says "ShuttleSet is natively 18 fine types —
  the earlier '35/25 classes' claim was refuted", while `stroke-classification.md:46-47` still says "ShuttleSet is 35 classes /
  25 merged; BadmintonDB is the 18-class set"; and v1 = train from scratch vs v2 = init from ShuttleSet checkpoint.
- **VLM pilot (BUILT, NOT RUN):** **VideoJev-4B** (MCG-NJU, Apache-2.0, per spec arXiv 2610.18615, HF `ZhuYuhan/VideoJev-4B`;
  Qwen3.5-4B multimodal LLM with a candidate-scorer head = softmax over text options) zero-shot on the 695 clips
  (`stroke/vlm_eval.py:37-52`, `stroke/vlm_model.py`; `features/stroke-vlm-pilot.md`). Protocol: dev = G5 (69 hits) picks one
  of 10 frozen presets (3 prompts × fps {2,10,30} + reversed-order bias probe); eval = other 4 games; CPU-fp32-vs-MPS probe
  (max |Δp| ≤ 0.05); tiers `closed` < 0.35 / `labeling_assist` 0.35–0.55 or top-2 ≥ 0.75 / `finetune_candidate` ≥ 0.55 (+ recall
  > 0 on ≥ 6/7 classes, lower confidence on Unsicher) / `gate` ≥ 0.70 (`stroke-vlm-pilot.md:38-84`). Results table empty;
  `VIDEOJEV_REVISION=None` (unpinned, `vlm_eval.py:41`); `vlm` extra not locked. Model ignores audio; expectation "between
  chance (1/7) and BST (0.40)". **35B-A3B** variant excluded (≈70 GB bf16 > 64 GB; quantized MPS paths drop the scorer head or
  need MLX) — cloud-only option (`stroke-vlm-pilot.md:30-34`). Never wired regardless of result (`stroke-vlm-pilot.md:104-106`).
- **Rejected:** **TypeSafe Jev 1.13/1.14** — hosted text-only decision API, no video, no open weights/on-prem → do not integrate
  (`stroke-classification.md:212-224`).
- **Honesty:** low confidence → `unknown`; `hit_disputed` flag from swing-cue disagreement planned (±0.3 s); net-play
  attribution occlusion-limited; smash sparse at 12 fps; shot counts would be estimates („Schätzung")
  (`stroke-classification.md:153-158,196-210`). Shot relabel in review UI deferred until stage 7 exists
  (`rally-shot-review.md:67-68`).

## Stage 8 — Insight aggregation (FR-11/12/13/14/17/26/27)
- **Built (pure, `worker/worker/insights.py`, `pipeline/aggregate.py`):** court-space distance; avg speed over moving samples
  (≥0.5 km/h); **top speed = p98** of step speeds (not max) (`insights.py:56-68`); sprints = rising edges ≥ **15 km/h**
  (`:71-82`); zones front/mid/rear by `net_depth` thirds + left/right halves (`:85-96`; `geometry.net_depth`); 5-min
  timeline buckets (`:99-110`); heatmap = **coarse 12×12 grid, top-4 cells → `{x,y,w,s}` kernels, σ=0.13** ("good enough for
  P0; a proper KDE/GMM can replace it", `:113-138`). Movement gated to in-rally spans (whole-clip fallback)
  (`aggregate.py:137-150`). Overlay keyframes ~10 Hz (`geometry.resample_10hz`), pose keyframes 12 fps in-rally / 6 Hz between
  (`overlay.py:50-54`). Tests: `test_insights.py` 12, `test_aggregate.py` 11, `test_aggregate_gating.py` 3.
- **Measured:** top speed fix — far-court players showed ~30–34 km/h max on `2026-06-14-doppel`, near 7–25 km/h; p98 +
  `SMOOTH_WINDOW=9` fixed it (`fix-later.md:64-80`); P1 acceptance top_kmh ~7–19 km/h after `smooth_track` (was ~46)
  (`p1-cv-vertical.md:31-34,47-50`).
- **Not built:** **shuttle speed** (`fastest` always `None`, "never faked from foot speed", `aggregate.py:70`; web would
  print „Schnellster Smash ≈ N km/h — Schätzung aus der Shuttle-Verfolgung", `ShotMixCard.tsx:30`); spec: homography + fps,
  labeled estimate, "~5–8 km/h MAE best case", 2D only, side-angle clips better (`30-ml-pipeline.md:116`; FR-17).
  **3D trajectory** — out of scope ("no 3D" in ROI order, `30-ml-pipeline.md:202`). **0.25 m raster heatmap** (FR-11
  `heatmap_uri`) — not written. **Doubles `pair_match_stats` (fb/ss/rot, FR-26)** — no worker writer; seed data only.
  Per-player shot counts (stage 7). Reaction time / footwork "efficiency" — listed as research-grade estimates only
  (`30-ml-pipeline.md:217-219`).
- **Landing in/out:** see Stage 6 (BUILT, estimate-grade, 0 located on the only real run).
- **Known issues:** far-court position uncertainty not surfaced per-point (RC2); lens bow biases zones/heatmaps
  (`fix-later.md:91-110`).
- **Spec acceptance:** zones sum ≈100 %; timeline length = ceil(durationMin/5); kernels render; spot-checks vs manual
  counts (`30-ml-pipeline.md:197`).

---

## Datasets, labeling & fine-tuning
- **Own-footage corpora in the repo's DB/caches (no external dataset used yet):**
  - f35d0984 / `2026-06-14-doppel` (30 s doubles, 4K/59.94): 185 player corrections (`detection_corrections`, 0004) via
    `worker verify` → `import-corrections`; 69 far-player re-adds; 112 box fixes (`fix-later.md:5-11,39-41`).
  - `2026-06-24-game1` (session 1, 30 fps ingest): shuttle labels via `verify --shuttle` (sparse clicks + linear
    interpolation within segments, contrast views), 424 visible frames → `shuttle_labels` (0005) → `export-shuttle-dataset`
    (byte-identical masked 512×288 crops + per-match `median.npz`) (`shuttle-tracking.md:114-129`).
  - Session 2 (`2026-06-30-session2-game1..5`, doubles, Osmo Action 6 Wide/155°): 738 stroke labels (Label Studio), 90
    rally corrections (0011), trainer rally reviews of all 5 games.
- **Seed datasets named (non-commercial, usable) but unused:** ShuttleSet/BadmintonDB (stroke taxonomy + broadcast
  pretrain), VideoBadminton (elevated-angle action), SportsMOT (tracking/ReID weights) (`30-ml-pipeline.md:200-202`).
  ShuttleSet-25 `_merged` checkpoint named for stage-7 v2 (`30-ml-pipeline.md:183`).
- **Fine-tune ROI order (spec):** shuttle (mandatory) → detector → stroke → court auto → pose → (no 3D) (`30-ml-pipeline.md:202`).
  Done: shuttle (1 clip). Detector/court/pose fine-tunes: not started. Stroke: from-scratch BST4 failed gate.
- **Tooling:** built = custom browser labelers in `worker/worker/verify.py` (player corrections, shuttle clicks) + Label Studio
  (strokes). Spec-named but not adopted: **CVAT**, **FiftyOne**, **SAM** pre-labeling (`30-ml-pipeline.md:203-204`; FR-37).
  Budget estimate ≈ 15–30 annotated rallies ≈ 1–3 weeks (`30-ml-pipeline.md:204`); shuttle labeling reality ~1 anchor every
  2–4 frames on arcs (`shuttle-finetune-plan.md:25-29,184-187`).
- **Missing answer keys blocking claims:** Session-2 shuttle labels (cross-session F1), landing answer key (≥2 sessions),
  partial-occlusion foot key, multi-clip direct-foot-drag key, hand-labeled rally boundaries (~20–30 rallies), Session-3
  stroke re-eval (cross-session).

## Apple Silicon execution (as built)
| Stage | Runtime as built | Spec intent |
|---|---|---|
| Decode | PyAV software decode (no hwaccel flag) → JPEG cache; proxy via ffmpeg VideoToolbox enc/dec | VideoToolbox HW decode via PyAV |
| Detect (YOLO11m) | PyTorch **MPS**, NaN probe → CPU; per-frame `track()` calls | CoreML candidate (fixed shape), benchmark vs MPS |
| Pose (RTMPose-m) | ONNXRuntime **CoreML EP** (`device='mps'`), NaN probe → CPU | CoreML sanctioned |
| Shuttle (TrackNetV3/WASB) | PyTorch **MPS** fp32, NaN probe → CPU; no CoreML | CoreML deferred; validate INT8 doesn't hurt tiny shuttle |
| Shuttle fine-tune | **Local MPS** with tracked `.cuda()`→device patches (exception to "cloud NVIDIA for fine-tuning") | `30-ml-pipeline.md:209-215` |
| Stroke BST4 training | MPS (auto) with CPU NaN probe; MPS has no float64 (fix `b4fe037`); MPS not bit-deterministic | — |
| VideoJev-4B pilot | MPS fp32 (bf16 fallback), sdpa attention (no flash-attn), CPU-vs-MPS probe gate; untested upstream on MPS | — |
- **No MLX anywhere** (rule; also cited as why the 35B VLM can't be quantized onto the Mac, `stroke-vlm-pilot.md:30-33`).
- No benchmarks of CoreML vs MPS are recorded; no INT8 quantization done. Runtime per stage not recorded except stroke
  training (~4.5 min/run) and proxy "~5× realtime" (`pipeline/proxy.py:5`).

---

## Cross-cutting findings (spec↔code drift / inconsistencies worth knowing before matching research)
1. **Stroke v2 spec missing**: `30-ml-pipeline.md:179` points to `features/stroke-classification-v2.md` (never committed);
   v1 and v2 disagree on ShuttleSet class counts and on scratch-vs-pretrained init.
2. **Production foot point = raw bbox bottom** (`USE_POSE_FOOT=0`, `FOOT_BIAS_K=0` defaults) even though pose ankle measured
   far better (far 1.6→0.5 m) — deliberate "off until reviewed" (`gate_track.py:30-39`; `fix-later.md:29-32`).
3. **Shuttle runs at 12 fps in production** while the promoted fine-tune was trained/evaluated at 30 fps; Session-2 coverage
   21–27 % is measured at 12 fps.
4. **Re-ID/interp provenance not exported** to the web (`overlay.py:156-158` drops `source`), contradicting
   `40-worker-contract.md:92-95`.
5. **Real shuttle overlay not rendered** in the web despite the "Shuttle" chip (`VideoOverlay.tsx:129`).
6. **Court auto-detect absent** (locked decision #4 / FR-6 unmet); generic starting trapezoid only.
7. **Lens undistortion absent** (FR-8 unmet) on footage confirmed to be barrel-distorted.
8. **Singles-first** (decision #3) vs all real footage being doubles.
9. Side-angle reduced pipeline and `from_stage/to_stage` params specified but unimplemented.
10. `worker/README.md` still says stage models are stubs (stale); CLAUDE.md branch name stale.

## Quick-match index — every alternative/candidate the repo already names
- Court: CourtKeyNet (MIT, planned auto-assist) · OpenCV findHomography (4–6 pts) · avoid PnLCalib (GPL-2.0), TennisCourtDetector (no license).
- Detection: YOLO11 (in use, AGPL) · RF-DETR (Apache, benchmark candidate; commercial fallback) · RTMDet (Apache) · far-half tiling / higher-res.
- Tracking/ReID: BoT-SORT (in use, ultralytics built-in) · BoT-SORT-ReID / Deep-OC-SORT via BoxMOT · gta-link (MIT, offline re-entry) · avoid Deep-EIoU · SportsMOT weights.
- Pose: RTMPose-m Halpe-26 (in use) · RTMPose-l · wholebody-133 · RTMO · YOLO11-pose · ViTPose++ / Sapiens (offline auto-label) · MonoTrack / MotionBERT (3D, optional).
- Shuttle: TrackNetV3 (in use, fine-tuned) · WASB (frozen baseline) · TrackNetV4 motion-attention (next lever) · TrackNetV5 ("incremental") · TrackNetV3 InpaintNet · MonoTrack-style parabola + Kalman/RTS · MOG2 frame-difference (avoid RAFT on MPS) · CLAHE-on-L.
- Rally/hit: rule-based segmenter (in use) · Sensors-2024 Shot Refinement fusion · HitNet-pattern GRU · audio-onset hit timing · GRU/BiLSTM sub-classifiers (out of scope).
- Stroke: BST/BST4 (built, failed gate) · ShuttleSet-25 `_merged` checkpoint (v2) · VideoJev-4B (pilot, not run) · VideoJev-35B-A3B (cloud-only) · TypeSafe Jev (rejected).
- Data/tooling: ShuttleSet, BadmintonDB, VideoBadminton, SportsMOT · Label Studio (in use) · CVAT, FiftyOne, SAM (named, unused).
- Geometry: DJI lens intrinsics / checkerboard undistort (deferred) · Osmo Action 6 "Standard (Dewarp)" capture (recommended) · second camera / per-court depth (durable fix for foreign shuttles, deferred).
