# 30 — ML / CV Pipeline

Offline batch on Mac Studio M4 Max (64 GB), **PyTorch MPS** primary. Non-distributed/non-commercial →
models chosen by quality. Each stage lists: model, license (for a future commercial revisit), fine-tune
need, acceptance threshold (measured on held-out **own-footage** labels before the stage is "done"),
and fallback. Benchmark candidates on the club's footage — literature numbers are clean-broadcast and optimistic.

## Stage 0 — Decode & preprocess
ffmpeg + VideoToolbox HW decode via **PyAV**. Confirm 10-bit HEVC for DJI. **Undistort DJI ultrawide first**
(rectilinear mode, or one-time intrinsics). **Decode once → cache frames** (or frame index) → every model
reads the cache. Media engine runs parallel to GPU/ANE.
*Accept:* deterministic frame count & timestamps; undistortion validated on a known straight line.

## Stage 1 — Court calibration (FR-5/6/7/8) — do FIRST
**Primary: semi-automatic 4–6-point homography** (OpenCV `findHomography`/`getPerspectiveTransform`, Apache).
Human confirms/drags corners + picks the court → the only approach robust to adjacent courts + overlapping lines.
**Auto-assist: CourtKeyNet** (MIT) proposes corners (single-court-trained → suggests, never decides; fine-tune for clutter).
*Avoid:* PnLCalib (GPL-2.0, soccer), TennisCourtDetector (no license).
*Accept:* reprojection error of confirmed corners < ~1 % of court width; metric distances sane (net 6.10 m, length 13.40 m).

## Stage 2 — Player detection (FR-9)
**Default Ultralytics YOLO11** (AGPL; best tooling + CoreML, integrated detect/pose/track). **Benchmark vs RF-DETR**
(Apache; best occlusion/domain-shift per research) on own footage; keep the winner. RTMDet (Apache) = OpenMMLab-stack option.
*Fine-tune:* yes, high ROI — ~500–1.5k own-footage bboxes (overhead+side, small adjacent-court players, blur, edges).
*Accept:* person AP ≥ target on held-out frames incl. small/edge players; ≥99 % recall on the 2/4 in-court players.

## Stage 3 — Court gating + tracking (FR-10)
**Gate:** project each detection's ankle midpoint through H; keep only those inside the court polygon (+margin,
temporal smoothing). Decisive for clutter. **Track** the filtered set with **BoT-SORT-ReID / Deep-OC-SORT**
(via BoxMOT, or YOLO11's built-in BoT-SORT) with camera-motion compensation. Enforce exactly 2/4 in-court tracks;
assign persistent roles by court half (near/far, left/right); re-anchor on ends change. **Offline re-entry**:
global tracklet association (**gta-link**, MIT). *Avoid Deep-EIoU (no license).* Do **not** rely on appearance
ReID alone (same-club kit) — spatial+motion continuity is primary.
*Accept:* IDF1/ID-switches under target on labeled clips; correct side/role assignment after manual confirm.

## Stage 4 — Pose (FR-24/25)
**RTMPose-m/-l top-down** (Apache), fed by the detector (2–4 crops → cheap). RTMPose-26 (body+feet) for footwork;
wholebody-133 for racket-hand only when needed. Fallbacks: RTMO (wide frames), YOLO11-pose (single-model path).
Auto-label hard frames offline with ViTPose++/Sapiens (label-only). Optional 3D later: MonoTrack/MotionBERT (approx).
*Accept:* keypoint AP adequate for foot placement & swing arm; sanity vs manual on lunges.

**Built (P1, foot-accuracy pull-forward):** RTMPose-m **body26 (Halpe-26)** run via **`rtmlib` on ONNXRuntime**
(no mmcv); `device='mps'` selects the **CoreML EP** (CLAUDE.md sanctions CoreML for RTMPose), one-time NaN probe
falls back to CPU. Top-down on the detector bboxes → `pose.parquet (frame,track,kp[],kp_score[],foot_x,foot_y,
foot_conf)`, the foot = mean of the confident ankle keypoints (`geometry.foot_keypoint`). Runs **before**
`gate_track` (`detect → pose → gate_track`) so the court position uses the real ankle, not the bbox bottom;
`gate_track` takes the confident pose foot (`USE_POSE_FOOT=1`, `POSE_CONF_THR`) else the Phase-2 bbox nudge —
the first two layers of the three-layer foot (pose → nudge → Phase-4 head/height grounding). The full keypoint
array (incl. head) is retained for that occlusion fallback + the stroke stage. *Measured (f35d0984, CoreML, 30 s,
99.8 % confident feet):* un-fitted pose cut far-court median error 1.6 → 0.5 m vs the raw bbox default; a clip-
fitted `FOOT_BIAS_K=0.16` still edged it (0.3 m) on its own box-drag answer key. Both off by default; verify per
clip with the `eval_corrections` full-mode scorecard. *Honesty:* foot keypoints on the tiny far player are noise-
amplified by the far homography (deferred RC2), so monocular far-court position stays an estimate („Schätzung").

## Stage 5 — Shuttle tracking + white-wall problem (FR-15/16/17)
**WASB (MIT) primary; TrackNetV3 (MIT) strong alternative — benchmark both.** White-shuttle-vs-white-wall is the
hardest case (video-camouflage: low appearance, distinct motion). Layered mitigations:
1. **Background-median subtraction** (per-clip; static tripod) removes the white wall — TrackNetV3 has it built-in.
2. **Motion cues**: multi-frame input; optional frame-difference channel (OpenCV MOG2; avoid RAFT on MPS).
3. **TrackNetV4 motion-attention** (now MIT) grafted on for small/blurry shuttles.
4. **Trajectory recovery**: TrackNetV3 inpainting + MonoTrack-style per-shot parabola + Kalman/RTS smoother —
   interpolate **within an arc only**, never across a racket contact.
5. **Targeted fine-tuning** on labeled white-wall frames + over-exposure/brightness + mixup augmentation — biggest lever.
6. Optional mild **CLAHE on L** front-end (train+infer on the same enhanced frames; verify it helps).
Speed = homography + fps, **labeled estimate** with uncertainty (~5–8 km/h MAE best case). **2D only.**
*Accept:* shuttle F1/recall on own white-wall test clips ≥ target; gaps over a wall bridged within an arc.

## Stage 6 — Rally segmentation + hit detection (FR-18/19/20)
Rallies from shuttle continuity + player motion energy + serve detector (shuttle low, players settled).
**Do not reuse broadcast rally/replay classifiers** (continuous stream, no replays). Hit frames = shuttle
direction reversal/speed spike near a player **fused with a swing detector** (Sensors-2024 "Shot Refinement"
pattern, ~90 % F1 on clean footage).
*Accept:* rally boundary tolerance and hit-frame F1 under target on labeled rallies.

## Stage 7 — Stroke classification (FR-21/22/23)
**BST (MIT, public weights)** — consumes player skeleton + shuttle xy + court xy (tiny transformer; hours to train
on MPS). Adopt **ShuttleSet 18-class BWF taxonomy collapsed to 6–8 coarse classes** first. Low-confidence → `unknown`.
*Fine-tune:* yes — ~300–1k own labeled strokes; iterate hard pairs (drop/net, clear/lift).
*Accept:* coarse-class accuracy ≥ target on held-out strokes; realistic ceiling 77–85 % on clean data, lower until tuned.

## Stage 8 — Insight aggregation (FR-11/12/13/14/26/27)
Pure functions from tracks/events (TDD these): distance (court-space displacement), avg/top speed, sprints
(threshold), zones %, 5-min timeline, heatmap KDE → gaussian kernels `{x,y,w,s}` (+0.25 m raster option),
shot counts, fastest, rally aggregates + `endLine`, doubles `fb/ss/rot`, and ~10 Hz overlay keyframes.
*Accept:* zones sum ≈100 %; timeline length = ceil(durationMin/5); kernels render to the design heatmap; spot-checks vs manual counts.

## Datasets & fine-tuning
**Seed (non-commercial, usable):** ShuttleSet/BadmintonDB (stroke taxonomy + broadcast pretrain), VideoBadminton
(elevated-angle action), SportsMOT (tracking/ReID weights). **Then fine-tune on own footage** for the overhead-gym +
white-wall domain gap. ROI order: **shuttle (mandatory) → detector → stroke → court auto → pose → (no 3D)**.
Tooling self-hosted on the Mac (privacy): CVAT (boxes/trajectories/keypoints), Label Studio (stroke/event timelines),
FiftyOne (curation/label-error mining); pre-label with SAM + pretrained models. Budget ≈ 15–30 annotated rallies ≈ 1–3 weeks.

## Apple Silicon execution
PyTorch **MPS** primary (`PYTORCH_ENABLE_MPS_FALLBACK=1`, fp32, validate vs CPU once for NaNs). **CoreML** only for
the two hot fixed-shape models (detector, RTMPose) — export at a single fixed input shape; benchmark vs MPS, keep
winner; validate INT8 doesn't hurt the tiny shuttle. **No MLX** (slow on Conv2d). Process locally; cloud NVIDIA only
for fine-tuning runs / large backfills.

## Honesty / estimates (must be surfaced in UI)
3D trajectory & absolute speed (monocular, not Hawk-Eye), reaction time, footwork "efficiency" absolutes, and
doubles ID through heavy occlusion are **estimates / research-grade** — show with uncertainty and a correction path.
