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
*Accept:* ≥99 % recall on the 2/4 in-court players; `eval_corrections` full-mode misses ≈ 0 after re-ID on
labeled clips (measured: 69 → 0 with k=0.16 / → 4 with k=0 on f35d0984) *(initial targets — revise as
answer-key clips accumulate)*.

## Stage 3 — Court gating + tracking (FR-10)
**Gate:** project each detection's ankle midpoint through H; keep only those inside the court polygon (+margin,
temporal smoothing). Decisive for clutter. **Track** the filtered set with **BoT-SORT-ReID / Deep-OC-SORT**
(via BoxMOT, or YOLO11's built-in BoT-SORT) with camera-motion compensation. Enforce exactly 2/4 in-court tracks;
assign persistent roles by court half (near/far, left/right); re-anchor on ends change. **Offline re-entry**:
global tracklet association (**gta-link**, MIT). *Avoid Deep-EIoU (no license).* Do **not** rely on appearance
ReID alone (same-club kit) — spatial+motion continuity is primary.
*Accept:* correct side/role assignment after manual confirm; detected median court error ≤ 0.5 m near /
≤ 1.0 m far half on the `eval_corrections` scorecard (measured 0.4/0.5 m un-fitted pose, 0.2/0.3 m clip-fitted
nudge on f35d0984) *(initial targets — revise as answer-key clips accumulate)*.

**Built (P1, Phase 4 — far-player robustness, RC3; identity rebuilt 2026-07-13):** when BoT-SORT drops a
small/low-contrast player and re-acquires them under a fresh id, the fragment must be re-associated or the
role has a hole. Three layers in `gate_track.build_role_tracks` (pure, unit-tested) before output:
1. **Half-aware primaries** (`geometry.select_role_primaries`): per court half, the most-present
   **coexisting pair** of tracks — two tracks alive at once are provably two people. Replaced the global
   top-N-by-count + relative `role_assign`, which on far-fragmented doubles games picked near tracks for the
   far roles (Session-2 postmortem: 18/20 role slots were near-half tracks — the far players were absent
   from every shipped stat/overlay; see `features/pose-skeleton-overlay.md` §Coverage).
2. **Framewise re-ID v2** (`geometry.stitch_fragments_framewise`, replaced `stitch_role_fragments`): a
   fragment attaches to a same-half role iff it fills per-frame holes (span overlap is irrelevant; brief
   double-tracking ≤ 20 % of its frames is tolerated), its boundary MEDIANS are velocity-continuous with
   the role's anchors (`base 1 m + 6 m/s · gap`; a near-role anchor that died across the net is a corpse,
   not evidence), and no second role is nearly as plausible (`STITCH_AMBIGUITY_M` — honest hole over
   partner swap). Sustained coexistence with the partner role is **mutual-exclusion evidence**: with the
   partner accounted for, the sole open role wins by elimination even when the boundary check glitches
   (id switches drift boxes exactly at handovers). Longest-first + fixpoint iteration lets later
   attachments disambiguate earlier drops. Tagged `source="reid"`; never fabricates.
3. **Bounded short-hole interp** (`geometry.fill_track_gaps`): holes of ≤ `GAP_FILL_MAX_FRAMES` (default 8) frames
   between two anchors are linearly interpolated (`source="interp"`); longer one-sided dropouts are **left
   absent** (never extrapolated). `tracks.parquet` carries the **`source`** column (`detected`/`reid`/`interp`) so
   the overlay surfaces re-ID/interp positions as estimates („Schätzung").
*Measured (f35d0984: far player B1 lost frames 284–355, 69 manual re-adds):* re-ID stitched the two dropped far
fragments back into B1 → the `eval_corrections` scorecard **misses 69 → 0** (k=0.16) / 69 → 4 (k=0 default),
recovering 63 real detections + 6 interp fills; the **detected** near/far cm is unchanged (re-ID/interp only add
previously-absent positions). The recovered far positions carry the deferred-RC2 depth amplification (~1.3 m),
reported in the scorecard's **separate** recovered bucket, never folded into the detected headline. *Honesty:*
re-ID recovers identity + lateral position (median ~0.1 m); far-court **depth** stays an estimate (deferred RC2).
The upstream **recall** cause (far-half detection) is still deferred — see specs/fix-later.md.

## Stage 4 — Pose (FR-24/25)
**RTMPose-m/-l top-down** (Apache), fed by the detector (2–4 crops → cheap). RTMPose-26 (body+feet) for footwork;
wholebody-133 for racket-hand only when needed. Fallbacks: RTMO (wide frames), YOLO11-pose (single-model path).
Auto-label hard frames offline with ViTPose++/Sapiens (label-only). Optional 3D later: MonoTrack/MotionBERT (approx).
*Accept:* keypoint AP adequate for foot placement & swing arm; sanity vs manual on lunges.

**Built (P1, foot-accuracy pull-forward):** RTMPose-m **body26 (Halpe-26)** run via **`rtmlib` on ONNXRuntime**
(no mmcv); `device='mps'` selects the **CoreML EP** (CLAUDE.md sanctions CoreML for RTMPose), one-time NaN probe
falls back to CPU. Top-down on the detector bboxes → `pose.parquet (frame,track,kp[],kp_score[],foot_x,foot_y,
foot_conf,foot_source)`, the foot = mean of the confident ankle keypoints (`geometry.foot_keypoint`). Runs
**before** `gate_track` (`detect → pose → gate_track`) so the court position uses the real ankle, not the bbox
bottom; `gate_track` takes the confident pose foot (`USE_POSE_FOOT=1`, `POSE_CONF_THR`) else the Phase-2 bbox
nudge. *Measured (f35d0984, CoreML, 30 s, 99.8 % confident feet):* un-fitted pose cut far-court median error
1.6 → 0.5 m vs the raw bbox default; a clip-fitted `FOOT_BIAS_K=0.16` still edged it (0.3 m) on its own box-drag
answer key. Both off by default; verify per clip with the `eval_corrections` full-mode scorecard. *Honesty:* foot
keypoints on the tiny far player are noise-amplified by the far homography (deferred RC2), so monocular far-court
position stays an estimate („Schätzung").

**Built (P1, Phase 4 — head/height occlusion grounding):** the **third layer** of the three-layer foot
(pose ankle → head/height → bbox nudge). A second pass (`pose.ground_occluded_feet`, pure helpers
`geometry.fit_head_foot_model` / `ground_foot_from_head`, unit-tested) learns, **per track**, the head→foot drop
from that player's own confident, feet-visible frames — measured as `(foot_y − head_y)/(hip_y − head_y)`, i.e. in
**torso lengths** (head kp 17, mid-hip kp 19, both visible under lower-body occlusion; *not* bbox height, which the
detector truncates — the RC1 problem this avoids). When the ankle is hidden but head+hip are confident it predicts
the foot from the head and marks `foot_source="head"` with a lower (head/hip) confidence so it stays an estimate.
`pose.parquet` gains a **`foot_source`** column (`ankle`/`head`/`none`). *Honest limits (it moves nothing on
f35d0984 — the lost far player has no detection there at all, so no head either; value is partial-occlusion,
mostly doubles P3):* real height varies ~±10 %, posture shifts the torso↔leg ratio (a deep lunge compresses it),
and the low standing camera foreshortens the drop. Verified by **synthetic unit test** (mask the ankle → recover
foot within tolerance); real/multi-clip verification awaits a partial-occlusion answer key (specs/fix-later.md).

## Stage 5 — Shuttle tracking + white-wall problem (FR-15/16/17)
**Production = the fine-tuned TrackNetV3 (MIT); WASB (MIT) is the frozen baseline; TrackNetV5 (ICIP 2026) is a
third backend under A/B — `features/shuttle-tracknetv5.md`.** White-shuttle-vs-white-wall is the
hardest case (video-camouflage: low appearance, distinct motion).
**Multi-court isolation (first-class):** training sessions run several courts at once → several shuttles in
frame. Isolate the **target** court before/around detection via the calibration mask (`mask_polygon_px` /
`geometry.point_in_poly`) and a court-region crop (`geometry.court_crop_box`) — the crop also restores the
tiny shuttle's effective resolution before a tracker downsizes to ~512×288. SOTA as of 2026 (web-checked):
WASB (BMVC2023) tops the PwC badminton benchmark; TrackNetV3 leads recall (inpainting); TrackNetV4 is TF/Keras
without published weights; the ICIP-2026 TrackNetV5 (PyTorch, from the V3 group) self-reports ~1 F1 point
over V3 on broadcast footage (an independent re-implementation saw none) and is being A/B'd here at 30 fps —
promoted only on a measured cross-session win (`features/shuttle-tracknetv5.md`). Layered mitigations:
1. **Background-median subtraction** (per-clip; static tripod) removes the white wall — TrackNetV3 has it built-in.
2. **Motion cues**: multi-frame input; optional frame-difference channel (OpenCV MOG2; avoid RAFT on MPS).
3. **Motion attention** for small/blurry shuttles — TrackNetV4's idea (its official code is TF/Keras with no
   published weights); TrackNetV5's gated motion prompt is the PyTorch successor under A/B.
4. **Trajectory recovery**: TrackNetV3 inpainting + MonoTrack-style per-shot parabola + Kalman/RTS smoother —
   interpolate **within an arc only**, never across a racket contact.
5. **Targeted fine-tuning** on labeled white-wall frames + over-exposure/brightness + mixup augmentation — biggest lever.
6. Optional mild **CLAHE on L** front-end (train+infer on the same enhanced frames; verify it helps).
Speed = homography + fps, **labeled estimate** with uncertainty (~5–8 km/h MAE best case). **2D only.**
*Accept:* held-out **same-session** F1 ≥ 0.90 @ tol=4px in the pinned 512×288 space (measured **0.953** on game1 —
in-sample, the 2026-07-02 re-measure with the recorded command; the earlier 0.958 was the first full-clip figure);
**cross-session** gate F1 ≥ 0.80 on a newly-labeled clip before trusting a new session unlabeled; gaps over a
wall bridged within an arc *(initial targets — revise as answer-key clips accumulate)*.

**Built (P1 slice 1 — thin baseline, run-both):** swappable backends behind `shuttle.ShuttleTracker`
(WASB HRNet + TrackNetV3 conv-UNet, both MIT, vendored + weights git-ignored; FakeTracker for weightless
end-to-end runs). Stage `pipeline/shuttle.py`: court crop → tracker peaks → `shuttle.track.build_shuttle_track`
(map crop→frame→court via `apply_homography`, drop <conf_thr, reject isolated spikes) → `shuttle.parquet
(t,u,v,visibility + debug)`; overlay emits subject='shuttle' (gap-aware via `split_on_gaps`). Pure math TDD'd
(`court_crop_box`/`map_detection_to_frame`/`reject_shuttle_outliers`/`split_on_gaps`/`build_shuttle_track`).
**Measured (f35d0984, eval_shuttle --compare):** TrackNetV3 81% / WASB 60% frame coverage, but **only 7–8%
of detections land on-court** and inter-frame court steps reach kilometres. *Honest finding (tracker-independent):*
(1) the **ground** homography can't place an **airborne** shuttle — pixels above the court horizon project
toward infinity, so court-space (u,v) is unusable for high shots, not just noisy; (2) the target court fills
the wide DJI frame, so a bbox crop doesn't isolate it from the side courts (detections hop between courts).
**Airspace-mask follow-up (same slice):** masking each frame to the target-court **airspace** (ground quad
raised up, `geometry.court_airspace_polygon`, `SHUTTLE_MASK=1`) before the tracker removed the cross-court
scatter and ~tripled on-court (TrackNetV3 7→26%, WASB 8→20%), but coverage *dropped* (46%/33%) and the
survivors cluster on the far net/players — so **isolation is solved, out-of-domain detection is the
residual**. **Label → fine-tune** was the #1 ML ROI and **has run** (Slice 2b, 2026-06-29/30, local MPS per the locked
plan: TrackNetV3 only, WASB frozen as the benchmark): fine-tuning on the trainer's own labeled footage took
F1 **0.483 → 0.958** on the labeled game1 clip — largely **in-sample** (rally-split within the one labeled
clip); see the measured table + honesty note in `features/shuttle-tracking.md`.
The fine-tuned checkpoint is **promoted to the production default** (`ckpts/TrackNet_best.pt`,
`SHUTTLE_TRACKER` defaults to `tracknetv3` — see `40-worker-contract.md` for the weights convention).
All 5 Session-2 games were processed with it on 2026-07-01 at 21–27% frame coverage (tight cross-session
band — an encouraging generalisation signal, but coverage-only). **Still open:** cross-session *scoring* —
label ≥1 Session-2 clip and run `eval_shuttle --labels` out-of-session (see `fix-later.md`). Then
image-space / horizon-guarded projection + mitigations 2–6 + speed.
See `specs/features/shuttle-tracking.md` + `specs/features/shuttle-finetune-plan.md`.

## Stage 6 — Rally segmentation + hit detection (FR-18/19/20) — BUILT (`rally_hit`)
Rallies from shuttle continuity + player motion energy + serve detector (shuttle low, players settled).
**Do not reuse broadcast rally/replay classifiers** (continuous stream, no replays). Hit frames = shuttle
direction reversal/speed spike near a player **fused with a swing detector** (Sensors-2024 "Shot Refinement"
pattern, ~90 % F1 on clean footage — the swing-detector fusion is stage 7's job).

**Built (`worker/rally/segment.py` pure + `pipeline/rally_hit.py` glue, TDD):** all trajectory/serve logic runs
in **normalized image space** (`px/W, py/H`), never court (u,v) — the ground homography can't place an airborne
shuttle. `group_trajectories` (keeps collinear smashes, breaks on gaps / non-collinear teleports) →
`reject_foreign_runs` (drops short neighbour-court runs that don't connect to a sustained run — *reduces* but
can't eliminate cross-court intrusion) → `merge_runs_into_rallies` (bridges white-wall dropouts, splits on long
gaps, discards sub-second) → per-rally `find_serve_frame` (players-settled cue, the reliable half of "shuttle
low + players settled" in 2D; falls back to first detection), `find_rally_end` (exposes the floor-contact
(x,y,u,v) candidate for the landing feature), provisional `count_hits` (vertical reversals — refined by stage 7).
Writes `rallies` + `events`(serve/hit/rally_end) and debug parquet; `aggregate` reads the spans to gate movement
to in-rally time. **Verified on game1** (labeled span): 6 rallies = the 6 labeled, dead time excluded (~11%
in-play), DB round-trip idempotent. Thresholds (`RALLY_*` env) are fit to one game — a 2nd session + a rally
labeler are needed before a quantitative boundary/hit-F1 claim.
*Accept:* rally boundaries within **±0.5 s** of hand-labeled boundaries *(provisional until the rally labeler
exists — today's only evidence is the game1 6/6 spot-check above; initial target, revise with labels)*.

**Landing + in/out (BUILT 2026-07-03 — `features/shuttle-landing-inout.md`):** the floor contact is the one
moment the ground homography is valid (the shuttle is on the floor), so in/out reduces to "which pixel is the
landing pixel, how sure". `worker/rally/landing.py` (pure, unit-tested): descent-flattens-to-rest selection in
image space, honest `located=false` degradation (never extrapolated), in/out vs the discipline's boundary with
a ±45 cm band, out-landings → `rallies.winner_side` + `player_shot_stats.errors` (`shot_type='unknown'` until
stage 7; singles only). Estimate-grade „Aus-Schläge"; the match score stays human-entered. `LANDING_*`
thresholds are untuned until a real-footage re-run; no located-rate/precision claim before the landing
answer key exists.

## Stage 7 — Stroke classification (FR-21/22/23)
**Governed by `features/stroke-classification-v2.md` (signed off 2026-07-13)** — v1
(`features/stroke-classification.md`) is superseded: its build failed the 0.70 gate (LOGO-CV 0.400,
GBDT spike 0.536) and was never wired. v2 in one line: learned hit detection (HitNet-pattern GRU + audio-onset timing
refinement) → 5-way hitter attribution → **BST-4-slot** (Chang et al., CVPRW 2026, arXiv:2502.21085,
MIT) from the ShuttleSet-25 `_merged` checkpoint, trained on **coarse-8** (17 fine labels for
labeling; ShuttleSet is natively 18 fine types — the earlier "35/25 classes" claim here mixed in the
checkpoint head sizes and was refuted). Low-confidence → `unknown`/„Unsicher" abstention.
*Accept:* the single ≥ 70 % gate is replaced by the **G0–G5 gate stack** (hit timing, attribution,
per-class recall, calibrated abstention, outcomes, trainer edit rate) — see v2 spec §9; below-gate
models are labeling-assist only, never pipeline.
*Side experiment (2026-10-18, not pipeline):* `features/stroke-vlm-pilot.md` — VideoJev-4B zero-shot on the
labeled clips, dev/eval protocol, result pending the Mac run. TypeSafe's Jev was evaluated and rejected
(text-only hosted API) — `features/stroke-classification.md` § Alternatives evaluated.

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
winner; validate INT8 doesn't hurt the tiny shuttle. **No MLX** (slow on Conv2d). Process locally; cloud NVIDIA
for fine-tuning runs / large backfills — **exception: the stage-5 TrackNetV3 shuttle fine-tune runs locally on
the Mac Studio M4 Max (MPS)** (small fine-tune; keeps the label→train→benchmark loop on one box against the
real frame cache/weights). This needs the vendored TrackNetV3 `.cuda()`→MPS patch in `worker/training/`
(`PYTORCH_ENABLE_MPS_FALLBACK=1` alone does NOT redirect explicit `.cuda()`). See
`specs/features/shuttle-finetune-plan.md` §6.

**Analysis cadence (user decision 2026-10-25 — `features/analysis-30fps.md`).** The earlier hybrid
(shuttle clips at 30 fps, movement at 12 fps) is superseded by a **full switch to 30 fps once measured**.
Note the "12 fps" caches really run at **10 fps** on 30 fps sources (ingest's keep rule); every frame-count
threshold is now a duration converted from the manifest's measured cadence, so the same code runs any
cadence. `DEFAULT_ANALYSIS_FPS` stays 12 until the 12-vs-30 table in that spec is filled on the Mac.

## Honesty / estimates (must be surfaced in UI)
3D trajectory & absolute speed (monocular, not Hawk-Eye), reaction time, footwork "efficiency" absolutes, and
doubles ID through heavy occlusion are **estimates / research-grade** — show with uncertainty and a correction path.
