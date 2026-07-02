# Fix-later / backlog

A running list of known issues and improvements we've consciously deferred, so they don't get lost.

## Player foot-position accuracy — bbox-bottom truncation (P1) — POSE STAGE BUILT
The foot point was the detection box's bottom-centre (`geometry.ankle_point`), but YOLO11's person box
ends ~1–3 % of frame height **above** the real feet (shins/ankles). The trainer's full review of clip
`f35d0984` (185 corrections) showed this in **112/112** box fixes pulling the foot **down**. Because the
homography amplifies far-court error, the resulting court-position error measured **~68 cm median near /
~158 cm median far** (recovered via the `worker/eval_corrections.py` scorecard, homography refit to 0.04 cm).

**Interim fix (done):** `geometry.foot_point(bbox, k)` pushes the foot down `k*box-height`; wired into
`gate_track` via env `FOOT_BIAS_K` (default 0 = off, identical to old behaviour). Fitted **k≈0.16** on
f35d0984 cut median court error to **~24 cm near / ~31 cm far** (~5× far) in-sample. Enable per-deployment;
refit `k` as more answer-key clips accumulate (in-sample fit will shrink out-of-sample).
**Durable fix (built, Phase 3):** the `pose` stage runs RTMPose-m body26 (Halpe-26, Apache) via `rtmlib`
on ONNXRuntime (CoreML EP, `device='mps'`, CPU NaN-fallback), top-down on the detector crops, and derives
the foot from the real ankle keypoint with per-keypoint confidence. It runs **before** `gate_track`
(`detect → pose → gate_track`), which uses the confident ankle (`USE_POSE_FOOT=1`) else the bbox nudge.
The full keypoint array (incl. head) is kept in `pose.parquet` for the Phase-4 occlusion fallback + stroke.
On f35d0984 (CoreML, 30 s, 99.8 % confident feet), full-mode scorecard medians (near/far court):
- raw bbox bottom (`FOOT_BIAS_K=0`, the default): **0.7 / 1.6 m**
- **pose ankle, un-fitted** (`USE_POSE_FOOT=1`): **0.4 / 0.5 m** (far p90 2.5 → 1.4 m) — big no-fit win
- clip-**fitted** nudge (`FOOT_BIAS_K=0.16`): **0.2 / 0.3 m** — still best, but fitted to this one clip

So pose is the durable generalizer (no per-clip constant) and a large improvement over the un-fitted
default; a hand-fitted `k` edges it on its own answer key, partly because that key was authored by dragging
the **box bottom** (so the GT foot ≈ box-bottom-centre, which the nudge reproduces by construction) and
partly because the residual far error is the deferred RC2 amplification, not the foot point. Both ship
**off by default** (`FOOT_BIAS_K=0`, `USE_POSE_FOOT=0`); enable per-deployment after review. A fair pose-vs-
nudge verdict needs a multi-clip / direct-foot-drag answer key — until then keep the nudge as primary and
pose for keypoints + Phase-4. **Phase 4 (planned):** head/height grounding when the ankle is occluded.
**Scorecard:** baseline `python -m worker.eval_corrections --corrections <json> --width W --height H`;
real-data `python -m worker.eval_corrections --mode full --video-id <id>` (loads detections + DB homography
+ pose.parquet; honours `FOOT_BIAS_K`/`USE_POSE_FOOT`) so each fix is proven, not eyeballed.

_Raised 2026-06-19 from the trainer's f35d0984 review; pose stage built + scored same day._

## Missed-detection gaps for the far player (P1) — re-ID + interp BUILT; recall still deferred
On f35d0984 the far player (B1) was "undetected" for ~6 s (idx 284–355), re-added by hand (69 `added`).
**Measure-first finding (2026-06-19):** it was *not* a fillable interpolation gap — B1 is a terminal dropout (lost
after 282, never re-acquired before the clip ends) and the true path is non-linear (13 m moved). But the far
player *was* re-detected through most of the span under short-lived track ids (64, 286–337; 68, 342–355) that
`enforce_n_tracks` discarded. **Built (Phase 4):** role re-ID (`stitch_role_fragments`) re-associates those
fragments with B1 + bounded short-hole interp (`fill_track_gaps`) — scorecard **misses 69 → 0** (k=0.16) / → 4
(k=0), recovering real detections; detected near/far cm unchanged; recovered far error (~1.3 m, deferred RC2 depth)
in a separate scorecard bucket. See specs/30-ml-pipeline.md stage 3.
**Still deferred (the upstream cause):** far-half detection **recall** — BoT-SORT fragments/loses the small,
low-contrast far player (`conf=0.25,imgsz=1280`); re-ID mitigates downstream but doesn't fix recall. Wants tiling
/ higher-res inference for the far half and/or stronger BoT-SORT ReID persistence. Also: **head/height occlusion
grounding** (`pose.ground_occluded_feet`) is built + unit-tested but **unverified on real data** — it can't help
this clip (no detection → no head) and needs a partial-occlusion answer key (mostly doubles, P3) to prove.

## Far-court top-speed inflation (P1 movement tier) — FIXED
The two players furthest from the camera showed an inflated **top speed** (~30–34 km/h on the
`2026-06-14-doppel` clip); near players are believable (~7–25 km/h). Cause: the far baseline is only
~610 px wide → very few pixels per metre, so ankle-point jitter is amplified there, and
`top_kmh = max(step speeds)` is the most outlier-sensitive statistic possible (a single bad frame-pair
sets it). It's the inherent monocular far-court noise, not a calibration error.

**Fix (done):** (1) robust top speed — `insights.avg_top_speed_kmh` now uses the **98th percentile**
(`insights.percentile`, linear-interp) of the step-speeds instead of `max`, so one jittery far-court
frame-pair no longer sets the headline; genuine sustained bursts survive (p98 ~= max on ≤2 samples, so
short tracks are unchanged). (2) **Stronger smoothing** — the `gate_track` call to
`geometry.smooth_track` now uses window **9** (~0.75 s @ ~12 Hz), env-tunable via `SMOOTH_WINDOW`
(default 9; `smooth_track`'s own default stays 5 for other callers). Both unit-tested
(`tests/test_insights.py`, `tests/test_geometry.py`). Still inherent monocular noise on the far court —
the deeper RC2 fixes (image-space filtering, distance-weighted uncertainty) remain below.

_Raised 2026-06-17 while reviewing the trainer's real calibration; fixed 2026-06-20._

## Automatic court detection (deferred — build-plan P4)
Today calibration is fully **manual**: the calibrate screen drops a generic starting trapezoid and the
trainer drags the 4 corners onto the court (the magnifier helps place them precisely). There is **no
automatic court detection yet** — the "far off" the trainer noticed is just that generic starting guess.
Auto-detect-with-manual-correction is locked decision #4, scheduled for P4. Likely approaches when we get
to it: CourtKeyNet (MIT) corner proposal, and/or reuse the player foot-point cloud we already compute to
pick the *right* court in the multi-court hall, then snap corners to nearby lines.
Deferred per the trainer 2026-06-17 — revisit later.

## Far-court geometry & camera motion (deferred — RC2, picked up later)
Three compounding geometry limits the trainer deferred on 2026-06-19 (filmed standing, moved little on
f35d0984 so low impact there):
- **Lens undistortion:** DJI ultrawide barrel distortion is **not** corrected (`ingest.py` records
  `undistorted: False`); it biases edge/far-court pixels where homography error is already amplified.
  Needs DJI intrinsics (checkerboard or lens profile). **Trainer visually confirmed 2026-06-24** on
  the new game footage (Osmo Action 6, Wide/155° FOV): painted court lines visibly **bow** under the
  straight calibration overlay, worst toward the frame edges/baselines. **Impact split:** *low* for the
  raw-pixel shuttle fine-tune (training frames, GT clicks, and inference all live in the same distorted
  pixel space, so the detector just learns the real appearance — no need to re-shoot the 6 clips);
  *real* for court-space geometry — a single 4-corner homography can't fit curved lines, so corners sit
  right but mid-line/edges drift, biasing player positions, heatmaps, court-space shuttle path and
  speed. **Mitigations:** (a) cheap, no-code — shoot future sessions in the Action 6's **Standard
  (Dewarp)** FOV (in-camera rectilinear correction → straight lines → clean homography; crops FOV, so
  mount slightly further back); (b) durable — lens-intrinsics calibration + undistort in ingest stage 0.
- **Distance-weighted uncertainty + image-space smoothing:** `smooth_track` averages in court space, which
  can't undo a bias and over-smooths fast far movement; filter the foot in image space (One-Euro/Kalman)
  and surface a position uncertainty that grows toward the far baseline (UI already says „Schätzung").
- **Camera motion:** a single static homography assumes a fixed camera; hand-held/standing drift shifts the
  court anchor over the clip. Revisit with per-segment re-calibration or auto-court-detect (P4).

## Foreign-court shuttle intrusion — reduced, not solved (2026-06-30)
On a single camera the target-court airspace mask + stage-6 rally continuity (`reject_foreign_runs`) cut
neighbouring-court shuttles a lot, but a foreign shuttle that lingers several in-mask frames with rally-like
motion still leaks into our track (visible in the wider-box A/B render: dots drift toward the right where the
next court is playing). **Monocular limit — a mitigation, not a fix.** Durable fixes (deferred): a 2nd camera
or per-court depth; tighter learned trajectory association once a 2nd-session fine-tune exists; or a
foreign-shuttle classifier. See `features/shuttle-tracking.md` "Honest limits".

## Wider airspace box needs a re-fine-tune (2026-06-30) — READY, gate is labeling effort
The box can be widened (`SHUTTLE_AIR_UP/DOWN/OUT`) to keep high clears + just-out landings in frame — the
geometry is correct — but the **current** fine-tuned TrackNet was trained on tight-box 512×288 crops, so a
wider crop regresses it (game1 A/B: F1 0.953→0.854, recall −13pp; train/inference crop mismatch). Defaults stay
at the legacy box. **To ship the wider box:** re-export the dataset crops at the wider geometry + re-fine-tune.
*Update 2026-07-02: no longer data-blocked — Session 2 (5 games) is recorded, ingested and processed. The
remaining gate is the labeling pass on Session-2 rallies (`verify --shuttle`), then re-export + re-fine-tune.*
Needed for the landing/in-out feature (`features/shuttle-landing-inout.md`), since out-landings fall outside
the tight box.

## Cross-session shuttle validation — READY, gate is labeling effort (2026-07-02)
The fine-tuned TrackNetV3 (promoted to production default) is only *scored* on session-1 game1 (F1 0.958).
The 5 Session-2 games processed at a tight 21–27% coverage band — an encouraging generalisation signal but
coverage-only, not precision/recall. **To close it:** label ≥1 Session-2 clip (`verify --shuttle`, white-wall
+ over-exposure rallies first) → `eval_shuttle --labels` out-of-session — the held-out metric
`shuttle-finetune-plan.md` always wanted. Target: cross-session F1 ≥ 0.80 before trusting new sessions
unlabeled (see `30-ml-pipeline.md` stage-5 targets).
