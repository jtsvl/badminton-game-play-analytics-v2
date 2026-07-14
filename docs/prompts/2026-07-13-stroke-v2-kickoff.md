# Stroke Classification v2 — fresh-start kickoff

You are starting a **fresh rebuild** of stroke classification for this badminton-analytics
monorepo (read `CLAUDE.md` first — persona, workflow, UI conventions, and locked decisions all
apply). The previous attempt never cleared its accuracy gate (best: GBDT 0.536 vs. a ≥0.70 gate)
and is NOT wired into the pipeline. You may inspect everything that exists, but you are **not
bound by any of its design choices**.

## 1. Scope — what you rebuild vs. what you reuse

**Rebuild from scratch (this project):**
- Hit detection: WHEN each stroke happens (current trajectory-reversal heuristic has ±0.2 s drift
  and 0.3 confidence — it is the known accuracy ceiling for everything downstream).
- Hitter attribution: WHO hit, as one of 4 doubles roles (vorne-links/vorne-rechts/hinten-links/
  hinten-rechts) or 2 singles roles. Full doubles attribution is first-class — all current
  footage is doubles. Net-play occlusion is the hard case; an honest „unsicher" output is
  required, not a forced guess.
- Stroke classification, per-stroke outcomes, and their UI (details below).

**Reuse as verified inputs (do NOT rebuild):**
- Court calibration + homography, shuttle tracking (fine-tuned TrackNetV3, F1 0.95), player
  detection/tracking, pose estimation (RTMPose Halpe-26), rally segmentation, the frame cache,
  Supabase schema/migrations flow, and the web app's overlay/report architecture.

**Reference, don't follow:** the old attempt lives on branch `feat/stroke-gbdt-traj` (worktree
`.claude/worktrees/stroke-gbdt-traj`) and in `specs/features/stroke-classification.md` /
`stroke-labeling.md`. Its clip-export + Label Studio import tooling worked well and may be
salvaged; its model choices are superseded by your Phase-0 research.

## 2. Stroke taxonomy (starting point — verify before building)

Label at **ShuttleSet-18 fine granularity**, train/report **8 coarse classes** until per-class
data supports finer. Coarse ← fine mapping:

| Coarse (trained) | Fine (labeled) |
|---|---|
| Aufschlag | kurzer Aufschlag · langer Aufschlag |
| Clear | Clear |
| Smash | Smash · Handgelenk-Smash |
| Drop | Drop · passiver Drop |
| Drive | Drive · flacher Treibball · Hinterfeld-Drive |
| Netzspiel | Netzdrop · Netz-Return · Cross-Netzdrop · Rush/Netzkill · Push |
| Lift | Lob/Lift · Abwehr-Lob |
| Abwehr | Abwehr-Drive |

Plus honesty options in labeling: „Unsicher" and „Kein Schlag" (never trained classes).
**Checkpoint:** as part of Phase 0, verify this taxonomy against the literature and against a
German coaching taxonomy (DBV terms), present any corrections to the user, and get sign-off
before any labeling starts. Adopting ShuttleSet's scheme 1:1 is deliberate — it maximizes
pretrained-model compatibility — so deviations need a stated reason.

## 3. Per-stroke outcomes & statistics

For every stroke: `rally continued` OR a rally-ending outcome. Rally-ending strokes get:
- **winner** (clean point) vs **error**, error type ∈ {Netz, Aus}, attributed to the correct
  player (a „Smash-Winner für A" and an „Aus-Schlag von B" are different stats).
- Note: net-cord detection (shuttle clipping the net) does not exist anywhere in the pipeline
  today — it is part of this project.
- **Forced vs unforced error** as a second-tier ESTIMATE (uncertainty surfaced in UI as
  „Schätzung", overridable in review). Even humans disagree here; never present it as fact.
- **Tactical patterns** as pure aggregation once strokes exist: per-rally stroke sequences
  (Aufschlag → Return → 3. Schlag …) enabling stats like „Punkte nach kurzem Aufschlag".

Existing signals to build on: rally segmentation emits `rally_end` + one `landing` per rally;
`specs/features/shuttle-landing-inout.md` classifies in/out (coverage currently poor — improving
it is in scope); `rallies.winner_side` exists as an estimate. Human-entered `matches.score` stays
authoritative and is never overwritten.

## 4. Recording environment (hard constraints)

- Public school sports hall: floor has **multiple overlapping court lines** for several sports;
  walls/background are **white**, and the shuttle is white (white-on-white contrast is a
  first-class requirement — already handled in tracking, but keep it in mind for any new
  video-consuming model).
- Single fixed camera (DJI, ultrawide, undistorted upstream), overhead-ish angle, 4K source,
  720p proxies for labeling. Both singles and doubles matches.

## 5. Phase 0 — Research (FIRST deliverable, before any code)

Do a thorough, source-cited research pass (web + papers), then present a written memo to the
user and STOP for sign-off. Cover:

1. **Stroke-classification models**: current SOTA on skeleton/trajectory/video inputs — BST
   (Badminton Stroke-type Transformer), TemPose, ST-GCN/PoseC3D-style skeleton models,
   video-native models; what pretrained weights actually exist for badminton (ShuttleSet
   checkpoints etc.), what input each needs (pose, shuttle track, hit frame), and realistic
   accuracy on amateur single-camera footage vs. the pro-broadcast datasets they're trained on.
2. **Datasets**: ShuttleSet/ShuttleSet22, VideoBadminton, BadmintonDB, CoachAI challenge data —
   size, taxonomy, license. Licensing note: this app is non-distributed/non-commercial; AGPL and
   NC licenses are fine (locked decision) — only hard-avoid deps with no license grant.
3. **Hit detection**: audio-based racket-impact detection (onset detection on the clip audio) —
   this is expected to become the primary hit-timing signal; plus A/V fusion approaches.
   Quantify expected timing precision vs. the current ±0.2 s drift.
4. **Hitter attribution** in doubles from one camera: what the literature does (pose-based swing
   detection near hit time, wrist/racket kinematics), honest limits under net-play occlusion.
5. **VLM pre-labeling**: Qwen3-VL (video-native) for pre-labeling stroke clips locally on a Mac
   Studio M4 Max / 64 GB (mlx-vlm or llama.cpp; the 30B-A3B MoE 4-bit or 8B variants). CRITICAL
   PRIOR: Qwen2.5-VL-7B with 9 sampled frames was piloted 2026-07-12 and FAILED at 0.267 top-1
   (frame sampling loses motion; artifacts in `.labeling/vlm_pilot/`). Qwen3-VL's video mode may
   fix exactly that — design a small measured pilot (~100 stratified clips, agreement vs. human
   labels) with an explicit adoption gate; do not assume it works.

Deliverable: research memo with citations + a recommended architecture (may be an ensemble/
staged design) + expected-accuracy honesty (state what's realistic with ~1–3 labeled sessions).
**User checkpoint — do not proceed without approval.**

## 6. Product requirements (user-approved feature set)

- **Confidence-aware overlay**: at each hit moment the video overlay shows stroke label + hitter,
  German labels, styled by confidence (confident = solid; low-confidence = muted „Schätzung"
  styling), consistent with the existing chip system (`OverlayChips.tsx`, `VideoOverlay.tsx`,
  time-keyed tracks `[{t,u,v}]`, `trackAt()` interpolation). Note: there is no discrete
  event-marker overlay yet — that's new UI, follow the existing SVG/no-chart-lib conventions.
- **Rally timeline strip**: horizontal strip under the video, one tick per stroke colored by
  coarse class, click-to-seek.
- **In-app stroke review**: correction flow modeled on the existing rally review — click a
  stroke, fix label/hitter/outcome, corrections are stored append-only (pattern: migration
  `0011_rally_corrections.sql`) and feed retraining. Label Studio remains for bulk labeling.
- **Active-learning loop**: model uncertainty + rare-class score drive the labeling queue;
  accept-rates tracked per model version; retraining is a one-command cycle. This serves the
  project's graduated-autonomy goal (auto-accept only once earned by measured edit rates).
- **Statistics**: per-player per-coarse-class counts, winner/error splits, tactical-pattern
  aggregates — shapes must follow `specs/20-data-model.md` conventions (`shots` = `[[type,count]]`).

## 7. Data plan

- **Relabel everything** under the fine taxonomy (user decision): re-export clips for all
  Session-2 games (738 hit events exist; the export tooling in `stroke_clips.py` is reusable),
  VLM pre-label if the pilot passes its gate, label in the local Label Studio (port 8010).
  The old 695 coarse labels serve as a cross-check/agreement signal only.
- **Re-ingest at 30 fps**: current feature parquets are 12 fps; temporal models want the full
  30 fps. Budget reprocessing accordingly.
- Labels always capture: fine stroke type, hitter (4-way), plus „Unsicher"/„Kein Schlag".

## 8. Ground rules

- **Spec-driven**: Phase 0 memo → then a new spec `specs/features/stroke-classification-v2.md`
  (supersedes the old spec explicitly) → then code. TDD for worker stages and aggregation math.
- **Gates before wiring**: define the acceptance gate in the spec — evaluation must be
  **cross-session** (train on session A, test on session B) once a second labeled session
  exists; until then LOGO-CV by video, and say plainly that same-session results overstate.
  Nothing ships to the pipeline below gate; a labeling-assist carve-out (never in pipeline) is
  allowed, mirroring the previous `labeling_assist_only` pattern.
- **Honest ML**: plausible-but-wrong is worse than absent. Per-class metrics always (rare
  classes hide inside means); confidence thresholds with abstention („Unsicher") are a feature.
- **User checkpoints**: after Phase-0 research, after the spec, after the VLM pilot, and before
  any promote/wiring decision. Run scope additions by the user first.
- Hardware: everything runs locally on the Mac Studio M4 Max 64 GB (PyTorch MPS primary, fp32,
  no MLX for conv-heavy models; CoreML only for hot fixed-shape models). No cloud training
  without asking.
