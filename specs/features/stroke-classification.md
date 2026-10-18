# Feature — Stroke classification (pipeline stage 7)

**Status: DESIGN LOCKED (2026-07-05) — build in progress on `feat/stroke-classification`.**
FR-21/22/23. See `specs/30-ml-pipeline.md` stage 7 and `specs/40-worker-contract.md`.
Nothing below is BUILT unless explicitly marked so.

## Goal
Classify each detected hit into a shot type from the coarse BWF-derived taxonomy and attribute it to the
hitting player → per-player `shots: [[type, count]]` in the report (FR-21), with everything the model
can't confidently call bucketed as **`unknown`** and surfaced in the UI, never hidden (FR-22).

## Current state (verified 2026-07-05)
- **Phase 2 (training + LOGO-CV eval) is BUILT and run — the acceptance gate FAILED, so the
  pipeline stays unwired.** In-tree, TDD'd: `stroke/model.py` (BST4 — the vendored BST_CG_AP
  generalized to the 4 quadrant slots with presence-masked AimPlayer/CleanGate; collision-safe
  vendor import), `stroke/train.py` (LOGO loop, τ sweep, `worker train-stroke` CLI), runbook
  `worker/training/bst_stroke_train.md` (vendored SHA + every command).
- **Measured 2026-07-05** (`PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker train-stroke
  --seed 42 [--lr 3e-4]`, full sanctioned sweep in the runbook): best mean LOGO-CV accuracy
  **0.400 ± 0.056** (pooled 0.403) vs the ≥ 0.70 gate; τ targets unsatisfiable (knee at 0 —
  no confidence separation); net + drive recall 0.00; Unsicher abstention 0.00. Diagnosis:
  train acc 0.945 vs held-out 0.402 — a **cross-game generalization gap**, not underfit; the
  sanctioned hyperparameter space is exhausted. First levers: near-half recovery v2 (below)
  and more labeled sessions — user decision before any further build.
- `stroke` is in `STAGE_ORDER` and `SIDE_ANGLE_STAGES` (`pipeline/__init__.py`) but **not** in `_REAL` —
  it resolves to the `_todo` no-op stub — and **not** in `RESUME_STAGES` (`__main__.py`). No
  `pipeline/stroke.py` exists.
- `player_shot_stats` counts come **only from `seed.sql`** today; `aggregate` writes only the
  `errors` column (out-landings, `shot_type='unknown'`, singles-gated).
- Provisional hit counting exists upstream: `rally_hit`'s `count_hits` writes `events` `hit` rows
  with `payload {provisional: true}`, confidence 0.3.
- **Labels are BUILT and imported** (see `features/stroke-labeling.md`): 695 training rows in
  `worker/worker/.labeling/stroke_labels.parquet` (drop 165, lift 153, unknown 141, clear 78,
  smash 61, serve 55, drive 22, net 20) + 43 `no_hit` rows in `hit_detection_eval.parquet`.
  All 5 Session-2 games, all doubles, one session/lighting; source video 30 fps, but the frame
  cache all stages read is the **12 fps analysis cadence** (`ingest.DEFAULT_ANALYSIS_FPS`).

## Model
**BST** — Chang et al., *BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition
in Racket Sports*, CVPRW 2026, arXiv:2502.21085,
<https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer> (MIT, public weights). Per-stream
TCNs (two player skeletons, shuttle trajectory, player court positions) → 2-layer transformer encoder.
Its expected inputs match our artifacts 1:1 — MMPose-style 2D skeletons, **TrackNetV3** shuttle
trajectory in [0,1], court positions from feet in [0,1].
Taxonomy: coarse 7 (FR-23): `clear, smash, drop, net, drive, lift, serve` + **`unknown`** below the
confidence threshold (FR-22). *(Correction to earlier revisions: ShuttleSet is 35 classes / 25 merged;
BadmintonDB is the 18-class set.)* Iterate the hard pairs (drop/net, clear/lift) before growing the
taxonomy.

## Locked decisions (user, 2026-07-05 — do not relitigate silently)
1. **Eval = leave-one-game-out CV**: 5 folds, whole-game splits keyed by `video_id` (G1 151, G2 207,
   G3 167, G4 144, G5 69 hits). Headline = mean accuracy ± spread, per-fold n and per-class recall
   reported. Honest caveat everywhere the number appears: one session/lighting → this measures
   cross-game, not cross-session generalization.
2. **Unsicher rows**: train the 7-class softmax on the **554 confident rows only**; the 141 `unknown`
   rows are a held-out confidence-calibration/eval set (a good model should be unconfident on them —
   the overlap is reported). `unknown` output = below-threshold abstention, never a trained class.
3. **Doubles (all current footage)**: refine `events.hit` for every match; write `player_shot_stats`
   **only for singles matches**. No heuristic hitter guessing — who-of-the-pair attribution is P3+
   (hitter labels banked).
4. **Hit-refinement scope = classify + flag only**: swing-cue disagreement → `hit_disputed: true` in
   the payload, rate reported. No re-timing, no hit deletion. The 43 `no_hit` labels are the stage-6
   hit-precision eval number (43/738 ≈ 5.8 % FP → ≈ 94.2 % precision).

## Design

### Player streams: 4 fixed quadrant slots + pose recovery (REVISED 2026-07-05)
The original two-stream nearest-per-half selector was **measured and rejected** at the Phase-1 gate
(all 554 hitter-labeled hits, all 5 games):
- the hitter's role track exists at the anchor for only **54.9 %** of hits (far half absent 40 % —
  `gate_track` loses the far pair for long stretches; ±1.0 s tolerance recovers < 3 pp);
- nearest-per-half picks the labeled hitter **76.6 %** when the track exists (time-aware eval —
  an earlier 6 % readout was an eval bug: track ids churn, role↔id lookups must be time-aware);
- every "which half hit" signal fails on this footage: trajectory vertical-flip **48 %**,
  shuttle-py vs image midline **49 %** (near/far centroids overlap in image y on 3 of 5 games).
  The `half` payload field is dropped from v1; singles attribution later comes from role tracks.

**Revised design (user-locked 2026-07-05):** one stream per court quadrant —
`near_left, near_right, far_left, far_right` — no hitter-picking logic at all. Slot membership is
the gate_track role (time-aware: the role segment with rows nearest the anchor), singles roles map
`near→near_left`, `far→far_left`. An absent player is an all-zeros stream. **Pose recovery** fills
slots that role tracks leave empty: pose.parquet sees everyone (incl. the lost far pair — only the
court-track bookkeeping drops them), so unassigned pose tracks whose precomputed feet
(`foot_x/foot_y`, `foot_conf`) project through the calibration homography into the court
(±margin, reusing `geometry.apply_homography`; quadrant = `v>0.5 → near`, `u<0.5 → left`, the
absolute form of `geometry.role_assign`'s sort) are slotted in; their position stream is the
per-frame foot projection. The 513 hitter labels **measure** slot coverage, never feed features:
Phase-1 gate = labeled hitter's slot filled ≥ ~85 % (reported split by source: track vs recovery).

**Phase-1 gate results (measured 2026-07-05, `export-stroke-dataset`, 695 samples, 554 labeled):**
- hitter's slot filled (strict slot name): **83.6 %** (463/554; 308 via track, **155 via recovery**
  — recovery is a third of all fills and lifted the far half from its 60 % track coverage).
- hitting half shows **≥ 1 player: near 94.0 % / far 100 %**; **both players: near 52.3 % /
  far 82.9 %** → the hitter's skeleton reaches the model in ≥ 67.1 % (guaranteed) to ~96.9 %
  (upper bound) of samples. The residual hole is mostly the near-half **partner** (mutual
  occlusion / feet at the frame edge), not the hitter.
- shuttle stream: median window coverage **1.00** (16-frame window), min 0.12, zero-coverage 0.
- Caveat for P3 attribution (not shot_type training): the labels' „links/rechts" are
  viewer-relative, slots are court-u-relative — the axis orientation per half must be verified
  before hitter labels are used as track-level ground truth (misses cluster as cross-half-only
  candidates, consistent with a left/right mirror on one half).
- Verdict: gate intent met (hitting half visible 94–100 %, dense shuttle stream); proceed to
  training with the LOGO-CV ≥ 70 % gate as arbiter. First lever if it fails: near-half
  recovery v2 for the missing partner.

### Skeleton + training init
Body26 (Halpe) indices 0–16 are exactly **COCO-17** → the stroke input is `kp[:17]`/`kp_score[:17]`.
**Train from scratch as v1** — the model is tiny (minutes–hours on MPS) and the broadcast-singles
pretrain transfers poorly to our steeper camera, 4-slot input, and sparse shuttle coverage (the
4-slot revision also breaks the pretrained 2-player input shape — from-scratch is now structural;
a pretrained-trunk fallback would need stream surgery and is correspondingly weaker). Variant
**BST_CG_AP** adapted to 4 player streams with shared TCN weights, **`seq_len=16`** ≈ ±0.67 s at
the pipeline's real **12 fps analysis cadence** (`ingest.DEFAULT_ANALYSIS_FPS = 12`, discovered at
the Phase-1 gate — the earlier "30 frames ≈ ±0.5 s @ 30 fps" actually spans 2.5 s and neighboring
hits). `pose_style=J_only` first. Sanctioned hyperparameter space only:
{BST_0 vs BST_CG_AP, J_only vs JnB_bone, LR, epochs, class-weighted CE}. 30-fps re-ingest
(`analysis_fps` job param) stays a separately measured later experiment if smash/drive recall
disappoints — it would not fix track coverage (user-locked 2026-07-05).

### Vendoring
Upstream BST cloned to `worker/.cache/_vendor/BST` (git-ignored; SHA pinned in
`worker/training/bst_stroke_train.md`). We import **only the model classes** via the `sys.path`
pattern of `shuttle/tracknetv3.py`; upstream's per-dataset training scripts and preprocessing are
not used. Dataset builder, training loop, eval, and threshold logic live in-tree and are TDD'd.
`worker/training/bst_mps.patch` exists only if the model files themselves are device-pinned.

### Windows + normalization (train == inference)
One pure `build_sample()` shared by the dataset exporter and the pipeline stage:
- **Window**: 16 frames centered on the anchor (nearest cached frame to `t_s` via `frames.json`);
  zero-pad at clip edges, pad count recorded.
- **Shuttle stream** `(16, 2)`: `(px/W, py/H)`; gaps ≤ 6 frames linearly interpolated **within an
  arc, never across the anchor** (the contact breaks the arc); undetected → `(0,0)`.
- **Joints** 4×`(16, 17, 2)` (slot order `near_left, near_right, far_left, far_right`):
  bbox-relative, center-aligned, normalized by bbox diagonal (BST convention); bbox = envelope of
  confident keypoints (pose.parquet stores no bbox); missing pose frames / empty slots → zeros.
- **Positions** 4×`(16, 2)`: `tracks.parquet` `(u,v)` joined nearest-t for role-track slots;
  per-frame homography-projected feet for pose-recovered slots; missing → zeros.
- fps guard: warn loudly if median frame dt deviates > 20 % from 1/12 (the 12 fps analysis cadence
  is a stage-7 input assumption). Train-time temporal jitter ±1 frame absorbs the observed ±0.2 s
  stage-6 anchor drift.

### Events write-back + stats plumbing
- `pipeline/stroke.py` (runs after `rally_hit`, before `aggregate`) enriches every `type='hit'` row
  in `events.parquet`: payload `{shot_type, confidence, hit_disputed, model_version}` replaces
  `{provisional: true}`; event `confidence` = model confidence. Re-uploads the artifact and calls the
  existing `store.replace_rallies(match_id, rallies, events)` once (events have no upsert key by
  design; consumers join `video_id + t_s` — ids regenerate, which is the documented contract).
- **Shot counts are written by `aggregate`, not by the stroke stage** (deviation from this spec's
  earlier phrasing, deliberate): `aggregate` runs in every slice, already owns the singles gate and
  the `track_label`→player binding, and `rally_sync` refreshes cache events from the DB — so a
  `remap`/`rally_review` job re-attributes shots **for free** (re-mapping, no re-classification) and
  `stroke` stays out of those slices. New `Store.upsert_player_shot_stats` sends only
  `count` + `share_pct` on conflict key `(match_id, player_id, shot_type)` — the column-split twin of
  `upsert_player_shot_errors` (errors-only), so the two writers coexist on the same row. `unknown`
  is counted as its own row (FR-22: surfaced, never hidden).
- `hit_disputed`: reuse `rally/swing.py::swing_events` — no swing activity from the selected players
  within ±0.3 s of the anchor → flagged. Rate reported per match.

### Confidence threshold τ (FR-22)
All predictions are out-of-fold from the LOGO-CV. Sweep τ over the softmax-max grid; pick the
smallest τ with OOF covered-accuracy ≥ 75 % subject to coverage ≥ 70 % on confident rows (if
unsatisfiable: take the accuracy-coverage knee and say so). Report at τ*: covered accuracy,
coverage, Unsicher-abstention rate (the honesty differential — if the model is confident on Unsicher
rows, report the overlap, don't tune it away), and unknown-rate **per match**. One global τ, shipped
in the model's sidecar JSON (`bst_best.json`: model_version, classes, variant, seq_len, τ, git SHA,
date). Weights at `models/weights/stroke/` (git-ignored), promoted-checkpoint convention, env
override `STROKE_MODEL` — all mirroring the stage-5 shuttle conventions.

### Build gaps this closes (registration points)
1. `pipeline/__init__.py`: add `"stroke"` to `_REAL`; create `pipeline/stroke.py` with `run(ctx)`.
2. `__main__.py`: add `"stroke"` to `RESUME_STAGES` after `rally_hit`, before `aggregate`; new
   `export-stroke-dataset` subcommand.
3. `store.py`: `upsert_player_shot_stats` as above.
4. Graceful degradation: missing weights or missing pose/tracks/shuttle → warn + leave events
   untouched; the stage must never crash the slice.

## Labels
~300–1k own-footage strokes via **Label Studio** — BUILT, see `features/stroke-labeling.md` for the
schema (coarse 7 + Unsicher/Kein Schlag + who-hit), the ±0.7 s clip window, export CLI usage and the
import CLI. All 738 Session-2 hit events are labeled and imported (2026-07-05): 695 training rows in
`stroke_labels.parquet` (141 `unknown`) + 43 `no_hit` in `hit_detection_eval.parquet` — the labeling
gate is cleared. **Training joins labels → parquets via `video_id + t_s`, never `event_id`**
(`events.id` uuids regenerate on every stage-6 re-run; `replace_rallies` is delete-then-insert).

## Acceptance
- Coarse accuracy **≥ 70 %** as the **LOGO-CV mean** on held-out own-footage games (literature
  ceiling 77–85 % on clean broadcast data; ours is harder: 12 fps analysis cadence, overhead
  angle, no netzoom).
  The model gate blocks pipeline wiring: no stage registration until the mean clears 70 %.
- `unknown`-rate reported per match in the eval output — an honest "couldn't call it" is required
  behaviour, not a failure mode to minimise at accuracy's expense.
- Per-class recall reported (drive n=22 and net n=20 are expected to lag — reported, never averaged
  away); confusion matrix for the hard pairs (drop/net, clear/lift).
- No metric enters this spec without the command (+ seed) that produced it.

## Honest limits
- **All current labels are doubles** — the earlier "singles first" framing is stale for this batch:
  the stage classifies doubles footage now; only per-player `player_shot_stats` stays singles-gated,
  and who-of-the-pair attribution (the banked 4-way hitter label) is P3+.
- **One session, one lighting, 5 games**: LOGO-CV measures cross-game, not cross-session
  generalization. A Session-3 re-eval is the real test (fix-later note).
- **Smash frames are sparse at the 12 fps analysis cadence** — the fastest strokes have the fewest
  usable frames; expect smash recall to lag until a higher-fps re-ingest (`analysis_fps` job param)
  is exercised.
- **Net-play attribution is occlusion-limited** — both players converge on the net; the overhead angle
  stacks them. `unknown` is the correct output there, not a guess.
- **Shuttle coverage is 21–27 %** per window on our footage — within-arc interpolation helps, but
  near-empty trajectory windows lean on pose alone; the exporter reports coverage per window.
- Shot counts feed `player_shot_stats.share_pct` and the report's shot table as **estimates** — the UI
  already labels derived numbers „Schätzung"; keep that framing.

## Alternatives evaluated (2026-10-18)

**TypeSafe Jev (1.13 / 1.14) — not a candidate.** Asked whether TypeSafe's Jev could beat the models
above. Jev is a hosted "System One" **decision model**: text state in ("natural-language text, sent as
a string, a JSON object, or an array of text values"), typed answers out (Choice / Score / Noul /
Box, each with probabilities), trained with "Reinforcement Learning for Calibrated Decisions",
"not fine-tuned or LoRA-adapted with customer data". TypeSafe's own docs: "Images, audio, and
video are not supported (yet)"; 1.14 (2026-10-14) adds still images behind an experimental
danger-zone header. Hosted API only, waitlisted, no open weights, no on-prem. A stroke is motion
over ~0.5 s, the worker runs locally on the Mac Studio, and our structured decisions (hit disputed,
quadrant) are geometric rules — so there is no place in stage 7 where Jev fits, and no evidence of
fine-grained sports competence anywhere (all performance claims are TypeSafe's own). **Decision: do
not integrate; revisit only if a video modality ships with open weights or an on-prem option.**

**VideoJev-4B (MCG-NJU, Apache-2.0) — unproven, being measured.** The video model that borrowed the
"System One" name (not a TypeSafe product): a Qwen3.5-4B multimodal LLM with a candidate-scorer head
that returns a softmax over text options for a clip. Pretrained video representation → plausibly a
different failure mode than BST's cross-game memorization; never evaluated on fine-grained sports
actions. **Zero-shot pilot on the 695 labeled clips: `features/stroke-vlm-pilot.md`** (dev game G5
picks the prompt, the other 4 games give the number; decision tiers closed / labeling_assist /
finetune_candidate / gate). **Result: pending the Mac run.** The 35B-A3B variant does not fit the
64 GB Mac (≈ 70 GB bf16) and is a cloud-only second row if the 4B lands in the assist band.
