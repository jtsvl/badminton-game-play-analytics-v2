# Feature — Stroke classification (pipeline stage 7)

**Status: DESIGN LOCKED (2026-07-05) — build in progress on `feat/stroke-classification`.**
FR-21/22/23. See `specs/30-ml-pipeline.md` stage 7 and `specs/40-worker-contract.md`.
Nothing below is BUILT unless explicitly marked so.

## Goal
Classify each detected hit into a shot type from the coarse BWF-derived taxonomy and attribute it to the
hitting player → per-player `shots: [[type, count]]` in the report (FR-21), with everything the model
can't confidently call bucketed as **`unknown`** and surfaced in the UI, never hidden (FR-22).

## Current state (verified 2026-07-05)
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
  All 5 Session-2 games, all doubles, one session/lighting, 30 fps.

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

### Player-stream selection (doubles-safe, one code path)
BST's native two player streams are kept; per hit window we feed the **nearest-to-shuttle track per
court half** at the anchor frame (image-space distance, shuttle `px,py` vs skeleton centroid; if the
shuttle is undetected at the anchor, nearest detection within ±3 frames). Half membership from
`tracks.role` prefix (`near_*`/`far_*`). The selector uses only pipeline artifacts → train and
inference are consistent by construction; in singles it degenerates to "the only player per half".
The 513 hitter-labeled rows **measure** the selector, never train it: Phase-1 go/no-go gate ≥ ~85 %
(nearest-per-half == labeled hitter on the hitting half).
**Hitting half** is inferred from the shuttle's vertical-direction flip at the anchor (robust slopes
over ~5 frames each side; `null` if under-determined), stored in the payload, validated against the
hitter labels' half. It drives singles attribution only — never doubles guessing.
Fallbacks, in order: median-distance selection over the central 11 frames; then 4 padded player
streams with shared TCN weights.

### Skeleton + training init
Body26 (Halpe) indices 0–16 are exactly **COCO-17** → the stroke input is `kp[:17]`/`kp_score[:17]`.
**Train from scratch as v1** — the model is tiny (minutes–hours on MPS) and the broadcast-singles
pretrain transfers poorly to our steeper camera, doubles selection, and sparse shuttle coverage.
The ShuttleSet-pretrained trunk (`--init_ckpt`, `strict=False`, fresh 7-way head) is a pure
init-switch fallback if scratch stalls below the gate. Variant **BST_CG_AP**, `seq_len=30`
(≈ ±0.5 s @ 30 fps), `pose_style=J_only` first. Sanctioned hyperparameter space only:
{BST_0 vs BST_CG_AP, J_only vs JnB_bone, LR, epochs, class-weighted CE}.

### Vendoring
Upstream BST cloned to `worker/.cache/_vendor/BST` (git-ignored; SHA pinned in
`worker/training/bst_stroke_train.md`). We import **only the model classes** via the `sys.path`
pattern of `shuttle/tracknetv3.py`; upstream's per-dataset training scripts and preprocessing are
not used. Dataset builder, training loop, eval, and threshold logic live in-tree and are TDD'd.
`worker/training/bst_mps.patch` exists only if the model files themselves are device-pinned.

### Windows + normalization (train == inference)
One pure `build_sample()` shared by the dataset exporter and the pipeline stage:
- **Window**: 30 frames centered on the anchor (nearest cached frame to `t_s` via `frames.json`);
  zero-pad at clip edges, pad count recorded.
- **Shuttle stream** `(30, 2)`: `(px/W, py/H)`; gaps ≤ 6 frames linearly interpolated **within an
  arc, never across the anchor** (the contact breaks the arc); undetected → `(0,0)`.
- **Joints** 2×`(30, 17, 2)`: bbox-relative, center-aligned, normalized by bbox diagonal (BST
  convention); bbox = envelope of confident keypoints (pose.parquet stores no bbox); missing pose
  frames → zeros.
- **Positions** 2×`(30, 2)`: `tracks.parquet` `(u,v)` joined nearest-t; missing → zeros.
- fps guard: warn loudly if median frame dt deviates > 20 % from 1/30 (30 fps ingest is a stage-7
  requirement). Train-time temporal jitter ±3 frames absorbs the observed ±0.2 s stage-6 anchor drift.

### Events write-back + stats plumbing
- `pipeline/stroke.py` (runs after `rally_hit`, before `aggregate`) enriches every `type='hit'` row
  in `events.parquet`: payload `{shot_type, confidence, hit_disputed, half, model_version}` replaces
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
  ceiling 77–85 % on clean broadcast data; ours is harder: 30 fps, overhead angle, no netzoom).
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
- **Smash frames are sparse at 30 fps** — the fastest strokes have the fewest usable frames; expect
  smash recall to lag until 60 fps re-ingest is exercised.
- **Net-play attribution is occlusion-limited** — both players converge on the net; the overhead angle
  stacks them. `unknown` is the correct output there, not a guess.
- **Shuttle coverage is 21–27 %** per window on our footage — within-arc interpolation helps, but
  near-empty trajectory windows lean on pose alone; the exporter reports coverage per window.
- Shot counts feed `player_shot_stats.share_pct` and the report's shot table as **estimates** — the UI
  already labels derived numbers „Schätzung"; keep that framing.
