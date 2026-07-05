# Feature — Stroke classification (pipeline stage 7)

**Status: SPEC ONLY — no code yet.** Captured honestly so it is build-ready when prioritised (next P3
build). FR-21/22/23. See `specs/30-ml-pipeline.md` stage 7 and `specs/40-worker-contract.md`.

## Goal
Classify each detected hit into a shot type from the coarse BWF-derived taxonomy and attribute it to the
hitting player → per-player `shots: [[type, count]]` in the report (FR-21), with everything the model
can't confidently call bucketed as **`unknown`** and surfaced in the UI, never hidden (FR-22).

## Current state (verified 2026-07-03)
- `stroke` is in `STAGE_ORDER` and `SIDE_ANGLE_STAGES` (`pipeline/__init__.py`) but **not** in `_REAL` —
  it resolves to the `_todo` no-op stub — and **not** in `RESUME_STAGES` (`__main__.py`). No
  `pipeline/stroke.py` exists.
- `player_shot_stats` rows are populated **only by `seed.sql`** today; the demo report renders those.
- Provisional hit counting already exists upstream: `rally_hit`'s `count_hits` (image-space vertical
  reversals) writes `events` `hit` rows with `payload {provisional: true}`, confidence 0.3.
- **Labeling tooling is BUILT** (`features/stroke-labeling.md`, 2026-07-03): `worker
  export-stroke-clips` cut all 738 Session-2 hits as real-video+audio clips; Label Studio 1.23
  runs self-hosted on the Mac. Labeling can start; the model below waits for labels.

## Inputs (all already produced by earlier stages)
- **`pose.parquet`** — full body26 `kp[]` / `kp_score[]` per track per frame (kept for exactly this
  stage), plus foot/`foot_source`.
- **`shuttle.parquet`** — `t, u, v, visibility, frame, px, py, conf` from the fine-tuned tracker.
- **`tracks.parquet`** — court-space player positions (`t, track_id, u, v, role, source`).
- **`events` `hit` rows** — stage 6's provisional hit frames, the anchor points to classify around.

## Model (per the locked stage-7 research)
**BST (MIT, public weights)** — a small transformer over player skeleton + shuttle xy + court xy;
hours to train on MPS. Taxonomy: **ShuttleSet 18 classes collapsed to the coarse 7** (FR-23):
`clear, smash, drop, net, drive, lift, serve` + **`unknown`** below the confidence threshold (FR-22).
Iterate the hard pairs (drop/net, clear/lift) before growing the taxonomy.

## Outputs / gaps to close when building
1. **Refine `events.hit`**: replace the provisional payload with `shot_type`, `player` (match_players
   attribution) and a real confidence; hit-fusion — the pose swing cue refines/confirms stage 6's
   velocity-reversal `count_hits` (they should converge; disagreement = a flag, not a silent pick).
2. **Write `player_shot_stats`**: a new **`Store.upsert_player_shot_stats` helper is required** — it does
   not exist today (only seed.sql fills that table). Unique key `(match_id, player_id, shot_type)` per
   `20-data-model.md`.
3. **Register the stage**: add `stroke` to `_REAL` and to `RESUME_STAGES` (after `rally_hit`, before
   `aggregate`), so the calibrate slice runs it and `aggregate` can consume shot counts.
4. `remap` interaction: shot attribution references player identity — a `remap` job after identity
   correction must also re-attribute shots (cheap: re-map track→player labels, no re-classification).

## Labels
~300–1k own-footage strokes via **Label Studio** — tooling built, see
`features/stroke-labeling.md` for the schema (coarse 7 + Unsicher/Kein Schlag + who-hit), the
±0.7 s clip window (wider than the ±0.5 s here, which was a swing-*search* margin, not a
viewing window), export CLI usage and the built import CLI. All 738 Session-2 hit events are
labeled and imported (2026-07-05): 695 training rows in `stroke_labels.parquet` (141 `unknown`)
+ 43 `no_hit` in `hit_detection_eval.parquet` — the labeling gate is cleared.

## Acceptance (initial targets — revise once own labels exist)
- Coarse accuracy **≥ 70 %** on held-out own-footage strokes to ship (literature ceiling 77–85 % on
  clean broadcast data; ours is harder: 30 fps, overhead angle, no netzoom).
- `unknown`-rate reported per match in the eval output — an honest "couldn't call it" is required
  behaviour, not a failure mode to minimise at accuracy's expense.

## Honest limits
- **Singles first** (locked decision #3); doubles attribution (who of the pair hit) is P3+.
- **Smash frames are sparse at 30 fps** — the fastest strokes have the fewest usable frames; expect
  smash recall to lag until 60 fps re-ingest is exercised.
- **Net-play attribution is occlusion-limited** — both players converge on the net; the overhead angle
  stacks them. `unknown` is the correct output there, not a guess.
- Shot counts feed `player_shot_stats.share_pct` and the report's shot table as **estimates** — the UI
  already labels derived numbers „Schätzung"; keep that framing.
