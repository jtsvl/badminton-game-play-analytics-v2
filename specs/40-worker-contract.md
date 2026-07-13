# 40 — Worker Contract

How the local CV/ML worker and Supabase cooperate. The worker is a long-running Python process on the Mac
Studio; it owns no HTTP surface — it **polls** `processing_jobs`, does work, and writes rows with the
**service-role key**.

## Job lifecycle
1. Web upload → insert `match` (`status='uploaded'`) + `match_videos` + a `processing_jobs` row (`stage='ingest'`, `status='queued'`).
2. Worker claims the oldest queued job atomically:
   ```sql
   update processing_jobs set status='running', claimed_by=$worker, claimed_at=now(), started_at=now()
   where id = (select id from processing_jobs where status='queued'
               order by created_at for update skip locked limit 1)
   returning *;
   ```
3. Worker advances `stage` and `progress` (0..1) as it runs; on success sets `status='succeeded'`, on error
   `status='failed'` + `error`. A crashed claim (running + stale `claimed_at`) is reclaimable after a timeout:
   every `set_stage` refreshes `claimed_at` (a per-stage **heartbeat**), and `claim_job` — when the queue is
   empty — reclaims a running job whose heartbeat is older than `WORKER_STALE_JOB_TIMEOUT_S` (default 3600 s),
   using the same conditional-update/`SKIP LOCKED` guard as a normal claim. Honest limits: the heartbeat
   granularity is one stage, so the timeout must outlast the longest single stage; and a poison job that kills
   its worker will be re-claimed once per timeout (no attempt counter — acceptable for one local worker).
4. Calibration is a **gate**: after `ingest`, set `match.status='calibrating'` and pause until the trainer confirms
   `match_videos.calibration` (FR-6); then enqueue the heavy stages. After aggregation, `match.status='review'`
   (or `done` if review skipped).
5. Live status to the web via Supabase **Realtime** on `processing_jobs` / `matches.status`.

## Stage order & I/O
Stages mirror `30-ml-pipeline.md`. Each stage reads the frame cache + prior artifacts and writes the next.
`stage` string ∈ `ingest · calibrate · detect · pose · gate_track · shuttle · rally_hit · stroke · aggregate · overlay · remap · rally_sync · proxy`.
(pose runs **before** gate_track: it supplies the ankle foot that gate_track projects instead of the bbox bottom.)

A job's `stage` selects the **slice** the worker runs (`__main__.py`):

| Job stage | Runs | Match status after |
|---|---|---|
| `ingest` | `[ingest, proxy]` (proxy = 720p playback artifact, never raises) | `calibrating` |
| `calibrate` | `RESUME_STAGES` = `[calibrate, detect, pose, gate_track, shuttle, rally_hit, stroke*, aggregate, overlay]` (*stroke lands here in the stage-7 build — `features/stroke-classification.md`; until then it is not in `RESUME_STAGES`) | `review` |
| `remap` | `[remap, rally_sync, aggregate, overlay]` (FR-35 — reuses cached `tracks.parquet`, no re-detect; `rally_sync` keeps DB rally spans/trainer edits) | `params.return_status` (default `done`) — lets backfills re-run overlay on `review` matches without skipping the PlayerReview gate |
| `rally_review` | `[remap, rally_sync, aggregate]` (FR-35 — trainer dropped/merged rallies; no overlay, it never reads rallies) | `params.return_status` (default `done`) |
| `proxy` | `[proxy]` standalone (backfill; `features/video-proxy.md`) | unchanged |

`rally_review`'s end status returns the match to where it was before the edit — editing at `review`
must not skip the PlayerReview confirmation; editing at `done` stays `done`.

| Stage | Reads | Writes |
|---|---|---|
| ingest | storage video | frame cache, `match_videos` fps/dims/duration |
| calibrate | frames, user corners | `court_calibrations` (homography, mask_polygon, state) |
| detect | frames | detections.parquet `(frame,track?,x,y,w,h,score)` |
| pose | frames, detections | pose.parquet `(frame,track,kp[],kp_score[],foot_x,foot_y,foot_conf,foot_source)` |
| gate_track | detections, pose, homography | tracks.parquet `(t,track_id,u,v,role,source)` (in-court only) |
| shuttle | frames, calibration (H + `mask_polygon_px`) | shuttle.parquet `(t,u,v,visibility[,frame,px,py,conf,tracker])` |
| rally_hit | shuttle, tracks, match_players (side binding) | `rallies` (+ `winner_side` from out landings) + `events`(serve/hit/rally_end/landing); debug `rallies.parquet`/`events.parquet` |
| stroke | pose, shuttle, tracks, events (stage-6 hit anchors), match discipline | enriched `events` hit payloads `{shot_type, confidence, hit_disputed, half, model_version}` via `replace_rallies`; debug `events.parquet` rewrite. **Not** `player_shot_stats` — counts are derived in `aggregate` (singles-gated) so `remap`/`rally_review` re-attribute without re-classification; `stroke` stays out of those slices |
| aggregate | tracks, **rallies** (in-rally gating), enriched events (shot counts) | `player_match_stats`, `player_shot_stats` (`count`+`share_pct` via `upsert_player_shot_stats`; `errors` column stays the landing writer's), `pair_match_stats`, `match_rally_stats` inputs |
| overlay | tracks, shuttle, pose + frames.json (skeleton), rallies (gating) | `track_keyframes` (~10 Hz), `pose_keyframes` (in-rally skeleton, `features/pose-skeleton-overlay.md`) |
| rally_sync | DB `rallies` + `events` (authoritative, incl. trainer edits) | cache `rallies.parquet`/`events.parquet` + re-uploaded artifacts, `ctx[rallies_path/events_path]`; `rally_corrections` rows when `job.params.corrections` is present (FR-35 eval corpus, `features/rally-shot-review.md`) |
| proxy | source video (local `videos/` or bucket) | `analysis-artifacts/<video_id>/proxy_720p.mp4` (TUS chunked — Kong kills single requests at ~60 s), `match_videos.proxy_path`. ffmpeg host dep; **never raises** (`features/video-proxy.md`) |
`rally_hit` runs all trajectory/serve logic in **normalized image space** (the ground homography can't place an
airborne shuttle); it writes the debug parquets so `aggregate` can gate player movement to in-rally spans
(whole-clip fallback when no rallies). Landing + in/out (BUILT — `features/shuttle-landing-inout.md`) is a
**sub-step of stage 6, not a wired stage**: `build_rallies` emits one `landing` event per rally
(`worker/rally/landing.py`), `rally_hit` sets `winner_side` from located out landings via the
track_label↔role binding, and `aggregate` writes `player_shot_stats.errors` (`shot_type='unknown'` until
stage 7; singles only; 0 overwrites stale counts after trainer drops).

**Intermediate artifacts = Parquet** in `analysis-artifacts` storage (or local cache), keyed by `video_id`.
Final aggregates → Postgres rows exactly matching `20-data-model.md` (the web's only read surface).

## Idempotency & re-runs
- Stages are **idempotent**: re-running a stage replaces its artifacts/rows for that `video_id`/`match_id` (upsert on the unique keys).
- A job can request a stage range (`params.from_stage`, `params.to_stage`) for re-processing after a model/label update.
- Side-angle videos (`angle='side'`) run a reduced stage set (skip court-metric movement; keep shuttle + pose).

## Contracts (must hold)
- Court coords normalized `u,v ∈ [0,1]`; never write pixel coords to web-facing rows.
- `player_match_stats.zones` ints sum ≈100; `timeline` length = ceil(durationMin/5); `heat` = `[{x,y,w,s}]`.
- All worker writes use service-role; never expose that key to the web.
- Each worker run logs to **`job_runs`** (job, started/finished, is_ok, detail) for audit/observability.
  (`started_at` defaults to the insert time ≈ `finished_at` — good enough for an audit row.)
- Identity mapping is reviewed **after** aggregation, not mid-pipeline: the resume slice auto-maps
  `match_players.track_label` by role and ends at `matches.status='review'`; the trainer confirms/corrects
  the mapping in `/match/:slug/review` (FR-35, see specs/features/player-identity-review.md) and a `remap`
  job recomputes stats/overlays, ending at `'done'`. The `awaiting_mapping` status stays in the enum as a
  **reserved, currently unset** value (kept to avoid a pointless migration).
- Estimates (`fastest`, speeds) carry their estimate nature (the UI labels them; don't present as measured).
- Stage-6 rally fields (`winner_side`, `end_shot_type`, `shot_count`) and the landing in/out call
  (`events.type='landing'`, BUILT) are **estimates decoupled from the human-entered `matches.score`** — surfaced
  as „Schätzung" with a confidence/uncertainty band, never as an umpire call or the scoreboard.
- Phase-4 recovery is self-labelling: `tracks.parquet.source` ∈ `detected`/`reid`/`interp` and
  `pose.parquet.foot_source` ∈ `ankle`/`head`/`none` mark re-ID'd, interpolated or head-grounded positions so the
  overlay surfaces them as „Schätzung" — never as a plain detection. The `eval_corrections` scorecard reports
  recovered (re-ID/interp) court error in a **separate** bucket, never folded into the detected near/far stats.

## Config / env
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL`, buckets, `WORKER_ID`,
`WORKER_POLL_INTERVAL_SECONDS`, `WORKER_STALE_JOB_TIMEOUT_S` (3600 — stale-claim reclaim, see lifecycle
step 3), `WORKER_FRAME_CACHE_DIR`, `WORKER_LOCAL_VIDEO_DIR` (decode a local copy in place instead of
downloading), `CALIB_FRAME_COUNT` (12 calibration snapshots, 0007), `PYTORCH_ENABLE_MPS_FALLBACK=1`.
See `.env.example`.
Foot/recovery tuning (all default off / conservative): `FOOT_BIAS_K` (bbox-nudge, 0), `USE_POSE_FOOT` (0),
`POSE_CONF_THR` (0.3), `GAP_FILL_MAX_FRAMES` (longest short hole the re-ID interp will fill, 8).
Shuttle (stage 5): `SHUTTLE_TRACKER` (tracknetv3|wasb|fake, **default `tracknetv3`** — the fine-tuned
own-footage winner; wasb is the frozen pre-fine-tune baseline), `SHUTTLE_CONF_THR` (0.5 — peak
below = a real dropout, not a fabricated point), `SHUTTLE_MAX_STEP` (0.5 court units, isolated-spike guard),
`SHUTTLE_MASK` (1 — black out outside the target-court airspace), `SHUTTLE_AIR_UP`/`SHUTTLE_AIR_SIDE`/
`SHUTTLE_AIR_DOWN`/`SHUTTLE_AIR_OUT` (airspace shape; down/out default 0 = legacy box — the fine-tune
was trained on that crop, see `specs/fix-later.md`), `SHUTTLE_BBOX_MARGIN`, `SHUTTLE_OVERLAY_GAP_S`
(0.3, overlay gap cap), `SHUTTLE_CHUNK_FRAMES` (240 — bounded-memory chunked prediction, see
features/shuttle-tracking.md; 0 = legacy single pass), `WASB_REPO`/`WASB_MODEL`,
`TRACKNETV3_REPO`/`TRACKNETV3_MODEL` (vendored repo + weights, git-ignored).
Stroke (stage 7, `features/stroke-classification.md`): `BST_REPO` (vendored BST clone, default
`worker/.cache/_vendor/BST`), `STROKE_MODEL` (default `models/weights/stroke/bst_best.pt`; sidecar
`bst_best.json` carries model_version/classes/variant/seq_len/τ/git SHA — same promoted-weights
convention as shuttle below).
Landing/in-out (stage-6 sub-step, `features/shuttle-landing-inout.md` — **all defaults untuned until a
real-footage run**): `LANDING_WINDOW_S` (0.5), `LANDING_MAX_WINDOW_DETS` (8), `LANDING_MIN_DETS` (4),
`LANDING_MAX_END_GAP_FRAMES` (2), `LANDING_EDGE_MARGIN` (0.04 → located=false),
`LANDING_UNCERTAIN_EDGE` (0.08 → forced uncertain), `LANDING_FLAT_VY_ABS` (0.25 norm/s),
`LANDING_FLAT_VY_FRAC` (0.35), `LANDING_BAND_CM` (45), `LANDING_MAX_OUT_CM` (100 — projections farther
outside every line degrade to `located=false, reason=implausible`: above-floor rests like racket catches;
validated against the 2026-07-03 real-footage false positive), `LANDING_NET_MARGIN_M` (1.0).
**Promoted-weights convention:** `models/weights/shuttle/ckpts/TrackNet_best.pt` is
the production checkpoint the code loads by default; the upstream baseline stays byte-identical as
`TrackNet_baseline.pt`, and each promoted fine-tune also keeps a named copy (e.g. `TrackNet_ft_session1.pt`)
so promotions are reversible and auditable.
`visibility` = tracker heatmap peak confidence in [0,1]; sub-threshold frames are absent (an honest gap).
`shuttle` **and** `rally_hit` are in `_REAL` **and** `RESUME_STAGES` (`worker/__main__.py`) — shuttle is
verified on real footage (see specs/features/shuttle-tracking.md "Wiring note"); `eval_shuttle` remains the
standalone bench/A-B harness.

## Operations (current reality — unbuilt pieces flagged as such)
- **Frame cache:** `<WORKER_FRAME_CACHE_DIR>/<video_id>/` holds the decoded JPEGs (`frames/NNNNNN.jpg`),
  the `frames.json` manifest, and the stage parquets (detections/pose/tracks/shuttle/rallies/events).
  **No automatic eviction exists.** Disk budget: ~0.8 MB per 4K JPEG → a ~7-min game at 12 fps analysis
  ≈ 4 GB (measured); at 30 fps ≈ 10 GB. It is safe to delete a video's cache dir once the match is
  `done`: final aggregates live in Postgres, parquet artifacts persist in the `analysis-artifacts`
  bucket, and `remap` re-downloads `tracks.parquet` on demand (`remap.py`). Re-labeling/re-benching a
  deleted clip needs a re-decode first.
- **Stale-job recovery:** the `claimed_at` heartbeat reclaim (lifecycle step 3) picks a stranded
  `running` job back up after `WORKER_STALE_JOB_TIMEOUT_S` (default 1 h). To recover one *immediately*
  (e.g. after a hard `kill` mid-job, observed 2026-07-01): `update processing_jobs set status='failed',
  error='...' where status='running' and claimed_by='<worker>'`, then re-enqueue.
- **Retries: none.** A `failed` job stays failed; re-enqueue manually. The worker survives job failures
  (records the error, keeps polling).

## Testing
Per-stage unit/integration tests on a short labeled clip assert the contracted artifacts/rows and numeric
sanity. Aggregation math is pure and TDD'd. What exists today: a `run_pipeline` orchestration test (fake
stages: order + progress + job_runs/finish wiring) and a stage-level chunked-shuttle test (synthetic
frames + FakeTracker: chunked ≡ single-pass parquet). A *real* ingest→aggregate e2e needs video decode +
model weights — heavier than the rest of the suite (pure/fake-backed, runs without the ml extra), so it
stays an aspiration for the Mac Studio, not CI. Judged & skipped 2026-07-01: caching `get_match_players`
across aggregate/overlay (one ~100 ms REST call per job vs minutes of GPU — not worth the ctx coupling).
