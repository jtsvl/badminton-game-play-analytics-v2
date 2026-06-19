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
   `status='failed'` + `error`. A crashed claim (running + stale `claimed_at`) is reclaimable after a timeout.
4. Calibration is a **gate**: after `ingest`, set `match.status='calibrating'` and pause until the trainer confirms
   `match_videos.calibration` (FR-6); then enqueue the heavy stages. After aggregation, `match.status='review'`
   (or `done` if review skipped).
5. Live status to the web via Supabase **Realtime** on `processing_jobs` / `matches.status`.

## Stage order & I/O
Stages mirror `30-ml-pipeline.md`. Each stage reads the frame cache + prior artifacts and writes the next.
`stage` string ∈ `ingest · calibrate · detect · pose · gate_track · shuttle · rally_hit · stroke · aggregate · overlay`.
(pose runs **before** gate_track: it supplies the ankle foot that gate_track projects instead of the bbox bottom.)

| Stage | Reads | Writes |
|---|---|---|
| ingest | storage video | frame cache, `match_videos` fps/dims/duration |
| calibrate | frames, user corners | `court_calibrations` (homography, mask_polygon, state) |
| detect | frames | detections.parquet `(frame,track?,x,y,w,h,score)` |
| pose | frames, detections | pose.parquet `(frame,track,kp[],kp_score[],foot_x,foot_y,foot_conf)` |
| gate_track | detections, pose, homography | tracks.parquet `(t,track_id,u,v,role)` (in-court only) |
| shuttle | frames, bg-median | shuttle.parquet `(t,u,v,visibility)` |
| rally_hit | shuttle, tracks | `rallies`, `events`(type=hit) |
| stroke | pose, shuttle, court xy | `events`(hit shot_type), `player_shot_stats` |
| aggregate | tracks, shots, rallies | `player_match_stats`, `pair_match_stats`, `match_rally_stats` inputs |
| overlay | tracks, shuttle, hits | `track_keyframes` (~10 Hz), hit markers |

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
- After detection+tracking, set `matches.status='awaiting_mapping'` until the trainer confirms `match_players.track_label`
  (identity mapping), then continue to stroke/aggregate.
- Estimates (`fastest`, speeds) carry their estimate nature (the UI labels them; don't present as measured).

## Config / env
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL`, buckets, `WORKER_ID`,
`WORKER_POLL_INTERVAL_SECONDS`, `WORKER_FRAME_CACHE_DIR`, `PYTORCH_ENABLE_MPS_FALLBACK=1`. See `.env.example`.

## Testing
Per-stage unit/integration tests on a short labeled clip assert the contracted artifacts/rows and numeric
sanity. Aggregation math is pure and TDD'd. An end-to-end test runs ingest→aggregate on a fixture and checks
the assembled report object matches expected shapes.
