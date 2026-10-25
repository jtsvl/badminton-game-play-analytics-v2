# Feature — 30 fps analysis cadence (measure 12 vs 30 on one game, then flip the default)

**Status: BUILT, NOT MEASURED (2026-10-25).** Every frame-count threshold in the worker is now a
duration/rate converted from the cache's *measured* cadence, a non-destructive bench driver exists, and the
eval tools report the metrics below. The 12-vs-30 numbers come from the Mac run (§ Measurement); until the
user has seen them `ingest.DEFAULT_ANALYSIS_FPS` **stays 12.0**. Branch `claude/new-session-kuz5k5`.
Supersedes the hybrid "shuttle 30 fps / movement 12 fps" recorded in `30-ml-pipeline.md`,
`shuttle-finetune-plan.md` §2b and `shuttle-tracking.md` (user decision 2026-10-25: full switch once measured).

## Why this exists — the cadence problem
1. **"12 fps" is really 10 fps.** `pipeline/ingest.py` keeps a frame when its PTS reaches
   `last_kept + 1/analysis_fps`, which only ever lands on the next source frame *after* the target instant.
   The real cadence is `source_fps / ceil(source_fps / requested)`, and `frames.json.analysis_fps` records
   the *request*. Replayed from the keep rule (`tests/test_manifest.py`):

   | source fps | requested 12 → actual | requested 30 → actual |
   |---|---|---|
   | 29.97 / 30 (**all Session-2 games**) | **9.99 / 10.00** | 29.97 / 30.00 |
   | 59.94 / 60 | 11.99 / 12.00 | 29.97 / 30.00 |
   | 50 | 10.00 | 25.00 |
   | 25 | 8.33 | 25.00 |

   Only the per-frame `t` values can be trusted. Every "N frames" constant was therefore tuned on a 10 fps
   cadence, i.e. meant `N × 0.1 s`.
2. **The promoted shuttle checkpoint was trained and scored at 30 fps** (game1, `shuttle-tracking.md`
   "Before/after"), while production ran it on the ~10 fps cache; every research dataset runs 25–30 fps
   (`reports/Badminton video analysis AI 2026.md` Risk 1). Symptoms attributed to it: 21–27 % Session-2
   shuttle coverage, 0 of 23 landings located, 1–2-frame swings.
3. **A 30 fps re-ingest of all 5 Session-2 games already happened** (draft PR #19, `worker/.cache-30fps`,
   stub store, no DB writes): coverage 24–31 % vs 21–27 % — with the OLD frame-count thresholds, i.e. the
   movement/rally/landing stages ran with gaps/windows 2.5–3× too short in time. Those frames are
   code-independent and are reused by the bench below (`--reuse-frames`).

## Locked decisions (user, 2026-10-25)
1. **Full switch** (shuttle *and* movement) to 30 fps once the measurement is in — not the hybrid.
2. **Seconds convention = `frames × 0.1 s`** (`manifest.REFERENCE_FPS = 10.0`, the real cadence of the caches
   the values were tuned on). The 12 fps leg of the bench therefore reproduces production exactly; 30 fps
   derives 2.4–3× the frame counts. The nominal `frames / 12` reading was rejected (it would shift the
   baseline by a frame per threshold).
3. **Measure first on one Session-2 game, same code both legs**; the default flips only after the user has
   seen the table. Mixed-cadence caches already exist (game1 at 30 fps), so thresholds come from each
   video's manifest, never from a global switch.
4. **Stroke (stage 7) stays 10 fps-bound.** BST4 is a fixed 16-step model; a seconds window would be 40
   frames it cannot consume. Stroke is not wired into the pipeline; `stroke/dataset._fps_guard` warns on a
   30 fps cache. Retraining is out of scope; only hit recall (stage 6) is measured here.
5. Licence/other research corrections, the ingest keep-rule itself, per-rally medians: out of scope.

## Design — as built
### One cadence helper (`worker/worker/manifest.py`, pure, TDD)
- `measured_fps(frames)` = 1 / median Δt of consecutive `t`; `manifest_fps(manifest)` = measured → recorded
  `cadence_fps`/`analysis_fps` → 12.0 (old hand-written manifests). `fps_from_frames_t((frame, t)…)` for
  detections with gaps (exact over the span). `frames_for_seconds(s, fps, minimum)` (half-up),
  `per_step(rate, fps)`, `seconds_for_frames`. Constants `REFERENCE_FPS = 10.0`, `NOMINAL_FPS = 12.0`.
- Ingest writes `cadence_fps` (measured) next to the requested `analysis_fps` in `frames.json` and logs
  "requested 12.0 → actual 10.00". `enqueue --analysis-fps` help derives from `DEFAULT_ANALYSIS_FPS`.

### Thresholds: durations/rates in, frame counts out (per clip, from the measured cadence)
The old frame-count names stay as module constants = the REFERENCE_FPS equivalents (direct callers and
tests); the stages derive the live values. Old frame-count env names are retired (listed in Config/env).

| Site | Was | Now (env, default) | @10 fps | @30 fps |
|---|---|---|---|---|
| `gate_track` gap fill | `GAP_FILL_MAX_FRAMES` 8 | `GAP_FILL_MAX_S` 0.8 | 8 | 24 |
| `gate_track` smoothing | `SMOOTH_WINDOW` 9 | `SMOOTH_WINDOW_S` 0.9 | 9 | 27 |
| `geometry.stitch_fragments_framewise` boundary medians | `BOUNDARY_K` 6 | `BOUNDARY_S` 0.6 (constant; `boundary_k` kwarg overrides) | 6 | 18 |
| `pipeline/shuttle` spike guard | `SHUTTLE_MAX_STEP` 0.5 /step | `SHUTTLE_MAX_STEP_PER_S` 5.0 | 0.5 | 0.167 |
| `rally/segment` jump | `RALLY_MAX_JUMP` 0.10 /step | `RALLY_MAX_JUMP_PER_S` 1.0 | 0.10 | 0.033 |
| `rally/segment` sustained run | `RALLY_MIN_RUN_LEN` 4 dets | `RALLY_MIN_RUN_S` 0.4 | 4 | 12 |
| `rally/segment` rally floor | `RALLY_MIN_DETS` 5 dets | `RALLY_MIN_DETS_S` 0.5 | 5 | 15 |
| `rally/segment` hit deadband | `RALLY_HIT_DEADBAND` 0.01 /step | `RALLY_HIT_DEADBAND_PER_S` 0.1 | 0.01 | 0.0033 |
| `rally/landing` window cap | `LANDING_MAX_WINDOW_DETS` 8 | `LANDING_MAX_WINDOW_S` 0.8 | 8 | 24 |
| `rally/landing` sparse floor | `LANDING_MIN_DETS` 4 | `LANDING_MIN_DETS_S` 0.4 | 4 | 12 |
| `rally/landing` straddle gap | `LANDING_MAX_END_GAP_FRAMES` 2 | `LANDING_MAX_END_GAP_S` 0.2 (compared on `t`) | 2 | 6 |
| `stroke/features` shuttle gap | `SHUTTLE_MAX_GAP` 6 | `SHUTTLE_MAX_GAP_S` 0.6 (constant) | 6 | 18 |
| `pipeline/detect` BoT-SORT lost-track buffer | `track_buffer` 30 (stock yaml; ultralytics hard-codes `frame_rate=30`) | `TRACK_BUFFER_S` 3.0 → per-cache `botsort_<n>f.yaml` copy of the stock config | 30 | 90 |

Plumbing: `build_rallies(..., fps=None)` (inferred from the detections' (frame, t) span when omitted,
echoed as `fps` in its result; `rally_hit.run` and `eval_rally` pass the manifest cadence) →
`thresholds_for_fps(fps)`; `detect_landing/landing_event(..., fps)` → `landing_counts(fps)`;
`build_sample` reads the cadence from the manifest list it already receives; `eval_shuttle`/`render` use
`manifest_fps` instead of the requested value. **One ordering change:** `detect_landing` now checks the
terminal gap (`straddle`) *before* the window-density test — at 30 fps a > 0.2 s terminal dropout also
thins the 0.5 s window, and "contact fell into a gap" is the honest reason; `located` is unchanged.

### Labels made on one cadence, scored on another
- `detection_corrections.frame_idx` / `corrections.json` are positions in the frames.json they were made
  on; every correction also carries `t_s`. `eval_corrections.remap_frame_idx` re-keys by nearest `t`
  (ties → earlier) before scoring; `full_from_pipeline` reports `frame_idx_remapped_by_t`.
- `rally_corrections` / reviewed `rallies` are seconds — cadence-neutral.
- Stroke labels are `video_id + t_s` — cadence-neutral (used by `eval_rally --hits`).
- `shuttle_labels.json` anchors are frame positions of the cache they were labelled in: label a 30 fps clip
  with `WORKER_FRAME_CACHE_DIR` pointing at the 30 fps cache (game1's labels already are 30 fps).

### Non-destructive bench (`python -m worker bench-fps`)
`worker/worker/bench_fps.py`: runs `[ingest, detect, pose, gate_track, shuttle, rally_hit]` for one video at
a chosen `--analysis-fps` into an isolated `--cache-dir` through a `ReadOnlyStore` (reads from Supabase,
every write a counted no-op; refuses any path named `.cache`). `--reuse-frames` skips ingest when the cache
already holds frames at that fps (PR #19's `worker/.cache-30fps`). Writes `bench.json`: per-stage seconds,
measured cadence, frame count, cache GB, blocked writes. `eval_rally` gained `--landings` (located N/M +
degraded-reason histogram), `--hits` (recall vs stroke labels — see honesty note), `--shuttle-parquet`
(score another tracker's detections) and prints the cadence it used.

### Unchanged on purpose — re-check with the measurement
`SETTLE_KMH` 4.0 (more samples per 0.5 s window, less smoothing → fewer "settled" serves), `SWING_MIN_TLPS`
3.0 / `RALLY_SWING_BRIDGE_MIN_STRENGTH` 4.0 (tuned on 10 fps strength distributions), `insights` p98 top
speed / 15 km/h sprints (per-step jitter ÷ 3× smaller dt — smoothing scales, the percentile does not),
`POSE_MAX_KEYFRAMES_PER_PLAYER` 10000 (in-rally keyframes go native → an 11-min game hits the cap once
in-rally time ≥ ~38 %; the 3.7–9.5 MB payload grows ~2.5× in-rally), `WORKER_STALE_JOB_TIMEOUT_S` 3600
(detect/pose/shuttle take ~2.5× longer; raise only if a measured stage exceeds ~2400 s).

## Measurement protocol (Mac Studio; both legs run the SAME code)
Game: `2026-06-30-session2-game1` (video `9aaa599b-47b5-4487-bdc9-9ad316c05850`, 30 fps source, doubles,
reviewed rallies = GT, 151 stroke labels). Session-2 has **no shuttle labels**, so shuttle F1 is n/a here;
coverage/gap structure is what the shuttle stage can report. `worker/.cache-30fps/<video>/frames` from
PR #19 is reused when present.
```bash
cd worker
G1=9aaa599b-47b5-4487-bdc9-9ad316c05850
uv run python -m worker bench-fps --video-id $G1 --analysis-fps 12 --cache-dir ../worker/.cache-12fps-bench
uv run python -m worker bench-fps --video-id $G1 --analysis-fps 30 --cache-dir ../worker/.cache-30fps --reuse-frames
for C in .cache-12fps-bench .cache-30fps; do
  export WORKER_FRAME_CACHE_DIR=$PWD/../worker/$C
  uv run python -m worker.eval_shuttle --video-id $G1 --tracker tracknetv3          # coverage, gaps, on-court %
  uv run python -m worker.eval_rally --slug 2026-06-30-session2-game1 --landings --hits   # boundary F1, noise, landings, hit recall
  uv run python -m worker.eval_pose_coverage --video-id $G1                          # player (skeleton) coverage
  cat ../worker/$C/$G1/bench.json                                                     # runtime, cadence, disk
done
```
Record every number with its command; `n` beside every rate.

## Results — 12 vs 30 fps, `2026-06-30-session2-game1` (*not run yet*)
| Metric | 12 fps requested (≈10 actual) | 30 fps | source |
|---|---|---|---|
| measured cadence (fps) / frames | *not run yet* | *not run yet* | `bench.json` |
| shuttle coverage % / gaps / longest gap s | *not run yet* | *not run yet* | `eval_shuttle` |
| shuttle F1 @4 px (n) | n/a — no Session-2 labels | n/a (label a 30 fps clip: `verify --shuttle`) | `eval_shuttle --labels` |
| rallies predicted / GT, boundary F1, false rallies, splits, missed, merges | *not run yet* | *not run yet* | `eval_rally` |
| landings located N/M + reasons | *not run yet* | *not run yet* | `eval_rally --landings` |
| hit recall @±0.25 s (n labels), no_hit re-proposed, candidates/label | *not run yet* | *not run yet* | `eval_rally --hits` |
| player coverage (skeleton) | *not run yet* | *not run yet* | `eval_pose_coverage` |
| runtime per stage (s) | *not run yet* | *not run yet* | `bench.json` |
| cache size (GB) | *not run yet* (≈4 expected) | *not run yet* (≈10 expected) | `bench.json` |

**Pre-registered decision rule** (set before the numbers): flip `DEFAULT_ANALYSIS_FPS` to 30 iff shuttle
coverage and landings located do not drop, rally boundary F1 stays within −0.02 and `missed`/`merges` stay 0,
hit recall does not drop, no stage exceeds ~2400 s (else raise `WORKER_STALE_JOB_TIMEOUT_S` first), and the
user accepts ~2.5× runtime and disk. Otherwise stay at 12 and record why. The flip itself is one line
(`ingest.DEFAULT_ANALYSIS_FPS`) plus the disk/runtime notes in `40-worker-contract.md`.

## Honest limits
- A 12-fps cache still exists for every production match; re-processing at 30 fps clobbers trainer rally
  edits (`replace_rallies`) and re-seeds track labels — the bench avoids this by never writing, but a real
  switch needs a re-run plan per match (not in this change).
- Hit *precision* is unmeasurable from the stroke labels (they only judged the old heuristic's candidates);
  `eval_rally --hits` reports recall, the re-proposal rate of the 43 known false positives and
  candidates/label, and says so.
- Stroke stays on the 10 fps cadence it was trained at; a 30 fps default means stage 7 needs a retrain
  before it can be wired (separate work).
- BoT-SORT's per-cache yaml is a copy of ultralytics' stock config; if the yaml helpers move, detect falls
  back to the stock `track_buffer=30` with a warning (never blocks detection).
- The ingest keep rule is unchanged: a requested 12 fps still yields 10 fps on 30 fps sources. It is
  documented, not fixed — the 30 fps default makes it moot on Session-2 footage.

## Config / env
New (durations/rates): `GAP_FILL_MAX_S` 0.8, `SMOOTH_WINDOW_S` 0.9, `SHUTTLE_MAX_STEP_PER_S` 5.0,
`RALLY_MAX_JUMP_PER_S` 1.0, `RALLY_MIN_RUN_S` 0.4, `RALLY_MIN_DETS_S` 0.5, `RALLY_HIT_DEADBAND_PER_S` 0.1,
`LANDING_MAX_WINDOW_S` 0.8, `LANDING_MIN_DETS_S` 0.4, `LANDING_MAX_END_GAP_S` 0.2, `TRACK_BUFFER_S` 3.0.
`SHUTTLE_CONF_THR`: unset → the backend's declared threshold (`tracker.conf_thr`, see
`shuttle-tracknetv5.md`), else 0.5; an explicit value always wins. Retired: `GAP_FILL_MAX_FRAMES`,
`SMOOTH_WINDOW`, `SHUTTLE_MAX_STEP`, `RALLY_MAX_JUMP`, `RALLY_MIN_RUN_LEN`, `RALLY_MIN_DETS`,
`RALLY_HIT_DEADBAND`, `LANDING_MAX_WINDOW_DETS`, `LANDING_MIN_DETS`, `LANDING_MAX_END_GAP_FRAMES`.
`frames.json` gains `cadence_fps`; `processing_jobs.params.analysis_fps` is the request.

## Verification
- `cd worker && uv run pytest` — new: `test_manifest.py` (cadence math incl. the keep-rule replay),
  `test_detect_tracker.py`, `test_bench_fps.py`, `test_eval_rally_cadence.py`; extended: `test_rally.py`
  (`thresholds_for_fps`, fps inference/echo), `test_landing.py` (counts per cadence, fps override, fixtures
  made cadence-consistent), `test_gate_track.py`, `test_shuttle.py` (`resolve_conf_thr`, rate),
  `test_eval_corrections.py` (`remap_frame_idx`). Whole worker suite 680 passed / 15 skipped in the
  container (2026-10-25; av/ultralytics-dependent tests skip there).
- Mac: the protocol above; paste the numbers + commands into the Results table.
