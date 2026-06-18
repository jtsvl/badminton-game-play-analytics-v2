# Feature: P1 — CV vertical (calibration + movement)

Status: **built + validated on real footage** (branch `feat/p1-calibration`); accurate full-clip numbers
pending the trainer's real corner calibration. Replaces the seeded dashboard with **real computed insights**
without changing the data contract (specs/20-data-model.md) — the web is the acceptance target.

## Contract spine (the load-bearing decision)
The 4 `court_calibrations.confirmed_points` (image pixels, order **FL, FR, NR, NL**) + `image_size {w,h}`
are the single source of truth. `(u,v)`: u left→right ∈[0,1], v far(0)→near(1) baseline.
- **Worker** derives image-px→(u,v) via `geometry.homography_from_corners` (pure DLT, identical to the
  frontend's `solveHomography`) and persists it in `court_calibrations.homography`; used for ankle
  projection + court gating.
- **Web** derives court→frame-fraction by reusing `solveHomography` (`lib/calibration.homographyFromCorners`)
  on the same corners — no inversion, no unit mismatch. The frontend never reads the worker's matrix.
- `track_keyframes.data` t is in **seconds** @ ~10 Hz. `VideoOverlay` nearness term = `clamp(v)`.

## Worker stages (worker/worker/pipeline/, registered in `_REAL`)
| stage | reads | writes |
|---|---|---|
| `ingest` | `match-videos/<path>` | decode-once JPEG frame cache + `frames.json` (PTS→t); `match_videos` fps/dims/duration; calibration still→`analysis-artifacts`; seeds `court_calibrations` (status `needs_review`); match→`calibrating` |
| `calibrate` | `confirmed_points` | `homography`, `mask_polygon_px`, `reprojection_error_m`, status `finalized` |
| `detect` | frame cache | `detections.parquet` (frame,track,x,y,w,h,score) px — YOLO11 `yolo11m` on MPS + BoT-SORT; one-time MPS-NaN→CPU fallback |
| `gate_track` | detections, H | `tracks.parquet` (t,track_id,u,v,role) in-court only — ankle=bbox bottom-center → H → court-polygon gate (+margin) → `enforce_n_tracks` → `role_assign` → `smooth_track`; auto-maps `match_players.track_label` by role |
| `aggregate` | tracks, match_players | `player_match_stats` (reuses `insights.py`; `net_depth` for zones; `fastest`=null) |
| `overlay` | tracks | `track_keyframes` (subject `player:<id>`, ~10 Hz) |

Gates honored by `__main__._slice_for(job.stage)`: `ingest`→[ingest]→`calibrating`; `calibrate`→[calibrate,
detect,gate_track,aggregate,overlay]→`review`. The trainer confirms corners between the two jobs.
`SUPABASE_DB_URL` is blank (Coolify API-only) → all DB I/O via supabase-py REST (`worker/worker/store.py`).

## Two correctness fixes (TDD'd in `worker/tests/`)
- `geometry.net_depth(v, role)` — `insights.zones` assumed depth `0=net..1=baseline`, but canonical v=0 is
  the FAR baseline and the net is at v=0.5; raw v gives meaningless front/rear. Aggregate transforms first.
- `geometry.smooth_track` — damps ankle-jitter that inflated `top_kmh` to ~46 km/h (→ ~19 km/h plausible).

## Web
- `/match/:slug/calibrate` (`pages/Calibrate.tsx` + `components/calibrate/*`): drag 4 corners over the still,
  live-projected BWF court template (`lib/courtTemplate.ts`), convexity guard, a **drag magnifier** (zoomed
  loupe + crosshair + projected grid, in the corner opposite the handle) for precise corner placement,
  Save → `confirmed_points` + match `processing` + enqueue `calibrate` job.
- Demo→real swap: `lib/storage.resolveVideoSrc` async signed URLs; `lib/calibration.overlayInputs` (real H +
  `track_keyframes`, demo fallback); `VideoOverlay` consumes H/tracks props, real seconds, N players, no
  shuttle; `useMatchReport` fetches calib+tracks and threads `ReportPlayer.playerId`.
- `supabase/migrations/0003_dev_calibration_write.sql` — DEV-ONLY anon read of `court_calibrations` +
  **is_test-scoped** writes + is_test storage read. Replace with player-link RLS before real sharing.

## Acceptance (measured on `doubles_match.MOV`, 4K/59.94fps/29.7s, slug `2026-06-14-doppel`)
- detect 18 raw tracks → gate_track **exactly 4 in-court** (drops adjacent-court people), quadrant roles.
- `player_match_stats`: zones sum ≈100 (depth + lateral), timeline len = ceil(dur/5), 4 heat kernels,
  `fastest`=null, top_kmh ~7–19 km/h (plausible). `track_keyframes` 10 Hz, u,v∈[0,1].
- web: typecheck + 36 tests + build green; worker: 39 tests green.

## Honest limits
Doubles per-player identity through occlusion = estimate (auto-mapped by court side/role, trainer-confirms).
DJI/lens undistortion is a no-op in P1 (rectilinear assumed; `frames.json.undistorted=false`). `fastest` +
shot-based rallies deferred to P2 (no shuttle stage). Validated run used a 15s capped cache + placeholder
corners; full accuracy needs the trainer's real calibration + a full ingest.
