# Feature: Player-Identity Review (`/match/:slug/review`)

Status: **built** (branch `claude/player-identity-review-i9sk5g`). The human-in-the-loop step
the pipeline already expects: the worker ends the calibrate-resume slice at `match.status='review'`
(`worker/worker/__main__.py`), and `gate_track._auto_map` only *estimates* the track→player mapping
from court-position role. This screen lets the trainer confirm or correct it, then mark the match done.
Implements the identity-mapping half of **FR-35**.

## Contract spine (the load-bearing decision)
`match_players` holds one row per `(match_id, side, slot)` (unique), each carrying `player_id` (the
registered player) and `track_label` (the detected track id, auto-seeded by role via `ROLE_TO_SIDESLOT`).
The mapping `track_label → player_id` is the contract between worker output and the report:
- `aggregate` writes `player_match_stats` per binding (track rows of `track_label` → that `player_id`).
- `overlay` writes `track_keyframes` keyed `subject = 'player:<player_id>'` from the same binding.

So a track's path is shown via the keyframes of the player it is *currently* bound to, and the
"available detected tracks" = the set of `track_label`s present across the match's rows. **Any** change
to `track_label` or `player_id` ⇒ stats + overlays are stale ⇒ recompute. A pure confirm needs none.

The common real correction is a **swap**: `geometry.role_assign` put the wrong person near/far in
singles, or swapped doubles quadrants. The trainer fixes which registered player and/or which detected
track sits in each side/slot.

## Status flow
`review` → Save:
- **pure confirm** (no row changed) ⇒ `matches.status='done'`, no job.
- **any change** ⇒ write the changed `match_players` rows, `matches.status='processing'`, enqueue
  `processing_jobs{stage:'remap'}`; the worker runs `[remap, aggregate, overlay]` → `done`.

`matches.status` enum already includes `review` and `done` (`0001_init.sql`).

## Worker — `remap` resume stage (`worker/worker/pipeline/remap.py`, registered in `_REAL`)
`aggregate`/`overlay` both require `ctx["tracks_path"]`, which a remap job (no `gate_track`) lacks.
`remap.run` resolves it cheaply — **no re-detect**:

| stage | reads | writes |
|---|---|---|
| `remap` | cached `tracks.parquet`, else S3 `analysis-artifacts/<video_id>/tracks.parquet` | sets `ctx["tracks_path"]` (downloads to the frame cache on a miss) |
| `aggregate` | tracks, edited `match_players` | recomputed `player_match_stats` |
| `overlay` | tracks, edited `match_players` | recomputed `track_keyframes` (`player:<id>`) |

`__main__._slice_for`: `stage=='remap' → (['remap','aggregate','overlay'], 'done')`.
Persistence assumption: `gate_track` both writes `tracks.parquet` to the frame cache **and** uploads it
to the artifact bucket, so remap is robust even on a fresh worker as long as the S3 artifact persists.
Only `tracks.parquet` is needed — aggregate/overlay never read frame JPEGs (overlay's shuttle read is
optional).

## Web (`pages/PlayerReview.tsx` + `components/review/*`, mirrors `Calibrate.tsx`)
- Route `/match/:slug/review` next to `/calibrate`.
- **Load**: `matches` by slug (`id,status,discipline`), `match_videos`, latest `court_calibrations`
  (`confirmed_points,image_size,frame_uri`), `match_players` joined to `players(name,first_name)`
  (same select shape as `useMatchReport`), `track_keyframes` where `subject like 'player:%'`. `H` from
  the confirmed corners via `lib/calibration.homographyFromCorners`; still frame via
  `lib/storage.signedUrl`. Every null/empty shape guarded.
- **UI** (`PlayerReviewBoard`): per-row card `[player name · side · slot · current track]` with two
  `<select>`s — registered player (the match's players) and detected track (the `track_label`s, plus „—").
  A read-only frame overlay (`PlayerReviewOverlay`) highlights the focused row's path, reusing
  `lib/homography.projectUV` + `lib/overlay.trackAt` (the report overlay engine, static here). German
  labels, light mode, brand tokens. „Bestätigen" (confirm, no change) + „Speichern" (save).
- **Save**: diff rows vs initial; `update` each changed `match_players` row; flip `matches.status`;
  enqueue `remap` only on a change; `navigate('/match/:slug')`.

## RLS (`supabase/migrations/0006_dev_review_write.sql`) — ⚠️ DEV ONLY
anon already has is_test-scoped `update` on `matches` and `insert/update` on `processing_jobs` (0003).
0006 adds the one missing grant: `update` on `match_players` + `dev_anon_review_update` policy gated on
the parent match being `is_test`. Replace with admin/service-role + player-link RLS before real sharing.

## Acceptance
- web: `npm run typecheck && npm test && npm run build` green; `PlayerReviewBoard` render smoke test
  (populated + empty/null fixtures) passes.
- worker: `pytest` green; `_slice_for('remap') == (['remap','aggregate','overlay'], 'done')`; existing
  slices unchanged; `remap.run` resolves the cached parquet (and falls back to S3 download).
- manual: open `/match/<test-slug>/review`, swap a mapping, Save → `match_players` updated,
  `matches.status` flipped, a `processing_jobs{stage:'remap'}` row enqueued; the report reflects the new
  mapping after the worker runs.

## Honest limits
Doubles per-player identity through occlusion is an estimate (auto-mapped by court side/role); this
screen is exactly where the trainer corrects it. Dev anon flow only (no auth yet). The recompute reuses
the cached `tracks.parquet` — it never re-detects, so it cannot fix a *detection/tracking* error, only the
identity binding. Rally drop/merge and shot relabel (the other halves of FR-35) are out of scope.
