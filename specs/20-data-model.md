# 20 — Data Model

Postgres on self-hosted Supabase. The schema is the **contract between worker and web** and mirrors the
frontend shapes in `handovers/**/mock-data.js`. Implemented by `supabase/migrations/0001_init.sql` (keep in
sync). Incorporates lessons adopted from the prior v1 schema (see "v1 lessons" at the end). ids are `uuid`
(`gen_random_uuid()`), timestamps `timestamptz default now()`.

## Conventions
- **Canonical storage, localize in the UI.** Store neutral/English values; the web maps to German:
  `discipline ∈ {singles,doubles}` → Einzel/Doppel; `result_side_a ∈ {win,loss}` → Sieg/Niederlage;
  `shot_type ∈ {clear,smash,drop,net,drive,lift,serve,unknown}` → German labels (`web/src/lib/labels.ts`).
- **Court coords** normalized `u,v ∈ [0,1]` (u left→right, v far→near baseline). Never store pixel coords on web-facing rows.
- **Heatmaps** = gaussian-kernel list `heat=[{x,y,w,s}]` (frontend `CourtHeatmap` shape); optional raster in `heatmap_uri`/`placement_uri`.
- **Shot distribution** = `player_shot_stats` rows → web builds `shots:[[shot_type,count]]` (sorted desc).
- **Units in names** (`*_s`, `*_m`, `*_pct`, `reprojection_error_m`); ML-derived rows carry `confidence`.
- `is_test` separates demo/dev data from real matches.

## Tables

### players
`id · name · first_name · email? · consent_date date?  (GDPR) · is_active bool · created_at`

### matches
`id · slug unique · played_on date · weekday · discipline∈{singles,doubles} · court int · score text ·
games jsonb [[a,b]…] · duration_min · rest_min · result_side_a∈{win,loss} ·
status∈{uploaded,calibrating,processing,awaiting_mapping,review,done,failed} ·
processed_at? · error? · notified_at?  (report link sent) · is_test · created_at`

### match_videos
`id · match_id → matches · storage_path · label · angle∈{overhead,side} · duration_s · fps · width · height · created_at`

### court_calibrations  (first-class state machine — decision #4, FR-5/6/10)
`id · video_id? · match_id? · court int? · status∈{proposed,needs_review,accepted,finalized,rejected} ·
source · frame_uri · frame_s · image_size jsonb · proposed_points jsonb · confirmed_points jsonb ·
mask_polygon_px jsonb  (in-court gating polygon) · homography jsonb · reprojection_error_m · confidence jsonb ·
auto_accepted bool · created_at · updated_at`

### match_players
`id · match_id · player_id · side∈{A,B} · slot int · track_label int?  (confirmed detected-track→player, FR-35)` — unique (match_id, side, slot)

### player_match_stats  → report `players[].stats`
`id · match_id · player_id · distance_m · avg_kmh · top_kmh · sprints · rest_time_s? ·
zones jsonb {front,mid,rear,left,right} · timeline jsonb [int…] · heat jsonb [{x,y,w,s}] ·
heatmap_uri? · placement_uri? · fastest?  (ESTIMATE)` — unique (match_id, player_id)

### player_shot_stats
`id · match_id · player_id · shot_type (canonical) · count · share_pct? · winners? · errors? · placements jsonb?` — unique (match_id, player_id, shot_type)
> `errors` is sourced from rally-ending **out**-landings per shot_type (`features/shuttle-landing-inout.md`,
> SPEC); the web sums them into „Aus-Schläge". An **estimate** — partial coverage (occlusion/sampling), a lower bound.

### pair_match_stats  (doubles)
`id · match_id · side∈{A,B} · label · fb · ss · rot` — unique (match_id, side)

### rallies
`id · match_id · idx · t_start_s · t_end_s · shot_count · winner_side? · end_shot_type? · confidence?` — unique (match_id, idx)
> Written by stage 6 (`rally_hit`, image-space segmentation). `winner_side`/`end_shot_type` are **estimates**
> (stage 6 / landing), **decoupled from `matches.score`** — never reconciled against the human-entered score.
> `shot_count` is a provisional reversal count (serve + contacts); `confidence` = in-rally detection density.

### events  (generic typed stream — raw hits/serves/bounces/rally markers)
`id · match_id · rally_id? · t_s · type ∈ {serve,hit,rally_end,landing,bounce,…} · payload jsonb · confidence?`
- `hit` payload `{player_id, shot_type, speed_kmh, from:[u,v], to:[u,v]}`.
- Stage 6 (`rally_hit`) emits `serve` `{settled:bool}`, provisional `hit` `{provisional:true}`, and
  `rally_end` `{x,y,u,v}` (image + court floor-contact candidate); stage 7 refines `hit.shot_type`.
- `landing` (SPEC — `features/shuttle-landing-inout.md`) `{located,call:in|out|uncertain,line,u?,v?,px?,py?,margin_cm?,band_cm,confidence}`
  — the rally-ending floor contact as an **estimate** („Schätzung"), never the match score (`'bounce'` reserved for raw mid-rally touches).

### track_keyframes  (overlay, ~10 Hz)
`id · video_id · subject ('player:<id>'|'shuttle') · data jsonb [{t,u,v}]` — unique (video_id, subject)

### processing_jobs  (work queue — claimed with FOR UPDATE SKIP LOCKED)
`id · match_id · video_id? · stage · status∈{queued,running,succeeded,failed,cancelled} · progress ·
params jsonb? · error? · claimed_by? · claimed_at? · started_at? · finished_at? · created_at`

### job_runs  (audit/observability log)
`id · job · match_id? · started_at · finished_at? · is_ok? · detail?`

### admin_users / deletion_log  (auth + GDPR)
`admin_users: user_id uuid (=auth.users.id) · created_at` ·
`deletion_log: id · player_name · requested_at · completed_at? · detail jsonb?  (FR-36)`

## Views
### match_rally_stats  (`security_invoker`)
Per match → `{count, avg_shots, avg_sec, long_shots, long_sec}` (duration derived from `t_end_s - t_start_s`).
`endLine` (ending-shot % summary) is computed in the app from `rallies.end_shot_type`. `outLine`
(out-landing % summary, „Aus-Schläge") is likewise **app-derived** from `events.type='landing'` (SPEC) —
neither is a stored column.

## RLS
- `service_role` (worker) bypasses RLS.
- `public.is_admin()` = `auth.uid() ∈ admin_users`. Policy `admin_all` grants authenticated **admins** full read/write on all tables.
- Anonymous read: **not** in `0001`. `0002_dev_anon_read.sql` (DEV ONLY) grants anon SELECT on report tables so the
  seeded demo renders before the auth UI exists; replaced by token-scoped player-link RLS in P4 (`features/sharing.md`).
- Privileges: anon SELECT, authenticated/service_role full (RLS still gates rows).

## Frontend mapping (report object the web assembles — `useMatchReport`)
```
report = {
  video:   match_videos[0] → { src(signed url), label },
  meta:    matches (+ dateParts, derived resultLine; discipline/result localized at render),
  rallies: match_rally_stats (+ endLine from rallies.end_shot_type),
  pairs:   pair_match_stats[] | null,
  players: match_players ⋈ players ⋈ player_match_stats ⋈ player_shot_stats,
}
```
`isViewer`/tab order come from the personal link, not stored on the row.

## v1 lessons adopted
Dedicated `court_calibrations` state machine + `mask_polygon_px` (vs. a calibration blob); `match_players.track_label`;
richer `matches.status` + `is_test`/`processed_at`/`notified_at`; richer `player_shot_stats` (winners/errors/placements);
GDPR `consent_date` + `deletion_log`; `admin_users` + `is_admin()` RLS; generic `events` stream + `job_runs` audit;
canonical storage + UI localization; unit-suffixed names + `confidence`. **Not** adopted: v1's bigint ids, enum types,
m/s speeds (we keep uuid, text+check, km/h).
