-- 0001_init.sql — Badminton Analytics initial schema (v2)
-- Implements specs/20-data-model.md. Incorporates lessons from the prior v1 schema:
-- dedicated court_calibrations state machine, match_players.track_label, richer status + is_test,
-- richer player_shot_stats, GDPR (consent_date + deletion_log), admin_users + is_admin() RLS,
-- a generic events stream + job_runs audit log. Canonical values stored (English); UI localizes.
-- Target: self-hosted Supabase (Coolify). Apply via Supabase MCP / CLI / psql.

create extension if not exists pgcrypto;  -- gen_random_uuid()

-- ─────────────────────────────────────────── core entities ───────────────────────────────────────────

create table if not exists players (
  id           uuid primary key default gen_random_uuid(),
  name         text not null,
  first_name   text not null,
  email        text,
  consent_date date,                              -- GDPR consent (esp. minors); null = not given
  is_active    boolean not null default true,
  created_at   timestamptz not null default now()
);

create table if not exists matches (
  id             uuid primary key default gen_random_uuid(),
  slug           text not null unique,
  played_on      date not null,
  weekday        text,
  discipline     text not null check (discipline in ('singles','doubles')),  -- canonical; UI → Einzel/Doppel
  court          int,
  score          text,
  games          jsonb not null default '[]',     -- [[a,b],...]
  duration_min   int,
  rest_min       int,
  result_side_a  text check (result_side_a in ('win','loss')),               -- canonical; UI → Sieg/Niederlage
  status         text not null default 'uploaded'
                 check (status in ('uploaded','calibrating','processing','awaiting_mapping','review','done','failed')),
  processed_at   timestamptz,
  error          text,
  notified_at    timestamptz,                      -- when the player report link was sent (FR-33)
  is_test        boolean not null default false,   -- demo/dev separation
  created_at     timestamptz not null default now()
);

create table if not exists match_videos (
  id           uuid primary key default gen_random_uuid(),
  match_id     uuid not null references matches(id) on delete cascade,
  storage_path text not null,
  label        text,
  angle        text not null default 'overhead' check (angle in ('overhead','side')),
  duration_s   numeric,
  fps          numeric,
  width        int,
  height       int,
  created_at   timestamptz not null default now()
);

-- Court calibration as a first-class state machine (decision #4 / FR-5/FR-6/FR-10). Adopted from v1.
create table if not exists court_calibrations (
  id                   uuid primary key default gen_random_uuid(),
  video_id             uuid references match_videos(id) on delete cascade,
  match_id             uuid references matches(id) on delete cascade,
  court                int,                          -- physical court # (lets a calibration be reused per setup)
  status               text not null default 'proposed'
                       check (status in ('proposed','needs_review','accepted','finalized','rejected')),
  source               text,                         -- 'auto' | 'manual' | model name
  frame_uri            text,                         -- the still used for calibration
  frame_s              real,
  image_size           jsonb,                        -- {w,h}
  proposed_points      jsonb,                        -- auto-detected 4 corners [[x,y]x4] (image px)
  confirmed_points     jsonb,                        -- human-corrected corners
  mask_polygon_px      jsonb,                        -- in-court polygon for player gating (FR-10), image px
  homography           jsonb,                        -- 3x3 image->metric-court
  reprojection_error_m real,                         -- acceptance metric
  confidence           jsonb,
  auto_accepted        boolean not null default false,
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

create table if not exists match_players (
  id          uuid primary key default gen_random_uuid(),
  match_id    uuid not null references matches(id) on delete cascade,
  player_id   uuid not null references players(id),
  side        text not null check (side in ('A','B')),
  slot        int  not null default 0,
  track_label int,                                   -- confirmed detected-track id → this player (review/FR-35)
  unique (match_id, side, slot)
);

-- ─────────────────────────────────────────── per-player stats ────────────────────────────────────────

create table if not exists player_match_stats (
  id           uuid primary key default gen_random_uuid(),
  match_id     uuid not null references matches(id) on delete cascade,
  player_id    uuid not null references players(id),
  distance_m   int,
  avg_kmh      numeric(4,1),
  top_kmh      numeric(4,1),
  sprints      int,
  rest_time_s  real,                                 -- per-player rest/idle (v1)
  zones        jsonb,                                -- {front,mid,rear,left,right} ints, %
  timeline     jsonb,                                -- [int,...] distance per 5-min bucket
  heat         jsonb,                                -- [{x,y,w,s}] movement-heatmap kernels
  heatmap_uri  text,                                 -- optional movement raster (FR-11)
  placement_uri text,                                -- optional shot-placement raster (v1)
  fastest      int,                                  -- estimated fastest shuttle speed (km/h) — ESTIMATE
  unique (match_id, player_id)
);

create table if not exists player_shot_stats (
  id         uuid primary key default gen_random_uuid(),
  match_id   uuid not null references matches(id) on delete cascade,
  player_id  uuid not null references players(id),
  shot_type  text not null,                          -- canonical: clear|smash|drop|net|drive|lift|serve|unknown
  count      int  not null default 0,
  share_pct  real,
  winners    int,
  errors     int,
  placements jsonb,                                  -- placement distribution for this shot type
  unique (match_id, player_id, shot_type)
);

create table if not exists pair_match_stats (
  id        uuid primary key default gen_random_uuid(),
  match_id  uuid not null references matches(id) on delete cascade,
  side      text not null check (side in ('A','B')),
  label     text,
  fb        int,   -- front-back % of active time
  ss        int,   -- side-by-side %
  rot       int,   -- rotation count
  unique (match_id, side)
);

-- ─────────────────────────────────────────── events & rallies ────────────────────────────────────────

create table if not exists rallies (
  id            uuid primary key default gen_random_uuid(),
  match_id      uuid not null references matches(id) on delete cascade,
  idx           int  not null,
  t_start_s     real,
  t_end_s       real,
  shot_count    int,
  winner_side   text check (winner_side in ('A','B')),
  end_shot_type text,                                -- canonical shot type
  confidence    real,
  unique (match_id, idx)
);

-- Generic typed event stream (adopted from v1) — raw hits/serves/bounces/rally markers etc.
create table if not exists events (
  id         uuid primary key default gen_random_uuid(),
  match_id   uuid not null references matches(id) on delete cascade,
  rally_id   uuid references rallies(id) on delete set null,
  t_s        real not null,
  type       text not null,                          -- 'hit'|'serve'|'bounce'|'rally_start'|'rally_end'|...
  payload    jsonb,                                  -- e.g. hit: {player_id, shot_type, speed_kmh, from:[u,v], to:[u,v]}
  confidence real
);

create table if not exists track_keyframes (
  id        uuid primary key default gen_random_uuid(),
  video_id  uuid not null references match_videos(id) on delete cascade,
  subject   text not null,                           -- 'player:<player_id>' | 'shuttle'
  data      jsonb not null,                          -- [{t,u,v}] court-normalized (~10 Hz); shuttle may add visibility
  unique (video_id, subject)
);

-- ─────────────────────────────────────────── work queue + audit ──────────────────────────────────────

create table if not exists processing_jobs (
  id          uuid primary key default gen_random_uuid(),
  match_id    uuid not null references matches(id) on delete cascade,
  video_id    uuid references match_videos(id) on delete cascade,
  stage       text,
  status      text not null default 'queued'
              check (status in ('queued','running','succeeded','failed','cancelled')),
  progress    numeric not null default 0,
  params      jsonb,
  error       text,
  claimed_by  text,
  claimed_at  timestamptz,
  started_at  timestamptz,
  finished_at timestamptz,
  created_at  timestamptz not null default now()
);

-- Lightweight run-history/audit log (adopted from v1) for observability across worker runs.
create table if not exists job_runs (
  id          uuid primary key default gen_random_uuid(),
  job         text not null,
  match_id    uuid references matches(id) on delete set null,
  started_at  timestamptz not null default now(),
  finished_at timestamptz,
  is_ok       boolean,
  detail      text
);

-- ─────────────────────────────────────────── admin + GDPR ────────────────────────────────────────────

create table if not exists admin_users (
  user_id    uuid primary key,                       -- references auth.users(id)
  created_at timestamptz not null default now()
);

create table if not exists deletion_log (
  id           uuid primary key default gen_random_uuid(),
  player_name  text not null,
  requested_at timestamptz not null default now(),
  completed_at timestamptz,
  detail       jsonb
);

-- ─────────────────────────────────────────── indexes ─────────────────────────────────────────────────

create index if not exists idx_match_videos_match  on match_videos(match_id);
create index if not exists idx_calib_video         on court_calibrations(video_id);
create index if not exists idx_calib_match         on court_calibrations(match_id);
create index if not exists idx_match_players_match  on match_players(match_id);
create index if not exists idx_pms_match           on player_match_stats(match_id);
create index if not exists idx_pss_match           on player_shot_stats(match_id);
create index if not exists idx_pairs_match         on pair_match_stats(match_id);
create index if not exists idx_rallies_match       on rallies(match_id);
create index if not exists idx_events_match        on events(match_id);
create index if not exists idx_events_rally        on events(rally_id);
create index if not exists idx_track_video         on track_keyframes(video_id);
create index if not exists idx_jobs_status_created on processing_jobs(status, created_at);

-- ─────────────────────────────────────────── rally aggregate view ────────────────────────────────────
-- Feeds the report's `rallies` object. Duration derived from t_end_s - t_start_s. endLine computed in app.

drop view if exists match_rally_stats;
create view match_rally_stats with (security_invoker = true) as
with r as (
  select match_id, shot_count, greatest(coalesce(t_end_s,0) - coalesce(t_start_s,0), 0) as dur_s
  from rallies
),
agg as (
  select match_id, count(*)::int as count,
         round(avg(shot_count)::numeric, 1) as avg_shots,
         round(avg(dur_s)::numeric, 1)      as avg_sec
  from r group by match_id
),
longest as (
  select distinct on (match_id) match_id, shot_count as long_shots, round(dur_s::numeric)::int as long_sec
  from r order by match_id, shot_count desc nulls last, dur_s desc nulls last
)
select a.match_id, a.count, a.avg_shots, a.avg_sec, l.long_shots, l.long_sec
from agg a left join longest l using (match_id);

-- ─────────────────────────────────────────── RLS ─────────────────────────────────────────────────────
-- service_role (worker) bypasses RLS. Writes/reads for authenticated ADMINS only (admin_users via is_admin()).
-- Anonymous/player-link read is added separately (0002 dev shortcut now; token-scoped in P4 sharing.md).

create or replace function public.is_admin() returns boolean
  language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.admin_users where user_id = auth.uid());
$$;

do $$
declare t text;
begin
  foreach t in array array[
    'players','matches','match_videos','court_calibrations','match_players','player_match_stats',
    'player_shot_stats','pair_match_stats','rallies','events','track_keyframes',
    'processing_jobs','job_runs','admin_users','deletion_log'
  ] loop
    execute format('alter table %I enable row level security;', t);
    execute format('drop policy if exists admin_all on %I;', t);
    execute format(
      'create policy admin_all on %I for all to authenticated using (public.is_admin()) with check (public.is_admin());', t);
  end loop;
end $$;

-- Privileges (RLS still gates rows). anon gets SELECT only; policies decide visibility.
grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update, delete on all tables in schema public to authenticated, service_role;
grant select on all tables in schema public to anon;
grant execute on function public.is_admin() to anon, authenticated, service_role;
