-- 0013_hit_corrections.sql — persist every hit-review trainer edit (add / delete / nudge) as an
-- append-only correction record: the answer-key + training corpus for the LEARNED hit-detector
-- (stroke-classification-v2.md §5 Stage 1 / §9 gate G0). Unlike rally_corrections (which scores a
-- threshold HEURISTIC), these correct the draft hit list a human turns into dense, unbiased
-- per-rally ground truth — the data the GRU/TCN hit-detector trains + evals on.
--
-- Mirrors 0011_rally_corrections.sql exactly: service-role only (RLS, no anon policy — the web
-- never reads this eval corpus); populated automatically from processing_jobs.params.corrections
-- by the worker's `hit_review` stage (service-role); APPEND-ONLY (re-editing the same hit across
-- saves is itself signal); match_id/video_id ON DELETE SET NULL (the correction signal outlives a
-- test match deleted in dev cleanup).
--
-- `op` distinguishes the three edits the hit-editor produces on the `events` hit list
-- (type='hit'), the unit of correction being a single hit at a video time `t_s`:
--   'add'    — a hit the heuristic never proposed; the web INSERTs the events row (see grant
--              below), `event_id` = the new row's id, `new_t_s` = its time, `orig_t_s` NULL.
--   'delete' — a false hit; the web DELETEs the events row, `event_id`/`orig_t_s` = its pre-edit
--              id/time, `new_t_s` NULL. (`hit_detection_eval.parquet`'s no_hit rows already
--              capture this signal from the labeling flow — this is its interactive twin.)
--   'nudge'  — timing off; the web UPDATEs events.t_s, `orig_t_s`→`new_t_s`.
--
-- event_id / rally_id are deliberately NOT foreign keys: a deleted hit's events id is gone by the
-- time this insert runs, and every surviving id goes stale on the next full detect/re-ingest — a
-- correction must outlive the hit it describes (same rationale as rally_corrections' header).

create table if not exists hit_corrections (
  id                 uuid primary key default gen_random_uuid(),
  match_id           uuid references matches(id)         on delete set null,
  video_id           uuid references match_videos(id)    on delete set null,
  job_id             uuid references processing_jobs(id) on delete set null, -- the hit_review job that wrote this row
  op                 text not null check (op in ('add', 'delete', 'nudge')),
  event_id           uuid not null,    -- events.id at edit time (NOT a FK — see header)
  rally_id           uuid,             -- events.rally_id at edit time (context; NOT a FK)
  rally_idx          int,              -- rallies.idx at edit time (human-readable order)
  orig_t_s           double precision, -- before-state (delete / nudge); NULL for add
  new_t_s            double precision, -- after-state  (add / nudge);   NULL for delete
  orig_confidence    double precision, -- the draft hit's confidence pre-edit (delete / nudge)
  orig_payload       jsonb,            -- the events.payload snapshot pre-edit (delete / nudge)
  created_at         timestamptz not null default now()
);

create index if not exists hit_corrections_match_idx on hit_corrections(match_id);
create index if not exists hit_corrections_video_idx on hit_corrections(video_id);

-- Lock it down: service-role (worker) bypasses RLS; no anon/authenticated policy — the web never
-- reads this offline evaluation/training data.
alter table hit_corrections enable row level security;

-- ── DEV ONLY: anon INSERT on events for /match/:slug/hits "add missed hit", SCOPED to is_test. ──
-- 0008 granted anon UPDATE/DELETE on events (nudge = UPDATE t_s, delete = DELETE row) and 0009 the
-- SELECT policy those WHERE clauses need; only INSERT ("add a hit the heuristic missed") is new.
-- A manually-added hit is a minimal events row (match_id, t_s, type='hit'); payload is NULL (no
-- CV-derived from/to/shot_type — downstream tolerates it). Same is_test guard, same REMOVE-before-
-- real-sharing caveat as 0008 (route through a service_role edge function for real auth).
grant insert on events to anon;

drop policy if exists dev_anon_event_insert on events;
create policy dev_anon_event_insert on events for insert to anon
  with check (exists (select 1 from matches m where m.id = events.match_id and m.is_test));
