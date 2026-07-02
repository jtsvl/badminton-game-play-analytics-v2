-- 0011_rally_corrections.sql — persist every rally-review trainer edit (drop/merge) as an
-- append-only correction record: the future answer-key/eval corpus for scoring stage 6
-- (`rally_hit`)'s segmentation HEURISTIC (module-level constants in worker/rally/segment.py —
-- RALLY_GAP_S/RALLY_BRIDGE_S/RALLY_MIN_COS/RALLY_SETTLE_KMH etc. — NOT a trained model, so this
-- is eval/threshold-tuning ground truth, not NN supervision).
--
-- Mirrors 0004_detection_corrections.sql's access pattern (service-role only, RLS with no anon
-- policy — the web never reads/writes this) but is populated AUTOMATICALLY: the web packs the
-- pre-edit snapshot + save plan into processing_jobs.params.corrections, a payload it already
-- sends on every rally-review save, and the worker's rally_sync stage (service-role, already
-- running on every such job) writes the rows here. No manual CLI step, unlike
-- detection_corrections' offline `worker verify` -> corrections.json -> import-corrections flow.
--
-- APPEND-ONLY (unlike detection_corrections'/shuttle_labels' upsert-on-conflict): editing the
-- same rally across separate review saves is itself signal, so repeats get their own row.
--
-- match_id/video_id use ON DELETE SET NULL (mirrors job_runs' audit-log precedent, not
-- detection_corrections' cascade) — this table's value is the accumulating correction signal,
-- which stays meaningful even if a test match is later deleted during routine dev cleanup.
--
-- `op` distinguishes the three row roles a review save produces (the client-side
-- RallySavePlan.rallyDeletes conflates the first two via its deleteEvents flag):
--   'drop'           — a whole rally judged noise (deleteEvents:true); gone from `rallies`.
--   'merge_absorbed' — a rally folded into a neighbour (deleteEvents:false, already re-parented);
--                      gone from `rallies`. `merged_into` names the surviving rally's id AT EDIT TIME.
--   'merge_keep'     — the surviving rally; `patch` carries exactly what was written (RallyPatch).
--
-- rally_id/merged_into are deliberately NOT foreign keys: a dropped/absorbed row's `rallies` id is
-- gone by the time this insert runs, and even a surviving row's id goes stale on the next full
-- stage-6 re-run (replace_rallies is delete-then-insert, fresh ids) — a correction must outlive
-- the rally it describes.

create table if not exists rally_corrections (
  id                  uuid primary key default gen_random_uuid(),
  match_id            uuid references matches(id) on delete set null,
  video_id            uuid references match_videos(id) on delete set null,
  job_id              uuid references processing_jobs(id) on delete set null, -- the rally_review job that wrote this row
  op                  text not null check (op in ('drop', 'merge_absorbed', 'merge_keep')),
  rally_id            uuid not null,    -- rallies.id at edit time (NOT a FK — see header)
  rally_idx           int,              -- rallies.idx at edit time (human-readable order; gaps allowed)
  merged_into         uuid,             -- op='merge_absorbed' only: the surviving rally's id at edit time
  orig_t_start_s      double precision, -- before-state (all ops) — the RallyReviewRow snapshot pre-edit
  orig_t_end_s        double precision,
  orig_shot_count     int,
  orig_winner_side    text,
  orig_end_shot_type  text,
  orig_confidence     double precision,
  patch               jsonb,            -- op='merge_keep' only: the RallyPatch actually written
  created_at          timestamptz not null default now()
);

create index if not exists rally_corrections_match_idx on rally_corrections(match_id);
create index if not exists rally_corrections_video_idx on rally_corrections(video_id);

-- Lock it down: service-role (worker) bypasses RLS; no anon/authenticated policy is granted, so
-- the web never sees this offline evaluation data.
alter table rally_corrections enable row level security;
